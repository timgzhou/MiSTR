"""Frozen Tessera-v2 embeddings of PASTIS for one configuration -> cached 128-d maps + cost.

    python -m exp.tessera.embed --config sea>mon-b1-attn [--dataset pastis|biomassters] [--splits ...] [--limit N]

Configurations
  fix-<level>                    one pass on <level> tokens (sea | bimon | mon | semi | raw | uniK)
  <c>><f>-b<n>-<pick>[-joint]    MiSTR, time only, per pixel: a coarse pass on level <c>,
                                 then each pixel re-encodes n of its coarse windows (seasons,
                                 2-month, months or half-months for c = sea / bimon / mon / semi) with the level-<f> tokens inside
                                 them (the fine tokens).
      n      windows per pixel; a fractional n (1.5) gives floor(n) windows to every pixel and
             one more to the frac(n) share of pixels with the highest entropy of the coarse
             model's prediction (the fix-<c> head, exp.tessera.head); the threshold is that
             quantile on val (on train for the train split), as OlmoEarth MiSTR's 1.5 seasons
      pick   attn     the windows the coarse pass's pool attends to most (S2 + S1 pool
                      weights summed per window) -- MiSTR's time selection
             tattn    the same with task-tuned pool queries (exp.tessera.pooltune; frozen
                      encoder, only the queries + a per-pixel head trained on the coarse pass);
                      selection only, the embedding keeps the pretrained pool
             low      the least attended (control)        rand   uniformly at random
             fix<j>   window j for every pixel             fixattn  the n windows with the
                      highest mean attention over the train split (one ranking for all pixels)
      n < 1  spatial selection too: only the frac(n) most uncertain pixels re-encode a window
             (b0.25 = top 25 % of pixels by coarse-prediction entropy, global val threshold)
      rpx    the extra window goes to random pixels instead (control for the entropy pick)
      joint  re-encode [coarse tokens of the other windows, fine tokens] as one sequence
             (default one-way: fine tokens read the cached coarse tokens and each other,
             coarse tokens unchanged, the pool reads both; cost = coarse + fine only)

Writes FEATURES/tessera/<config>/<split>.npy (N, 128, 128, 128) fp16 embeddings and
<split>_stats.json (GMACs per patch, tokens per pixel, picks per window, mean window
attention). Targets are written once to FEATURES/tessera/targets_<split>.npy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import torch

from exp.common.paths import FEATURES
from exp.tessera import model as tm
from exp.tessera.inputs import DATASETS, LEVEL_KEY, LEVEL_W, level_w, to_patch
from exp.tessera.tasks import SPECS

FINENESS = {"sea": 0, "bimon": 1, "mon": 2, "semi": 3, "raw": 4}   # bimon does not nest in sea

OUT = FEATURES / "tessera"
CHUNK = 4096


@dataclass
class Cfg:
    name: str
    coarse: str
    fine: str | None = None
    n: float = 0.0
    pick: str = "attn"
    joint: bool = False
    rpx: bool = False           # fractional n: the extra window to RANDOM pixels (control)
    nm: int = 12                # months in the dataset's window (YieldSAT: 24), set by run()

    @classmethod
    def parse(cls, s: str) -> "Cfg":
        if s.startswith("fix-"):
            return cls(s, s[4:])
        m = re.fullmatch(r"(sea|bimon|mon|semi)>(mon|semi|raw)-b([\d.]+)-(attn|tattn|low|rand|fix\d+|fixattn)(-rpx)?(-joint)?", s)
        if not m:
            raise ValueError(f"bad config {s!r}")
        if FINENESS[m[2]] <= FINENESS[m[1]]:
            raise ValueError(f"{s}: the fine level must be finer than the coarse one")
        return cls(s, m[1], m[2], float(m[3]), m[4], bool(m[6]), bool(m[5]))

    @property
    def win_key(self):          # which index of a token names its coarse window
        return LEVEL_KEY[self.coarse]

    @property
    def W(self):
        return level_w(self.coarse, self.nm) if self.coarse in LEVEL_W else self.nm


# tattn: task-tuned pool queries per coarse level (exp.tessera.pooltune), [(w, b) S2, (w, b) S1]
_TUNED: dict = {}


def window_scores(coarse_tok, cp, W, tuned=None):
    """(P, W) summed pool weight of each window's coarse tokens (S2 + S1). tuned: score with
    task-tuned queries instead of the pretrained ones (the embedding itself is unchanged)."""
    P = cp[0].logits.shape[0]
    s = torch.zeros(P, W, device=cp[0].logits.device)
    for i, (tok, p) in enumerate(zip(coarse_tok, cp)):
        if p is None:                   # a modality without tokens (YieldSAT S1)
            continue
        lg = p.logits if tuned is None else \
            (p.h.float() @ tuned[i][0].T + tuned[i][1])[..., 0].masked_fill(~p.valid, tm.NEG)
        w = torch.softmax(lg, 1)
        s.index_add_(1, tok.win, w)
    return s


def select(cfg: Cfg, score, avail, unc_extra, fixattn_rank, gen):
    """(P, W) bool windows each pixel re-encodes."""
    P, W = score.shape
    if cfg.pick in ("attn", "tattn"):
        key = score
    elif cfg.pick == "low":
        key = -score
    elif cfg.pick == "rand":
        key = torch.rand(P, W, device=score.device, generator=gen)
    elif cfg.pick == "fixattn":
        key = torch.zeros(P, W, device=score.device)
        key[:, fixattn_rank] = torch.arange(W, 0, -1, device=score.device, dtype=torch.float)
    else:                                       # fix<j>
        key = torch.zeros(P, W, device=score.device)
        key[:, int(cfg.pick[3:])] = 1.0
    key = key.masked_fill(~avail, -math.inf)
    n = torch.full((P,), int(math.floor(cfg.n)), device=score.device)
    if unc_extra is not None:
        n = n + unc_extra.long()
    if cfg.pick.startswith("fix") and cfg.pick != "fixattn":
        n = n.clamp(max=1)
    order = key.argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(W, device=score.device).expand(P, W))
    return (rank < n[:, None]) & avail


def encode_chunk(m, cfg: Cfg, toks, sl, unc_extra, fixattn_rank, gen, dims):
    """embeddings (p, 128), MACs (p,), tokens (coarse, fine) (p,), picks (W,), score (W,)."""
    mods = ("S2", "S1A")
    bbs = (m.s2_backbone, m.s1_backbone)
    C = {"S2": 10, "S1A": 2}
    c_tok = [toks[(cfg.coarse, mo)] for mo in mods]
    for t in c_tok:
        t.win = t.window(cfg.coarse) if cfg.coarse in LEVEL_W else t.month
    cv = [t.valid[sl] for t in c_tok]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cp = [tm.coarse_pass(bb, t.x[sl], t.doy, v, keep_trace=cfg.fine is not None and not cfg.joint)
              if t.T else None for bb, t, v in zip(bbs, c_tok, cv)]
    # an empty modality pools to zeros, as Tessera's AttentionPooling does for T = 0
    zero = lambda: torch.zeros(cv[0].shape[0], dims.D, device=cv[0].device)
    nc = [v.sum(1) for v in cv]
    macs = sum(tm.macs_tokens(dims, C[mo], n, n) for mo, n in zip(mods, nc)) + tm.macs_reducer(dims)
    P = cv[0].shape[0]
    W = cfg.W
    score = window_scores(c_tok, cp, W, _TUNED.get(cfg.coarse) if cfg.pick == "tattn" else None) \
        if cfg.coarse in LEVEL_W else None
    nf_tot = torch.zeros(P, device=cv[0].device)
    picks = torch.zeros(W, device=cv[0].device)
    if cfg.fine is None:
        z = [tm.pool(p.h, p.logits) if p is not None else zero() for p in cp]
    else:
        f_tok = [toks[(cfg.fine, mo)] for mo in mods]
        f_win = [t.window(cfg.coarse) for t in f_tok]
        fv = [t.valid[sl] for t in f_tok]
        avail = torch.zeros(P, cfg.W, dtype=torch.bool, device=score.device)
        for w, v in zip(f_win, fv):                    # a window with a valid fine token
            avail |= (torch.nn.functional.one_hot(w, cfg.W).bool()[None] & v[..., None]).any(1)
        sel = select(cfg, score, avail, unc_extra, fixattn_rank, gen)
        picks = sel.float().sum(0)
        z = []
        for i, (bb, t, w, v, p, mo) in enumerate(zip(bbs, f_tok, f_win, fv, cp, mods)):
            if p is None:
                z.append(zero()); continue
            fs = sel[:, w] & v                              # (P, Tf) fine tokens re-encoded
            nf = fs.sum(1)
            nf_tot += nf
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg.joint:
                    keep_c = p.valid & ~sel[:, c_tok[i].win] & cv[i]
                    x = torch.cat([c_tok[i].x[sl], t.x[sl]], 1)
                    doy = torch.cat([c_tok[i].doy, t.doy])
                    jp = tm.coarse_pass(bb, x, doy, torch.cat([keep_c, fs], 1), keep_trace=False)
                    z.append(tm.pool(jp.h, jp.logits))
                    nj = keep_c.sum(1) + nf
                    macs = macs + tm.macs_tokens(dims, C[mo], nj, nj)
                else:
                    fp = tm.fine_pass(bb, p, t.x[sl], t.doy, fs)
                    z.append(tm.pool(torch.cat([p.h, fp.h], 1), torch.cat([p.logits, fp.logits], 1)))
                    macs = macs + tm.macs_tokens(dims, C[mo], nf, nc[i] + nf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = tm.fuse(m, z[0], z[1]).float()
    sc = score.sum(0) if score is not None else torch.zeros(1, device=e.device)
    return e, macs, nc[0] + nc[1], nf_tot, picks, sc, sum((~v.any(1)).sum() for v in cv if v.shape[1])


def run(cfg: Cfg, split: str, m, dims, device, limit=None, workers=6, seed=0, dataset="pastis"):
    for lv in (cfg.coarse, cfg.fine):
        if lv and not lv.startswith("uni") and lv not in DATASETS[dataset]["levels"]:
            raise ValueError(f"{dataset} has no level {lv!r}")
    ds = DATASETS[dataset]["items"](split, limit)
    cfg.nm = DATASETS[dataset]["n_months"]
    HW = DATASETS[dataset]["hw"]
    OUT = SPECS[dataset].out            # (shadows the module's PASTIS root)
    out_dir = OUT / (cfg.name + (f"_lim{limit}" if limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = np.lib.format.open_memmap(out_dir / f"{split}.npy.tmp", "w+", np.float16, (len(ds), HW, HW, 128))
    tgt_path = OUT / f"targets_{split}{'_lim' + str(limit) if limit else ''}.npy"
    tgt = None if tgt_path.exists() else np.zeros(
        (len(ds), HW, HW), np.int16 if SPECS[dataset].kind == "classification" else np.float32)
    # uncertainty (fractional n): entropy of the fix-<coarse> head's prediction
    unc = thr = None
    if cfg.fine and cfg.n != math.floor(cfg.n) and not cfg.rpx:
        frac = cfg.n - math.floor(cfg.n)
        src = OUT / (f"fix-{cfg.coarse}" + (f"_lim{limit}" if limit else "")) / "head"
        unc = np.load(src / f"entropy_{split}.npy", mmap_mode="r")
        ref = np.load(src / f"entropy_{'train' if split == 'train' else 'val'}.npy", mmap_mode="r")
        thr = float(np.quantile(np.asarray(ref[:, ::4, ::4], dtype=np.float32), 1 - frac))
    fixattn_rank = None
    if cfg.pick == "fixattn":
        st = json.load(open(OUT / (f"fix-{cfg.coarse}" + (f"_lim{limit}" if limit else "")) / "train_stats.json"))
        fixattn_rank = torch.as_tensor(np.argsort(-np.asarray(st["window_attention"])), device=device)
    if cfg.pick == "tattn":
        q = torch.load(OUT / f"pooltune_{cfg.coarse}" / "query.pt", map_location=device)
        _TUNED[cfg.coarse] = [(q["s2_w"].float(), q["s2_b"].float()), (q["s1_w"].float(), q["s1_b"].float())]
    gen = torch.Generator(device=device).manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4)
    stats = {"config": cfg.name, "split": split, "gmacs": [], "coarse_tokens": [], "fine_tokens": [],
             "ids": [], "no_token_px": 0}
    picks_tot = score_tot = None
    t0 = time.time()
    for i, item in enumerate(loader):
        pt = to_patch(item, device)
        levels = {cfg.coarse} | ({cfg.fine} if cfg.fine else set())
        toks = {(lv, mo): pt.level(lv, mo) for lv in levels for mo in ("S2", "S1A")}
        P = pt.hw[0] * pt.hw[1]
        ue = None
        if cfg.fine and cfg.rpx:
            ue = torch.rand(P, device=device, generator=gen) < (cfg.n - math.floor(cfg.n))
        if unc is not None:
            ue = torch.as_tensor(np.asarray(unc[i], dtype=np.float32).reshape(-1) > thr, device=device)
        es, ms, ncs, nfs = [], 0.0, 0.0, 0.0
        for s in range(0, P, CHUNK):
            sl = slice(s, s + CHUNK)
            with torch.no_grad():
                e, mc, nc, nf, pk, sc, bad = encode_chunk(m, cfg, toks, sl, None if ue is None else ue[sl],
                                                         fixattn_rank, gen, dims)
            es.append(e); ms += mc.sum().item(); ncs += nc.sum().item(); nfs += nf.sum().item()
            picks_tot = pk if picks_tot is None else picks_tot + pk
            score_tot = sc if score_tot is None else score_tot + sc
            stats["no_token_px"] += int(bad)
        emb[i] = torch.cat(es).view(*pt.hw, 128).half().cpu().numpy()
        if tgt is not None:
            tgt[i] = item["target"]
        stats["gmacs"].append(ms / 1e9); stats["coarse_tokens"].append(ncs / P)
        stats["fine_tokens"].append(nfs / P); stats["ids"].append(item["id"])
        if i % 200 == 0:
            print(f"  [{split}] {i}/{len(ds)}  {stats['gmacs'][-1]:.0f} GMACs  "
                  f"tok {ncs / P:.1f}+{nfs / P:.1f}  {time.time() - t0:.0f}s", flush=True)
    emb.flush(); del emb
    (out_dir / f"{split}.npy.tmp").rename(out_dir / f"{split}.npy")
    if tgt is not None:            # parallel jobs write the same targets: atomic rename
        tmp = tgt_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, tgt)
        os.replace(tmp, tgt_path)
    npx = len(ds) * HW * HW
    stats["window_picks"] = (picks_tot / npx).tolist()
    stats["window_attention"] = (score_tot / npx).tolist()
    stats["mean_gmacs"] = float(np.mean(stats["gmacs"]))
    stats["mean_tokens"] = float(np.mean(stats["coarse_tokens"]) + np.mean(stats["fine_tokens"]))
    stats["unc_threshold"] = thr
    json.dump(stats, open(out_dir / f"{split}_stats.json", "w"))
    print(f"[{split}] done {len(ds)} patches in {time.time() - t0:.0f}s: {stats['mean_gmacs']:.0f} GMACs/patch, "
          f"{stats['mean_tokens']:.1f} tokens/px, picks {np.round(stats['window_picks'], 3).tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset", default="pastis", choices=sorted(DATASETS))
    a = ap.parse_args()
    cfg = Cfg.parse(a.config)
    dev = torch.device("cuda")
    m = tm.load(dev)
    dims = tm.Dims.of(m)
    for split in a.splits.split(","):
        if (SPECS[a.dataset].out / cfg.name / f"{split}.npy").exists() and not a.limit:
            print(f"[{split}] exists, skip"); continue
        run(cfg, split, m, dims, dev, a.limit, a.workers, dataset=a.dataset)


if __name__ == "__main__":
    main()
