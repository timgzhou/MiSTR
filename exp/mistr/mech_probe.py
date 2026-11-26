"""Mechanistic probes of a trained MiSTR model (one-way attention), test split.

    python -u -m exp.mistr.mech_probe [--limit N] [--ablations a,b] <run name> [...]

The fine-token blocks are re-run with explicit softmax attention (OneWayEncoder._cross_blocks'
arithmetic, attention in fp32 instead of FlashAttention bf16; checked against the model's own
fine tokens, row "check"). Each fine query token's keys fall in one of GROUPS:

    self            the query token itself
    crop            the other tokens of its crop (other bandsets / modalities, same window+month)
    hr_month        fine tokens of the same month at other windows
    hr_window       fine tokens of the same window at other months
    hr_season       fine tokens at other windows and months of the same season
    hr_other        fine tokens at other windows of other seasons
    lr_season_loc   coarse tokens of the crop's season at the coarse cell(s) under the crop
    lr_season_far   coarse tokens of the crop's season at other cells
    lr_other_loc    coarse tokens of other seasons at the cell(s) under the crop
    lr_other_far    coarse tokens of other seasons at other cells

A. attention (ablation "none"): per block x head x group, the queries' mean attention mass, the
   group's mean share of the keys (uniform attention), and per block x group the norm of the
   group's contribution to the attention output (sum_k p_k v_k through proj / ls1), plus the
   attention update's and the MLP update's norm relative to the residual stream
   -> results/<task>/mech_attn.csv
B. ablations: the fine tokens re-encoded with key groups masked out at every block (out of the
   training distribution: a dependence test, not a deployable model) -> test metric
C. decoder: with one-way attention the season tokens are the coarse pass's, so at a pixel
   covered by crops the output is EXACTLY (1 - w) coarse + w crop (w = the crops' share of the
   pool, crop = the crops' own pooled map). Per covered labelled pixel: w, the coarse / crop /
   final decisions (classification) or errors (regression), who wins when coarse and crop
   disagree, and whether w is high where the crop is the better one
   B + C -> results/<task>/mech_decoder.csv (one row per run x ablation x statistic)
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from torch import nn  # noqa: E402

import mistr.core as _pkg  # noqa: E402
from exp.common.tasks import IGNORE, TASKS  # noqa: E402
from exp.mistr.train import set_attn, val_threshold  # noqa: E402
from mistr.baselines import build_model, build_selector_for  # noqa: E402
from mistr.core.config import load  # noqa: E402
from mistr.core.data import gpu_loader  # noqa: E402
from mistr.core.encoder import Packed  # noqa: E402
from mistr.core.selection import CoarsePass  # noqa: E402

GROUPS = ("self", "crop", "hr_month", "hr_window", "hr_season", "hr_other",
          "lr_season_loc", "lr_season_far", "lr_other_loc", "lr_other_far")
G = {g: i for i, g in enumerate(GROUPS)}
# the four reported (+ own crop kept apart inside "this month")
COARSE4 = {"this_month_hr": ("self", "crop", "hr_month"),
           "this_season_lr": ("lr_season_loc", "lr_season_far"),
           "other_month_hr": ("hr_window", "hr_season", "hr_other"),
           "other_season_lr": ("lr_other_loc", "lr_other_far")}
LR = ("lr_season_loc", "lr_season_far", "lr_other_loc", "lr_other_far")
ABLATIONS = {
    "none": (),
    "no_lr": LR,                                            # fine tokens read only fine tokens
    "no_lr_other_season": ("lr_other_loc", "lr_other_far"),
    "no_lr_this_season": ("lr_season_loc", "lr_season_far"),
    "lr_local_only": ("lr_season_far", "lr_other_far"),     # coarse context only under the crop
    "no_hr_others": ("hr_month", "hr_window", "hr_season", "hr_other"),   # own crop + coarse
    "no_hr_other_months": ("hr_window", "hr_season", "hr_other"),
    "own_crop_only": tuple(g for g in GROUPS if g not in ("self", "crop")),
}


# -- token metadata ---------------------------------------------------------------------------
def lr_meta(lr: Packed, cp: int):
    """Per valid coarse token: (sample, slot, cell row, cell col), in lr.x's order."""
    S, N = lr.keep.shape
    cols = []
    for m, shape in lr.dims.items():
        inner = tuple(shape[1:-1])                           # (h, w, t, bandsets)
        if len(inner) != 4:
            raise ValueError(f"{m}: token shape {shape}, want (S, h, w, t, bandsets, D)")
        h, w, t, nb = inner
        yy, xx, tt, _ = torch.meshgrid(*(torch.arange(n) for n in inner), indexing="ij")
        cols.append(torch.stack([tt.flatten(), yy.flatten(), xx.flatten()], 1))
    meta = torch.cat(cols).to(lr.keep.device)                # (N, 3)
    samp = torch.arange(S, device=lr.keep.device)[:, None].expand(S, N)
    keep = lr.keep
    return samp[keep], meta[None].expand(S, N, 3)[keep]      # (N_lr,), (N_lr, 3)


class Groups:
    """Padded per-sample query / key tables and each (query, key) pair's group."""

    def __init__(self, lr: Packed, crops: Packed, plan, season, cp: int, lo: int, B: int):
        dev = lr.x.device
        l_s, l_m = lr_meta(lr, cp)
        n_crop = len(crops.seqlens)
        c_id = torch.repeat_interleave(torch.arange(n_crop, device=dev), crops.seqlens)
        c_s = plan.b[c_id]
        N_lr, N_c = len(l_s), len(c_id)
        n_lr_b = torch.bincount(l_s, minlength=B)
        n_c_b = torch.bincount(c_s, minlength=B)
        st_lr = torch.cumsum(n_lr_b, 0) - n_lr_b
        rank_lr = torch.arange(N_lr, device=dev) - st_lr[l_s]
        order = torch.argsort(c_s, stable=True)
        st_c = torch.cumsum(n_c_b, 0) - n_c_b
        rank_c = torch.empty_like(order)
        rank_c[order] = torch.arange(N_c, device=dev) - st_c[c_s[order]]
        self.Kmax = int((n_lr_b + n_c_b).max())
        self.Qmax = int(n_c_b.max())
        kidx = torch.full((B, self.Kmax), -1, dtype=torch.long, device=dev)
        kidx[l_s, rank_lr] = torch.arange(N_lr, device=dev)
        kidx[c_s, n_lr_b[c_s] + rank_c] = N_lr + torch.arange(N_c, device=dev)
        qidx = torch.full((B, self.Qmax), -1, dtype=torch.long, device=dev)
        qidx[c_s, rank_c] = torch.arange(N_c, device=dev)
        self.kidx, self.qidx, self.N_lr, self.N_c = kidx, qidx, N_lr, N_c
        # key metadata over [lr tokens; fine tokens]
        c_t = plan.t[c_id]
        c_slot = season[plan.b, plan.t][c_id]
        c_py, c_px = plan.py[c_id] + lo, plan.px[c_id] + lo
        fine = torch.cat([torch.zeros(N_lr, dtype=torch.bool, device=dev),
                          torch.ones(N_c, dtype=torch.bool, device=dev)])
        slot = torch.cat([l_m[:, 0], c_slot])
        month = torch.cat([torch.full((N_lr,), -1, device=dev, dtype=torch.long), c_t])
        crop = torch.cat([torch.full((N_lr,), -1, device=dev, dtype=torch.long), c_id])
        tok = torch.arange(N_lr + N_c, device=dev)
        loc = torch.cat([torch.full((N_lr,), -1, device=dev, dtype=torch.long),
                         c_py * 4096 + c_px])
        cy, cx = l_m[:, 1], l_m[:, 2]
        k = kidx.clamp(min=0)
        q = qidx.clamp(min=0) + N_lr                          # into the key tables
        Kf, Kslot, Kmon, Kcrop, Ktok, Kloc = (v[k][:, None, :] for v in
                                              (fine, slot, month, crop, tok, loc))
        Qslot, Qmon, Qcrop, Qtok, Qloc = (v[q][:, :, None] for v in (slot, month, crop, tok, loc))
        # coarse cell under the crop: footprint [py, py + patch) overlaps [cy cp, (cy + 1) cp)
        pa = plan.patch
        qpy, qpx = c_py[qidx.clamp(min=0)][:, :, None], c_px[qidx.clamp(min=0)][:, :, None]
        ky = torch.cat([cy, torch.zeros(N_c, dtype=torch.long, device=dev)])[k][:, None, :]
        kx = torch.cat([cx, torch.zeros(N_c, dtype=torch.long, device=dev)])[k][:, None, :]
        local = ((ky * cp < qpy + pa) & ((ky + 1) * cp > qpy)
                 & (kx * cp < qpx + pa) & ((kx + 1) * cp > qpx))
        same_s = Kslot == Qslot
        g_lr = torch.where(same_s, torch.where(local, G["lr_season_loc"], G["lr_season_far"]),
                           torch.where(local, G["lr_other_loc"], G["lr_other_far"]))
        same_m, same_l = Kmon == Qmon, Kloc == Qloc
        g_hr = torch.where(Ktok == Qtok, G["self"],
               torch.where(Kcrop == Qcrop, G["crop"],
               torch.where(same_m, G["hr_month"],
               torch.where(same_l, G["hr_window"],
               torch.where(same_s, G["hr_season"], G["hr_other"])))))
        grp = torch.where(Kf, g_hr, g_lr)
        valid = (kidx >= 0)[:, None, :] & (qidx >= 0)[:, :, None]
        self.grp = grp.masked_fill(~valid, -1).to(torch.int8)       # (B, Qmax, Kmax)
        self.qvalid = qidx >= 0


# -- the fine-token blocks with explicit attention ---------------------------------------------
@torch.no_grad()
def fine_blocks(model, lr: Packed, trace, crops: Packed, grp: Groups, drop=(), stats=None):
    """OneWayEncoder._cross_blocks (cross_hr), explicit attention; drop: key groups masked at
    every block. stats: dict accumulated in place (ablation none). -> the fine tokens (normed)."""
    mix = model.mix
    enc = mix.encoder
    xs = mix._typed(crops, lr.x.dtype)
    kidx, qidx, qv = grp.kidx, grp.qidx, grp.qvalid
    gmask = grp.grp.long()
    keep = gmask >= 0
    for g in drop:
        keep &= gmask != G[g]
    k_ok = kidx.clamp(min=0)
    q_ok = qidx.clamp(min=0)
    for li, (blk, x_lr) in enumerate(zip(enc.blocks[mix.L:], trace)):
        a = blk.attn
        nh = a.num_heads
        heads = lambda v: v.view(len(v), nh, -1)                          # noqa: E731
        with torch.autocast("cuda", dtype=torch.bfloat16):
            n_lr, n_c = blk.norm1(x_lr), blk.norm1(xs)
            q = a.q_norm(heads(a.q(n_c)))
            k = torch.cat([a.k_norm(heads(a.k(n_lr))), a.k_norm(heads(a.k(n_c)))])
            v = torch.cat([heads(a.v(n_lr)), heads(a.v(n_c))])
        Qp, Kp, Vp = q.float()[q_ok], k.float()[k_ok], v.float()[k_ok]   # (B, Q|K, H, dh)
        s = torch.einsum("bqhd,bkhd->bhqk", Qp, Kp) * a.scale
        s = s.masked_fill(~keep[:, None], float("-inf"))
        p = s.softmax(-1).nan_to_num(0.0)                                 # (B, H, Q, K)
        o = torch.einsum("bhqk,bkhd->bqhd", p, Vp)                        # (B, Q, H, dh)
        out = torch.zeros(len(xs), nh, o.shape[-1], device=xs.device)
        out[qidx[qv]] = o[qv]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk.ls1(a.proj_drop(a.proj(out.reshape(len(out), -1).to(torch.bfloat16))))
        if stats is not None:
            accumulate(stats, li, p, Vp, gmask, qv, a, blk, xs, y)
        xs = xs + y.to(xs.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = blk.ls2(blk.mlp(blk.norm2(xs)))
        if stats is not None:
            r = xs.float().norm(dim=-1)
            stats.setdefault(("mlp_rel", li), []).append((m.float().norm(dim=-1) / r).sum().item())
        xs = xs + m.to(xs.dtype)
    return enc.norm(xs)


def _ls_scale(blk):
    g = getattr(blk.ls1, "gamma", None)
    return g.float() if g is not None else None


def accumulate(st, li, p, Vp, gmask, qv, a, blk, xs, y):
    """Sums over valid queries: attention mass and key share per (head, group), value-weighted
    output norm per group, attention update norm / residual norm."""
    nq = qv.sum().item()
    st[("nq", li)] = st.get(("nq", li), 0) + nq
    Wp = a.proj.weight.float()                                            # (D, D)
    gam = _ls_scale(blk)
    nk = (gmask >= 0).sum(-1).clamp(min=1).float()                        # (B, Q)
    tot = None
    for g, gi in G.items():
        mk = (gmask == gi)
        mass = (p * mk[:, None]).sum(-1)                                  # (B, H, Q)
        share = mk.sum(-1).float() / nk                                   # (B, Q)
        key = ("mass", li, g)
        st[key] = st.get(key, 0) + (mass * qv[:, None]).sum((0, 2)).cpu()     # (H,)
        st[("share", li, g)] = st.get(("share", li, g), 0) + (share * qv).sum().item()
        og = torch.einsum("bhqk,bkhd->bqhd", p * mk[:, None], Vp).flatten(2) @ Wp.T
        if gam is not None:
            og = og * gam
        st[("vnorm", li, g)] = st.get(("vnorm", li, g), 0) + (og.norm(dim=-1) * qv).sum().item()
        tot = og if tot is None else tot + og
    r = xs.float().norm(dim=-1)
    st[("attn_rel", li)] = st.get(("attn_rel", li), 0) + (y.float().norm(dim=-1) / r).sum().item()


# -- decoder ----------------------------------------------------------------------------------
def to_img(model, t):
    """(B, Q, X, 16) per ps4 cell -> (B, X, 64, 64), cropped like the decoder's map."""
    rd = model.readout
    H = int(round(t.shape[1] ** 0.5))
    img = rearrange(t, "b (y x) k (i j) -> b k (y i) (x j)", y=H, x=H, i=rd.crop_px, j=rd.crop_px)
    if rd.out_px is not None and img.shape[-1] > rd.out_px:
        lo = (img.shape[-1] - rd.out_px) // 2
        img = img[..., lo:lo + rd.out_px, lo:lo + rd.out_px]
    return img


@torch.no_grad()
def decode_parts(model, lr_F, hr_F, p, season, present):
    """-> final map, crop share w (B, 64, 64), crops-only map (B, C, 64, 64)."""
    lo = model.lo
    off = not p.on_grid or lo
    img, (wgt, cand, _, _) = model.readout(
        model.head.linear, lr_F, season, present, hr_F, (p.b, p.t, p.y, p.x), None,
        None if _pkg.FAST else torch.ones_like(p.b),
        hr_px=(p.py + lo, p.px + lo) if off else None, hr_fp=p.patch, parts=True)
    NS = lr_F.shape[1]
    if wgt.ndim == 3:
        wgt = wgt[..., None].expand(*wgt.shape, cand.shape[-1])
    wc = wgt[:, :, NS:]                                                   # (B, Q, M, 16)
    w = wc.sum(2)                                                         # (B, Q, 16)
    crop = torch.einsum("bqkp,bqkcp->bqcp", wc / w.clamp(min=1e-12)[:, :, None], cand[:, :, NS:])
    return img, to_img(model, w[:, :, None])[:, 0], to_img(model, crop)


class DecoderStats:
    def __init__(self, kind):
        self.kind = kind
        self.n = {}
        self.w, self.d = [], []             # per covered pixel: w, coarse - crop "badness"

    def add(self, k, v):
        self.n[k] = self.n.get(k, 0) + float(v)

    def update(self, coarse, final, crop, w, target):
        cov = w > 0
        if self.kind == "classification":
            y = target
            ok = cov & (y != IGNORE)
            yc = y.clamp(min=0)
            pc, pf, pr = (t.argmax(1) for t in (coarse, final, crop))
            ce = lambda t: -t.float().log_softmax(1).gather(1, yc[:, None])[:, 0]  # noqa: E731
            dce = ce(coarse) - ce(crop)                       # > 0: the crop is the better one
            c_ok, r_ok, f_ok = pc == y, pr == y, pf == y
            dis = ok & (pc != pr)
            self.add("px", ok.sum())
            self.add("w_sum", w[ok].sum())
            self.add("acc_coarse", (c_ok & ok).sum())
            self.add("acc_crop", (r_ok & ok).sum())
            self.add("acc_final", (f_ok & ok).sum())
            self.add("disagree", dis.sum())
            self.add("dis_final_is_crop", (dis & (pf == pr)).sum())
            self.add("dis_final_is_coarse", (dis & (pf == pc)).sum())
            self.add("dis_crop_right", (dis & r_ok).sum())
            self.add("dis_coarse_right", (dis & c_ok).sum())
            self.add("fix", (ok & ~c_ok & f_ok).sum())
            self.add("break", (ok & c_ok & ~f_ok).sum())
            self.add("oracle_gate", (ok & (c_ok | r_ok)).sum())
            a, b = dis & r_ok & ~c_ok, dis & c_ok & ~r_ok
            self.add("n_crop_wins", a.sum())
            self.add("w_crop_wins", w[a].sum())
            self.add("n_coarse_wins", b.sum())
            self.add("w_coarse_wins", w[b].sum())
            self.w.append(w[ok].cpu())
            self.d.append(dce[ok].cpu())
        else:
            y = target if target.ndim == 3 else target[:, 0]
            ok = cov & torch.isfinite(y)
            ec, er, ef = ((t[:, 0].float() - y).abs() for t in (coarse, crop, final))
            self.add("px", ok.sum())
            self.add("w_sum", w[ok].sum())
            for nm, e in (("coarse", ec), ("crop", er), ("final", ef)):
                self.add(f"se_{nm}", (e[ok] ** 2).sum())
                self.add(f"ae_{nm}", e[ok].sum())
            for g in (0.5,):
                eg = ((1 - g) * coarse[:, 0].float() + g * crop[:, 0].float() - y).abs()
                self.add(f"se_gate{g}", (eg[ok] ** 2).sum())
            better = ok & (er < ec)
            self.add("crop_better", better.sum())
            self.add("w_crop_better", w[better].sum())
            self.add("w_crop_worse", w[ok & ~(er < ec)].sum())
            self.add("oracle_gate_se", (torch.minimum(ec, er)[ok] ** 2).sum())
            self.w.append(w[ok].cpu())
            self.d.append((ec - er)[ok].cpu())

    def rows(self):
        n = self.n
        px = max(n.get("px", 0), 1)
        out = {"covered_px": n.get("px", 0), "w_mean": n.get("w_sum", 0) / px}
        w = torch.cat(self.w) if self.w else torch.zeros(0)
        d = torch.cat(self.d) if self.d else torch.zeros(0)
        if len(w):
            for qq in (0.1, 0.25, 0.5, 0.75, 0.9):
                out[f"w_q{int(qq * 100)}"] = float(torch.quantile(w[:2_000_000].float(), qq))
            rw = w.argsort().argsort().float()
            rd = d.argsort().argsort().float()
            out["spearman_w_vs_crop_advantage"] = float(torch.corrcoef(torch.stack([rw, rd]))[0, 1])
        if self.kind == "classification":
            dis = max(n.get("disagree", 0), 1)
            out.update({
                "acc_coarse": n["acc_coarse"] / px, "acc_crop": n["acc_crop"] / px,
                "acc_final": n["acc_final"] / px, "acc_oracle_gate": n["oracle_gate"] / px,
                "disagree_rate": n["disagree"] / px,
                "dis_final_follows_crop": n["dis_final_is_crop"] / dis,
                "dis_final_follows_coarse": n["dis_final_is_coarse"] / dis,
                "dis_crop_right": n["dis_crop_right"] / dis,
                "dis_coarse_right": n["dis_coarse_right"] / dis,
                "fix_rate": n["fix"] / px, "break_rate": n["break"] / px,
                "w_when_crop_wins": n["w_crop_wins"] / max(n["n_crop_wins"], 1),
                "w_when_coarse_wins": n["w_coarse_wins"] / max(n["n_coarse_wins"], 1)})
        else:
            nb = n.get("crop_better", 0)
            out.update({f"rmse_{k}": (n[f"se_{k}"] / px) ** 0.5
                        for k in ("coarse", "crop", "final", "gate0.5")})
            out.update({f"mae_{k}": n[f"ae_{k}"] / px for k in ("coarse", "crop", "final")})
            out.update({"rmse_oracle_gate": (n["oracle_gate_se"] / px) ** 0.5,
                        "crop_better_rate": nb / px,
                        "w_when_crop_better": n["w_crop_better"] / max(nb, 1),
                        "w_when_crop_worse": n["w_crop_worse"] / max(px - nb, 1)})
        return out


def season_of(model, b):
    return model._inputs(b.ms, b.ss, b.season, b.present)[2]


# -- driver -----------------------------------------------------------------------------------
def load_run(name, bs):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ck["config"], f)
    cfg = load(f.name)
    task = TASKS[cfg.task]
    model = build_model(task, cfg).cuda()
    vbs = cfg.train.batch_size or task.batch_size[16]
    dl = {"val": gpu_loader(task, "val", vbs, 8, False), "test": gpu_loader(task, "test", bs, 8, False)}
    b0 = next(iter(dl["val"]))
    sel = build_selector_for(task, cfg, model, (b0.ms, b0.ss))
    model.load_state_dict(ck["model"])
    # pre-rename checkpoints also keep the threshold's time-entropy table tab_t
    selector = ck["selector"]
    if isinstance(selector, dict):
        selector = {k: v for k, v in selector.items() if not k.endswith("tab_t")}
    else:
        selector = [{k: v for k, v in sd.items() if k != "tab_t"} for sd in selector]
    nn.ModuleList(sel.modules()).load_state_dict(selector)
    set_attn(model, cfg.model.attention)
    if model.mix.attn != "cross_hr":
        raise ValueError(f"{name}: attention {cfg.model.attention}, the probe is for one_way")
    if cfg.model.crop_time != "month" or cfg.model.coarse_time != "season":
        raise ValueError(f"{name}: crop_time {cfg.model.crop_time} / coarse_time {cfg.model.coarse_time}")
    model.eval()
    return cfg, task, model, sel, dl


@torch.no_grad()
def one(name, limit, ablations, bs):
    cfg, task, model, sel, dl = load_run(name, bs)
    val_threshold(model, sel, dl["val"], limit)
    meters = {a: task.meter() for a in ablations}
    meters["coarse_pass"] = task.meter()
    dstats = {a: DecoderStats(task.kind) for a in ablations}
    stats, check, seasons_n = {}, [], []
    for i, b in enumerate(dl["test"]):
        if limit and i >= limit:
            break
        target = b.target.cuda()
        ctx = CoarsePass.from_batch(model, b)
        lr_F = ctx.out0.coarse_features
        # (fp32, as decode_parts: the coarse pass itself decodes under bf16 autocast)
        coarse = model.decode(lr_F, lr_F.new_zeros(0, lr_F.shape[-1]), [], season_of(model, b),
                              b.present.cuda()).float()
        meters["coarse_pass"].update(coarse, target)
        plan = sel(ctx).plan
        ms, ss, season, present = model._inputs(b.ms, b.ss, b.season, b.present)
        plans, _ = model.split(plan)
        if len(plans) != 1:
            if not plans:       # no crops in this batch: every ablation = the coarse pass
                for a in ablations:
                    meters[a].update(coarse, target)
                continue
            raise ValueError("several crop groups: not supported")
        p = plans[0]
        lr = ctx.coarse_tokens
        trace = ctx.coarse_cache[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            crops = model.tokenize_crops(ms, p, season, lr, ss)
        grp = Groups(lr, crops, p, season, model.cfg.coarse_patch, model.lo, ctx.B)
        sl = season[p.b, p.t]                        # distinct crop seasons per sample with crops
        nd = torch.zeros(ctx.B, season.max() + 1, device=sl.device).index_put_(
            (p.b, sl), torch.ones_like(sl, dtype=torch.float), accumulate=True)
        has = nd.sum(1) > 0
        seasons_n.append(((nd > 0).sum(1)[has].float().sum().item(), has.sum().item()))
        for a in ablations:
            x = fine_blocks(model, lr, trace, crops, grp, ABLATIONS[a],
                            stats if a == "none" else None)
            hr_F = crops.pooled(x).float()[:, 0, 0, 0]
            final, w, crop = decode_parts(model, lr_F, hr_F, p, season, present)
            final = final.float()
            meters[a].update(final, target)
            dstats[a].update(coarse, final, crop.float(), w, target)
            if a == "none":
                ref = model(b.ms, b.ss, b.season, b.present, plan, None, None, lr, ctx.coarse_cache)
                rel = ((hr_F - ref.fine_features).norm(dim=-1)
                       / ref.fine_features.norm(dim=-1)).max().item()
                same = ((final.argmax(1) == ref.prediction.argmax(1)).float().mean().item()
                        if task.kind == "classification" else
                        (final - ref.prediction.float()).abs().max().item())
                mix = ((1 - w[:, None]) * coarse + w[:, None] * crop.float() - final).abs().max().item()
                check.append((rel, same, mix))
        if i % 10 == 0:
            print(f"  batch {i}: crops/sample {len(plan) / ctx.B:.1f}, check {check[-1]}", flush=True)
    # -- rows
    head = {"run": name, "task": cfg.task}
    drows = []
    for a in list(ablations) + ["coarse_pass"]:
        for k, v in meters[a].compute().items():
            drows.append({**head, "ablation": a, "stat": f"test_{k}", "value": float(v)})
        if a in dstats:
            for k, v in dstats[a].rows().items():
                drows.append({**head, "ablation": a, "stat": k, "value": float(v)})
    if check:
        c = torch.tensor(check)
        for j, k in enumerate(("check_fine_rel_err_max", "check_pred_agree_or_maxdiff",
                               "check_mix_identity_maxdiff")):
            v = c[:, j].max() if j != 1 or task.kind != "classification" else c[:, j].min()
            drows.append({**head, "ablation": "none", "stat": k, "value": float(v)})
    if seasons_n:
        t = torch.tensor(seasons_n)
        drows.append({**head, "ablation": "none", "stat": "crop_seasons_per_sample",
                      "value": float(t[:, 0].sum() / t[:, 1].sum().clamp(min=1))})
    arows = []
    layers = sorted({k[1] for k in stats if k[0] == "nq"})
    for li in layers:
        nq = stats[("nq", li)]
        for g in GROUPS:
            m = stats[("mass", li, g)] / nq
            for h in range(len(m)):
                arows.append({**head, "layer": li, "head": h, "group": g, "mass": float(m[h]),
                              "key_share": stats[("share", li, g)] / nq,
                              "vnorm": stats[("vnorm", li, g)] / nq})
        arows.append({**head, "layer": li, "head": -1, "group": "_attn_update_rel",
                      "mass": stats[("attn_rel", li)] / nq})
        arows.append({**head, "layer": li, "head": -1, "group": "_mlp_update_rel",
                      "mass": sum(stats[("mlp_rel", li)]) / nq})
    if not limit:
        for rows, fn in ((drows, "mech_decoder.csv"), (arows, "mech_attn.csv")):
            out = Path(f"results/{cfg.task}/{fn}")
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    summarize(pd.DataFrame(arows), pd.DataFrame(drows))


def summarize(A, D):
    if len(A):
        a = A[A["head"] >= 0].groupby(["layer", "group"])[["mass", "key_share", "vnorm"]].mean()
        mass = a["mass"].unstack()
        share = a["key_share"].unstack()
        print("\nattention mass per block (mean over heads, queries):")
        four = pd.DataFrame({k: mass[list(v)].sum(1) for k, v in COARSE4.items()})
        four_u = pd.DataFrame({k: share[list(v)].sum(1) for k, v in COARSE4.items()})
        print(four.round(3).to_string())
        print("uniform (key share):")
        print(four_u.round(3).to_string())
        print("all groups, mean over blocks: mass / key share / vnorm share")
        vn = a["vnorm"].unstack()
        print(pd.DataFrame({"mass": mass.mean(), "key_share": share.mean(),
                            "vnorm_share": (vn.T / vn.sum(1)).T.mean()}).round(3).to_string())
        rel = A[A["head"] < 0].pivot(index="layer", columns="group", values="mass")
        print(rel.round(3).T.to_string())
    print("\ndecoder / ablations:")
    print(D.pivot(index="stat", columns="ablation", values="value").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="batches (smoke: nothing written)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    abl = args.ablations.split(",")
    if abl[0] != "none":
        raise ValueError("--ablations must start with none (the check and the attention stats)")
    for n in args.runs:
        one(n, args.limit, abl, args.bs)
