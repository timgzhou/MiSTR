"""Task-tuned decoder attention for frozen Tessera: train only the two attention-pool queries
(S2, S1: Linear(D, 1) each) + a per-pixel MLP head on the coarse pass, encoder frozen.

    python -m exp.tessera.pooltune --coarse sea [--dataset pastis|biomassters] [--px_train 1024] [--epochs 15]

Why: in the frozen MiSTR round the windows were picked with Tessera's PRETRAINED pool, which is
task-agnostic (~17 % of pixels re-encoded Oct-Mar, worth nothing), and a fixed window for every
pixel beat per-pixel picks. OlmoEarth MiSTR selects with its decoder attention, which is trained
on the task. This is the cheap equivalent: the tuned queries score the windows, the per-pixel
embeddings stay those of the pretrained model (exp.tessera.embed pick `tattn`).

1. cache   the frozen coarse pass (level `sea` or `mon`) on a random subset of labelled pixels
           of every train / val patch: final tokens per window slot (n, W, D) bf16 + masks
2. train   two arms on the cache, same recipe: `tuned` (queries trainable, init = pretrained)
           and `frozen` (queries fixed, the control); z = pool -> frozen dim_reducer -> MLP head
3. write   pooltune_<coarse>/query.pt (tuned queries), result.json (val mIoU of both arms, the
           share of pixels whose top window each arm picks, agreement with the pretrained pick)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from exp.tessera import model as tm
from exp.tessera.inputs import DATASETS, level_w, to_patch
from exp.tessera.tasks import SPECS, Spec


@torch.no_grad()
def build_cache(m, coarse: str, split: str, n_px: int, dev, workers=6, seed=0, limit=None,
                dataset="pastis"):
    W = level_w(coarse, DATASETS[dataset]["n_months"])
    D = tm.Dims.of(m).D
    ds = DATASETS[dataset]["items"](split, limit)
    spec = SPECS[dataset]
    rng = np.random.default_rng(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4)
    H = {"S2": [], "S1A": []}; V = {"S2": [], "S1A": []}; WI = {"S2": [], "S1A": []}; Y = []
    T_max = {"S2": W, "S1A": 2 * W}      # <= one token per window and orbit (BioMassters: asc + desc)
    t0 = time.time()
    for i, item in enumerate(loader):
        pt = to_patch(item, dev)
        y = torch.as_tensor(item["target"]).reshape(-1)
        lab = np.where(spec.labelled(y).numpy())[0]
        if len(lab) == 0:
            continue
        px = torch.as_tensor(rng.choice(lab, min(n_px, len(lab)), replace=False), device=dev)
        for mod, bb in (("S2", m.s2_backbone), ("S1A", m.s1_backbone)):
            t = pt.level(coarse, mod)
            win = t.window(coarse)
            T = t.T
            h = torch.zeros(len(px), T_max[mod], D, dtype=torch.bfloat16, device=dev)
            v = torch.zeros(len(px), T_max[mod], dtype=torch.bool, device=dev)
            wi = torch.zeros(len(px), T_max[mod], dtype=torch.int8, device=dev)
            if T:                    # (an empty modality stays all-masked zeros: pools to 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = tm.coarse_pass(bb, t.x[px], t.doy, t.valid[px], keep_trace=False)
                h[:, :T] = p.h.to(torch.bfloat16); v[:, :T] = t.valid[px]; wi[:, :T] = win.to(torch.int8)
            H[mod].append(h.cpu()); V[mod].append(v.cpu()); WI[mod].append(wi.cpu())
        Y.append(spec.target(y[px.cpu()]))
        if i % 300 == 0:
            print(f"  cache {coarse}/{split} {i}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
    return {"h2": torch.cat(H["S2"]), "v2": torch.cat(V["S2"]), "w2": torch.cat(WI["S2"]),
            "h1": torch.cat(H["S1A"]), "v1": torch.cat(V["S1A"]), "w1": torch.cat(WI["S1A"]),
            "y": torch.cat(Y), "W": torch.tensor(W)}


class Readout(nn.Module):
    """pool (queries init from Tessera) -> frozen dim_reducer -> per-pixel MLP head."""

    def __init__(self, m, tune: bool, n_out: int = 19):
        super().__init__()
        self.q2 = nn.Linear(m.s2_backbone.attn_pool.query.in_features, 1)
        self.q1 = nn.Linear(m.s1_backbone.attn_pool.query.in_features, 1)
        self.q2.load_state_dict(m.s2_backbone.attn_pool.query.state_dict())
        self.q1.load_state_dict(m.s1_backbone.attn_pool.query.state_dict())
        for p in (*self.q2.parameters(), *self.q1.parameters()):
            p.requires_grad_(tune)
        self.red = m.dim_reducer.eval()
        for p in self.red.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Linear(256, n_out))

    def weights(self, h2, v2, h1, v1):
        w2 = self.q2(h2.float())[..., 0].masked_fill(~v2, tm.NEG).softmax(1)
        w1 = self.q1(h1.float())[..., 0].masked_fill(~v1, tm.NEG).softmax(1)
        return w2, w1

    def forward(self, h2, v2, h1, v1):
        w2, w1 = self.weights(h2, v2, h1, v1)
        z = torch.cat([(w2[..., None] * h2.float()).sum(1), (w1[..., None] * h1.float()).sum(1)], -1)
        self.red.eval()
        return self.head(self.red(z))


def batches(c, bs, perm=None):
    n = len(c["y"])
    idx = perm if perm is not None else torch.arange(n, device=c["y"].device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield c["h2"][j], c["v2"][j], c["h1"][j], c["v1"][j], c["y"][j]


@torch.no_grad()
def evaluate(net, c, spec: Spec):
    net.eval()
    meter = spec.meter()
    for h2, v2, h1, v1, y in batches(c, 32768):
        meter.update(net(h2, v2, h1, v1), y)
    return meter.compute()[spec.metric]


@torch.no_grad()
def top_window(net, c):
    """(n,) the window with the highest summed S2 + S1 pool weight (as exp.tessera.embed)."""
    out, W = [], int(c["W"])
    for i in range(0, len(c["y"]), 32768):
        j = slice(i, i + 32768)
        w2, w1 = net.weights(c["h2"][j], c["v2"][j], c["h1"][j], c["v1"][j])
        s = torch.zeros(len(w2), W, device=w2.device)
        s.scatter_add_(1, c["w2"][j].long(), w2 * c["v2"][j])
        s.scatter_add_(1, c["w1"][j].long(), w1 * c["v1"][j])
        out.append(s.argmax(1))
    return torch.cat(out)


def train(m, c_tr, c_va, tune: bool, epochs, lr, seed, dev, spec: Spec):
    torch.manual_seed(seed)
    net = Readout(m, tune, spec.n_out).to(dev)
    sign = 1 if spec.higher_is_better else -1
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)
    bs = 8192
    steps = epochs * ((len(c_tr["y"]) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    best, best_state = -float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(c_tr["y"]), device=dev)
        for h2, v2, h1, v1, y in batches(c_tr, bs, perm):
            loss = spec.loss(net(h2, v2, h1, v1), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        v = evaluate(net, c_va, spec)
        if sign * v > best:
            best, best_state = sign * v, {k: t.detach().clone() for k, t in net.state_dict().items()}
        print(f"== [{'tuned' if tune else 'frozen'}] epoch {ep + 1} loss {loss.item():.3f} val {spec.metric} "
              f"{v:.4f} (best {sign * best:.4f})", flush=True)
    net.load_state_dict(best_state)
    return net, sign * best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", required=True, choices=("sea", "bimon", "mon", "semi"))
    ap.add_argument("--px_train", type=int, default=None, help="pixels per train patch (default sea 1024, bimon 768, mon 512, semi 256)")
    ap.add_argument("--px_val", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_debug", type=int, default=None)
    ap.add_argument("--dataset", default="pastis", choices=sorted(SPECS))
    a = ap.parse_args()
    dev = torch.device("cuda")
    m = tm.load(dev)
    spec = SPECS[a.dataset]
    out = spec.out / (f"pooltune_{a.coarse}" + (f"_lim{a.limit_debug}" if a.limit_debug else ""))
    out.mkdir(parents=True, exist_ok=True)
    # pixels per train sample: PASTIS's per-patch count (~6-9 GB cache over its 1455 patches),
    # scaled so every dataset caches about the same total (YieldSAT: 8703 tiles, 24 months)
    n_tr = a.px_train
    if n_tr is None:
        per = {"sea": 1024, "bimon": 768, "mon": 512, "semi": 256}[a.coarse]
        total = per * 1455 // (DATASETS[a.dataset]["n_months"] // 12)
        n_tr = max(16, min(per, total // len(DATASETS[a.dataset]["items"]("train"))))
    caches = {}
    for split, n in (("train", n_tr), ("val", a.px_val)):
        f = out / f"cache_{split}.pt"
        if f.exists() and "w2" in torch.load(f, mmap=True):        # (older caches: no window index)
            caches[split] = torch.load(f)
        else:
            caches[split] = build_cache(m, a.coarse, split, n, dev, seed=a.seed, limit=a.limit_debug,
                                        dataset=a.dataset)
            torch.save(caches[split], f)
        caches[split] = {k: t.to(dev) for k, t in caches[split].items()}
        print(f"{split}: {len(caches[split]['y'])} pixels", flush=True)
    res = {"coarse": a.coarse, "px_train": n_tr, "epochs": a.epochs, "lr": a.lr, "seed": a.seed}
    nets = {}
    for tune in (False, True):
        net, best = train(m, caches["train"], caches["val"], tune, a.epochs, a.lr, a.seed, dev, spec)
        nets[tune] = net
        res[f"{'tuned' if tune else 'frozen'}_val_{spec.metric}"] = best
    W = level_w(a.coarse, DATASETS[a.dataset]["n_months"])
    top_f, top_t = top_window(nets[False], caches["val"]), top_window(nets[True], caches["val"])
    res["picks_pretrained"] = (torch.bincount(top_f, minlength=W).float() / len(top_f)).tolist()
    res["picks_tuned"] = (torch.bincount(top_t, minlength=W).float() / len(top_t)).tolist()
    res["agreement"] = (top_f == top_t).float().mean().item()
    q = nets[True]
    torch.save({"s2_w": q.q2.weight.detach().cpu(), "s2_b": q.q2.bias.detach().cpu(),
                "s1_w": q.q1.weight.detach().cpu(), "s1_b": q.q1.bias.detach().cpu()}, out / "query.pt")
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print("RESULT", json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in res.items()}),
          flush=True)


if __name__ == "__main__":
    main()
