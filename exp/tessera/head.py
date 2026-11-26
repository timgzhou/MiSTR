"""Heads on cached frozen Tessera embeddings (exp.tessera.embed) -> mIoU (PASTIS) / RMSE
(BioMassters, exp.tessera.tasks: the head also predicts a log-variance = its uncertainty).

    python -m exp.tessera.head --config sea>mon-b1-attn [--dataset pastis|biomassters]
                               [--heads unet,linear,mlp] [--limit N]

Tessera pools time inside its encoder, so a head only maps the 128-d pixel map to classes.
  unet           spatial context from neighbouring pixels (the encoder has none)
  linear / mlp   per pixel (1x1 convs): the clean time-only comparison -- a pixel's prediction
                 uses only its own embedding. Results: results/tessera/pastis_frozen_<head>.csv

Tessera's own PASTIS protocol: frozen 128-d pixel embeddings + a lightweight UNet. Every
configuration gets the same head and recipe, so the head is constant cost and the
comparison is between the embeddings (i.e. the temporal input the encoder saw).

  head    UNet, widths 64-128-256-512 (3 poolings), 2 x (conv3x3-BN-ReLU) per stage,
          transposed-conv upsampling with skips, 1x1 to 19 classes; full 128 px patches
  train   folds 1-3, flips + rot90, CE (void 19 ignored), AdamW 1e-3 / wd 1e-2, cosine,
          batch 16; all embeddings held on the GPU (fp16)
  select  the epoch with the best val mIoU; test at that epoch (SegMeter, as every PASTIS run)

Writes FEATURES/tessera/<config>/head/{result.json, entropy_<split>.npy} (per-pixel entropy
of the prediction, used by the 1.5-window MiSTR configs of the coarse level) and appends a
row to results/tessera/pastis_frozen.csv.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from exp.tessera.tasks import SPECS, Spec


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, cin=128, widths=(64, 128, 256, 512), nc=19):
        super().__init__()
        self.enc = nn.ModuleList([block(cin, widths[0])] +
                                 [block(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths[:-1], widths[1:])])
        self.dec = nn.ModuleList([block(2 * a, a) for a in widths[:-1]])
        self.out = nn.Conv2d(widths[0], nc, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(reversed(self.up), reversed(self.dec)):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.out(x)


def head_gmacs(net, hw=128) -> float:
    tot = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            tot[0] += out.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1] // mod.groups
        elif isinstance(mod, nn.ConvTranspose2d):
            tot[0] += inp[0].numel() * mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1]
    hs = [mm.register_forward_hook(hook) for mm in net.modules() if isinstance(mm, (nn.Conv2d, nn.ConvTranspose2d))]
    with torch.no_grad():
        net.eval()(torch.zeros(1, 128, hw, hw, device=next(net.parameters()).device))
    for h in hs:
        h.remove()
    return tot[0] / 1e9


def load(spec: Spec, cfg_dir: Path, split: str, limit):
    """CPU tensors: embeddings (N, H, W, 128) fp16, targets as spec.target()."""
    x = torch.from_numpy(np.load(cfg_dir / f"{split}.npy", mmap_mode="r")[:limit or None].copy())
    y = np.load(spec.out / f"targets_{split}{'_lim' + str(limit) if limit else ''}.npy")[:limit or None]
    return x, spec.target(torch.from_numpy(np.asarray(y)))


def augment(x, y, gen):
    k = int(torch.randint(4, (1,), generator=gen))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(1, generator=gen) < 0.5:
        x, y = x.flip(3), y.flip(2)
    return x, y


@torch.no_grad()
def evaluate(net, spec: Spec, x, y, bs=32, entropy=False):
    """x / y on any device (batches are moved to the net's); entropy: also the per-pixel
    uncertainty (spec.uncertainty: entropy / predicted log-variance)."""
    net.eval()
    dev = next(net.parameters()).device
    meter = spec.meter(boundary_px=2)
    ent = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = net(xb).float()
        meter.update(lo, y[i:i + bs].to(dev))
        if entropy:
            ent.append(spec.uncertainty(lo).cpu())
    return meter.compute(), (torch.cat(ent).numpy() if entropy else None)


def build(kind: str, nc: int) -> nn.Module:
    """unet: spatial context (Tessera's PASTIS protocol). linear / mlp: per pixel (1x1 convs),
    so the prediction uses only that pixel's embedding, i.e. only what the encoder's temporal
    input gave it."""
    if kind == "unet":
        return UNet(nc=nc)
    if kind == "linear":
        return nn.Conv2d(128, nc, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                             nn.Conv2d(256, nc, 1))
    raise ValueError(kind)


def train_head(kind, a, xs, cfg_dir, dev, spec: Spec):
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    # seed > 0 (appendix seed repeats): its own dir, so seed 0's head (whose entropy maps
    # the fractional-window configs read) is never skipped over or overwritten
    out = cfg_dir / (("head" if kind == "unet" else f"head_{kind}") + (f"_seed{a.seed}" if a.seed else ""))
    if (out / "result.json").exists() and not a.limit:
        print(f"{kind}: {out / 'result.json'} exists, skip"); return
    net = build(kind, spec.n_out).to(dev).to(memory_format=torch.channels_last)
    gm = head_gmacs(net, xs["train"][0].shape[1])
    print(f"{cfg_dir.name} [{kind}]: head {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params, "
          f"{gm:.2f} GMACs/patch", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    xtr, ytr = xs["train"]
    steps = a.epochs * ((len(xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    sign = 1 if spec.higher_is_better else -1
    best, best_state, best_ep, t0 = -float("inf"), None, -1, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(xtr), a.bs):
            idx = perm[i:i + a.bs].to(dev)
            xb, yb = augment(xtr[idx].permute(0, 3, 1, 2), ytr[idx], gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = spec.loss(net(xb.float().contiguous(memory_format=torch.channels_last)), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            v, _ = evaluate(net, spec, *xs["val"])
            if sign * v[spec.metric] > best:
                best, best_ep = sign * v[spec.metric], ep + 1
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"== [{kind}] epoch {ep + 1} loss {loss.item():.3f} val {spec.metric} {v[spec.metric]:.4f} "
                  f"(best {sign * best:.4f} @{best_ep}) {time.time() - t0:.0f}s", flush=True)
    net.load_state_dict(best_state)
    out.mkdir(exist_ok=True)
    res = {"config": a.config, "head": kind, "limit": a.limit, "seed": a.seed, "epochs": a.epochs,
           "best_epoch": best_ep, "head_gmacs": gm}
    for s in ("train", "val", "test"):
        r, ent = evaluate(net, spec, *xs[s], entropy=kind == "unet")
        if kind == "unet":     # the fractional-window configs read the UNet head's uncertainty
            np.save(out / f"entropy_{s}.npy", ent)
        res[s] = {k: v for k, v in r.items() if not k.startswith(("iou_", "biou_"))}
    st = json.load(open(cfg_dir / "test_stats.json"))
    res["test_encoder_gmacs"] = st["mean_gmacs"]
    res["test_tokens_per_px"] = st["mean_tokens"]
    res["test_window_picks"] = st["window_picks"]
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(f"TEST {a.config} [{kind}]: {spec.metric} {res['test'][spec.metric]:.4f}  val {res['val'][spec.metric]:.4f}  "
          f"encoder {st['mean_gmacs']:.0f} + head {gm:.1f} GMACs/patch, {st['mean_tokens']:.1f} tokens/px", flush=True)
    if not a.limit:
        csv_path = spec.csv(kind)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        extra = (("overall_acc", "test_acc"), ("biou", "test_biou")) if spec.kind == "classification" \
            else (("mae", "test_mae"), ("r2", "test_r2"))
        row = {"config": a.config, "seed": a.seed, "epochs": a.epochs, "best_epoch": best_ep,
               f"val_{spec.metric}": res["val"][spec.metric], f"test_{spec.metric}": res["test"][spec.metric],
               **{c: res["test"][k] for k, c in extra},
               "encoder_gmacs": st["mean_gmacs"], "head_gmacs": gm, "tokens_per_px": st["mean_tokens"],
               "window_picks": json.dumps([round(p, 3) for p in st["window_picks"]])}
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            new = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--heads", default="unet,linear,mlp", help="unet (spatial) | linear | mlp (per pixel)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--dataset", default="pastis", choices=sorted(SPECS))
    a = ap.parse_args()
    dev = torch.device("cuda")
    spec = SPECS[a.dataset]
    cfg_dir = spec.out / (a.config + (f"_lim{a.limit}" if a.limit else ""))
    xs = {s: load(spec, cfg_dir, s, a.limit) for s in ("train", "val", "test")}
    xs["train"] = tuple(t.to(dev) for t in xs["train"])         # val / test stay on the CPU
    print(f"{cfg_dir.name}: train {len(xs['train'][0])} val {len(xs['val'][0])} test {len(xs['test'][0])}", flush=True)
    for kind in a.heads.split(","):
        train_head(kind, a, xs, cfg_dir, dev, spec)


if __name__ == "__main__":
    main()
