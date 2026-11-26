"""Write notebooks/*.ipynb from the cell sources below (outputs empty; run them with Jupyter).

    python tools/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''import os, sys
# run from the repo root (the notebooks import the repo's packages)
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
assert torch.cuda.is_available(), "the models run on a GPU (OlmoEarth's FlashAttention path)"'''

PASTIS = [
    ("md", """# MiSTR on PASTIS: one checkpoint, eight test tiles

PASTIS crop-type segmentation (20 classes, 64 x 64 px tiles of 10 m Sentinel-2 + Sentinel-1, one
agricultural year). The model is MiSTR on OlmoEarth Tiny: a **coarse pass** encodes seasonal
composites as 8 x 8 px tokens ($S_8T_\\text{sea}$); the selector then picks half of the 4 x 4 px
points (where the coarse prediction is most uncertain) and, for each, the season that decoder
attention ranks highest; each point is re-encoded at that season's months as **4 x 4 px monthly
tokens** ($S_4T_\\text{mon}$) that read the cached coarse tokens (one-way attention); the
decoder pools both.

Needs: `checkpoints/mistr_pastis_S8Tsea+S4Tmon_q50_woracle50_ep100_seed0_9a7509fd.pt` and
`demo/pastis_test_samples.npz` (see the README, *Demo*). A GPU; about a minute."""),
    ("code", SETUP),
    ("code", '''from exp.demo import load_pretrained, load_samples, predict
from exp.bench.flop_trace import count_macs
from mistr_paper.data.datasets.pastis import CLASS_NAMES

RUN = "mistr_pastis_S8Tsea+S4Tmon_q50_woracle50_ep100_seed0_9a7509fd"
task, cfg, model, selector = load_pretrained(f"checkpoints/{RUN}.pt")
print(f"coarse tokens: S{cfg.model.coarse_patch} seasons | windows: {cfg.model.window_px} px, "
      f"{cfg.select.percentage:.0%} of the points, fine tokens S{cfg.model.fine_patch or cfg.model.window_px} months"
      f" | backbone OlmoEarth {cfg.model.backbone} | {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")'''),
    ("md", """## Predict

`predict` runs the coarse pass, the selector and the second pass; it also returns the coarse
pass's own prediction (what the model outputs where nothing is re-encoded)."""),
    ("code", '''batch = load_samples("demo/pastis_test_samples.npz")
with count_macs() as c:
    out = predict(model, selector, batch)
B = batch.target.shape[0]
print(f"{B} tiles, {out['windows_per_sample'].float().mean():.0f} high-res tokens per tile on average "
      f"(point-months), {c.total / B / 1e9:.1f} GMACs per tile (coarse pass + selector + second pass)")

def miou(logits):
    m = task.meter()
    m.update(logits, batch.target)
    return m.compute()["primary"] * 100

from exp.viz.round1_main import mistr_cost, with_test_cost      # how Fig 03 computes the test cost
d = with_test_cost(pd.read_csv(f"results/{task.name}/mistr.csv", low_memory=False), task.name)
paper = d[d.run == RUN].iloc[-1]
gmacs_test = mistr_cost(paper, d)
print(f"mIoU on these {B} tiles: MiSTR {miou(out['logits']):.1f}, coarse pass alone {miou(out['coarse_logits']):.1f}")
print(f"paper, whole test set (1,984 tiles): MiSTR {paper.test_primary * 100:.1f} mIoU at {gmacs_test:.1f} GMACs per tile")'''),
    ("md", """The full test set is evaluated in batches of other tiles; in bfloat16 a tile's logits can
change in the last digits with its batch (> 99.7 % of pixels keep their class), so numbers on
these eight tiles can differ slightly from a run over the whole split.

## Look at the tiles

Per tile: an RGB view of the input (seasonal composites, averaged), the label, the coarse pass,
MiSTR, and the points re-encoded at high resolution (boxes, coloured by month)."""),
    ("code", '''from olmoearth_pretrain.data.constants import Modality
order = list(Modality.SENTINEL2_L2A.band_order)
rgb_idx = [order.index(b) for b in ("B04", "B03", "B02")]

def rgb(i):
    x = batch.ss.sentinel2_l2a[i].float().mean(2)[..., rgb_idx].cpu().numpy()   # (H, W, 3), normalized
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    return np.clip((x - lo) / (hi - lo), 0, 1)

cmap = plt.get_cmap("tab20", 20)
def label_img(y):
    y = y.astype(float); y[y == 19] = np.nan     # 19: void (not scored)
    return y

fig, axes = plt.subplots(B, 5, figsize=(11, 2.2 * B))
plan = out["plan"]
for i in range(B):
    y = batch.target[i].cpu().numpy()
    p0, p1 = out["coarse_logits"][i].argmax(0).cpu().numpy(), out["logits"][i].argmax(0).cpu().numpy()
    ok = y != 19
    acc = lambda p: 100 * (p == y)[ok].mean() if ok.any() else float("nan")
    panels = [(rgb(i), f"{batch.ids[i]}"), (label_img(y), "label"), (label_img(p0), f"coarse pass ({acc(p0):.0f}% px)"),
              (label_img(p1), f"MiSTR ({acc(p1):.0f}% px)"), (rgb(i), f"{int(out['windows_per_sample'][i])} hi-res tokens")]
    for ax, (img, title) in zip(axes[i], panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=19, interpolation="nearest") if img.ndim == 2 else ax.imshow(img)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    m = (plan.b == i).cpu()
    for t, py, px in zip(plan.t[m].tolist(), plan.py[m].tolist(), plan.px[m].tolist()):
        axes[i, 4].add_patch(Rectangle((px - 0.5, py - 0.5), plan.patch, plan.patch, fill=False, lw=0.6,
                                       ec=plt.get_cmap("twilight")(t / 12)))
plt.tight_layout()'''),
    ("md", """## Where to go from here

- train this run (or any other of the paper's) from its exact config:
  `python -m exp.mistr.paper_runs train --match 'S8Tsea\\+S4Tmon_q50_woracle50_ep100_seed0'`
- Fig 03 (all datasets, all budgets) from the shipped results: `python -m exp.viz.round1_main --paper`
- every run in the paper: `exp/mistr/paper_configs.py`; every figure: README, *Reproduce the paper*"""),
]

GEOID = [
    ("md", """# MiSTR for flood tracking on GEOID-Flood

A sequence of 32 Sentinel-1 passes (same orbit, 512 x 512 px at 10 m, VV/VH) around a mapped flood;
the model labels every pass as dry / permanent water / flood (causal: a pass only sees earlier
ones). Fixed resolution encodes every pass at one patch size; MiSTR encodes all passes at 128 px
patches, then re-encodes the 32 px windows its selector ranks highest (budget q: a share of all
windows of the series) at 32 px.

Needs: `checkpoints/geoidchange/{trkO_128c32_qr_s0,trkO_fix128_cont_s0,trkO_fix32_cont_s0}.pt` and
`demo/geoid_test_series.npz` (two test series, the event at pass 16; see the README, *Demo*). A GPU."""),
    ("code", SETUP),
    ("code", '''from exp.demo import load_geoid, load_geoid_samples, predict_geoid
from exp.viz.geoid_common import miou, CMAP

names, items = load_geoid_samples("demo/geoid_test_series.npz")
C = "checkpoints/geoidchange"
models = {"fixed S128": (load_geoid(f"{C}/trkO_fix128_cont_s0.pt", 128), 0.0),
          "MiSTR S128 & S32, q 8 %": (load_geoid(f"{C}/trkO_128c32_qr_s0.pt", 128, 32), 0.08),
          "fixed S32": (load_geoid(f"{C}/trkO_fix32_cont_s0.pt", 32), 0.0)}
res = {(n, m): predict_geoid(net, it, q) for n, it in zip(names, items) for m, (net, q) in models.items()}
for (n, m), o in res.items():
    k = o["event"]
    print(f"{n:16} {m:24} 3-class mIoU on the labelled pass {100 * miou(o['pred'][k - 1], o['label']):5.1f}"
          f"   tokens {o['tokens']:5d}")'''),
    ("md", """The paper scores 1,000+ test series with the crop budget shared across a batch of 8 series
(so flood-free series spend little); here each series has its own budget, so MiSTR's numbers
differ slightly from the paper's per-series values.

Passes around the labelled date (VV backscatter; label outlines on its date; predictions per model):"""),
    ("code", '''show = range(-3, 4)            # passes around the labelled one (pass k; prediction for pass t = pred[t - 1])
for n, it in zip(names, items):
    o0 = res[(n, "fixed S128")]
    k, days = o0["event"], o0["days"]
    fig, axes = plt.subplots(1 + len(models), len(show), figsize=(1.6 * len(show), 1.7 * (1 + len(models))))
    for j, dk in enumerate(show):
        t = k + dk
        axes[0, j].imshow(it[0][t, 0], cmap="gray", vmin=-25, vmax=0)      # VV, dB
        axes[0, j].set_title("labelled pass" if dk == 0 else f"{days[t]:+.0f} d", fontsize=8)
        if dk == 0:
            axes[0, j].contour(o0["label"] == 2, levels=[0.5], colors="cyan", linewidths=0.5)
        for r, m in enumerate(models, 1):
            o = res[(n, m)]
            axes[r, j].imshow(o["pred"][t - 1] + 1, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
            if o["windows"] is not None:
                sel = o["windows"]["sel_t"] == t
                for y, x in zip(o["windows"]["win_y"][sel], o["windows"]["win_x"][sel]):
                    axes[r, j].add_patch(Rectangle((x - 0.5, y - 0.5), 32, 32, fill=False, lw=0.4, ec="w"))
            if j == 0:
                axes[r, j].set_ylabel(m, fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(n, fontsize=9); plt.tight_layout()'''),
]


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / name)
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    write("01_pastis_mistr.ipynb", PASTIS)
    write("02_geoid_flood_tracking.ipynb", GEOID)
