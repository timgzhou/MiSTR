"""Temporal CutMix on a 32-pass GEOID-Flood training series, as the flood-tracking models see it (exp.geoidchange.tiletrack
LongTrack, data=long2, T = 32, the event at any position, no reference, flood-centred boxes, never pasted into pass 0).
round=O (default): passes within 28 d of the event dropped (training only), no ignore label, CutMix into
10-40 % of the passes. round=N: dense series, the event's flood px ignored within 30 d, 1-3 pasted passes.

The 32 passes in three blocks of up to 11; per block three rows: the unaugmented S1 VV (dB), the augmented input with the
pasted boxes outlined (dashed) -- each box copies the labelled date's pixels (VV + VH) onto another pass -- and the
per-pass training target: the GEOID-Flood label on its date and inside the pasted boxes; elsewhere the label's
permanent water, dry otherwise; the label's flood pixels ignored on passes within 30 d of the event; pass 0 has no
target. Augmentation is forced on (mix_p = 1) for the figure.

    python -m exp.viz.geoid_cutmix_long_example k=16 rec=EMSR352-1-282_q2 out=figures/paper/11_GEOID-Cutmix-sample
      (Fig 17; [split=train seed=0 rec=<series> out=<stem>])
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from exp.geoidchange.tiletrack import LongTrack  # noqa: E402

T, BLOCK = 32, 11                    # 11 passes per row: 3 blocks (16 too small, 8 too long)
LONG2 = "data/GEOID-Flood-tileseries-long2/cache"
PAPER_REC = "EMSR352-1-282_q2"          # the training series of Fig 17
MIX, INK, IGN, RED = "#e8590c", "#0b0b0b", "#c8c7c2", "#c0392b"
TCMAP = ListedColormap([IGN, "#f7f7f5", "#5cc8b8", "#2a78d6"])        # ignore, dry, permanent water, flood
plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 8})


def draw(ds, i, path):
    x, days, valid, k, y, rday, yt = ds[i]
    boxes, raw = ds.mixed, ds.x_raw
    nb = (T + BLOCK - 1) // BLOCK
    hr = sum([[1, 1, 1] + ([0.32] if b < nb - 1 else []) for b in range(nb)], [])   # spacer row between blocks
    fig, axs = plt.subplots(len(hr), BLOCK, figsize=(6.875, 6.875 / BLOCK * 0.98 * sum(hr) + 0.5),
                            gridspec_kw=dict(wspace=0.04, hspace=0.08, height_ratios=hr))
    for b in range(nb - 1):
        for a in axs[4 * b + 3]:
            a.axis("off")
    ax = np.stack([axs[4 * b + r] for b in range(nb) for r in range(3)])       # the 3 * nb image rows
    for t in range(T):
        b, c = divmod(t, BLOCK)
        r0 = 3 * b
        a0 = ax[r0, c]
        a0.set_title("labeled\nevent date" if t == k else f"{int(round(rday[t])):+d} d", fontsize=8, pad=2,
                     color=RED if t == k else INK, fontweight="bold" if t == k else "normal")
        a0.imshow(raw[t, 0], cmap="gray", vmin=-25, vmax=0)
        ax[r0 + 1, c].imshow(x[t, 0], cmap="gray", vmin=-25, vmax=0)
        a = ax[r0 + 2, c]
        if t == 0:
            for sp in a.spines.values():
                sp.set_visible(False)
            a.text(0.5, 0.5, "no\ntarget", fontsize=6.5, color="#8a8984", style="italic",
                   ha="center", va="center", transform=a.transAxes)
        else:
            a.imshow(yt[t - 1] + 1, cmap=TCMAP, vmin=0, vmax=3, interpolation="nearest")
        for tb, y0, y1, x0, x1 in boxes:
            if tb == t:
                for r in (1, 2):
                    ax[r0 + r, c].add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec=MIX,
                                                      lw=0.9, ls=(0, (3, 1.5))))
        if not valid[t]:
            a0.set_title("invalid", fontsize=6.5)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    for t in range(T, nb * BLOCK):                    # unused slots of the last row
        for r in range(3):
            ax[3 * (nb - 1) + r, t % BLOCK].axis("off")
    for b in range(nb):
        for r, n in enumerate(["S1 VV", "input\n(CutMix)", "target"]):
            ax[3 * b + r, 0].text(-0.06, 0.5, n, fontsize=8, ha="right", va="center", transform=ax[3 * b + r, 0].transAxes)
    handles = [Patch(fc="#f7f7f5", ec="#9a9994", lw=0.5, label="dry"),
               Patch(fc="#5cc8b8", label="permanent water"), Patch(fc="#2a78d6", label="flood"),
               Patch(fc=IGN, label="ignored (unlabeled)" if LongTrack.near_days == 0 else
                     "ignored (unlabelled; flood px within 30 d of the event)"),
               Patch(fc="none", ec=MIX, lw=0.9, ls=(0, (3, 1.5)), label="CutMix paste from the labeled event date")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8, handlelength=1.3,
               handletextpad=0.4, bbox_to_anchor=(0.55, -0.025), columnspacing=0.9)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.985, bottom=0.04)
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(path, "k", k, "boxes at passes", sorted({b[0] for b in boxes}))


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    split, seed = kw.get("split", "train"), int(kw.get("seed", 0))
    LongTrack.root, LongTrack.min_hist, LongTrack.ref_n = LONG2, 30.0, 0
    LongTrack.mix_p, LongTrack.keep_raw, LongTrack.mix_ref, LongTrack.mix_flood = 1.0, True, False, True
    if kw.get("round", "O") == "O":              # round O: no passes within 28 d of the event, no ignore label,
        LongTrack.gap_days, LongTrack.near_days = 28.0, 0.0       # CutMix into 10-40 % of the passes
        LongTrack.mix_frac = (0.1, 0.4)
    else:                                         # round N: dense, flood px ignored within 30 d, 1-3 pasted passes
        LongTrack.gap_days, LongTrack.near_days, LongTrack.mix_frac = 0.0, 30.0, None
    ds = LongTrack(split, T, augment=True)
    ds.k_post = int(kw.get("k", 16))
    out = "figures/geoidchange/cutmix_long_examples"
    os.makedirs(out, exist_ok=True)
    rec = kw.get("rec", PAPER_REC)
    i = next(i for i, r in enumerate(ds.recs) if f"{r['key']}_q{r['quad']}" == rec)
    np.random.seed(seed * 1000 + i)               # the CutMix draw (boxes, passes) of this series
    draw(ds, i, kw.get("out") or f"{out}/{split}_{rec}")


if __name__ == "__main__":
    main(sys.argv[1:])
