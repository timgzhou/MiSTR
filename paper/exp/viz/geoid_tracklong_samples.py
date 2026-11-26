"""Fig 18 (App. K): two fixed GEOID-Flood test series with a flood (round O, T = 32, no reference),
one block of 4 rows each: S1 VV with the label as class outlines on its date (as fig 06), fixed S_L, MiSTR S_L & S_C,
fixed S_C (scores on the labeled date in the label column); passes k-3 .. k+3 around the labelled date (the model sees all T passes, causal). Fixed models default to
the training-matched +8-epoch continuations (fixed=cont). Model outputs are cached per series (GPU needed only once):
cache/tracklong_samples/<rec>_<combo>_q<q>_k<k>_<fixed>.pkl.

Series (RECS): EMSR417-1-5_q2 (persistent flood: visible -18 .. +6 d) and EMSR456-5-2_q0 (short-lived: one
capture).

    python -m exp.viz.geoid_tracklong_samples [recs=A,B,C combo=128c32 q=0.08 k=16 fixed=cont batch=8
                                               out=figures/paper/13_GEOID-sample-full.pdf]
"""
import os
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from exp.viz.geoid_tracklong_paper import CROP, GRIDC, INK, RED, grid  # noqa: E402
from exp.viz.geoid_common import CMAP  # noqa: E402

RECS = "EMSR417-1-5_q2,EMSR456-5-2_q0"   # paper: persistent + short-lived flood
CACHE = "cache/tracklong_samples"
plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 8})


def compute(rec, kw):
    """x window, label, predictions, crops, tokens, scores of one series (cached)."""
    combo, q, k_post, fixed = kw["combo"], float(kw["q"]), int(kw["k"]), kw["fixed"]
    path = f"{CACHE}/{rec}_{combo}_q{q:g}_k{k_post}_{fixed or 'base'}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from exp.geoidchange.tiletrack import LongTrack
    from exp.viz import geoid_common as tl
    from exp.viz.geoid_common import miou
    LongTrack.root = "data/GEOID-Flood-tileseries-long2/cache"
    ds = LongTrack("test", 32)
    ds.k_post = k_post
    tl.Q_UNIT, tl.BATCH = "series", int(kw["batch"])
    ms = tl.models(32, 0, 0, q, combo, "trkO_", "_qr")
    if fixed == "cont":
        ms = [(m[0], m[1], m[2].replace("_s0", "_cont_s0"), *m[3:]) if m[5] == 0 else m for m in ms]
    i = next(i for i, r in enumerate(ds.recs) if f"{r['key']}_q{r['quad']}" == rec)
    preds, k, y, rd, valid = tl.run(ms, ds, i, 0)
    x = ds[i][0]
    out = dict(rec=rec, k=k, y=y, rd=rd, x=np.asarray(x[:, 0]), ms=ms, toks={m[0]: preds[m[0]][2] for m in ms},
               pred={m[0]: preds[m[0]][0] for m in ms}, boxes={m[0]: preds[m[0]][1] for m in ms},
               score={m[0]: (miou(preds[m[0]][0][k - 1], y), tl.fbiou(preds[m[0]][0][k - 1], y)) for m in ms})
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


def draw(samples, show, path):
    """Full text width (6.875 in): scores sit in the label column so the rows stay tight (2 samples per page)."""
    nrow, W = 4, show[1] - show[0]
    left, gap, fig_w = 1.08, 0.5, 6.875                   # label column (in), gap rows between samples (cells)
    cw = (fig_w - left) / W / 1.0                          # cell (in)
    heights = sum([[1] * nrow + ([gap] if b < len(samples) - 1 else []) for b in range(len(samples))], [])
    fig_h = sum(heights) * cw * 1.04
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(heights), W, left=left / fig_w, right=1.0, top=1.0, bottom=0.0,
                          wspace=0.03, hspace=0.05, height_ratios=heights)
    r0 = 0
    for b, s in enumerate(samples):
        k, y, rd, x = s["k"], s["y"], s["rd"], s["x"]
        L, C = s["ms"][1][3], s["ms"][1][4]
        rows = [("S1 VV", None)] + [(m[0], m) for m in s["ms"]]
        for c, t in enumerate(range(k + show[0], k + show[1])):
            for r, (key, m) in enumerate(rows):
                a = fig.add_subplot(gs[r0 + r, c])
                a.set_xticks([]), a.set_yticks([])
                if m is None:
                    a.imshow(x[t], cmap="gray", vmin=-25, vmax=0)
                    a.set_title("labeled event date" if t == k else f"{int(round(rd[t])):+d} d", fontsize=7.5,
                                pad=2, color=RED if t == k else INK, fontweight="bold" if t == k else "normal")
                    if t == k:
                        for cls, col in ((1, "#5cc8b8"), (2, "#2a78d6")):
                            a.contour((y == cls).astype(float), levels=[0.5], colors=[col], linewidths=0.6)
                    if c == 0:                              # sample header, in the label column at title height
                        a.text(-left / cw, 1.0 + 2 / 72 / cw, f"Sample {b + 1}: {s['rec'].split('-')[0]}",
                               fontsize=8.5, fontweight="bold", ha="left", va="bottom", transform=a.transAxes)
                else:
                    a.imshow(s["pred"][key][t - 1] + 1, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
                    grid(a, m[3])
                    bx = s["boxes"][key]
                    if bx is not None:
                        sel = bx["sel_t"] == t
                        for yy, xx in zip(bx["win_y"][sel], bx["win_x"][sel]):
                            a.add_patch(Rectangle((xx - .5, yy - .5), m[4], m[4], fill=False, ec=CROP, lw=0.3))
                if c == 0:
                    if m is None:
                        name = "S1 VV + label"
                    else:
                        mi, bi = s["score"][key]
                        name = (f"MiSTR $S_{{{L}}}$ & $S_{{{C}}}$" if key == "mix" else f"fixed $S_{{{m[3]}}}$") + \
                            f"\n{s['toks'][key] / 1000:.1f}k tokens\nmIoU {100 * mi:.0f}\nflood bIoU {100 * bi:.0f}"
                    a.text(-0.06, 0.5, name, fontsize=7.5, ha="right", va="center", transform=a.transAxes,
                           linespacing=1.15)
        r0 += nrow + 1
    handles = [Line2D([], [], color="#2a78d6", lw=1.2, label="label: flood"),
               Line2D([], [], color="#5cc8b8", lw=1.2, label="label: permanent water"),
               Patch(fc="#f7f7f5", ec="#9a9994", lw=0.5, label="dry"),
               Patch(fc="#5cc8b8", label="permanent water"), Patch(fc="#2a78d6", label="flood"),
               Line2D([], [], color=GRIDC, lw=0.6, alpha=0.7, label="token grid"),
               Patch(fc="none", ec=CROP, lw=0.9, label=f"selected $S_{{{C}}}$ token")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5 + left / fig_w / 2, -0.004), ncol=4,
               frameon=False, fontsize=7.5, handlelength=1.3, handletextpad=0.4, columnspacing=1.2)
    fig.savefig(path, dpi=200 if path.endswith(".png") else 300, bbox_inches="tight")
    plt.close(fig)
    print(path, f"{fig_w:.2f} x {fig_h:.2f} in", flush=True)


def main(argv):
    kw = dict(recs=RECS, combo="128c32", q="0.08", k="16", fixed="cont", batch="8", show="-3:4",
              out="figures/geoidchange/tracklong_samples/13_samples.pdf") | dict(a.split("=", 1) for a in argv)
    samples = [compute(r, kw) for r in kw["recs"].split(",")]
    for s in samples:
        print(s["rec"], {m: tuple(round(100 * v, 1) for v in sc) for m, sc in s["score"].items()}, s["toks"])
    os.makedirs(os.path.dirname(kw["out"]), exist_ok=True)
    draw(samples, tuple(int(v) for v in kw["show"].split(":")), kw["out"])


if __name__ == "__main__":
    main(sys.argv[1:])
