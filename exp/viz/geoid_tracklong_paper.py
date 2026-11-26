"""Paper figures 06 / 13: one GEOID-Flood long test series (round O, T = 32, no reference) in the 7-pass layout of
the Nov-11 renders (exp.viz.geoid_track_example): the passes k-3 .. k+3 around the labelled date (the model sees
all T passes, causal; only the window is drawn). The crop budget is batch-shared as in evaluation: the series is
run inside a batch of BATCH test series (exp.viz.geoid_common.run), its own crops drawn.

compact=1 (fig 06, main text): S1 VV (the label as class outlines on its date) + MiSTR's predictions.
compact=0 (fig 13, appendix):  S1 VV, the label row, fixed S_L, MiSTR S_L & S_C, fixed S_C (+ extra fixed).

    python -m exp.viz.geoid_tracklong_paper [rec=EMSR275-1-4_q1 k=16 show=-3:4 combo=64c32 q=0.04 batch=8
                                             compact=0 extra= out=figures/geoidchange/tracklong_paper/<stem>]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from exp.geoidchange.tiletrack import LongTrack  # noqa: E402
from exp.viz import geoid_common as tl  # noqa: E402
from exp.viz.geoid_common import CMAP, miou  # noqa: E402

CROP, GRIDC, INK, RED = "#e8590c", "#3d3d3a", "#0b0b0b", "#c0392b"
GREY = "#8a8984"
plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 8})


def grid(a, step, n=512):
    for g in range(step, n, step):
        a.axhline(g - .5, color=GRIDC, lw=0.25, alpha=0.5)
        a.axvline(g - .5, color=GRIDC, lw=0.25, alpha=0.5)


def blank(a, text=None):
    for sp in a.spines.values():
        sp.set_visible(False)
    if text:
        a.text(0.5, 0.5, text, fontsize=7.5, color=GREY, style="italic", ha="center", va="center",
               transform=a.transAxes)


def draw(ds, i, ms, show, compact, q, path):
    x = ds[i][0]
    preds, k, y, rd, valid = tl.run(ms, ds, i, 0)
    cols = [t for t in range(k + show[0], k + show[1]) if 0 <= t < len(x)]
    score = {key: (miou(preds[key][0][k - 1], y), tl.fbiou(preds[key][0][k - 1], y)) for key, *_ in ms}
    best = max(score, key=lambda m: sum(score[m]))
    rows = [m for m in ms if m[0] == "mix"] if compact else ms
    nr = 1 + (0 if compact else 1) + len(rows)
    fig, ax = plt.subplots(nr, len(cols), figsize=(6.875, 2.35 if compact else nr * 0.98 + 0.45),
                           gridspec_kw=dict(wspace=0.04, hspace=0.16 if compact else 0.2), squeeze=False)
    for c, t in enumerate(cols):
        a = ax[0, c]
        a.set_title("labeled event date" if t == k else f"{int(round(rd[t])):+d} d", fontsize=8, pad=3,
                    color=RED if t == k else INK, fontweight="bold" if t == k else "normal")
        a.imshow(x[t, 0], cmap="gray", vmin=-25, vmax=0)
        if compact and t == k:                                         # the label, as class outlines
            for cls, col in ((1, "#5cc8b8"), (2, "#2a78d6")):
                a.contour((y == cls).astype(float), levels=[0.5], colors=[col], linewidths=0.7)
        r0 = 1
        if not compact:
            a = ax[1, c]
            if t == k:
                a.imshow(y + 1, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
            else:
                blank(a, "no label")
            r0 = 2
        for r, (key, lab, ck, lr, cp, qq) in enumerate(rows):
            a = ax[r0 + r, c]
            if t == 0:
                blank(a, "no prediction" if r == len(rows) // 2 else None)
                continue
            p, bx, _, _ = preds[key]
            p = p[t - 1]
            a.imshow(p + 1, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
            grid(a, lr)
            if bx is not None:
                sel = bx["sel_t"] == t
                for yy, xx in zip(bx["win_y"][sel], bx["win_x"][sel]):
                    a.add_patch(Rectangle((xx - .5, yy - .5), cp, cp, fill=False, ec=CROP, lw=0.3))
            if t == k:
                mi, bi = score[key]
                a.set_title(f"mIoU {100 * mi:.0f}  flood bIoU {100 * bi:.0f}", fontsize=7, pad=2,
                            fontweight="bold" if (key == best and not compact) else "normal")
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    mix = next(m for m in ms if m[0] == "mix")
    L, C = mix[3], mix[4]

    def name(m):
        key, lab, ck, lr, cp, qq = m
        if key == "mix" and compact:           # fig 06: no token count
            return f"MiSTR\n$S_{{{L}}}$ & $S_{{{C}}}$"
        if key == "mix":
            lab = f"MiSTR $S_{{{L}}}$ & $S_{{{C}}}$"
        return f"{lab}\n{preds[key][2]} tokens"
    names = ["S1 VV"] + ([] if compact else ["GEOID-Flood label\n(one date only)"]) + [name(m) for m in rows]
    for r, n in enumerate(names):
        ax[r, 0].text(-0.06, 0.5, n, fontsize=8, ha="right", va="center", transform=ax[r, 0].transAxes)
    handles = ([Line2D([], [], color="#2a78d6", lw=1.2, label="label: flood"),
                Line2D([], [], color="#5cc8b8", lw=1.2, label="label: permanent water")] if compact else []) + [
        Patch(fc="#f7f7f5", ec="#9a9994", lw=0.5, label="dry"),
        Patch(fc="#5cc8b8", label="permanent water"), Patch(fc="#2a78d6", label="flood"),
        Line2D([], [], color=GRIDC, lw=0.6, alpha=0.7, label=f"$S_{{{L}}}$ token grid" if compact else "token grid"),
        Patch(fc="none", ec=CROP, lw=0.9, label=f"selected $S_{{{C}}}$ token")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7 if compact else 7.5,
               handlelength=1.3, handletextpad=0.4, bbox_to_anchor=(0.55, -0.03 if compact else -0.005),
               columnspacing=0.9)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.9 if compact else 0.95, bottom=0.1 if compact else 0.07)
    fig.savefig(path, dpi=200 if path.endswith(".png") else 300, bbox_inches="tight")
    plt.close(fig)
    print(path, {key: tuple(round(v, 3) for v in s) for key, s in score.items()},
          {key: preds[key][2] for key, *_ in ms}, flush=True)


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    seed, q, combo = int(kw.get("seed", 0)), float(kw.get("q", 0.04)), kw.get("combo", "64c32")
    prefix = kw.get("prefix", "trkO_")
    LongTrack.root = kw.get("long_root", "data/GEOID-Flood-tileseries-long2/cache")
    ds = LongTrack("test", 32)
    ds.k_post = int(kw.get("k", 16))
    tl.Q_UNIT, tl.BATCH = kw.get("q_unit", "series"), int(kw.get("batch", 8))
    ms = tl.models(32, 0, seed, q, combo, prefix, kw.get("suffix", "_qr"))
    for extra in kw.get("extra", "").split(","):
        if extra:
            ms.insert(-1 if int(extra) > ms[-1][3] else len(ms),
                      (f"fix{extra}", f"fixed $S_{{{extra}}}$", f"{prefix}fix{extra}_s{seed}", int(extra), int(extra), 0.0))
    if kw.get("fixed") == "cont":                  # training-matched fixed models (+8 epochs, appendix)
        ms = [(m[0], m[1], m[2].replace(f"_s{seed}", f"_cont_s{seed}"), *m[3:]) if m[5] == 0 else m for m in ms]
    ms = [m for m in ms if os.path.exists(f"{tl.CK}/{m[2]}.pt")]
    show = tuple(int(v) for v in kw.get("show", "-3:4").split(":"))
    rec = kw.get("rec", "EMSR275-1-4_q1")
    i = next(i for i, r in enumerate(ds.recs) if f"{r['key']}_q{r['quad']}" == rec)
    out = kw.get("out") or (f"figures/geoidchange/tracklong_paper/{rec}_{combo}_q{q:g}_k{ds.k_post}"
                            f"{'_compact' if kw.get('compact') == '1' else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw(ds, i, ms, show, kw.get("compact") == "1", q, out)


if __name__ == "__main__":
    main(sys.argv[1:])
