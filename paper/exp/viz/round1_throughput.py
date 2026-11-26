"""Paper figure: accuracy vs MEASURED inference time per sample (full text width, one row):

    [shared legend] [Tiny PASTIS] [Tiny YieldSAT] [Base PASTIS]

each panel's subtitle naming its backbone.
Fig 04's hues (exp.viz.baselines_r1): MiSTR = orange stars (every resolution
pair and budget alike), APT / AnyUp / vid-TLDR in their colours, the fixed-resolution models in
ink (seasonal hollow squares, monthly filled diamonds) as lone points, not joined.

Runs: round 1 seed 0 (exp.viz.round1_main.rows) for Tiny; the `_base_` round (seed 0; no S14,
no S8+S4) for Base. Baselines: Tiny panels (PASTIS, YieldSAT), Fig 04's set (exp.viz.baselines_r1.baselines:
best-val lr; YieldSAT hides baseline points with RMSE > 1.6; APT, AnyUp fine-tuned, vid-TLDR training-free; Upsample Anything has no throughput
measurement). Fixed points beyond 2x MiSTR's slowest are left out, as Fig 03 does.

Time per sample = 1000 / samples_s from exp.bench.mistr_throughput (results/<task>/throughput.csv:
L40S, test split, data already on the GPU, mistr_paper.core.FAST on), each run at its fastest batch
size of 32 / 64 / 128 / 256. A run without a measurement takes its configuration's (another
seed's); configurations never measured are left out.

    python -m exp.viz.round1_throughput  -> figures/paper/21_MiSTR-throughput.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import common as bp  # noqa: E402
from exp.viz import baselines_r1 as b1  # noqa: E402
from exp.viz import datasets_tradeoff as dt  # noqa: E402
from exp.viz import common as rp  # noqa: E402
from exp.viz import round1_main as r1  # noqa: E402

# (task, backbone, dataset, subtitle, lower is better)
PANELS = [("pastis", "tiny", "PASTIS", "OlmoEarth Tiny, mIoU", False),
          ("yieldsat_raw", "tiny", "YieldSAT", "OlmoEarth Tiny, RMSE (flipped)", True),
          ("pastis", "base", "PASTIS", "OlmoEarth Base, mIoU", False)]
BASELINES = ("apt", "anyup", "tldr")
FIXED = {"sea": ("s", rp.SURFACE), "mon": ("D", rp.INK)}       # marker, face
CLIP = {("yieldsat_raw", "tiny"): 1.6}   # baseline points worse than this are hidden (axis range)
TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10)
STAR, DOT = 60, 20                                              # pt^2
OUT = "figures/paper/21_MiSTR-throughput.pdf"


def cfg(run):
    return run.str.replace(r"_seed\d+_[0-9a-f]+$", "", regex=True)


def base_rows(task):
    """round1_main.rows for the OlmoEarth Base round: its runs carry `_base_` in the name."""
    d = pd.read_csv(f"results/{task}/mistr.csv", low_memory=False)
    if "eval_percentage" in d:
        d = d[d.eval_percentage.isna()]
    d = d[d["model.backbone"].fillna("tiny") == "base"].drop_duplicates("run", keep="last")
    d = d[d.run.str.replace("_base_", "_").str.match(r1.ROUND1)]
    return r1.pct(r1.with_test_cost(d, task), task)       # (mIoU in 0-100, as r1.rows)


def rows(task, backbone):
    """round1_main's rows with the `gmacs` column holding ms per sample (cost_on_test: as is)."""
    d = r1.rows(task) if backbone == "tiny" else base_rows(task)
    f = Path(f"results/{task}/throughput.csv")
    if not f.exists():
        return d.iloc[:0]
    t = pd.read_csv(f).groupby("run").samples_s.max()
    by_cfg = t.groupby(cfg(t.index.to_series())).mean()
    s = d.run.map(t).fillna(cfg(d.run).map(by_cfg))
    d = d.assign(gmacs=1000 / s, cost_on_test=True)
    return d[d.gmacs.notna()]


def panel(ax, task, backbone, name, sub, lower):
    d = rows(task, backbone)
    ours = [p for k in dt.PAIRS for p in r1.mistr(d, task, *k, None)]
    top = 2 * max(x for x, _ in ours)
    fx = [t for t in r1.fixed(task, d) if t[2] <= top]
    print(f"== {name} {backbone}")
    for fam, (mk, face) in FIXED.items():
        p = [(x, y) for f, _, x, y in fx if f == fam]
        ax.scatter(*zip(*p), s=16, marker=mk, facecolors=face, edgecolors=rp.INK,
                   linewidths=.8, alpha=.75, zorder=2)
        for f, ps, x, y in sorted(t for t in fx if t[0] == fam):
            print(f"  fixed S{ps}T{fam}  {x:6.3f} ms  {y:.4f}")
    if backbone == "tiny":
        g = b1.baselines(d, lower)
        for key in BASELINES:
            q = g[key]
            if (task, backbone) in CLIP:
                c = CLIP[task, backbone]
                out = q.test_primary > c if lower else q.test_primary < c
                if out.any():
                    print(f"  {key}: hiding {out.sum()} beyond {c}: "
                          f"{q[out].test_primary.round(3).tolist()}")
                q = q[~out]
            ax.scatter(q.gmacs, q.test_primary, s=DOT, color=bp.COLOR[key], alpha=.45,
                       edgecolors=bp.COLOR[key], linewidths=.8, zorder=3)
            print(f"  {key}: {len(q)} configurations, {q.gmacs.min():.2f}-{q.gmacs.max():.2f} ms")
    ax.scatter(*zip(*ours), s=STAR, marker="*", color=bp.OURS, edgecolors=rp.SURFACE,
               linewidths=.4, zorder=5)
    for x, y in sorted(ours):
        print(f"  MiSTR {x:6.3f} ms  {y:.4f}")

    ax.set_xscale("log")
    ax.margins(.08)
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in TICKS if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    if lower:
        ax.invert_yaxis()                   # (up is better in every panel)
    ax.set_box_aspect(1)
    ax.set_facecolor(rp.SURFACE)
    ax.grid(alpha=.2, lw=.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("latency/sample (ms, 1/throughput)")
    ax.set_title(name, loc="left", fontweight="bold", pad=11)
    # (the subtitle in runs: "Base" in bold, so the right-hand panel reads as the other model)
    x, r = 0, ax.figure.canvas.get_renderer()
    for part in __import__("re").split(r"(Base)", sub):
        t = ax.text(x, 1.015, part, transform=ax.transAxes, va="bottom",
                    fontweight="bold" if part == "Base" else "normal")
        x += t.get_window_extent(r).width / ax.get_window_extent(r).width


def legend(ax):
    ax.axis("off")
    star = plt.Line2D([], [], ls="none", marker="*", ms=9, color=bp.OURS, mec=rp.SURFACE,
                      mew=.4)
    dots = [plt.Line2D([], [], ls="none", marker="o", ms=4.6, color=bp.COLOR[k], alpha=.45,
                       mec=bp.COLOR[k], mew=.8) for k in BASELINES]
    fixed = [plt.Line2D([], [], ls="none", marker=mk, ms=4, mfc=face, mec=rp.INK, mew=.8,
                        alpha=.75) for mk, face in FIXED.values()]
    labels = ([b1.NAMES["mistr"]] + [b1.NAMES[k] for k in BASELINES]
              + [b1.NAMES["sea"], b1.NAMES["mon"]])
    ax.legend([star, *dots, *fixed], labels, loc="center left", frameon=False, handlelength=1,
              handletextpad=.5, labelspacing=.55, borderaxespad=0)


def main():
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(6.875, 2.15), facecolor=rp.SURFACE)
        gs = fig.add_gridspec(1, 4, width_ratios=[.58, 1, 1, 1], wspace=.38, left=.0,
                              right=.995, bottom=.2, top=.86)
        for spec, cell in zip(PANELS, (1, 2, 3)):
            panel(fig.add_subplot(gs[cell]), *spec)
        legend(fig.add_subplot(gs[0]))
        fig.axes[0].set_ylabel("test metric $\\rightarrow$")
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, facecolor=rp.SURFACE, bbox_inches="tight", pad_inches=.02)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
