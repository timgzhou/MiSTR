"""MiSTR on TerraMind-1.0-tiny, space-only selection, in the style of fig 03
(exp.viz.datasets_tradeoff): MiSTR as stars, one colour per resolution pair, against the
fixed-resolution TerraMind models of the same (season) series only. Every token of a run comes
from the season composites (model.crop_time season); PASTIS, PLANTEUR, BioMassters, YieldSAT.
(The PASTIS month-series runs, model.coarse_time month, stay in results/pastis/mistr.csv but are
not drawn: see PANELS.)

    python -m exp.viz.terramind_tradeoff          -> figures/terramind_tradeoff.png
    python -m exp.viz.terramind_tradeoff paper    -> figures/paper/15_TerraMind-MiSTR.pdf (the
                                                     appendix figure: fig 03's fonts and sizes)
"""
import sys
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from exp.viz import common as rp  # noqa: E402
from exp.viz.datasets_tradeoff import FAM, PAIR_COL  # noqa: E402

S4S2_COL = plt.get_cmap("Set2").colors[6]   # (S4+S2: not in fig 03; the palette's next colour)

OUT = Path("figures/terramind_tradeoff.png")
PAPER_OUT = Path("figures/paper/15_TerraMind-MiSTR.pdf")
K, LABEL_PT = 1.0, None     # marker / line scale, point label size (the paper figure: smaller)
STAR = 13                   # MiSTR star size before K (the paper figure: smaller, the gap to
#                             the fixed curve stays visible)
NAME = re.compile(r"^(?P<kind>fixed|mistr)_(?P<tk>[a-z_]+?)_S(?P<ps>\d+)T(?P<t>sea|mon)"
                  r"(?:\+S(?P<cp>\d+)T(?P<ct>sea|mon)(?:-w(?P<w>\d+))?_q(?P<q>[\d.]+)_woracle\d+)?"
                  r"_terramind_tiny_ep\d+_seed(?P<seed>\d)_[0-9a-f]{8}$")
PANELS = [   # task, title, T of every token, metric, lower is better
    ("pastis", "PASTIS, season series", "sea", "mIoU", False),
    # (PASTIS month series dropped from the figure it repeats the season panel, and
    # the other datasets have no month runs; its runs stay in results/pastis/mistr.csv)
    ("planteur", "PLANTEUR, season series", "sea", "mIoU", False),
    ("biomassters", "BioMassters, season series", "sea", "RMSE", True),
    ("yieldsat_raw", "YieldSAT, season series", "sea", "RMSE", True),
]


def load(task) -> pd.DataFrame:
    """one row per configuration: means over the seeds that have a test cost (x = GMACs, y =
    test_primary), + xse / yse (standard error, 0 for one seed) and n (seeds)."""
    d = pd.read_csv(f"results/{task}/mistr.csv", low_memory=False).drop_duplicates("run", keep="last")
    c = pd.read_csv(f"results/{task}/test_cost.csv").drop_duplicates("run", keep="last")
    d = d.merge(c[["run", "gmacs_test"]], on="run", how="left", suffixes=("_row", ""))
    d = pd.concat([d, d.run.str.extract(NAME)], axis=1).dropna(subset=["kind"])
    # whole-test cost: test_cost.csv (measured afterwards, older runs) or the run's own row
    # (exp.mistr.train measures gmacs_test at the end of training, the seed 1/2 runs)
    row = d["gmacs_test_row"] if "gmacs_test_row" in d else pd.Series(float("nan"), index=d.index)
    d = d.assign(x=d.gmacs_test.fillna(row), ps=d.ps.astype(int), q=d.q.astype(float))
    if task in ("pastis", "planteur"):          # mIoU in 0-100, as the paper
        d["test_primary"] = d.test_primary * 100
    d = d[d.x.notna()]
    keys = ["kind", "tk", "ps", "t", "cp", "ct", "w", "q"]
    g = d.fillna({k: "" for k in ("cp", "ct", "w")}).fillna({"q": -1}).groupby(keys)
    out = g.agg(x=("x", "mean"), test_primary=("test_primary", "mean"), xse=("x", "sem"),
                yse=("test_primary", "sem"), n=("x", "size")).reset_index()
    out[["xse", "yse"]] = out[["xse", "yse"]].fillna(0)
    out = out.replace({"cp": {"": None}, "ct": {"": None}, "w": {"": None}, "q": {-1: float("nan")}})
    print(f"{task}: seeds per point {out.n.value_counts().sort_index().to_dict()}")
    return out


def panel(ax, task, title, t, metric, lower):
    d = load(task)
    f = d[(d.kind == "fixed") & (d.t == t)].sort_values("x")
    ls, mk, filled, label = FAM[t]
    ax.errorbar(f.x, f.test_primary, yerr=f.yse, fmt="none", ecolor=rp.INK, elinewidth=0.6,
                capsize=1.2, alpha=0.75, zorder=3)      # (+- 1 s.e. over seeds)
    ax.plot(f.x, f.test_primary, ls + mk, color=rp.INK, lw=1.2 * K, ms=7 * K, alpha=0.75,
            mfc=rp.INK if filled else rp.SURFACE, mec=rp.INK, mew=1.4 * K, zorder=3,
            label=label.replace("fixed resolution", "fixed TerraMind"))
    ends = (f.ps.max(), f.ps.min())
    for _, r in f.iterrows():
        if LABEL_PT and r.ps not in ends:     # (paper: the two ends named, as fig 03)
            continue
        ax.annotate(rf"$S_{{{r.ps}}}T_{{{t}}}$", (r.x, r.test_primary), xytext=(4 * K, -4 * K),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=LABEL_PT or 8.5, zorder=5)
    ours = d[(d.kind == "mistr") & (d.t == t) & (d.ct == t)]
    for (ps, cp, w), g in ours.fillna({"w": ""}).groupby(["ps", "cp", "w"]):
        g = g.sort_values("q")
        col = PAIR_COL.get((ps, f"c{w}p{cp}" if w else f"c{cp}"), S4S2_COL)
        ax.errorbar(g.x, g.test_primary, xerr=g.xse, yerr=g.yse, fmt="none", ecolor=col,
                    elinewidth=0.6, capsize=1.2, zorder=5)     # (GMACs vary: adaptive selection)
        # (paper: lone stars, as fig 03)
        ax.plot(g.x, g.test_primary, "*" if LABEL_PT else "-*", color=col, lw=2.4 * K,
                ms=STAR * K, mec=col, mew=0 if LABEL_PT else 1.2 * K,
                zorder=6, label=rf"MiSTR $S_{{{ps}}}T_{{{t}}}+S_{{{cp}}}T_{{{t}}}$")
        for _, r in g.iterrows() if not LABEL_PT else ():
            ax.annotate(f"q{r.q:g}", (r.x, r.test_primary), xytext=(-6 * K, 4 * K),
                        textcoords="offset points", ha="right", fontsize=(LABEL_PT or 8.5) - 1,
                        color=rp.MUTED,
                        zorder=7)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=rp.GRID, lw=0.8 * K)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if lower:
        ax.invert_yaxis()
    if LABEL_PT:            # (paper: titles, labels and legends set by paper())
        return
    ax.set_ylabel(f"test {metric}" + (" (flipped)" if lower else "") + " $\\rightarrow$")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("GMACs per sample, test (log scale)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=2.2)


PAPER_TITLE = {   # (task, series) -> panel title, subtitle (fig 03's two lines)
    ("pastis", "sea"): ("PASTIS", "crop types, mIoU, seasons"),
    ("pastis", "mon"): ("PASTIS", "crop types, mIoU, months"),
    ("planteur", "sea"): ("PLANTEUR", "crop types, mIoU"),
    ("biomassters", "sea"): ("BioMassters", "biomass, RMSE (flipped)"),
    ("yieldsat_raw", "sea"): ("YieldSAT", "yield, RMSE (flipped)"),
}
PAPER_PAIRS = [(16, "c8"), (12, "c12p6"), (8, "c4"), (6, "c6p3"), (4, "c4p2")]


def paper():
    """The appendix figure in fig 03's style (exp.viz.datasets_tradeoff.draw_paper): a 2 x 2
    grid in ONE CVPR column (3.25 in), Times text, STIX maths, TrueType fonts, lone stars, the fixed
    curves' two ends named, MiSTR's pairs (colour) and the fixed baselines in legends below."""
    global K, LABEL_PT, STAR
    K, LABEL_PT, STAR = 0.45, 6, 8
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        # (2 x 2 grid in one column, the two legends side by side below it)
        fig, grid = plt.subplots(2, 2, figsize=(3.3, 3.9), facecolor=rp.SURFACE)
        axes = grid.flat[:len(PANELS)]
        for i, (ax, spec) in enumerate(zip(axes, PANELS)):
            ax.set_facecolor(rp.SURFACE)
            panel(ax, *spec)
            ax.margins(0.1)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo, hi * 2.4)       # (room for the end labels)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(labelsize=6.5, pad=1.5)
            ax.set_xlabel("GMACs per sample")
            if i % 2 == 0:
                ax.set_ylabel("test metric $\\rightarrow$")
            name, sub = PAPER_TITLE[(spec[0], spec[2])]
            ax.set_title(name, loc="left", fontweight="bold", pad=10)
            ax.text(0, 1.02, sub, fontsize=6.5, transform=ax.transAxes, va="bottom")
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.94, wspace=0.32,
                            hspace=0.45)
        star = dict(marker="*", ls="none", ms=11 * K, mew=0)
        h = [plt.Line2D([], [], color=PAIR_COL.get(k, S4S2_COL), **star) for k in PAPER_PAIRS]
        lab = [rf"$S_{{{ps}}}$ & $S_{{{int(g[1:].split('p')[-1])}}}$" if "p" in g
               else rf"$S_{{{ps}}}$ & $S_{{{g[1:]}}}$" for ps, g in PAPER_PAIRS]
        leg = fig.legend(h, lab, loc="upper left", borderaxespad=0, ncol=3, frameon=False,
                         title="\u2605 MiSTR TerraMind (ours)",
                         title_fontproperties={"weight": "bold"}, alignment="left",
                         handlelength=1.2, columnspacing=1.2)
        fig.add_artist(leg)
        h = [plt.Line2D([], [], color=rp.INK, marker=FAM["sea"][1], ms=7 * K, ls=FAM["sea"][0],
                        lw=1.2 * K, mfc=rp.SURFACE, alpha=0.75)]
        leg2 = fig.legend(h, ["seasonal inputs"], loc="upper left", borderaxespad=0,
                          frameon=False, title="fixed-res TerraMind",
                          title_fontproperties={"weight": "bold"}, alignment="left",
                          handlelength=2.6)
        r = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        span = to_fig.transform_bbox(Bbox.union([ax.get_tightbbox(r) for ax in axes]))
        b1 = to_fig.transform_bbox(leg.get_window_extent(r))
        leg.set_bbox_to_anchor((span.x0, span.y0 - 0.01), transform=fig.transFigure)
        leg2.set_bbox_to_anchor((b1.x1 + 0.04, span.y0 - 0.01), transform=fig.transFigure)
        fig.savefig(PAPER_OUT, facecolor=rp.SURFACE, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    K, LABEL_PT, STAR = 1.0, None, 13
    print(PAPER_OUT)


def main():
    if sys.argv[1:] == ["paper"]:
        return paper()
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 5.2),
                             facecolor=rp.SURFACE)
    for ax, spec in zip(axes, PANELS):
        ax.set_facecolor(rp.SURFACE)
        panel(ax, *spec)
    fig.suptitle("MiSTR on TerraMind-1.0-tiny, space-only selection, mean +- s.e. over seeds",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150, facecolor=rp.SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
