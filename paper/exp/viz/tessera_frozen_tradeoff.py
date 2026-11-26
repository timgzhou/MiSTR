"""Frozen Tessera-v2-small on PASTIS: test mIoU vs GMACs, fixed temporal resolution vs MiSTR
(time-only, per-pixel window selection), one panel per head trained on the cached embeddings.

    python -m exp.viz.tessera_frozen_tradeoff     -> figures/tessera_frozen_tradeoff.png (all pairs per
                                                     head) + _grid.png (heads x pairs, zoomed)
                                                     + figures/tessera_frozen_simple.png (UNet,
                                                     composites vs task-tuned MiSTR, stars)
    python -m exp.viz.tessera_frozen_tradeoff paper  -> figures/paper/17_Tessera-MiSTR.pdf (appendix,
                                                     after TerraMind: PASTIS only, one CVPR column,
                                                     fig 15's fonts / sizes / legend layout)

Reads results/tessera/pastis_frozen{,_mlp,_linear}.csv (exp.tessera.head). x = encoder + head
GMACs per 128 px patch (the head is the same for every point of a panel). Drawn:
  fixed       grey: composites sea -> mon -> semi -> raw (filled, solid) and K evenly spaced
              raw frames uni8 -> 16 -> 24 -> raw (hollow, dashed)
  per pair    colour = coarse > fine (sea>mon, sea>raw, mon>raw)
    attn      solid line, circles: pretrained-pool selection at 0.25 / 0.5 / 1 / 1.5 / 2 (/3)
              windows per px (below 1: only the most uncertain pixels re-encode, by entropy)
    attn-rpx  +: the same at 0.25 / 0.5 with random pixels
    tattn     dashed line, diamonds: task-tuned pool queries (exp.tessera.pooltune), if present
    fixed win dotted line, hollow squares: the same windows for every pixel (the train-split
              attention ranking; for sea that is Jul-Sep at 1 window)
    random    x: one random window per pixel
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = Path("results/tessera")
OUT = Path("figures/tessera_frozen_tradeoff.png")
OUT_GRID = Path("figures/tessera_frozen_tradeoff_grid.png")
OUT_SIMPLE = Path("figures/tessera_frozen_simple.png")
HEADS = [("unet", "UNet head (spatial context)", "pastis_frozen.csv"),
         ("mlp", "per-pixel MLP head", "pastis_frozen_mlp.csv"),
         ("linear", "per-pixel linear head", "pastis_frozen_linear.csv")]
PAIRS = ["sea>mon", "sea>raw", "mon>raw"]
COL = dict(zip(PAIRS, plt.get_cmap("Set2").colors[:3]))
COL.update({"bimon>mon": "#e7298a", "mon>semi": "#66a61e", "semi>raw": "#d95f02"})
SIMPLE_PAIRS = ["sea>mon", "bimon>mon", "mon>semi", "mon>raw", "semi>raw"]
LEVEL_NAME = {"sea": "seasonal", "bimon": "2-month", "mon": "monthly", "semi": "half-month", "raw": "raw"}
FIXED_LABEL = {"sea": "sea", "mon": "mon", "semi": "½-mon", "raw": "raw",
               "uni8": "8", "uni16": "16", "uni24": "24"}
GREY = "0.45"


def load(fname):
    """one row per config: the mean over head seeds (exp.tessera.head --seed; embeddings, pool
    queries and picks are seed 0's), + <metric>_se (standard error, 0 for one seed) and n_seeds."""
    df = pd.read_csv(RES / fname).drop_duplicates(["config", "seed"], keep="last")
    df["gmacs"] = df.encoder_gmacs + df.head_gmacs
    if "test_miou" in df:                     # mIoU in 0-100, as the paper
        df["test_miou"] = df.test_miou * 100
    g = df.groupby("config")
    out = g[[c for c in df.select_dtypes("number").columns if c != "seed"]].mean()
    for c in ("test_miou", "test_rmse"):
        if c in df:
            out[f"{c}_se"] = g[c].sem().fillna(0)
    out["n_seeds"] = g.size()
    return out


def series(df, pair, pick, budgets):
    rows = [(b, df.loc[f"{pair}-b{b}-{pick}"]) for b in budgets if f"{pair}-b{b}-{pick}" in df.index]
    return [b for b, _ in rows], [r.gmacs for _, r in rows], [r.test_miou for _, r in rows]


def fixed_window(df, pair):
    """the windows of the train-split attention ranking, the same for every pixel."""
    keys = [f"{pair}-b0.25-fixattn", f"{pair}-b0.5-fixattn"] + \
        ([f"{pair}-b1-fix3", f"{pair}-b2-fixattn"] if pair.startswith("sea")
         else [f"{pair}-b{b}-fixattn" for b in (1, 2, 3)])
    keys = [k for k in keys if k in df.index]
    return [df.loc[k].gmacs for k in keys], [df.loc[k].test_miou for k in keys]


def panel(ax, df, title, pairs=PAIRS):
    fx = df[df.index.str.startswith("fix-")].sort_values("gmacs")
    comp = fx.loc[[c for c in ("fix-sea", "fix-mon", "fix-semi", "fix-raw") if c in fx.index]]
    uni = fx.loc[[c for c in ("fix-uni8", "fix-uni16", "fix-uni24", "fix-raw") if c in fx.index]]
    ax.plot(comp.gmacs, comp.test_miou, "-", color=GREY, lw=1.2, zorder=1)
    ax.plot(uni.gmacs, uni.test_miou, "--", color=GREY, lw=1, zorder=1)
    for cfg, r in fx.iterrows():
        lv = cfg[4:]
        hollow = lv.startswith("uni")
        ax.plot(r.gmacs, r.test_miou, "o", ms=5, mec=GREY, mfc="white" if hollow else GREY, zorder=2)
        ax.annotate(FIXED_LABEL.get(lv, lv), (r.gmacs, r.test_miou), xytext=(3, -9),
                    textcoords="offset points", fontsize=7, color="0.35")
    budgets = ("0.25", "0.5", "1", "1.5", "2", "3")
    for pair in pairs:
        c = COL[pair]
        bs, x, y = series(df, pair, "attn", budgets)
        ax.plot(x, y, "-o", color=c, lw=1.6, ms=5, zorder=4)
        for b, xi, yi in zip(bs, x, y):
            ax.annotate(b, (xi, yi), xytext=(-4, 5), textcoords="offset points", fontsize=6.5, color=c)
        _, x, y = series(df, pair, "tattn", budgets)
        if x:
            ax.plot(x, y, "--D", color=c, lw=1.4, ms=4.5, zorder=4)
        x, y = fixed_window(df, pair)
        ax.plot(x, y, ":s", color=c, lw=1.2, ms=5, mfc="white", zorder=3)
        _, x, y = series(df, pair, "attn-rpx", budgets)
        ax.plot(x, y, "+", color=c, ms=8, mew=1.5, zorder=3)
        k = f"{pair}-b1-rand"
        if k in df.index:
            ax.plot(df.loc[k].gmacs, df.loc[k].test_miou, "x", color=c, ms=6, mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 3000, 5000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if len(pairs) == 1:     # zoom to this pair's range (+ the fixed points around it)
        xs = [v for k, v in df.gmacs.items() if k.startswith(pairs[0])]
        lo, hi = min(xs) / 1.35, max(xs) * 1.35
        sel = df[(df.gmacs >= lo) & (df.gmacs <= hi)]
        ax.set_xlim(lo, hi)
        ax.set_ylim(sel.test_miou.min() - .8, sel.test_miou.max() + .8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("GMACs per 128 px patch (encoder + head)")
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


DS_TITLE = {"pastis": "PASTIS", "planteur": "PLANTEUR", "biomassters": "BioMassters (centre 128 px)",
            "yieldsat": "YieldSAT (64 px field tiles)"}
DS_ORDER = ("pastis", "planteur", "biomassters", "yieldsat")


def draw_simple(ax, dataset, legend=True):
    """UNet head only; fixed = composites (what MiSTR's tokens are); MiSTR = task-tuned pool
    attention, unconnected stars labelled by windows per pixel, one colour per pair."""
    f = RES / f"{dataset}_frozen.csv"
    if not f.exists():
        ax.set_axis_off(); return False
    df = load(f.name)
    metric = "test_miou" if "test_miou" in df.columns else "test_rmse"
    df["test_miou"] = df[metric]          # (the y column the helpers read)
    comp = df.loc[[c for c in ("fix-sea", "fix-bimon", "fix-mon", "fix-semi", "fix-raw") if c in df.index]]
    comp = comp.sort_values("gmacs")
    ax.plot(comp.gmacs, comp.test_miou, "-o", color=GREY, lw=1.2, ms=5, zorder=1)
    for cfg, r in comp.iterrows():
        ax.annotate(LEVEL_NAME[cfg[4:]], (r.gmacs, r.test_miou), xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color="0.35")
    pairs = [p for p in SIMPLE_PAIRS if any(k.startswith(p + "-") and k.endswith("-tattn") for k in df.index)]
    for pair in pairs:
        bs, x, y = series(df, pair, "tattn", ("0.25", "0.5", "1", "1.5", "2", "3"))
        ax.plot(x, y, "*", color=COL[pair], ms=12, mec="white", mew=0.6, lw=0, zorder=3)
        for b, xi, yi in zip(bs, x, y):
            ax.annotate(b, (xi, yi), xytext=(-3, 7), textcoords="offset points", fontsize=7, color=COL[pair])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("GMACs per sample (encoder + head)")
    ax.set_ylabel("test mIoU" if metric == "test_miou" else "test RMSE (t/ha, lower is better)")
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(handles=[Line2D([], [], color=GREY, marker="o", lw=1.2, label="fixed resolution (composites)")] +
                  [Line2D([], [], color=COL[p], marker="*", ms=12, lw=0,
                          label="MiSTR " + " → ".join(LEVEL_NAME[x] for x in p.split(">"))) for p in pairs],
                  loc="lower right" if metric == "test_miou" else "upper right", fontsize=7.5, frameon=False)
    ax.set_title(DS_TITLE[dataset], fontsize=10)
    return True


def simple(dataset="pastis"):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if not draw_simple(ax, dataset):
        plt.close(fig); return
    ax.set_title(f"Frozen Tessera-v2-small on {DS_TITLE[dataset]} (UNet head, 1 seed)\n"
                 "MiSTR: task-tuned attention; numbers = windows per pixel", fontsize=9.5)
    fig.tight_layout()
    out = OUT_SIMPLE if dataset == "pastis" else OUT_SIMPLE.with_name(f"tessera_frozen_simple_{dataset}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


def simple_all():
    """the four datasets in one 2 x 2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, ds in zip(axes.flat, DS_ORDER):
        draw_simple(ax, ds)
    fig.suptitle("Frozen Tessera-v2-small: fixed temporal resolution (composites) vs MiSTR with task-tuned "
                 "attention (UNet head, 1 seed; numbers = windows per pixel)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_SIMPLE.with_name("tessera_frozen_simple_all.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


PAPER_OUT = Path("figures/paper/17_Tessera-MiSTR.pdf")
PAPER_PAIRS = ["sea>mon", "bimon>mon", "mon>semi", "mon>raw", "semi>raw"]   # coarse -> fine
PAPER_LEVEL = {"sea": "season", "bimon": "2-month", "mon": "month", "semi": "half-month", "raw": "raw"}


def paper():
    """PASTIS, frozen Tessera-v2-small, UNet head: composites at each temporal resolution vs MiSTR
    with task-tuned decoder attention (lone stars, one colour per pair, Set2 in pair order as
    fig 15), in fig 15's style (exp.viz.terramind_tradeoff.paper)."""
    from exp.viz import common as rp
    from matplotlib.ticker import LogLocator, NullFormatter
    K = 0.45
    df = load("pastis_frozen.csv")
    comp = df.loc[[f"fix-{lv}" for lv in PAPER_LEVEL]].sort_values("gmacs")
    col = dict(zip(PAPER_PAIRS, plt.get_cmap("Set2").colors))
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.3, 2.3), facecolor=rp.SURFACE)
        ax.set_facecolor(rp.SURFACE)
        ax.plot(comp.gmacs, comp.test_miou, "--s", color=rp.INK, lw=1.2 * K, ms=7 * K, alpha=0.75,
                mfc=rp.SURFACE, mec=rp.INK, mew=1.4 * K, zorder=3)
        ax.errorbar(comp.gmacs, comp.test_miou, yerr=comp.test_miou_se, fmt="none", ecolor=rp.INK,
                    elinewidth=0.6, capsize=1.2, alpha=0.75, zorder=3)     # (+- 1 s.e. over head seeds)
        for cfg, r in comp.iterrows():
            low = cfg == "fix-sea"           # (season: beside the point, clear of the x axis)
            ax.annotate(PAPER_LEVEL[cfg[4:]], (r.gmacs, r.test_miou), xytext=(4, 0) if low else (3, -5),
                        textcoords="offset points", ha="left", va="center" if low else "top",
                        fontsize=6, zorder=5)
        for pair in PAPER_PAIRS:
            bs, x, y = series(df, pair, "tattn", ("0.25", "0.5", "1", "1.5", "2", "3"))
            e = [df.loc[f"{pair}-b{b}-tattn", "test_miou_se"] for b in bs]
            ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col[pair], elinewidth=0.6, capsize=1.2, zorder=5)
            ax.plot(x, y, "*", color=col[pair], ms=5, mew=0, zorder=6)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(0.08)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 1.25)          # (room for the "raw" label)
        ax.tick_params(pad=1.5)
        ax.grid(True, color=rp.GRID, lw=0.8 * K)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("GMACs per sample")
        ax.set_ylabel("test mIoU $\\rightarrow$")
        ax.set_title("PASTIS", loc="left", fontweight="bold", pad=10)
        ax.text(0, 1.02, "crop types, mIoU, frozen Tessera v2 + UNet head", fontsize=6.5,
                transform=ax.transAxes, va="bottom")
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.88)
        star = dict(marker="*", ls="none", ms=11 * K, mew=0)
        h = [Line2D([], [], color=col[p], **star) for p in PAPER_PAIRS]
        lab = [" & ".join(PAPER_LEVEL[x] for x in p.split(">")) for p in PAPER_PAIRS]
        leg = fig.legend(h, lab, loc="upper left", borderaxespad=0, ncol=2, frameon=False,
                         title="\u2605 MiSTR Tessera v2 (ours)", title_fontproperties={"weight": "bold"},
                         alignment="left", handlelength=1.2, columnspacing=1.0)
        fig.add_artist(leg)
        h = [Line2D([], [], color=rp.INK, marker="s", ms=7 * K, ls="--", lw=1.2 * K,
                    mfc=rp.SURFACE, alpha=0.75)]
        leg2 = fig.legend(h, ["composites"], loc="upper left", borderaxespad=0, frameon=False,
                          title="fixed-res Tessera v2", title_fontproperties={"weight": "bold"},
                          alignment="left", handlelength=2.6)
        from matplotlib.transforms import Bbox
        r = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        span = to_fig.transform_bbox(Bbox.union([ax.get_tightbbox(r)]))
        leg.set_bbox_to_anchor((span.x0, span.y0 - 0.02), transform=fig.transFigure)
        b1 = to_fig.transform_bbox(leg.get_window_extent(r))
        leg2.set_bbox_to_anchor((b1.x1 + 0.05, span.y0 - 0.02), transform=fig.transFigure)
        fig.savefig(PAPER_OUT, facecolor=rp.SURFACE, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"Wrote {PAPER_OUT}")


def main():
    import sys
    if sys.argv[1:] == ["paper"]:
        return paper()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (_, title, f) in zip(axes, HEADS):
        panel(ax, load(f), title)
    axes[0].set_ylabel("test mIoU")
    style = [Line2D([], [], color=GREY, marker="o", lw=1, label="fixed: composites (sea / mon / ½-mon) → raw"),
             Line2D([], [], color=GREY, marker="o", mfc="white", ls="--", lw=1, label="fixed: K evenly spaced frames → raw"),
             Line2D([], [], color="k", marker="o", lw=1.6, label="MiSTR, pretrained-pool attention (windows/px)"),
             Line2D([], [], color="k", marker="D", ls="--", lw=1.4, label="MiSTR, task-tuned pool attention"),
             Line2D([], [], color="k", marker="s", mfc="white", ls=":", lw=1.2, label="same windows for every pixel"),
             Line2D([], [], color="k", marker="x", lw=0, mew=1.5, label="random window per pixel"),
             Line2D([], [], color="k", marker="+", lw=0, ms=8, mew=1.5, label="b < 1: random pixels (not entropy)")]
    pairs = [Line2D([], [], color=COL[p], lw=3, label=p.replace(">", " → ")) for p in PAIRS]
    fig.legend(handles=style + pairs, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Frozen Tessera-v2-small on PASTIS: fixed temporal resolution vs MiSTR (time only, per pixel); "
                 "test mIoU, 1 seed", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    # one pair per panel: rows = heads, columns = pairs
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    for i, (_, htitle, f) in enumerate(HEADS):
        df = load(f)
        for j, pair in enumerate(PAIRS):
            panel(axes[i, j], df, f"{pair.replace('>', ' → ')} — {htitle}", [pair])
            if i < 2:
                axes[i, j].set_xlabel("")
        axes[i, 0].set_ylabel("test mIoU")
    fig.legend(handles=style, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Frozen Tessera-v2-small on PASTIS, one pair per panel (zoomed); test mIoU, 1 seed", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT_GRID, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_GRID}")
    for ds in DS_ORDER:
        simple(ds)
    simple_all()


if __name__ == "__main__":
    main()
