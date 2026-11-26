"""Fig 04 (baselines) from round 1 ONLY: PASTIS, results/pastis/mistr.csv (BioMassters left out). the
style (marker area = parameters added to the fixed model), its data swapped for round 1's:

- MiSTR and the fixed curves = Fig 03's runs (exp.viz.round1_main: test-split cost, all resolution
  pairs, q 1/8, 1/4, 1/2). MiSTR adds NO parameters (no crop-type embedding, the entropy threshold
  is a statistic): checkpoint keys == the fixed model's.
- vid-TLDR: training-free only (upstream; merging on the fixed monthly model, `_ep0_`); our
  fine-tuned-with-merging extension (`-ft`) is left out.
- AnyUp fine-tuned per pixel, S16/12/8.
- APT (both scorers, keeps .25/.5/.75, seasons + months), Upsample Anything
  (features on the bilinear LP head, predictions on the fixed model): best-val lr per config.
  Upsample Anything drawn as ONE method, +0 in both spaces (the per-pixel head
  counts as the base head's; feature vs prediction space goes to the appendix). Not fine-tuned:
  too expensive (exp/baselines/bench_upa_ft.py: ~90 h / 50 ep with the fit in the loop).
- Left out as before: the bilinear control.

Parameters counted from round-1 checkpoints: base 6,731,377; APT +184,704 (patch_attn
+ zero_conv), AnyUp +881,747 (upsampler + pixel head); UPA features' +3,667 pixel head drawn as 0.

    python -m exp.viz.baselines_r1          -> figures/baselines_r1.png (PASTIS, single column)
    python -m exp.viz.baselines_r1 --png    -> figures/baselines_r1.png
    python -m exp.viz.baselines_r1 --task yieldsat_raw -> figures/baselines_r1_yieldsat.png
    python -m exp.viz.baselines_r1 --side [--png] -> PASTIS + YieldSAT side by side, the paper's
                                                    04_Baselines.pdf (PNG: baselines_r1_side.png)
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import common as bp  # noqa: E402
from exp.viz import datasets_tradeoff as dt  # noqa: E402
from exp.viz import common as rp  # noqa: E402
from exp.viz import round1_main as r1  # noqa: E402

APT, ANYUP = 184_704, 881_747          # (ANYUP: PASTIS' count, the marker-area scale everywhere)
# per task: base = the fixed S12 checkpoint's parameters (its head sized for 12 x 12 px; YieldSAT
# fixed_yieldsat_raw_S16Tsea's 6,252,737 with the S12 head); AnyUp = its upsampler
# + the pixel head, 192 -> outputs. RMSE: y runs downward, better is up.
TASKS = {
    "pastis": dict(base=6_731_377, anyup=ANYUP, label="PASTIS test mIoU", lower=False,
                   ylim=(40, 60.5), strip=(26, 40), xlim=(1.4, 220),     # mIoU in 0-100
                   yticks={45: "45", 50: "50", 55: "55", 60: "60"},
                   strip_ticks={30: "30", 35: "35"}, xticks=(2, 5, 10, 20, 50, 100, 200),
                   out=None, png="figures/baselines_r1.png"),
    "yieldsat_raw": dict(base=6_231_121, anyup=878_080 + 193, label="YieldSAT test RMSE",
                         lower=True, ylim=(1.48, 1.31), strip=(1.70, 1.48), xlim=(.7, 220),
                         yticks={1.35: "1.35", 1.4: "1.40", 1.45: "1.45"},
                         strip_ticks={1.5: "1.50", 1.6: "1.60"}, xticks=(1, 2, 5, 10, 20, 50, 100, 200),
                         out=None, png="figures/baselines_r1_yieldsat.png"),
}
FIGSIZE = (3.9, 3.2)
LEGEND_COLS = 4                 # methods legend below the plot, 2 rows (7 did not fit one row)
NAMES = {"mistr": "MiSTR (ours)", "apt": "APT", "anyup": "AnyUp", "upa": "Upsample Anything",
         "tldr": "vid-TLDR", "sea": r"$S_{\text{Fixed-Res}}T_{\text{season}}$",
         "mon": r"$S_{\text{Fixed-Res}}T_{\text{month}}$"}
AREA_MAX = 260                          # pt^2 at AnyUp's count (single column: smaller than bp's)
STAR = 70


def area(n):
    return bp.AREA_FLOOR * .8 + n / ANYUP * (AREA_MAX - bp.AREA_FLOOR * .8)


def best_lr(d, lower=False):
    """One row per configuration: the lr with the best val metric (test where val is missing)."""
    keys = ["baseline.method", "baseline.tokens", "baseline.upsampler", "baseline.space",
            "baseline.r", "baseline.keep", "baseline.scorer", "model.coarse_patch",
            "train.epochs"]
    d = d.assign(sel=d.val_primary.fillna(d.test_primary) * (-1 if lower else 1))
    return d.sort_values("sel").groupby(keys, dropna=False).tail(1)


def baselines(d, lower=False):
    """{key: rows} per method drawn."""
    b = best_lr(d[d["baseline.method"].notna()], lower)
    m, up = b["baseline.method"], b["baseline.upsampler"]
    return {
        "apt": b[m == "apt"],
        # (S6 AnyUp, left out: the paper version predates it; budget caveat)
        "anyup": b[(m == "upsample") & (up == "anyup") & (b["train.epochs"] > 0)
                   & (b["model.coarse_patch"] != 6)
                   & ~b["train.head_only"].astype(bool)],
        "upa_feat": b[(m == "upsample") & (up == "upa") & (b["baseline.space"] == "features")],
        "upa_pred": b[(m == "upsample") & (up == "upa") & (b["baseline.space"] == "predictions")],
        "tldr": b[(m == "tldr") & (b["train.epochs"] == 0)],
    }


def mistr_points(d, task):
    pts = []
    for ps, geo in dt.PAIRS:
        pts += r1.mistr(d, task, ps, geo, None)
    return sorted(pts)


def panel(fig, cell, task, param_key=True, scale=1.0, label_size=9, title_above=False,
          key_size=7, key_box=False):
    """One dataset into gridspec cell `cell`: the main axes over the compressed strip and (param_key)
    its own parameters-added key at the strip's lower right. scale: marker areas and line
    weights (the figure* draws at the old 11 pt); title_above: the label as a title above the
    main axes' top-left corner instead of inside it; key_size: the key's font (title + 1).
    key_box: a light frame round the key. -> (main, strip) axes."""
    T = TASKS[task]
    BASE, XMAX = T["base"], T["xlim"][1]
    d = r1.rows(task)
    fx = r1.fixed(task, d)
    groups = baselines(d, T["lower"])
    methods = (("apt", "apt", APT), ("anyup", "anyup", T["anyup"]),
               ("upa_feat", "upa", 0),
               ("upa_pred", "upa", 0), ("tldr", "tldr", 0))

    gs = cell.subgridspec(2, 1, height_ratios=[5, .9], hspace=.06)
    ax, low = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.sharex(low)
    for a in (ax, low):
        a.set_facecolor(rp.SURFACE)

    for fam, ls, mk, mfc in (("sea", "--", "s", rp.SURFACE), ("mon", "-", "D", rp.INK)):
        p = sorted((x, y) for f, _, x, y in fx if f == fam and x <= XMAX)
        ax.plot(*zip(*p), ls, color=rp.INK, lw=.9 * scale ** .5, marker=mk, ms=2.8 * scale ** .5,
                mfc=mfc, alpha=.75,
                zorder=2, label=NAMES[fam])
    table = []
    for key, col, n in methods:
        r = groups[key]
        table += [(key, x.run, x.gmacs, x.test_primary) for x in r.itertuples()]
        inside = r.test_primary.between(*sorted(T["ylim"]))
        for a, q in ((ax, r[inside]), (low, r[~inside])):
            a.scatter(q.gmacs, q.test_primary, s=area(n) * scale, color=bp.COLOR[col], alpha=.45,
                      edgecolors=bp.COLOR[col], linewidths=.8, zorder=3, clip_on=False,
                      label=NAMES[col] if a is ax and key != "upa_pred" else None)
    pts = mistr_points(d, task)
    ax.scatter(*zip(*pts), s=STAR * scale, marker="*", color=bp.OURS, edgecolors=rp.SURFACE,
               linewidths=.4, zorder=5, label=NAMES["mistr"])

    print(pd.DataFrame(pts, columns=["gmacs", "miou"]).round(3).to_string(index=False))
    print(pd.DataFrame(table, columns=["method", "run", "gmacs", "miou"]).round(3)
          .sort_values(["method", "gmacs"]).to_string(index=False))

    ax.set_xscale("log")
    ax.set_xlim(*T["xlim"])
    ax.set_ylim(*T["ylim"])
    ax.set_yticks(list(T["yticks"]), list(T["yticks"].values()))
    low.set_ylim(*T["strip"])
    low.set_xticks(list(T["xticks"]), [f"{x:g}" for x in T["xticks"]])
    low.set_yticks(list(T["strip_ticks"]), list(T["strip_ticks"].values()))
    low.set_xlabel("GMACs per sample")
    if title_above:
        ax.set_title(T["label"], loc="left", fontsize=label_size, fontweight="bold", pad=6)
    else:
        ax.text(.015, .985, T["label"], transform=ax.transAxes, ha="left", va="top",
                fontsize=label_size, fontweight="bold")
    for a in (ax, low):
        a.minorticks_off()
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.2, lw=.5)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    for a, y, k in ((ax, 0, 1), (low, 1, 5 / .9)):           # the y axis break
        a.plot([-.015, .015], [y - .012 * k, y + .012 * k], color=rp.INK, lw=.7,
               transform=a.transAxes, clip_on=False)

    if not param_key:
        return ax, low
    # parameters added: boxed, one column, anchored at the plot's lower-right corner (the strip's,
    # below the y break) and reaching up across the break; methods + fixed curves below the plot
    share = lambda n: f"{n / BASE * 100:.2g}%"                                   # noqa: E731
    dot = lambda k, n: plt.scatter([], [], s=area(n) * scale, color=bp.COLOR[k],  # noqa: E731
                                   alpha=.45, edgecolors=bp.COLOR[k], linewidths=.8)
    zero = (plt.scatter([], [], s=STAR * scale * .65, marker="*", color=bp.OURS, edgecolors=rp.SURFACE,
                        linewidths=.4), dot("tldr", 0), dot("upa", 0))
    par = fig.legend([zero, dot("apt", APT), dot("anyup", T["anyup"])],
                    ["+0", f"+{bp.count(APT)} ({share(APT)})",
                     f"+{bp.count(T['anyup'])} ({share(T['anyup'])})"],
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=.6)},
                    title="parameters added", loc="lower right", bbox_to_anchor=(1, .15),
                    bbox_transform=low.transAxes,
                    ncol=1, handlelength=3.0, handletextpad=.5,
                    # (boxed: tighter, the frame shows the extent; the big marker still clears)
                    labelspacing=.95 if key_box else 1.35, borderpad=.7 if key_box else .65,
                    alignment="left", frameon=key_box, fancybox=False,
                    borderaxespad=.4 if key_box else 0, fontsize=key_size,
                    title_fontproperties={"size": key_size + 1, "weight": "bold"})
    par.set_zorder(6)
    if key_box:
        par.get_frame().set(edgecolor=rp.MUTED, facecolor=rp.SURFACE, linewidth=.6, alpha=.9)
    return ax, low


def methods_legend(fig, ax, order, fontsize, title=None, sizes=(45, 22), **kw):
    """The methods + fixed curves legend, its handles from main axes ax ("": a blank slot)."""
    h, names = ax.get_legend_handles_labels()
    by = dict(zip(names, h)) | {"": plt.Line2D([], [], alpha=0)}
    leg = fig.legend([by[NAMES.get(k, k)] for k in order], [NAMES.get(k, k) for k in order],
                     frameon=False, handlelength=1.6, handletextpad=.35, columnspacing=.9,
                     borderaxespad=0, fontsize=fontsize, title=title, alignment="left",
                     title_fontproperties={"size": fontsize, "weight": "bold"}, **kw)
    for t, k in zip(leg.legend_handles, order):
        if hasattr(t, "set_sizes"):
            t.set_sizes([sizes[0] if k == "mistr" else sizes[1]])
    return leg


def shared_key(fig, tasks, scale, fontsize, **kw):
    """One parameters-added key over several datasets, ascending: counts and shares of each
    task's fixed model as ranges where they differ (AnyUp's head, the base)."""
    def span(vals, fmt):
        lo, hi = fmt(min(vals)), fmt(max(vals))
        return lo if lo == hi else f"{lo}–{hi}"
    pct = lambda n: span([n(t) / TASKS[t]["base"] * 100 for t in tasks], lambda v: f"{v:.1f}")
    cnt = lambda n: span([n(t) for t in tasks], bp.count).replace("k–", "–")  # noqa: E731
    apt, anyup = (lambda t: APT), (lambda t: TASKS[t]["anyup"])                   # noqa: E731
    dot = lambda k, n: plt.scatter([], [], s=area(n) * scale, color=bp.COLOR[k],  # noqa: E731
                                   alpha=.45, edgecolors=bp.COLOR[k], linewidths=.8)
    zero = (plt.scatter([], [], s=STAR * scale * .65, marker="*", color=bp.OURS,
                        edgecolors=rp.SURFACE, linewidths=.4), dot("tldr", 0), dot("upa", 0))
    big = max(TASKS[t]["anyup"] for t in tasks)
    return fig.legend([zero, dot("apt", APT), dot("anyup", big)],
                      ["+0, 0%", f"+{cnt(apt)}, {pct(apt)}%", f"+{cnt(anyup)}, {pct(anyup)}%"],
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=.6)},
                      title="parameters added", frameon=False, alignment="left",
                      handlelength=3.0, handletextpad=.5, labelspacing=.9, borderaxespad=0,
                      fontsize=fontsize, title_fontproperties={"size": fontsize, "weight": "bold"},
                      **kw)


def stack(fig, legends, x, top, bottom=None, gap=.04):
    """Legends top-down at figure x, the first at `top` (figure fractions); bottom: gaps equal so
    the last ends there, else `gap` between them."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    hs = [leg.get_window_extent().transformed(inv).height for leg in legends]
    if bottom is not None:
        gap = max(0.0, (top - bottom - sum(hs)) / max(1, len(hs) - 1))
    y = top
    for leg, h in zip(legends, hs):
        leg.set_loc("upper left")
        leg.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        y -= h + gap


def save(fig, out):
    fig.savefig(out, dpi=300 if out.endswith(".png") else None, facecolor=rp.SURFACE,
                bbox_inches="tight", pad_inches=.02)
    print(f"wrote {out}")


def draw(out, task="pastis"):
    """One dataset, single column; the methods legend below in LEGEND_COLS columns."""
    bp.style()
    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=FIGSIZE, facecolor=rp.SURFACE)
    gs = fig.add_gridspec(1, 1, left=.09, right=.99, top=.99, bottom=.22)
    ax, _ = panel(fig, gs[0], task)
    # columns (filled top-down): MiSTR | APT, AnyUp | Upsample Anything, vid-TLDR | fixed res
    methods_legend(fig, ax, ["mistr", "", "apt", "anyup", "upa", "tldr", "sea", "mon"], 7,
                   loc="lower center", bbox_to_anchor=(.54, 0), ncol=LEGEND_COLS,
                   labelspacing=.3)
    save(fig, out)


SIDE = ("pastis", "yieldsat_raw")
SIDE_OUT, SIDE_PNG = "figures/paper/04_Baselines.pdf", "figures/baselines_r1_side.png"


SIDE_SCALE = 1.3            # marker areas at 11 pt (figure*: 330 / 260)


def draw_side(out):
    """PASTIS and YieldSAT side by side (figure*, 11 pt): each
    panel titled above its top-left corner with its own parameters-added key (the counts differ:
    AnyUp's head, the base); a column at the right of two titled legends, methods and fixed
    resolution baselines, from the panels' top."""
    bp.style()
    fig = plt.figure(figsize=(11.5, 3.3), facecolor=rp.SURFACE)
    top, bottom = .9, .15
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .52], wspace=.16, left=.055, right=.995,
                          top=top, bottom=bottom)
    kw = dict(scale=SIDE_SCALE, label_size=12, title_above=True, key_size=10,
              key_box=True)
    ax, low = panel(fig, gs[0], SIDE[0], **kw)
    ax2, _ = panel(fig, gs[1], SIDE[1], **kw)
    sizes = (95, 40)
    legs = [methods_legend(fig, ax, ["mistr", "apt", "anyup", "upa", "tldr"], 12.5,
                           title="methods", sizes=sizes, labelspacing=.3),
            methods_legend(fig, ax, ["sea", "mon"], 12.5, title="fixed resolution baselines",
                           labelspacing=.3)]
    stack(fig, legs, ax2.get_position().x1 + .02, top)      # (just past the right panel)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--task", default="pastis", choices=list(TASKS))
    ap.add_argument("--side", action="store_true", help="PASTIS + YieldSAT, figure*")
    a = ap.parse_args()
    if a.side:
        draw_side(SIDE_PNG if a.png else SIDE_OUT)
        return
    t = TASKS[a.task]
    draw(t["png"] if a.png or not t["out"] else t["out"], a.task)


if __name__ == "__main__":
    main()
