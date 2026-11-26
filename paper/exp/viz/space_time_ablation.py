"""Appendix apx:separate (two single-column figures, 2 x 2 panels): spatial and temporal resolution each help, with diminishing returns. Two figures,
each varying one axis of fixed-resolution OlmoEarth tiny (regular grid, no crops) with the other held:

    python -m exp.viz.space_time_ablation
      -> figures/paper/09_patch_size_ablation.pdf          seasonal composites, patch 16..4, ROUND 1,
                                                           + legacy S2 (hollow, dashed; added)
                                                           (results/<task>/mistr.csv: Fig 03's fixed runs)
      -> figures/paper/10_temporal_resolution_ablation.pdf 16 px patches, annual / seasonal / monthly
                                                           composites, ROUND 1 (fig 03's protocol), mean
                                                           +- s.e. over seeds; T_year = the *_year tasks the time figure is on round 1 (it was on the legacy inject_fft runs, 50 epochs, whose
seasonal points trained at batch 32 against batch 8 for the others: PLANTEUR's legacy T_sea fell below
T_year). Every acquisition (T_acq) stays legacy (one seed, half the epochs; PASTIS and YieldSAT only),
hollow on a dashed segment like S2 in the space figure: the new code has no per-acquisition input and
its cost (9-17x monthly GMACs) is not worth retraining. (A single cadence x patch grid figure,
09_space-time-grid, was tried and dropped: the appendix shows that each axis helps, not which
to buy.)
"""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import pandas as pd

from exp.viz import round1_main as r1

# task -> (title, subtitle, higher is better)
TASKS = {"pastis": ("PASTIS", "mIoU", True), "planteur": ("PLANTEUR", "mIoU", True),
         "biomassters": ("BioMassters", "RMSE", False), "yieldsat_raw": ("YieldSAT", "RMSE", False)}
# legacy inference GMACs per sample at 16 px patches, traced with --flops: {task: {cadence: GMACs}}
CAD = ("year", "season", "all", "raw")
TEX = {"year": r"$T_\mathrm{year}$", "season": r"$T_\mathrm{sea}$", "all": r"$T_\mathrm{mon}$",
       "raw": r"$T_\mathrm{acq}$"}
CAD_NAME = {"year": "1 annual composite", "season": "4 seasonal composites", "all": "12 monthly composites",
            "raw": "every acquisition"}
RAMP = ("#a1dab4", "#41b6c4", "#2c7fb8", "#253494")    # YlGnBu: finer = darker
MUTED, GRID, LINE = "#52514e", "#e8e7e3", "#c9c8c2"


def space_rows(task):
    """-> DataFrame (P, x GMACs, y, ys = s.e. over seeds, n, hollow): round 1 fixed seasonal runs (Fig 03's
    fixed runs, seeds averaged, cost on test) + S2, which round 1 does not have: the legacy one-seed run
    (half the epochs; cost traced on val batches, results/<task>/inject_flops.csv), drawn hollow on a
    dashed segment. (+- s.e. bars, as in the time figure.)"""
    d = r1.rows(task)
    g = d[d.run.str.startswith("fixed_") & (d["model.coarse_time"] == "season")]
    f = [dict(P=int(p), x=r.gmacs.mean(), y=r.test_primary.mean(), ys=r.test_primary.sem(), n=len(r),
              hollow=False) for p, r in g.groupby("model.coarse_patch")]
    pt = legacy_point(task, "lrps2")
    if pt:
        f.append(dict(P=2, x=pt[0], y=pt[1], ys=float("nan"), n=1, hollow=True))
    return pd.DataFrame(f, columns=["P", "x", "y", "ys", "n", "hollow"]).sort_values("x")


def legacy_point(task, tag):
    """(GMACs, metric) of a legacy one-seed fixed run inject_fft_<task>_..._native_set_<tag>_... (half
    the epochs; cost traced on val batches, results/<task>/inject_flops.csv), or None: the points round 1
    does not have (S2, T_acq), drawn hollow on a dashed segment."""
    run = rf"inject_fft_{task}_tiny_none_native_set_{tag}_lr0.001_ep\d+"
    leg = pd.read_csv(f"results/{task}/inject_fft.csv", low_memory=False)
    leg = leg[leg.run.astype(str).str.fullmatch(run + r"(_seed\d)?")]
    with open(f"results/{task}/inject_flops.csv") as fh:     # (rows have uneven trailing fields)
        rd = csv.reader(fh)
        hd = next(rd)
        fl = [float(r[hd.index("gmacs")]) for r in rd if re.fullmatch(run, r[hd.index("run")])]
    if not (len(leg) and fl):
        return None
    return fl[-1], float(leg.test_primary.mean()) * (100 if task in r1.PCT_TASKS else 1)


def time_rows(task):
    """-> DataFrame (steps, x GMACs, y, ys = standard error over seeds), ROUND 1 16 px fixed runs:
    T_sea / T_mon of fig 03 (results/<task>/mistr.csv) + T_year (results/<task>_year/mistr.csv,
    the *_year tasks: one composite of all month slots)."""
    out = []
    for steps, t, coarse in (("year", f"{task}_year", "season"), ("season", task, "season"),
                             ("all", task, "month")):
        d = r1.rows(t)
        if not len(d):
            continue
        g = d[d.run.str.startswith("fixed_") & (d["model.coarse_patch"] == 16)
              & (d["model.coarse_time"] == coarse)]
        if not len(g):
            continue
        y = g.test_primary * (100 if task in r1.PCT_TASKS and t not in r1.PCT_TASKS else 1)
        out.append(dict(steps=steps, x=g.gmacs.mean(), y=y.mean(), ys=y.sem(), n=len(g), hollow=False))
    pt = legacy_point(task, "lrraw")       # (T_acq: legacy, half the epochs -- too costly to retrain)
    if pt:
        out.append(dict(steps="raw", x=pt[0], y=pt[1], ys=float("nan"), n=1, hollow=True))
    return pd.DataFrame(out, columns=["steps", "x", "y", "ys", "n", "hollow"]).sort_values("x")


def style(ax, title, sub, up, ys):
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()      # 1-3-10 ticks up to ~2.5 decades, else decades only (no collisions)
    ax.xaxis.set_major_locator(mt.LogLocator(subs=(1, 3) if hi / lo < 400 else (1,), numticks=8))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    if up:      # mIoU in 0-100, as in fig 03
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.invert_yaxis()
        if max(ys) < 10:
            ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mt.MaxNLocator(5))
    ax.margins(x=0.12, y=0.14)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(r"$\bf{" + title + "}$  " + sub + ("" if up else ", flipped"), loc="left", fontsize=8)
    ax.set_xlabel("GMACs/sample")


# inches: figure width, left margin (y label + ticks), right margin, gap between columns (the right
# panels' tick labels), axes size, top (panel titles), gap between rows (x ticks + next title), x label
G = dict(W=3.25, left=0.47, right=0.04, wgap=0.40, top=0.20, axh=1.06, hgap=0.40, xlab=0.36)
G["axw"] = (G["W"] - G["left"] - G["right"] - G["wgap"]) / 2


def draw(out, panels, legend, note, ncol=None):
    """Column figure (3.25 in): panels on a 2 x 2 grid. panels: [(task, DataFrame x/y, [legend key per
    point])]; legend: [(key, label, colour)] in order. One grey line per panel, points coloured by the
    varied resolution. 3 panels: the legend fills the 4th cell; 4 panels: legend below, in ncol
    columns (default one row) under the note as its title."""
    col = {k: c for k, _, c in legend}
    four = len(panels) == 4
    # FIXED geometry in inches: every figure drawn here has the same width and the same
    # axes boxes, so figs 7 and 8 line up side by side in the paper; only the legend strip below
    # differs. (tight_layout / bbox_inches="tight" sized the panels by tick labels and legend.)
    rows = -(-len(legend) // (ncol or len(legend))) if four else 0
    leg_h = (0.17 * rows + (0.15 if ncol else 0) + 0.06) if four else 0.0
    H = G["top"] + 2 * G["axh"] + G["hgap"] + G["xlab"] + leg_h
    fig = plt.figure(figsize=(G["W"], H))
    axes = []
    for r in range(2):
        for c in range(2):
            x0 = G["left"] + c * (G["axw"] + G["wgap"])
            y0 = H - G["top"] - G["axh"] - r * (G["axh"] + G["hgap"])
            axes.append(fig.add_axes((x0 / G["W"], y0 / H, G["axw"] / G["W"], G["axh"] / H)))
    for ax, (task, d, keys) in zip(axes, panels):
        title, sub, up = TASKS[task]
        hol = d.hollow.values if "hollow" in d else [False] * len(d)
        solid = d[~pd.Series(hol, index=d.index)]
        ax.plot(solid.x, solid.y, "-", color=LINE, lw=1.2, zorder=2)
        for i in [i for i, h in enumerate(hol) if h and i > 0]:    # dashed into a separately trained point
            ax.plot(d.x.iloc[i - 1:i + 1], d.y.iloc[i - 1:i + 1], "--", color=LINE, lw=1.2, zorder=2)
        for x, y, k, h in zip(d.x, d.y, keys, hol):
            ax.plot(x, y, "o", ms=4.5, color=col[k], mfc="white" if h else col[k],
                    mec=col[k] if h else "white", mew=1.0 if h else 0.6, zorder=3)
        if "ys" in d:                        # (+-1 s.e. over seeds; none for a 1-seed point)
            for x, y, e, k in zip(d.x, d.y, d.ys, keys):
                if pd.notna(e):
                    ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col[k], elinewidth=0.9,
                                capsize=1.8, capthick=0.9, zorder=2.5)
        style(ax, title, sub, up, list(d.y))
    for i, ax in enumerate(axes[:len(panels)]):
        if i + 2 < len(panels):              # a panel below: no x label
            ax.set_xlabel("")
    axes[0].set_ylabel(r"metric $\rightarrow$ better")
    axes[2].set_ylabel(r"metric $\rightarrow$ better")
    hk = {k for _, d, keys in panels if "hollow" in d for k, h in zip(keys, d.hollow) if h}
    h = [plt.Line2D([], [], marker="o", ms=4.5, ls="none", color=c, mfc="white" if k in hk else c,
                    mec=c if k in hk else "white", mew=1.0 if k in hk else 0.6) for k, _, c in legend]
    labs = [lab for _, lab, _ in legend]
    if four:
        # one row of patch sizes; the held cadence (note) is in the caption
        cx = (G["left"] + G["axw"] + G["wgap"] / 2) / G["W"]      # (centred under the panels)
        fig.legend(h, labs, loc="lower center", ncol=ncol or len(h), frameon=False, fontsize=7.5,
                   columnspacing=0.9, handlelength=0.8, handletextpad=0.2, borderaxespad=0,
                   bbox_to_anchor=(cx, 0.03 / H), **({"title": note, "title_fontsize": 7} if ncol else {}))
    else:
        lx = axes[3]
        lx.axis("off")
        lx.legend(h, labs, loc="center left", frameon=False, fontsize=7, handlelength=0.8,
                  handletextpad=0.3, labelspacing=0.7, borderaxespad=0, title=note, title_fontsize=7,
                  alignment="left")
    fig.savefig(out)        # (no bbox_inches="tight": the fixed width is what keeps figures aligned)
    plt.close(fig)
    print(out)


INK = "#0b0b0b"


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    sp = [(t, space_rows(t)) for t in TASKS]
    ps = sorted({p for _, d in sp for p in d.P}, reverse=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("r", RAMP)
    leg = [(p, f"$S_{{{p}}}$", matplotlib.colors.to_hex(ramp(i / (len(ps) - 1)))) for i, p in enumerate(ps)]
    draw("figures/paper/09_patch_size_ablation.pdf", [(t, d, list(d.P)) for t, d in sp], leg,
         r"all: $T_\mathrm{sea}$ (4 seasonal composites)")
    tp = [(t, time_rows(t)) for t in TASKS]
    leg = [(c, f"{TEX[c]}: {CAD_NAME[c]}", RAMP[i]) for i, c in enumerate(CAD)]   # 4th cell: room for names
    draw("figures/paper/10_temporal_resolution_ablation.pdf", [(t, d, list(d.steps)) for t, d in tp], leg,
         r"all: $S_{16}$ (16 px patches)", ncol=2)

if __name__ == "__main__":
    main()
