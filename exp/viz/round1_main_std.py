"""Appendix companion of fig 03: the same round-1 points with their seed spread. One row of 4
panels (one per dataset) at the full text width: fixed-resolution curves = mean line + a +-1 standard error band,
MiSTR = mean star + +-1 standard error bars (y; x too, where the crop count varies across seeds).
Also writes the table of every point (mean +- standard error, GMACs).

    python -m exp.viz.round1_main_std          -> figures/round1_main_std.png (preview)
    python -m exp.viz.round1_main_std --paper  -> figures/paper/22_MiSTR-result-std.pdf
                                                  + figures/paper/22_MiSTR-result-std.tex (table)
Bars = standard error of the mean over the seeds landed, s / sqrt(n) (s = sample std, ddof 1);
a point with one seed has no bar.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import datasets_tradeoff as dt  # noqa: E402
from exp.viz import common as rp  # noqa: E402
from exp.viz import round1_main as r1  # noqa: E402
from exp.viz.common import pair_name  # noqa: E402

SEED = r"_seed\d+_[0-9a-f]+$"


def geo_of(r):
    fp = int(r["model.fine_patch"]) if pd.notna(r["model.fine_patch"]) else 0
    w = int(r["model.window_px"])
    return f"c{w}p{fp}" if fp else f"c{w}"


def points(task):
    """One row per plotted point: kind (sea / mon / mistr), patch, pair geo, q, n seeds, GMACs
    and metric mean / standard error."""
    d = r1.rows(task)
    d = d[d.run.str.match("(fixed|mistr)_")].copy()
    d["x"] = [r1.mistr_cost(r, d) if r.run.startswith("mistr_") else r.gmacs
              for _, r in d.iterrows()]
    d["point"] = d.run.str.replace(SEED, "", regex=True)
    out = []
    for p, g in d.groupby("point"):
        r = g.iloc[0]
        ps = int(r["model.coarse_patch"])
        if p.startswith("fixed_"):
            kind, geo, q = r1.TIME[r["model.coarse_time"]], "", float("nan")
        else:
            kind, geo = "mistr", geo_of(r)
            q = float(re.search(r"_q([\d.]+)_", p).group(1)) / 100
            if (ps, geo) not in dt.PAIRS:
                continue
        out.append(dict(kind=kind, ps=ps, geo=geo, q=q, n=len(g),
                        x=g.x.mean(), xs=g.x.sem(), y=g.test_primary.mean(),
                        ys=g.test_primary.sem()))
    return pd.DataFrame(out)


def panel(ax, task, title, ylabel, lower):
    P = points(task)
    for fam in ("sea", "mon"):
        f = P[P.kind == fam].sort_values("x")
        ls, mk, filled, _ = dt.FAM[fam]
        ys = f.ys.fillna(0)
        ax.fill_between(f.x, f.y - ys, f.y + ys, color=rp.INK, alpha=0.12, lw=0, zorder=2)
        ax.plot(f.x, f.y, ls + mk, color=rp.INK, lw=0.7, ms=3.8,
                mfc=rp.INK if filled else rp.SURFACE, mec=rp.INK, mew=0.8, alpha=0.75, zorder=3)
        for i in (0, len(f) - 1):     # (as fig 03: S16 ends + the monthly top end named)
            if i and fam == "sea":
                continue
            x, y, ps = f.x.iloc[i], f.y.iloc[i], f.ps.iloc[i]
            xy, ha, va = ((3, 0), "left", "center") if i else ((2, -2), "left", "top")
            ax.annotate(rf"$S_{{{ps}}}T_{{{fam}}}$", (x, y), xytext=xy,
                        textcoords="offset points", ha=ha, va=va, fontsize=7.5, zorder=5)
    m = P[P.kind == "mistr"]
    for (ps, geo), g in m.groupby(["ps", "geo"]):
        col = dt.PAIR_COL[(ps, geo)]
        # (x bars only where the crop count moves the cost visibly: > 1% of the point's GMACs)
        xs = g["xs"].where(g["xs"] > 0.01 * g.x).fillna(0)
        ax.errorbar(g.x, g.y, yerr=g.ys.fillna(0), xerr=xs, fmt="none", ecolor=col,
                    elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=5)
        ax.plot(g.x, g.y, "*", color=col, ms=7, mec=rp.INK, mew=0.25, zorder=6)
    ax.set_xscale("log")
    if lower:
        ax.invert_yaxis()
    ax.grid(True, color=rp.GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    name, kind = title.rstrip(")").split(" (")
    ax.set_title(name, loc="left", fontweight="bold", pad=13)
    ax.text(0, 1.02, f"{kind}, {ylabel.removeprefix('test ')}{' (flipped)' if lower else ''}",
            fontsize=9, transform=ax.transAxes, va="bottom")
    ax.margins(0.1)      # (room for the end labels, as fig 03)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 3.2)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    return P


def table(pts, out):
    """Every point: GMACs and test metric mean +- standard error, datasets side by side."""
    def fmt(r, task):
        k = 1        # (mIoU already 0-100 from r1.rows)
        dg = {"pastis": 1, "planteur": 1, "biomassters": 2}.get(task, 3)
        s = f"{r.y * k:.{dg}f}"
        s += f" $\\pm$ {r.ys * k:.{dg}f}" if r.n > 1 else " (1 seed)"
        return f"{r.x:.1f} & {s}"
    keys = []
    for fam in ("sea", "mon"):
        keys += [(fam, ps, "", None) for ps in sorted(
            {int(p) for P in pts.values() for p in P[P.kind == fam].ps}, reverse=True)]
    for ps, geo in dt.PAIRS:
        keys += [("mistr", ps, geo, q) for q in dt.QS]
    rows = []
    for fam, ps, geo, q in keys:
        name = (rf"$S_{{{ps}}}T_{{{fam}}}$" if fam != "mistr" else
                pair_name(ps, geo).replace("+", "$+$") + f", $q_s{{=}}{q:g}$")
        cells = []
        for task, P in pts.items():
            sel = P[(P.kind == fam) & (P.ps == ps)]
            if fam == "mistr":
                sel = sel[(sel.geo == geo) & ((sel.q - q).abs() < 1e-9)]
            cells.append(fmt(sel.iloc[0], task) if len(sel) else "-- & --")
        rows.append(name + " & " + " & ".join(cells) + r" \\")
        if (fam, ps) in (("sea", 4), ("mon", 6)):
            rows.append(r"\midrule")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for t in
                      ("PASTIS mIoU (\\%)", "PLANTEUR mIoU (\\%)", "BioMassters RMSE",
                       "YieldSAT RMSE"))
    tex = "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Every point of \cref{fig:result-std}: test cost (GMACs per sample) and test "
        r"metric, mean $\pm$ standard error over seeds.}",
        r"\label{tab:result-std}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l rl rl rl rl}", r"\toprule",
        " & " + head + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        "Model" + " & GMACs & metric" * 4 + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    Path(out).write_text(tex)
    print(out)


LOWER = {"biomassters", "yieldsat_raw"}
TASK_HEAD = {"pastis": "PASTIS (mIoU)", "planteur": "PLANTEUR (mIoU)",
             "biomassters": r"BioMassters (RMSE $\downarrow$)",
             "yieldsat_raw": r"YieldSAT (RMSE $\downarrow$)"}


def matched_rows(P, task):
    """Per fixed model with a cheaper MiSTR point: the cheapest MiSTR point at least as accurate
    (strict), else the most accurate MiSTR point that is cheaper. -> list of dicts; improvement
    > 0 = MiSTR better (mIoU points, or RMSE reduction)."""
    lo, M, out = task in LOWER, P[P.kind == "mistr"], []
    F = P[P.kind != "mistr"]
    for fam in ("sea", "mon"):
        for _, f in P[P.kind == fam].sort_values("ps", ascending=False).iterrows():
            # (dominated: another fixed model is no more expensive and at least as accurate)
            dom = bool(((F.x <= f.x) & ((F.y <= f.y) if lo else (F.y >= f.y))
                        & ((F.x < f.x) | (F.y != f.y))).any())
            cheap = M[M.x <= f.x]
            if not len(cheap):
                continue
            ok = cheap[(cheap.y <= f.y) if lo else (cheap.y >= f.y)]
            m = ok.loc[ok.x.idxmin()] if len(ok) else cheap.loc[
                cheap.y.idxmin() if lo else cheap.y.idxmax()]
            out.append(dict(fam=fam, ps=int(f.ps), y=f.y, x=f.x, better=len(ok) > 0, dom=dom,
                            imp=(f.y - m.y) if lo else (m.y - f.y), save=f.x / m.x,
                            se=float(np.hypot(f.ys, m.ys))))
    return out


def best_rows(P, task):
    """Per fixed model with a cheaper MiSTR point: the most accurate MiSTR point at no more
    GMACs (ties -> the cheaper). Same dict fields as matched_rows."""
    lo, M, out = task in LOWER, P[P.kind == "mistr"], []
    F = P[P.kind != "mistr"]
    for fam in ("sea", "mon"):
        for _, f in P[P.kind == fam].sort_values("ps", ascending=False).iterrows():
            dom = bool(((F.x <= f.x) & ((F.y <= f.y) if lo else (F.y >= f.y))
                        & ((F.x < f.x) | (F.y != f.y))).any())
            cheap = M[M.x <= f.x]
            if not len(cheap):
                continue
            m = cheap.sort_values(["y", "x"], ascending=[lo, True]).iloc[0]
            better = bool((m.y <= f.y) if lo else (m.y >= f.y))
            out.append(dict(fam=fam, ps=int(f.ps), y=f.y, x=f.x, better=better, dom=dom,
                            imp=(f.y - m.y) if lo else (m.y - f.y), save=f.x / m.x,
                            se=float(np.hypot(f.ys, m.ys))))
    return out


CAP_ACC = (
    r"\caption{\textbf{Compute at matched accuracy.} For each fixed-resolution model, the cheapest "
    r"MiSTR configuration that is at least as accurate (\textcolor{mbetter}{teal} $\Delta$), or, "
    r"if no cheaper configuration is as accurate, the most accurate cheaper one "
    r"(\textcolor{mworse}{maroon} $\Delta$). $\Delta$: MiSTR's metric gain over the fixed model "
    r"(mIoU points; RMSE reduction), $\pm$ its standard error over seeds; saving: the fixed "
    r"model's GMACs over MiSTR's. $^\dagger$, gray: off the fixed-resolution Pareto front, i.e.\ "
    r"dominated by a cheaper, at least as accurate fixed-resolution model. Bold: the largest "
    r"saving at matched accuracy against a Pareto-optimal fixed model. $S_{16}T_\text{sea}$ is "
    r"omitted, as no MiSTR configuration is cheaper. Means over seeds of the points in "
    r"\cref{fig:result}; GMACs per sample on the test split.}",
)
CAP_GMACS = (
    r"\caption{\textbf{Accuracy at matched compute.} For each fixed-resolution model, the most "
    r"accurate MiSTR configuration that needs no more GMACs; $\Delta$ is \textcolor{mbetter}{teal} "
    r"where MiSTR is better and \textcolor{mworse}{maroon} where it is worse. $\Delta$: MiSTR's "
    r"metric gain over the fixed model (mIoU points; RMSE reduction), $\pm$ its standard error "
    r"over seeds; saving: the fixed model's GMACs over MiSTR's. $^\dagger$, gray: off the "
    r"fixed-resolution Pareto front, i.e.\ dominated by a cheaper, at least as accurate "
    r"fixed-resolution model. Bold: the largest gain against a Pareto-optimal fixed model. "
    r"$S_{16}T_\text{sea}$ is omitted, as no MiSTR configuration is cheaper. Means over seeds of "
    r"the points in \cref{fig:result}; GMACs per sample on the test split.}",
)


def matched_table(pts, out, mode="acc"):
    """apx:matched-compute, a full-width table: one shared model column, then per dataset the
    fixed model (metric, GMACs) and the MiSTR point of matched_rows (delta, GMACs saving).
    Bold = largest saving among the better rows = the intro's x-fold numbers.
    mode="gmacs": the MiSTR point of best_rows instead (accuracy at matched compute);
    bold = largest gain against a non-dominated fixed model."""
    dg = {"pastis": 1, "planteur": 1, "biomassters": 2}
    pick, key = (matched_rows, "save") if mode == "acc" else (best_rows, "imp")
    R = {t: {(r["fam"], r["ps"]): r for r in pick(P, t)} for t, P in pts.items()}
    keys = list(next(iter(R.values())))
    assert all(list(v) == keys for v in R.values()), "datasets list different fixed models"
    top = {t: max((r[key] for r in v.values() if r["better"] and not r["dom"]), default=None)
           for t, v in R.items()}
    rows, fam0 = [], None
    for fam, ps in keys:
        if fam0 and fam != fam0:
            rows.append(r"\midrule")
        fam0 = fam
        cells = []
        for t, v in R.items():
            r, d = v[(fam, ps)], dg.get(t, 3)
            col = "mbetter" if r["better"] else "mworse"
            sv = f"{r['save']:.1f}$\\times$"
            dl = f"{r['imp']:+.{d}f}".replace("-", "$-$")
            if r["better"] and not r["dom"] and r[key] == top[t]:
                if key == "save":
                    sv = rf"\textbf{{{sv}}}"
                else:
                    dl = rf"\textbf{{{dl}}}"
            if np.isfinite(r["se"]):   # (s.e. of the difference; none for a 1-seed point)
                dl += rf"{{\tiny$\pm${r['se']:.{d}f}}}"
            if r["dom"]:      # (dominated fixed model: all four cells grey, metric marked)
                g = r"\textcolor{mdom}"
                cells.append(rf"{g}{{{r['y']:.{d}f}$^\dagger$}} & {g}{{{r['x']:.1f}}} & "
                             rf"{g}{{{dl}}} & {g}{{{sv}}}")
                continue
            cells.append(rf"{r['y']:.{d}f} & {r['x']:.1f} & \textcolor{{{col}}}{{{dl}}} & "
                         rf"\textcolor{{mbetter}}{{{sv}}}")   # (saving: always teal)
        rows.append(rf"$S_{{{ps}}}T_\text{{{fam}}}$ & " + " & ".join(cells) + r" \\")
    n = len(R)
    tex = "\n".join([
        r"\providecolor{mbetter}{RGB}{0,128,128}", r"\providecolor{mworse}{RGB}{128,0,0}",
        r"\providecolor{mdom}{gray}{0.6}",
        r"\begin{table*}[t]", r"\centering",
        *(CAP_ACC if mode == "acc" else CAP_GMACS),
        r"\label{tab:matched-" + ("compute" if mode == "acc" else "accuracy") + "}",
        r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l@{\hspace{8pt}}rrrr" + r"!{\color{black!25}\vrule}rrrr" * (n - 1) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{\textbf{{{TASK_HEAD[t]}}}}}" for t in R)
        + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(n)),
        " & " + " & ".join([r"\multicolumn{2}{c}{Fixed Res} & \multicolumn{2}{c}{MiSTR}"] * n)
        + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(2 * n)),
        "Model" + r" & metric & GMACs & $\Delta$ & saving" * n + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    Path(out).write_text(tex)
    print(out, {k: round(v, 1) for k, v in top.items()})



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    out = Path("figures/paper/22_MiSTR-result-std.pdf" if a.paper else "figures/round1_main_std.png")
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=(6.875, 3.1), facecolor=rp.SURFACE)
        pts = {}
        for ax, spec in zip(axes, dt.SEG):
            ax.set_facecolor(rp.SURFACE)
            pts[spec[0]] = panel(ax, *spec)
            ax.set_box_aspect(1.2)
            ax.set_xlabel("GMACs/sample")
            ax.set_ylabel("test metric $\\rightarrow$" if ax is axes[0] else "")
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.9, wspace=0.3)
        star = dict(marker="*", ls="none", ms=7, mew=0)
        h = [plt.Line2D([], [], color=dt.PAIR_COL[k], **star) for k in dt.PAIRS]
        leg = fig.legend(h, [pair_name(*k).replace("+", "$ & $") for k in dt.PAIRS],
                         loc="upper left", borderaxespad=0, ncol=3, frameon=False,
                         title="\u2605 MiSTR (ours), bars: \u00b11 s.e.",
                         title_fontproperties={"weight": "bold",
                                               "family": ["Liberation Serif", "DejaVu Sans"]},
                         alignment="left", handlelength=1.2, columnspacing=1.2)
        fig.add_artist(leg)
        h = [plt.Line2D([], [], color=rp.INK, marker=dt.FAM[f][1], ms=3.8, ls=dt.FAM[f][0],
                        lw=0.7, mfc=rp.INK if dt.FAM[f][2] else rp.SURFACE, alpha=0.75)
             for f in ("sea", "mon")]
        h.append(plt.Rectangle((0, 0), 1, 1, color=rp.INK, alpha=0.12, lw=0))
        leg2 = fig.legend(h, ["seasonal inputs", "monthly inputs", "$\\pm$1 s.e."],
                          loc="upper left", borderaxespad=0, frameon=False,
                          title="fixed-res baselines", title_fontproperties={"weight": "bold"},
                          alignment="left", handlelength=2.6)
        # (the two legends side by side, centred under the panels, as in fig 03)
        r = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        span = to_fig.transform_bbox(Bbox.union([ax.get_tightbbox(r) for ax in axes]))
        w1, w2 = (to_fig.transform_bbox(lg.get_window_extent(r)).width for lg in (leg, leg2))
        left = (span.x0 + span.x1 - w1 - 0.05 - w2) / 2
        leg.set_bbox_to_anchor((left, span.y0 - 0.02), transform=fig.transFigure)
        leg2.set_bbox_to_anchor((left + w1 + 0.05, span.y0 - 0.02), transform=fig.transFigure)
        fig.savefig(out, dpi=None if out.suffix == ".pdf" else 300, facecolor=rp.SURFACE,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(out)
    table(pts, out.with_suffix(".tex") if a.paper else Path("figures/round1_main_std.tex"))
    matched_table(pts, Path("figures/paper/23_matched-compute.tex") if a.paper
                  else Path("figures/round1_matched_compute.tex"))
    matched_table(pts, Path("figures/paper/23_matched-accuracy.tex") if a.paper
                  else Path("figures/round1_matched_accuracy.tex"), mode="gmacs")


if __name__ == "__main__":
    main()
