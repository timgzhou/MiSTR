"""Season-range PASTIS runs (App. C, flexible time budget).

One checkpoint per (pair, q) trained with 1 / 1.5 / 2 seasons per picked window drawn per batch
(select.extra_steps_choices=0,0.5,1, extra season = anchor pick select.extra_pick=disagree), tested
at 1, 1.5 and 2 seasons (train.eval_percentages=q:0.5,q:1). Against the round-1 fixed curves; test mIoU vs test GMACs.

    python -m exp.viz.season_range_tradeoff  -> figures/paper/20_flexible-temporal-budget.pdf
        (appendix; all pairs x budgets x tested seasons as unconnected points)
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz.round1_main import mistr_cost, pct, round1_only, with_test_cost  # noqa: E402

INK, INK2, GRID, SURF, FIX = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29d"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # palette slots 1-3
PAIRS = ["S16+S8", "S14+S7", "S12+S6", "S10+S5", "S8+S4", "S6+S3"]
SEASONS = ((None, "1 season", BLUE, "o"), ("q:0.5", "1.5 seasons", AQUA, "s"),
           ("q:1", "2 seasons", ORANGE, "^"))


def key(run):
    m = re.match(r"mistr_pastis_(S\d+)Tsea\+(S\d+)Tmon.*?_q([\d.]+)_", run)
    return (f"{m[1]}+{m[2]}", float(m[3])) if m else (None, None)


def load():
    d = pd.read_csv("results/pastis/mistr.csv", low_memory=False)
    new = d[d.run.str.contains("_xset0-0.5-1_xdis_woracle50_ep100")].drop_duplicates(
        ["run", "eval_percentage"], keep="last")
    new = new.assign(pair=[key(r)[0] for r in new.run], q=[key(r)[1] for r in new.run],
                     g=new.gmacs_test)
    r1 = round1_only(d[d.eval_percentage.isna()]).drop_duplicates("run", keep="last")
    r1 = with_test_cost(r1, "pastis")
    fx = r1[r1.run.str.startswith("fixed_")].assign(g=lambda x: x.gmacs)
    fx = pct(fx, "pastis").groupby(["model.coarse_time", "model.coarse_patch"], as_index=False).agg(
        g=("g", "mean"), test_primary=("test_primary", "mean"), yse=("test_primary", "sem"))
    fx["yse"] = fx.yse.fillna(0)                      # (mean +- s.e. over seeds)
    m = r1[r1.run.str.startswith("mistr_")]
    m = m.assign(g=[mistr_cost(r, r1) for _, r in m.iterrows()],
                 pair=[key(r)[0] for r in m.run], q=[key(r)[1] for r in m.run])
    # mean +- s.e. over seeds per (pair, q, tested seasons) (seeds 1 2:)
    new = pct(new, "pastis").assign(ev=lambda x: x.eval_percentage.fillna("")).groupby(
        ["pair", "q", "ev"], as_index=False).agg(
        g=("g", "mean"), gse=("g", "sem"), test_primary=("test_primary", "mean"),
        yse=("test_primary", "sem"), n=("g", "size"))
    new[["gse", "yse"]] = new[["gse", "yse"]].fillna(0)
    new["eval_percentage"] = new.ev.replace("", None)
    print("seeds per point:", new.n.value_counts().to_dict())
    return new, pct(m, "pastis"), fx      # mIoU in 0-100


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))


def fixed(ax, fx, lo=0, hi=1e9, names=True, ylo=0):
    for t, ls, mk, lab in (("season", (0, (4, 3)), "s", "fixed seasonal"),
                           ("month", "-", "D", "fixed monthly")):
        g = fx[fx["model.coarse_time"] == t].sort_values("g")
        ax.plot(g.g, g.test_primary, ls=ls, color=FIX, lw=.9, marker=mk, ms=2.5, zorder=1,
                label=lab)
        ax.errorbar(g.g, g.test_primary, yerr=g.yse, fmt="none", ecolor=FIX, elinewidth=.5,
                    capsize=1, zorder=1)
        if names:
            for _, r in g[(g.g > lo) & (g.g < hi) & (g.test_primary > ylo + .8)].iterrows():
                tt = "sea" if t == "season" else "mon"
                ax.annotate(rf"$S_{{{int(r['model.coarse_patch'])}}}T_{{{tt}}}$",
                            (r.g, r.test_primary), xytext=(3, -7) if t == "season" else (4, -6),
                            textcoords="offset points", fontsize=5, color=INK2)


def pick(p, ev):
    return p[p.eval_percentage.isna() if ev is None else p.eval_percentage == ev].sort_values("g")


def single(new, fx):
    fig, ax = plt.subplots(figsize=(3.3, 2.55), facecolor=SURF)
    style(ax)
    lo, hi = new.g.min() / 1.3, new.g.max() * 1.3
    ylo = new.test_primary.min() - 1
    fixed(ax, fx, lo, hi, ylo=ylo)
    for ev, lab, c, mk in SEASONS:
        s = pick(new, ev)
        ax.errorbar(s.g, s.test_primary, xerr=s.gse, yerr=s.yse, fmt="none", ecolor=c,
                    elinewidth=.5, capsize=1, zorder=3)
        ax.plot(s.g, s.test_primary, ls="none", color=c, marker=mk, ms=4.2, mec=SURF, mew=.6,
                zorder=4, label=f"MiSTR, {lab}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, new.test_primary.max() + 1)
    ax.set_xlabel("GMACs per SITS", fontsize=6.5)
    ax.set_ylabel("test mIoU", fontsize=6.5)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(loc="lower right", frameon=False, fontsize=6, labelcolor=INK, handlelength=1.8)
    # (no in-figure title: the caption says it, as every other appendix figure)
    return fig


OUT = Path("figures/paper/20_flexible-temporal-budget.pdf")


def main():
    new, _, fx = load()
    plt.rcParams.update({"font.size": 6.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
    fig = single(new, fx)
    fig.savefig(OUT, facecolor=SURF, bbox_inches="tight", pad_inches=0.02)
    print(OUT)


if __name__ == "__main__":
    main()
