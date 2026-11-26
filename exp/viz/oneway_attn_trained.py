"""App. G: the second pass's attention, trained. PASTIS, S16+S8 and
S8+S4, q 1/8, 1/4, 1/2; every run = stage 2 (50 ep) from ONE shared round-1 oracle warm-up per
(pair, q, seed) (logs/launch_attn_ablation.sh), so the three attention designs differ only in
model.attention: one_way (MiSTR), joint (self-attention), fine_reads_coarse (cross-attention).
Test mIoU against test GMACs per sample, means over seeds (figure: unjoined points, no bars; table:
+- standard error); the fixed seasonal + monthly curves (fig 03's round-1 runs) for context.

    python -m exp.viz.oneway_attn_trained          -> figures/oneway_attn_trained.png (preview)
    python -m exp.viz.oneway_attn_trained --paper  -> figures/paper/24_one-way-attention-trained.pdf
                                                     + figures/paper/oneway-trained.tex (table)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from exp.viz import datasets_tradeoff as dt  # noqa: E402
from exp.viz import common as rp  # noqa: E402
from exp.viz import round1_main as r1  # noqa: E402

TASK = "pastis"
PAIRS = {"S16Tsea+S8Tmon": (16, 8), "S8Tsea+S4Tmon": (8, 4)}
QS = (0.125, 0.25, 0.5)
FRAC = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}
# (name as in table/oneway-cost.tex, colour, marker, marker size)
MODES = {"one_way": ("one-way attention (MiSTR)", "#1b7f79", "*", 8.5),
         "joint": ("self-attention", "#d95f02", "o", 4.2),
         "fine_reads_coarse": ("cross-attention", "#7570b3", "s", 4.0)}


def runs():
    """The ablation's stage-2 rows, test cost (as fig 03) and mIoU in 0-100."""
    p = Path(f"results/{TASK}/mistr.csv")
    d = pd.read_csv(p, low_memory=False).drop_duplicates("run", keep="last")
    if "eval_percentage" in d:
        d = d[d.eval_percentage.isna()]
    d = r1.pct(r1.with_test_cost(d, TASK), TASK)
    ref = r1.rows(TASK)                       # (round-1 fixed runs: the cost model's crop-free pass)
    m = d[d["train.init_from"].fillna("").str.contains(r"_woracle50_ep50_seed\d")
          & d.run.str.contains(r"_warm_ep50_seed\d")
          & (d["select.where"] == "uncertainty") & (d["select.when"] == "attention")
          & (d["model.backbone"].fillna("tiny") == "tiny")].copy()
    m["pair"] = m.run.str.extract("(" + "|".join(k.replace("+", r"\+") for k in PAIRS) + ")")[0]
    m = m[m.pair.notna() & m["select.percentage"].isin(QS)]
    m["x"] = [r1.mistr_cost(r, ref) for _, r in m.iterrows()]
    return m, ref


def points(m):
    g = m.groupby(["pair", "select.percentage", "model.attention"])
    P = g.agg(n=("x", "size"), x=("x", "mean"), xs=("x", "sem"),
              y=("test_primary", "mean"), ys=("test_primary", "sem")).reset_index()
    return P.rename(columns={"select.percentage": "q", "model.attention": "attn"})


def draw(P, ref, out):
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.3, 2.9), facecolor=rp.SURFACE)
        ax.set_facecolor(rp.SURFACE)
        # fixed-resolution curves (fig 03's round-1 runs), cut to within 2x of the plotted costs;
        # only each curve's two ends named (as fig 03)
        for fam in ("sea", "mon"):
            ls, mk, filled, _ = dt.FAM[fam]
            f = sorted((x, y, ps) for fm, ps, x, y in r1.fixed(TASK, ref) if fm == fam)
            f = [t for t in f if P.x.min() / 2 <= t[0] <= P.x.max() * 2]
            if not f:
                continue
            ax.plot([t[0] for t in f], [t[1] for t in f], ls + mk, color=rp.INK, lw=0.7,
                    ms=3.6, mfc=rp.INK if filled else rp.SURFACE, mec=rp.INK, mew=0.8,
                    alpha=0.6, zorder=2)
            for x, y, ps in (f[0], f[-1]):
                ax.annotate(rf"$S_{{{ps}}}T_\mathrm{{{fam}}}$", (x, y), xytext=(3, -2),
                            textcoords="offset points", ha="left", va="top", fontsize=7.5,
                            color=rp.MUTED, zorder=5)
        for attn, (_, col, mk, ms) in MODES.items():     # mixed resolution: lone points
            g = P[P.attn == attn]
            ax.plot(g.x, g.y, mk, color=col, ms=ms, mec=rp.INK, mew=0.25, ls="none", zorder=6)
        for pair, (c, fn) in PAIRS.items():        # pair name over its group of points
            g = P[P.pair == pair]
            ax.annotate(rf"$S_{{{c}}}T_\mathrm{{sea}}$ & $S_{{{fn}}}T_\mathrm{{mon}}$",
                        ((g.x.min() * g.x.max()) ** 0.5, g.y.max()), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, color=rp.GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(0.08, 0.12)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 1.6)                  # (room for the right-end labels)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("GMACs/sample")
        ax.set_ylabel(r"PASTIS test mIoU $\rightarrow$")
        # two legends side by side under the axes (as fig 03): mixed resolution | fixed
        h = [plt.Line2D([], [], color=c, marker=mk, ms=ms * 0.9, mec=rp.INK, mew=0.25, ls="none")
             for _, c, mk, ms in MODES.values()]
        leg = fig.legend(h, [v[0] for v in MODES.values()], loc="upper left",
                         bbox_to_anchor=(0.0, -0.075), frameon=False, fontsize=7.5,
                         title="mixed resolution",
                         title_fontproperties={"weight": "bold", "size": 7.5},
                         alignment="left", handlelength=1.2, borderaxespad=0)
        fig.add_artist(leg)
        h = [plt.Line2D([], [], color=rp.INK, marker=dt.FAM[f][1], ms=3.6, ls=dt.FAM[f][0],
                        lw=0.7, mfc=rp.INK if dt.FAM[f][2] else rp.SURFACE, alpha=0.6)
             for f in ("sea", "mon")]
        fig.legend(h, ["seasonal inputs", "monthly inputs"], loc="upper left",
                   bbox_to_anchor=(0.62, -0.075), frameon=False, fontsize=7.5,
                   title="fixed resolution", title_fontproperties={"weight": "bold", "size": 7.5},
                   alignment="left", handlelength=2.4, borderaxespad=0)
        fig.savefig(out, dpi=None if out.suffix == ".pdf" else 300, facecolor=rp.SURFACE,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(out)


def table(P, out):
    """Pair x q rows, per design: GMACs and test mIoU mean +- standard error (one column wide:
    the pair named once per block, short design names)."""
    short = {"one_way": "one-way (MiSTR)", "joint": "self-attention",
             "fine_reads_coarse": "cross-attention"}
    rows = []
    for pair, (c, fn) in PAIRS.items():
        for i, q in enumerate(QS):
            cells = []
            for attn in MODES:
                r = P[(P.pair == pair) & (P.q == q) & (P.attn == attn)]
                if not len(r):
                    cells.append("-- & --")
                    continue
                r = r.iloc[0]
                s = f"{r.y:.1f}" + (f"$\\pm${r.ys:.1f}" if r.n > 1 else "")
                cells.append(f"{r.x:.1f} & {s}")
            name = rf"$S_{{{c}}}$ \& $S_{{{fn}}}$" if i == 0 else ""
            rows.append(f"{name} & {FRAC[q]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{short[k]}}}" for k in MODES)
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Trained second-pass attention designs} (\cref{fig:oneway-trained}). "
        r"PASTIS; every model is trained from the same guided warm-up per configuration and "
        r"seed and differs only in its second-pass attention. Pairs $S_c$ \& $S_f$ = "
        r"$S_cT_\text{sea}$ \& $S_fT_\text{mon}$. GMACs per sample and test mIoU, mean $\pm$ "
        r"standard error over three seeds.}",
        r"\label{tab:oneway-trained}", r"\scriptsize", r"\setlength{\tabcolsep}{2.3pt}",
        r"\begin{tabular}{l c rl rl rl}", r"\toprule", " & & " + head + r" \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Pair & $q$" + " & GMACs & mIoU" * 3 + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    Path(out).write_text(tex)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    m, ref = runs()
    P = points(m)
    print(P.assign(y=P.y.round(2), ys=P.ys.round(2), x=P.x.round(1)).to_string(index=False))
    print("cost on test:", m.cost_on_test.value_counts().to_dict())
    out = Path("figures/paper/24_one-way-attention-trained.pdf" if a.paper
               else "figures/oneway_attn_trained.png")
    draw(P, ref, out)
    table(P, Path("figures/paper/oneway-trained.tex") if a.paper
          else Path("figures/oneway_attn_trained.tex"))


if __name__ == "__main__":
    main()
