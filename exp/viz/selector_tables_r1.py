"""App. B selector tables under the round-1 protocol: where (tab:selector) and
when (tab:time-selector), PASTIS, 3 seeds, test and val mIoU (mean +- standard error) and GMACs/SITS.

Rows: the method = fig 03 MiSTR (entropy where + attention when, round 1, 100 ep); the other arms =
stage-2 runs warm-started from the shared stage-1 runs (exp.mistr.paper_configs); "no HR"
= fig 03 fixed S16Tsea / S8Tsea. Writes figures/paper/{spatial,temporal}-selector.tex (copy to
overleaf table/ with the same names).
    python -m exp.viz.selector_tables_r1
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from exp.viz.round1_main import mistr_cost, round1_only, with_test_cost

PAIRS = (("16", "8"), ("8", "4"))
QS = (12.5, 25.0, 50.0)
WHERE = (("none", r"no \hrtok{}"), ("where-random", "random"), ("where-grid", "grid"),
         ("method", "entropy"))
WHEN = (("none", r"no \hrtok{}"), ("when-random", "random"), ("when-grid", "grid"),
        ("when-entropy", "per-season entropy"), ("method", "attention"))


def runs() -> pd.DataFrame:
    d = pd.read_csv("results/pastis/mistr.csv", low_memory=False).drop_duplicates("run", keep="last")
    d = with_test_cost(d, "pastis")     # gmacs = the whole-test-split cost where measured (= fig 03)
    rows = []
    for _, x in d.iterrows():
        f = re.match(r"fixed_pastis_S(16|8)Tsea_ep100_seed(\d)_[0-9a-f]+$", x.run)
        m = re.match(r"mistr_pastis_S(\d+)Tsea\+S(\d+)Tmon_q([\d.]+)_(?:(where-\w+|when-\w+)_)?"
                     r"(warm_ep50|woracle50_ep100)_seed(\d)_[0-9a-f]+$", x.run)
        if f:
            cp = f.group(1)
            for q in QS:
                rows.append(dict(pair=cp, q=q, arm="none", seed=int(f.group(2)), x=x))
            continue
        if not m:
            continue
        if m.group(5) == "warm_ep50":     # stage 2 from OUR shared stage 1 only
            if not m.group(4) or not re.match(r"mistr_pastis_.*_woracle50_ep50_seed\d_[0-9a-f]+$",
                                              str(x["train.init_from"])):
                continue
            arm = m.group(4)
        elif m.group(4):
            continue
        else:
            arm = "method"
        rows.append(dict(pair=m.group(1), q=float(m.group(3)), arm=arm, seed=int(m.group(6)), x=x))
    r = pd.DataFrame(rows)
    r["test"] = [100 * x.test_miou for x in r.x]
    r["val"] = [100 * x.val_miou for x in r.x]
    # cost over the WHOLE test split, as fig 03 (the 4-val-batch trace is never used here)
    # a MiSTR run without a whole-test measurement: fig 03's estimate (mistr_cost: the fixed model's
    # test cost + the traced cost per crop x its test crops), from the same round-1 fixed rows
    r1 = round1_only(d)
    r["gmacs"] = [x.gmacs if x.cost_on_test else mistr_cost(x, r1) if a != "none" else float("nan")
                  for x, a in zip(r.x, r.arm)]
    # fixed models' cost is deterministic: a run from before gmacs_test takes its seed-mates' value
    for (p, q, a), g in r[r.arm == "none"].groupby(["pair", "q", "arm"]):
        r.loc[g.index, "gmacs"] = g.gmacs.fillna(g.gmacs.mean())
    if r.gmacs.isna().any():
        raise SystemExit(f"no gmacs_test: {sorted(set(r[r.gmacs.isna()].arm))}")
    return r.drop(columns="x")


def stats(r, pair, q, arm):
    s = r[(r.pair == pair) & (r.q == q) & (r.arm == arm)]
    if len(s) != 3:
        raise SystemExit(f"S{pair} q{q} {arm}: {len(s)} seeds")
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))      # noqa: E731
    return dict(test=s.test.mean(), tse=se(s.test), val=s.val.mean(), vse=se(s.val),
                gmacs=s.gmacs.mean(), gse=se(s.gmacs))


def cell(v, e, best, d=1):
    """mean +- standard error (every number in these tables carries its error bar)"""
    m = rf"\textbf{{{v:.{d}f}}}" if best else f"{v:.{d}f}"
    return rf"{m}\,{{\footnotesize$\pm${e:.{d}f}}}"


def block(r, arms, cols, key, ekey, title, ncol):
    S = {(a, c): stats(r, *c, a) for a, _ in arms for c in cols}
    out = [rf"\multicolumn{{{ncol + 1}}}{{l}}{{\emph{{{title}}}}} \\"]
    for a, label in arms:
        cells = []
        for c in cols:
            best = max(S[(b, c)][key] for b, _ in arms if b != "none")
            if key == "gmacs":
                cells.append(cell(S[(a, c)]["gmacs"], S[(a, c)]["gse"], False, 2))
            else:
                cells.append(cell(S[(a, c)][key], S[(a, c)][ekey], a != "none" and S[(a, c)][key] == best))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


def where_table(r):
    cols = [(p, q) for p, _ in PAIRS for q in QS]
    nw = {"16": 64, "8": 256}
    head = " & ".join(str(int(q / 100 * nw[p])) for p, q in cols)
    body = (block(r, WHERE, cols, "test", "tse", "test mIoU", 6) + [r"\midrule"]
            + block(r, WHERE, cols, "gmacs", None, "GMACs per SITS (test)", 6))
    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{\textbf{Where to re-encode.} PASTIS mIoU and compute at equal \hrtok{} budget: "
        r"$n$ points per sample (on average for entropy, which uses one global threshold), i.e.\ a "
        r"share $q$ of all \Shi{} points, each re-encoded at the months of the season picked by "
        r"decoder attention. Only \emph{where} differs. Mean $\pm$ standard error over three seeds; "
        r"GMACs vary across seeds because selection is adaptive (\cref{apx:selection}).}",
        r"\label{tab:selector}", r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{$S_{16}T_\text{sea}$ \& $S_8T_\text{mon}$} & "
        r"\multicolumn{3}{c}{$S_8T_\text{sea}$ \& $S_4T_\text{mon}$} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}", rf"$n$ & {head} \\",
        r"$q$ & " + " & ".join(["1/8", "1/4", "1/2"] * 2) + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def when_table(r):
    cols = [(p, 25.0) for p, _ in PAIRS]
    S = {(a, c): stats(r, *c, a) for a, _ in WHEN for c in cols}
    rows = []
    for a, label in WHEN:
        cells = []
        for c in cols:
            for key, ekey in (("test", "tse"),):     # (test only: val is shown nowhere else)
                best = max(S[(b, c)][key] for b, _ in WHEN if b != "none")
                cells.append(cell(S[(a, c)][key], S[(a, c)][ekey],
                                  a != "none" and S[(a, c)][key] == best))
            cells.append(cell(S[(a, c)]["gmacs"], S[(a, c)]["gse"], False, 2))
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        if a == "none":
            rows.append(r"\midrule")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{\textbf{When to re-encode.} PASTIS mIoU and compute at equal \hrtok{} budget: the "
        r"same entropy-picked points ($q = 1/4$), each re-encoded at the months of one season; only "
        r"\emph{which} season differs (grid: seasons cycle over the lattice of points). Mean $\pm$ "
        r"standard error over three seeds; GMACs vary across seeds because selection is adaptive "
        r"(\cref{apx:selection}).}",
        r"\label{tab:time-selector}", r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r" & \multicolumn{2}{c}{$S_{16}T_\text{sea}$ \& $S_8T_\text{mon}$} & "
        r"\multicolumn{2}{c}{$S_8T_\text{sea}$ \& $S_4T_\text{mon}$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & test mIoU & GMACs & test mIoU & GMACs \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def main():
    r = runs()
    out = Path("figures/paper")
    (out / "spatial-selector.tex").write_text(where_table(r))
    (out / "temporal-selector.tex").write_text(when_table(r))
    print(f"wrote {out}/spatial-selector.tex, {out}/temporal-selector.tex")


if __name__ == "__main__":
    main()
