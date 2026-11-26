"""App. G one-way analysis tables from exp.mistr.mech_probe, mean +- standard error over seeds
:

    tab:oneway-attn     attention mass / key share (%) of the hr queries per key group
    tab:oneway-mask     test-metric change when a key group is masked out (test time only)
    tab:oneway-decoder  the decoder's hr weight w and lr / hr / final accuracy (RMSE)

Runs: fig 03's round-1 MiSTR q 1/4, PASTIS S16+S8 + S8+S4, PLANTEUR / BioMassters / YieldSAT
S16+S8, seeds 0 1 2 (results/<task>/mech_{attn,decoder}.csv; a run probed twice keeps its last
rows). Also prints every number App. G's prose quotes. Writes figures/paper/oneway-{attn,mask,
decoder}.tex (copy to overleaf table/ with the same names).
    python -m exp.viz.oneway_probe_tables
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = [("pastis", "16", "8", "PASTIS S16+S8"), ("pastis", "8", "4", "PASTIS S8+S4"),
        ("planteur", "16", "8", "PLANTEUR S16+S8"), ("biomassters", "16", "8", "BioMassters S16+S8"),
        ("yieldsat_raw", "16", "8", "YieldSAT S16+S8")]
PCT = {"pastis", "planteur"}                     # mIoU / accuracy in 0-100
DEC = {"pastis": 1, "planteur": 1, "biomassters": 2, "yieldsat_raw": 3}   # metric-change decimals
COLS4 = {"this_hr": ("self", "crop", "hr_month"), "this_lr": ("lr_season_loc", "lr_season_far"),
         "other_hr": ("hr_window", "hr_season", "hr_other"), "other_lr": ("lr_other_loc", "lr_other_far"),
         "same": ("self", "crop", "hr_window", "lr_season_loc", "lr_other_loc")}
MASK = [("no_lr", "all"), ("no_hr_others", "other points' hr"), ("own_crop_only", "all but own point")]
# (own-season / other-season lr splits dropped: a parallel argument)


def tex(label: str) -> str:
    """'PASTIS S16+S8' -> 'PASTIS $S_{16}$ \\& $S_8$' (the paper's notation)."""
    s = lambda n: f"$S_{{{n}}}$" if len(n) > 1 else f"$S_{n}$"
    return re.sub(r"S(\d+)\+S(\d+)", lambda m: rf"{s(m[1])} \& {s(m[2])}", label)


def run_re(task, cp, w):
    ep = "woracle10_ep20" if task == "biomassters" else "woracle50_ep100"
    return re.compile(rf"mistr_{task}_S{cp}Tsea\+S{w}Tmon_q25_{ep}_seed(\d)_[0-9a-f]{{8}}$")


def load(task, cp, w):
    rx = run_re(task, cp, w)
    A = pd.read_csv(f"results/{task}/mech_attn.csv")
    D = pd.read_csv(f"results/{task}/mech_decoder.csv")
    A, D = A[A.run.str.match(rx)], D[D.run.str.match(rx)]
    D = D.drop_duplicates(["run", "ablation", "stat"], keep="last")
    A = A.drop_duplicates(["run", "layer", "head", "group"], keep="last")
    seeds = sorted({rx.match(r).group(1) for r in D.run})
    if seeds != ["0", "1", "2"]:
        raise SystemExit(f"{task} S{cp}+S{w}: seeds {seeds}")
    return A, D


def se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v))


def pm(m, e, d, sign=False):
    """mean +- s.e.; 2+ decimals drop the leading zero (.731), 0-1 decimals keep it (0.5)."""
    s = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
    es = f"{e:.{d}f}"
    if d >= 2:
        s, es = s.replace("0.", ".", 1), es.replace("0.", ".", 1)
    return rf"{s.replace('-', '$-$')}\,{{\scriptsize$\pm${es}}}"


def attn_stats(A):
    """per run: % mass / % key share of each column group (mean over blocks and heads)."""
    out = {}
    for run, g in A[A["head"] >= 0].groupby("run"):
        a = g.groupby(["layer", "group"])[["mass", "key_share"]].mean().groupby("group").mean()
        out[run] = {k: (100 * a.loc[list(v), "mass"].sum(), 100 * a.loc[list(v), "key_share"].sum())
                    for k, v in COLS4.items()}
    return out


def attn_table(data):
    lines = []
    for (task, cp, w, label), (A, D) in data.items():
        s = attn_stats(A)
        cells = []
        for k in COLS4:
            m = [s[r][k][0] for r in s]
            cells.append(pm(np.mean(m), se(m), 1))     # (mass only; key shares are in the text)
        lines.append(f"{tex(f'{label:<19}')}& " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{What MiSTR's \hrtok{} tokens attend to}: mean attention (\%) that "
        r"\hrtok{} queries give each key group (over all blocks and heads), mean $\pm$ standard "
        r"error over three seeds. ``This month'' \hrtok{} = the \hrtok{} tokens of the query's own "
        r"point and of the other selected points in the same month. ``Same loc.'' = the same spatial "
        r"point in other months, and the \lrtok{} cells under it. Pairs $S_c$ \& $S_f$ = "
        r"$S_cT_\text{sea}$ \& $S_fT_\text{mon}$.}",
        r"\label{tab:oneway-attn}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l ccccc}", r"\toprule",
        r" & \multicolumn{2}{c}{this month / season} & \multicolumn{2}{c}{other months / seasons} & same \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"MiSTR config & \hrtok{} & \lrtok{} & \hrtok{} & \lrtok{} & loc. \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def metric(D, task, abl):
    """per-seed test metric (0-100 for PCT tasks) of one ablation, in run order."""
    v = D[(D.ablation == abl) & (D.stat == "test_primary")].set_index("run").value.sort_index()
    return v * (100 if task in PCT else 1)


def mask_table(data):
    lines = []
    for (task, cp, w, label), (A, D) in data.items():
        full = metric(D, task, "none")
        cells = []
        for abl, _ in MASK:
            dlt = (metric(D, task, abl) - full).values
            cells.append(pm(dlt.mean(), se(dlt), DEC[task], sign=True))
        lines.append(f"{tex(f'{label:<19}')}& " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"% exp.viz.oneway_probe_tables, from exp.mistr.mech_probe",
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Which context the trained model relies on}: change in the test metric "
        r"when a key group is masked out of the \hrtok{} tokens' attention at every block, at test "
        r"time only, for a model trained with all tokens (mIoU points$\uparrow$ on PASTIS and "
        r"PLANTEUR, RMSE$\downarrow$ in t/ha on BioMassters and YieldSAT), mean $\pm$ standard error over "
        r"three seeds. Pairs $S_c$ \& $S_f$ = $S_cT_\text{sea}$ \& "
        r"$S_fT_\text{mon}$.}",
        r"\label{tab:oneway-mask}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc}", r"\toprule",
        r" & \multicolumn{3}{c}{masked out} \\", r"\cmidrule(lr){2-4}",
        r" & all & other & all but \\",
        r"MiSTR config & \lrtok{} & \hrtok{} & own spatial point \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def dstat(D, k):
    return D[(D.ablation == "none") & (D.stat == k)].set_index("run").value.sort_index().values


def decoder_table(data):
    lines = []
    for (task, cp, w, label), (A, D) in data.items():
        wm, q10, q90 = dstat(D, "w_mean"), dstat(D, "w_q10"), dstat(D, "w_q90")
        cls = task in PCT
        k = "acc" if cls else "rmse"
        v = {n: dstat(D, f"{k}_{n}") * (100 if cls else 1) for n in ("coarse", "crop", "final")}
        best = (max if cls else min)(v, key=lambda n: v[n].mean())
        d = 1 if cls else 3
        cells = [rf"\textbf{{{pm(v[n].mean(), se(v[n]), d)}}}" if n == best else pm(v[n].mean(), se(v[n]), d)
                 for n in v]
        wcell = f"{pm(wm.mean(), se(wm), 2)} ({q10.mean():.2f}--{q90.mean():.2f})".replace("0.", ".")
        lines.append(f"{tex(label)} & {wcell} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{How the decoder combines \lrtok{} and \hrtok{} tokens}, at pixels covered "
        r"by \hrtok{} tokens. With one-way attention the \lrtok{} tokens are unchanged by the "
        r"\hrtok{} tokens, so the output there is exactly $(1-w)\,p_{\lrtok{}} + w\,p_{\hrtok{}}$, "
        r"where $p_{\lrtok{}}$ and $p_{\hrtok{}}$ are the pooled predictions of the \lrtok{} and "
        r"\hrtok{} tokens and $w$ is the \hrtok{} tokens' total weight in the decoder's pool. Mean "
        r"$\pm$ standard error over three seeds ($w$ percentiles: means). The combined prediction "
        r"beats both inputs on every run. Regression RMSE is in normalized target units. Pairs $S_c$ "
        r"\& $S_f$ = $S_cT_\text{sea}$ \& $S_fT_\text{mon}$.}",
        r"\label{tab:oneway-decoder}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l c ccc}", r"\toprule",
        r" & \hrtok{} weight & \multicolumn{3}{c}{acc.\ (\%)$\uparrow$ / RMSE$\downarrow$} \\",
        r"\cmidrule(lr){3-5}",
        r"MiSTR config & mean (p10--p90) & \lrtok{} & \hrtok{} & final \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def prose(data):
    """The numbers App. G's prose quotes, recomputed (seed means; per-seed where it says 'every')."""
    print("\n== prose checks")
    lr_m, lr_u, same_m, same_u, ws = [], [], [], [], []
    for (task, cp, w, label), (A, D) in data.items():
        s = attn_stats(A)
        m = {k: np.mean([s[r][k][0] for r in s]) for k in COLS4}
        u = {k: np.mean([s[r][k][1] for r in s]) for k in COLS4}
        lr_m.append(m["this_lr"] + m["other_lr"]); lr_u.append(u["this_lr"] + u["other_lr"])
        same_m.append(m["same"]); same_u.append(u["same"]); ws.append(dstat(D, "w_mean").mean())
        full = metric(D, task, "none")
        rel = {a: ((metric(D, task, a) - full) / full * 100).values
               for a in ("no_lr", "no_lr_this_season", "no_lr_other_season", "no_hr_others", "own_crop_only")}
        sign = 1 if task in PCT else -1             # >0 = masking hurts
        hurt = {a: -sign * rel[a] for a in rel}
        print(f"{label}: own-season |change| % per seed {np.round(np.abs(rel['no_lr_this_season']), 2)}; "
              f"other-season hurts more than own-season per seed "
              f"{(hurt['no_lr_other_season'] > hurt['no_lr_this_season']).tolist()}; own-crop-only "
              f"hurts most per seed {[bool(hurt['own_crop_only'][i] == max(h[i] for h in hurt.values())) for i in range(3)]}")
        fol = dstat(D, "dis_final_follows_crop")
        print(f"   w {dstat(D, 'w_mean').round(3)}  final follows crop when disagreeing "
              f"{(100 * fol).round(1) if len(fol) else '-'}")
        cls = task in PCT
        k = "acc" if cls else "rmse"
        v = {n: dstat(D, f"{k}_{n}") for n in ("coarse", "crop", "final")}
        ok = (v["final"] > np.maximum(v["coarse"], v["crop"])) if cls else \
             (v["final"] < np.minimum(v["coarse"], v["crop"]))
        print(f"   final beats both per seed {ok.tolist()}; crop weaker than coarse per seed "
              f"{((v['crop'] < v['coarse']) if cls else (v['crop'] > v['coarse'])).tolist()}")
    print(f"lr mass {min(lr_m):.0f}-{max(lr_m):.0f}% vs keys {min(lr_u):.0f}-{max(lr_u):.0f}% "
          f"(prose: 38-52 vs 36-52)")
    print(f"same loc keys {min(same_u):.0f}-{max(same_u):.0f}% attention {min(same_m):.0f}-{max(same_m):.0f}% "
          f"(prose: 1-5 / 13-23)")
    print(f"w {min(ws):.2f}-{max(ws):.2f} (prose: 0.58-0.74)")


def main():
    data = {r: load(*r[:3]) for r in ROWS}
    out = Path("figures/paper")
    (out / "oneway-attn.tex").write_text(attn_table(data))
    (out / "oneway-mask.tex").write_text(mask_table(data))
    (out / "oneway-decoder.tex").write_text(decoder_table(data))
    print(f"wrote {out}/oneway-{{attn,mask,decoder}}.tex")
    prose(data)


if __name__ == "__main__":
    main()
