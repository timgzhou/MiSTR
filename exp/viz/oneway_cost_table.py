"""Table oneway-cost (appendix G): inference cost of the second pass's attention modes, from
results/pastis/oneway_cost.csv (exp.mistr.oneway_cost on the round-1 S8+S4 q 25 % seed-0 run).

    python -m exp.viz.oneway_cost_table   -> figures/paper/oneway-cost.tex
"""
from pathlib import Path

import pandas as pd

CSV = Path("results/pastis/oneway_cost.csv")
OUT = Path("figures/paper/oneway-cost.tex")
ROWS = [("none", "coarse_pass_only"), ("cross-attention", "fine_reads_coarse"),
        ("one-way attention", "one_way"), ("self-attention", "joint")]
FINE_PX, MONTHS_PER_SEASON = 4, 3          # 8 px windows as 4 px tokens: 4 tokens per window-month


def main():
    d = pd.read_csv(CSV).drop_duplicates("attention", keep="last").set_index("attention")
    base = d.loc["coarse_pass_only"]
    coarse_tokens = 64 // 8 * (64 // 8) * 4 * 4    # 8x8 grid x 4 seasons x 4 band groups (S2: 3, S1: 1)
    fine = round(d.loc["one_way", "tokens_per_block"] - coarse_tokens)
    n = int(d.loc["one_way", "compute_samples"])
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{\textbf{Inference cost of the second pass's attention.} None: the \lrtok{} pass alone.",
             r"Cross-attention: \hrtok{} tokens read the cached \lrtok{} tokens only. One-way attention (MiSTR):",
             r"\hrtok{} tokens read the cached \lrtok{} tokens and each other. Self-attention: all \lrtok{} and",
             r"\hrtok{} tokens re-encoded jointly. PASTIS (64$\times$64 px, 12 months in 4 seasons), \lrtok{} =",
             r"8$\times$8 px seasonal tokens, \hrtok{} = 4$\times$4 px monthly tokens, 25\% of points selected",
             rf"with one season each ({fine // 4} selected point-months, {fine} \hrtok{{}} tokens per sample). Same model and selections;",
             rf"GMACs per sample, mean over {n} test samples, including the \lrtok{{}} pass. $^\dagger$Summed over both passes.}}",
             r"\label{tab:oneway-cost}", r"\footnotesize", r"\begin{tabular}{l c c c}", r"\toprule",
             r"Attention & tokens / block$^\dagger$ & GMACs (attn.) & 2nd pass \\", r"\midrule"]
    for label, key in ROWS:
        r = d.loc[key]
        tok = coarse_tokens if key == "coarse_pass_only" else round(r.tokens_per_block)
        g, ga = f"{r.gmacs:.1f}", f"({r.gmacs_attention:.1f})"
        second = "--" if key == "coarse_pass_only" else f"{r.gmacs - base.gmacs:.1f}"
        if key == "one_way":
            g, second = rf"\textbf{{{g}}}", rf"\textbf{{{second}}}"
        lines.append(f"{label:<19} & {tok} & {g + ' ' + ga:<11} & {second} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
