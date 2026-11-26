"""App. A, table token-counts: tokens per SITS of fixed-resolution OlmoEarth Tiny (seed 0) on the
64 x 64 px tiles, from results/<task>/mistr.csv (tokens_per_block, mean over the training split).
A patch size without a run (S2 seasonal, S4 monthly) is the next coarser row, rounded, times the
ratio of patch counts (ceil(64 / p)^2) -- the caption's "scaled from S16" up to 2 tokens on YieldSAT.

    python -m exp.viz.token_counts_table        # -> figures/paper/token-counts.tex (the tabular rows)
"""
import csv
import math
import re
from pathlib import Path

TASKS = ("pastis", "planteur", "biomassters", "yieldsat_raw")
ROWS = (("sea", (16, 12, 8, 6, 4, 2)), ("mon", (16, 12, 8, 6, 4)))
OUT = Path("figures/paper/token-counts.tex")


def tokens(task):
    """{(patch, time): tokens per block} of the seed-0 OlmoEarth Tiny fixed runs."""
    rx = re.compile(rf"^fixed_{task}_S(\d+)T(sea|mon)_ep\d+_seed0_[0-9a-f]{{8}}$")
    out = {}
    for r in csv.DictReader(open(f"results/{task}/mistr.csv")):
        m = rx.match(r["run"])
        if m and r.get("tokens_per_block"):
            out[(int(m[1]), m[2])] = float(r["tokens_per_block"])
    return out


def count(tok, p, t):
    if (p, t) in tok:
        return tok[(p, t)]
    return round(count(tok, 2 * p, t)) * (math.ceil(64 / p) / math.ceil(64 / (2 * p))) ** 2


def main():
    tok = {t: tokens(t) for t in TASKS}
    lines = []
    for i, (t, ps) in enumerate(ROWS):
        if i:
            lines.append(r"\midrule")
        for p in ps:
            vals = " & ".join(f"{round(count(tok[d], p, t)):,}".replace(",", "{,}") for d in TASKS)
            lines.append(rf"$S_{{{p}}}T_\text{{{t}}}$ & {vals} \\")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text(), end="")


if __name__ == "__main__":
    main()
