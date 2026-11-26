"""Fig 03 (the paper's main result) drawn from round 1 only:
exp.viz.datasets_tradeoff's paper layout and style, its data functions swapped for ones reading
results/<task>/mistr.csv (fixed_* and mistr_* runs; no legacy rows mixed in).

    python -m exp.viz.round1_main          -> figures/round1_main.png (preview)
    python -m exp.viz.round1_main --paper  -> figures/paper/03_MiSTR-result.pdf

The paper's protocol for every dataset (decided): fixed + MiSTR from OlmoEarth, 100 ep
(BioMassters 20), MiSTR = label-oracle warm-up for the first half, then the selector.
Means over whatever seeds have landed.
"""
from pathlib import Path

import pandas as pd

from exp.viz import datasets_tradeoff as dt

TIME = {"season": "sea", "month": "mon"}
# tasks whose primary metric is a fraction (mIoU): the paper reports it in 0-100
PCT_TASKS = {"pastis", "pastis_clean", "planteur"}


def pct(d: pd.DataFrame, task: str) -> pd.DataFrame:
    """test_primary / val_primary x 100 on PCT_TASKS (the paper's 0-100 scale); RMSE tasks unchanged."""
    if task in PCT_TASKS:
        d = d.assign(**{c: d[c] * 100 for c in ("test_primary", "val_primary") if c in d})
    return d


def rows(task):
    p = Path(f"results/{task}/mistr.csv")
    if not p.exists():
        return pd.DataFrame(columns=["run"])
    d = pd.read_csv(p, low_memory=False)
    if "eval_percentage" in d:
        d = d[d.eval_percentage.isna()]
    if "model.backbone" in d:          # (the tiny round only: base runs are their own figure)
        d = d[d["model.backbone"].fillna("tiny") == "tiny"]
    return pct(with_test_cost(round1_only(d.drop_duplicates("run", keep="last")), task), task)


ROUND1 = (r"^(mistr_[a-z_]+_S\d+Tsea\+S\d+Tmon(-w\d+)?_q[\d.]+_woracle\d+_ep\d+"
          r"|fixed_[a-z_]+_S\d+T(sea|mon)_ep\d+)_seed\d+_[0-9a-f]+$")


def round1_only(d: pd.DataFrame) -> pd.DataFrame:
    """fixed_ / mistr_ rows of round 1 only: the plain configurations at the default lr. Out:
    other experiments sharing the csv (lr sweeps, _warm_ continuations, space-only S..Tsea crops,
    when-* selectors, ...). Baselines (apt_, tldr_, upsample_) pass through."""
    fm = d.run.str.match("(fixed|mistr)_")
    ok = d.run.str.match(ROUND1)
    if "train.lr" in d:
        ok &= d["train.lr"].fillna(1e-3) == 1e-3
    # MiSTR: the oracle warm-up is the first HALF of the epochs (50 + 50, BioMassters 10 + 10);
    # out: warm-up length pilots that share the name pattern (BioMassters woracle5_ep20)
    w = d.run.str.extract(r"_woracle(\d+)_ep(\d+)_").astype(float)
    ok &= ~d.run.str.startswith("mistr_") | (2 * w[0] == w[1])
    return d[~fm | ok]


def with_test_cost(d: pd.DataFrame, task: str) -> pd.DataFrame:
    """gmacs -> the cost over the whole test split where measured (train.py's gmacs_test, or
    exp.mistr.measure_test_cost's results/<task>/test_cost.csv); cost_on_test marks those rows
    (the rest keep the 4-val-batch trace, MiSTR's corrected by dt.test_cost)."""
    f = Path(f"results/{task}/test_cost.csv")
    t = d.gmacs_test if "gmacs_test" in d else pd.Series(float("nan"), index=d.index)
    if f.exists():
        m = pd.read_csv(f).drop_duplicates("run", keep="last").set_index("run").gmacs_test
        t = t.fillna(d.run.map(m))
    has = t.notna()
    return d.assign(gmacs_trace=d.gmacs, gmacs=t.where(has, d.gmacs), cost_on_test=has)


def mistr_cost(r, d) -> float:
    """A MiSTR run's cost on test: measured when it was (cost_on_test); else the crop-free pass
    (the fixed seasonal model at that patch: on test if measured) + the traced cost per crop x the
    test crops -- per crop from the TRACE's own crop-free pass, so val and test never mix."""
    if r.cost_on_test:
        return r.gmacs
    f = d[d.run.str.startswith("fixed_") & (d["model.coarse_time"] == "season")
          & (d["model.coarse_patch"] == r["model.coarse_patch"])]
    cc, tc = r.compute_crops_per_sample, r.test_crops_per_sample
    if not len(f) or not cc or pd.isna(cc) or pd.isna(tc):
        return r.gmacs
    return f.gmacs.mean() + (r.gmacs - f.gmacs_trace.mean()) / cc * tc


def fixed(task, d):
    """[(family, patch, GMACs, metric)], mean over seeds."""
    f = d[d.run.str.startswith("fixed_")]
    out = []
    for (t, ps), g in f.groupby(["model.coarse_time", "model.coarse_patch"]):
        # (no DROP_FIXED here: round 1's YieldSAT S4Tsea, 1.351 at ~34 GMACs, is level with the
        # monthly curve, not dominated -- kept)
        out.append((TIME[t], int(ps), g.gmacs.mean(), g.test_primary.mean()))
    return out


def mistr(d, task, ps, geo, base):
    """(GMACs at the test crop count, metric) per q_s of MiSTR at (season patch, crop geo)."""
    m = d[d.run.str.startswith("mistr_") & (d["model.coarse_patch"] == ps)]
    fp = m["model.fine_patch"].fillna(0).astype(int)
    w = m["model.window_px"].astype(int)
    g = m[pd.Series([(f"c{a}p{b}" if b else f"c{a}") == geo for a, b in zip(w, fp)],
                     index=m.index, dtype=bool)]
    pts = []
    for _, r in g.groupby(g.run.str.extract(r"_q([\d.]+)_")[0]):
        pts.append((sum(mistr_cost(x, d) for _, x in r.iterrows()) / len(r),
                    r.test_primary.mean()))
    return sorted(pts)


def ours(d, task, ps, keep, base=None):
    return mistr(d, task, ps, f"c{ps // 2}", base) if keep == "one" else []


def sched_pair(d, task, ps, geo, base=None):
    return mistr(d, task, ps, geo, base)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="-> figures/paper/03_MiSTR-result.pdf")
    a = ap.parse_args()
    dt.rows = rows
    dt.fixed, dt.ours, dt.sched_pair = fixed, ours, sched_pair
    dt.XT = "-xhr"     # (the paper's layout: one colour per resolution pair, no controls)
    dt.draw_paper("seg", out="figures/paper/03_MiSTR-result.pdf" if a.paper
                  else "figures/round1_main.png")


if __name__ == "__main__":
    main()
