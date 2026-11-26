"""The learning rate of a baseline's lr sweep with the best val metric, for the sweep-once
protocol (scripts/slurm/baselines.sh: one configuration per method and dataset is swept, every
other configuration of that method uses its best lr):

    python -m exp.baselines.best_lr --task pastis --run '^apt_pastis_S12Tsea-w6-keep0.5_warm_ep50_'
    -> prints e.g. 0.001 (exit 1 while fewer than --expect sweep runs have finished)
"""
import argparse
import sys

import pandas as pd

from exp.common.tasks import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True, help="regex on the run name (the swept configuration)")
    ap.add_argument("--expect", type=int, default=3, help="sweep runs that must have finished")
    ap.add_argument("--batch", type=int, default=0, help="only runs at this train.batch_size")
    a = ap.parse_args()
    d = pd.read_csv(f"results/{a.task}/mistr.csv", low_memory=False)
    d = d[d.run.astype(str).str.contains(a.run) & d["eval_percentage"].isna()
          if "eval_percentage" in d else d.run.astype(str).str.contains(a.run)]
    if a.batch:
        d = d[d["train.batch_size"] == a.batch]
    d = d.drop_duplicates("run", keep="last")
    if len(d) < a.expect:
        print(f"{len(d)}/{a.expect} sweep runs finished for {a.run}", file=sys.stderr)
        sys.exit(1)
    lower = not TASKS[a.task].meter().higher_is_better
    d = d.sort_values("val_primary", ascending=lower)
    for _, r in d.iterrows():
        print(f"  lr {r['train.lr']:g}: val {r.val_primary:.4f} test {r.test_primary:.4f}", file=sys.stderr)
    print(f"{d['train.lr'].iloc[0]:g}")


if __name__ == "__main__":
    main()
