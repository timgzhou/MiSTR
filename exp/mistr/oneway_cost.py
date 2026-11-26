"""Inference cost of one-way vs full (joint) self-attention in MiSTR's second pass, same model and
same selections (the attention mode only changes who reads whom; the weights are shared).

    python -u -m exp.mistr.oneway_cost [--batches N] <run name>

Per test sample (exp.bench.flop_trace, everything run at test time): the coarse pass alone, and
coarse pass + second pass with attention
    one_way            fine tokens read the cached coarse tokens and each other (MiSTR)
    fine_reads_coarse  fine tokens read only the cached coarse tokens
    joint              one sequence of coarse + fine tokens, re-encoded from scratch
-> results/<task>/oneway_cost.csv
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exp.mistr.mech_probe import load_run  # noqa: E402
from exp.mistr.train import measure_compute, set_attn, val_threshold  # noqa: E402
from mistr.core.selection import CoarsePass  # noqa: E402


@torch.no_grad()
def coarse_only(model, dl, batches):
    from exp.bench.flop_trace import count_macs
    n = 0
    with count_macs() as count:
        for i, b in enumerate(dl):
            if i >= batches:
                break
            CoarsePass.from_batch(model, b).out0
            n += b.target.shape[0]
    return {"gmacs": round(count.total / n / 1e9, 3),
            "gmacs_attention": round(count.attn / n / 1e9, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("run")
    a = ap.parse_args()
    cfg, task, model, sel, dl = load_run(a.run, 8)
    val_threshold(model, sel, dl["val"])
    rows = [{"run": a.run, "attention": "coarse_pass_only",
             **coarse_only(model, dl["test"], a.batches)}]
    for att in ("one_way", "fine_reads_coarse", "joint"):
        set_attn(model, att)
        c = measure_compute(model, sel, dl["test"], a.batches)
        rows.append({"run": a.run, "attention": att, **c})
    set_attn(model, cfg.model.attention)
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    out = Path(f"results/{cfg.task}/oneway_cost.csv")
    d.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
