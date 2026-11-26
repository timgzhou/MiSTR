"""Test-time check of select.input_mask (never pick a window without S2 input; YieldSAT: outside
the field polygon), no retraining: each trained MiSTR run evaluated on test with the mask off and
on, the window threshold refit on val each time (as at test time), both arms in ONE process
(bfloat16 differs across processes).

    python -m exp.mistr.eval_input_mask [--limit N] <run name> [...]
    python -m exp.mistr.eval_input_mask --task yieldsat_raw      # every round-1 MiSTR run, seed 0

-> results/<task>/input_mask_swap.csv (one row per run x arm: test metrics, test crops per
sample, GMACs over 8 test batches)
"""
from exp.common import olmo_bootstrap  # noqa: F401
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mistr.train import evaluate, measure_compute, set_attn, val_threshold  # noqa: E402
from mistr.baselines import build_model, build_selector_for  # noqa: E402
from mistr.core.config import load  # noqa: E402
from mistr.core.data import gpu_loader  # noqa: E402

ROUND1 = r"^mistr_{task}_S\d+Tsea\+S\d+Tmon(-w\d+)?_q[\d.]+_woracle\d+_ep\d+_seed0_[0-9a-f]+$"


def one(name, limit):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ck["config"], f)
    cfg = load(f.name)
    task = TASKS[cfg.task]
    out = Path(f"results/{cfg.task}/input_mask_swap.csv")
    if out.exists() and name in set(pd.read_csv(out).run):     # resumable
        print(f"{name}: done, skipped", flush=True)
        return
    model = build_model(task, cfg).cuda()
    bs = cfg.train.batch_size or task.batch_size[16]
    dl = {s: gpu_loader(task, s, bs, 8, False) for s in ("val", "test")}
    b0 = next(iter(dl["val"]))
    sel = build_selector_for(task, cfg, model, (b0.ms, b0.ss))
    model.load_state_dict(ck["model"])
    nn.ModuleList(sel.modules()).load_state_dict(ck["selector"])
    set_attn(model, cfg.model.attention)
    q0 = sel.cfg
    rows = []
    for mask in (False, True):
        sel.cfg = replace(q0, input_mask=mask)
        val_threshold(model, sel, dl["val"], limit)
        t, crops = evaluate(model, sel, task, dl["test"], limit)
        g = measure_compute(model, sel, dl["test"], 8)["gmacs"]
        rows.append({"run": name, "task": cfg.task, "input_mask": int(mask),
                     "percentage": q0.percentage, "crops_per_sample": round(crops, 2),
                     "gmacs_8batches": g, **{f"test_{k}": round(float(v), 5) for k, v in t.items()
                                             if torch.as_tensor(v).numel() == 1}})
        print(f"{name} | input_mask {int(mask)} | crops {crops:7.2f} | {g:6.2f} GMACs | "
              + " ".join(f"{k} {float(t[k]):.4f}" for k in ("primary", "rmse") if k in t), flush=True)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    del model, sel, dl
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task", default="", help="every round-1 MiSTR seed-0 checkpoint of this task")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    runs = list(a.runs)
    if a.task:
        pat = re.compile(ROUND1.format(task=re.escape(a.task)))
        runs += sorted(p.stem for p in Path("checkpoints").glob(f"mistr_{a.task}_*.pt")
                       if pat.match(p.stem))
    for n in runs:
        one(n, a.limit)
