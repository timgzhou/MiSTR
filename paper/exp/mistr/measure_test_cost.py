"""The inference cost of finished runs over the WHOLE test split (exp.mistr.train.test_cost), for
runs trained before train.py recorded it: results/<task>/test_cost.csv, one row per
run (gmacs_test, ...). The figures prefer these to the 4-val-batch trace. Resumable: runs already
in the file are skipped.

    python -m exp.mistr.measure_test_cost --task pastis [--kind fixed] [--match <regex>]
"""
from exp.common import olmo_bootstrap  # noqa: F401
import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mistr.train import set_attn, test_cost, val_threshold  # noqa: E402
from exp.common.runlog import _append  # noqa: E402
from mistr_paper.baselines import build_model, build_selector_for  # noqa: E402
from mistr_paper.core.config import load  # noqa: E402
from mistr_paper.core.data import gpu_loader  # noqa: E402


def measure(name: str) -> dict:
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ck["config"], f)
    cfg = load(f.name)
    if cfg.name() != name:
        raise ValueError(f"{name}: the checkpoint's config names itself {cfg.name()}")
    task = TASKS[cfg.task]
    model = build_model(task, cfg).cuda()
    bs = cfg.train.batch_size or task.batch_size[16]
    dl = {s: gpu_loader(task, s, bs, 8, False) for s in ("val", "test")}
    b0 = next(iter(dl["val"]))
    sel = build_selector_for(task, cfg, model, (b0.ms, b0.ss))
    model.load_state_dict(ck["model"])
    nn.ModuleList(sel.modules()).load_state_dict(ck["selector"])
    set_attn(model, cfg.model.attention)
    val_threshold(model, sel, dl["val"])        # (as at test time; a no-op without a threshold)
    out = test_cost(model, sel, dl["test"])
    del model, sel, dl
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kind", default="fixed", help="run-name prefix: fixed / mistr / ...")
    ap.add_argument("--match", default="", help="regex the run must also match")
    ap.add_argument("--runs", default="", help="a file of run names (instead of --kind/--match)")
    a = ap.parse_args()
    d = pd.read_csv(f"results/{a.task}/mistr.csv", low_memory=False)
    if "eval_percentage" in d:
        d = d[d.eval_percentage.isna()]
    runs = sorted(r for r in d.run.drop_duplicates() if r.startswith(a.kind + "_")
                  and (not a.match or pd.Series([r]).str.contains(a.match).iloc[0]))
    if a.runs:
        runs = [r for r in Path(a.runs).read_text().split() if r]
    out = Path(f"results/{a.task}/test_cost.csv")
    done = set(pd.read_csv(out).run) if out.exists() else set()
    for r in runs:
        if r in done:
            continue
        if not Path(f"checkpoints/{r}.pt").exists():
            print(f"{r}: no checkpoint", flush=True)
            continue
        c = measure(r)
        _append(out, {"run": r, **c})
        print(f"{r}: {c['gmacs_test']:.3f} GMACs on test", flush=True)


if __name__ == "__main__":
    main()
