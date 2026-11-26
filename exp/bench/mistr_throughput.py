"""Inference throughput (samples/s) of trained runs -- MiSTR, fixed, the baselines -- on the test
split, data already on the GPU (the model's speed, not the loader's), batch size fixed for every
run, with a per-stage breakdown (each stage timed with CUDA events after a synchronize):

    coarse    MiSTR.uniform: tokenize the coarse tokens + encoder (+ the trace) + readout
    select    the selector on the coarse pass (uncertainty, attention time, the threshold)
    crops     MiSTR.tokenize_crops: crop + embed the fine windows
    encode    the fine tokens through the blocks (one-way: reading the cached coarse trace)
    decode    pooling + the readout with the crops

    python -m exp.bench.mistr_throughput --batch 32 --batches 20 <run> [<run> ...]
      [--runs <file of run names>] [--skip_done]   (resumable: runs already in the csv skipped)
      [--profile]   also print torch.profiler's top CUDA / CPU ops of one timed pass
    -> results/<task>/throughput.csv (one row per run)
"""
from exp.common import olmo_bootstrap  # noqa: F401
import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mistr.train import set_attn, step, val_threshold  # noqa: E402
from exp.common.runlog import _append  # noqa: E402
from mistr.encoders.olmoearth import _to_device  # noqa: E402
from mistr.baselines import build_model, build_selector_for  # noqa: E402
from mistr.core.config import load  # noqa: E402
from mistr.core.data import gpu_loader  # noqa: E402
from mistr.core.data_cpu import Batch  # noqa: E402
from mistr.core.model import MiSTR  # noqa: E402

STAGES = defaultdict(float)
TIMING = [False]


@contextmanager
def stage(name):
    if not TIMING[0]:
        yield
        return
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        STAGES[name] += time.perf_counter() - t


def instrument():
    """Wrap the stage functions (class level: every model / selector instance)."""
    def wrap(cls, fn, name):
        orig = getattr(cls, fn)

        def w(*a, **k):
            with stage(name):
                return orig(*a, **k)
        setattr(cls, fn, w)
    wrap(MiSTR, "uniform", "coarse")
    wrap(MiSTR, "tokenize_crops", "crops")
    wrap(MiSTR, "encode", "encode")
    wrap(MiSTR, "decode", "decode")


def on_gpu(b: Batch) -> Batch:
    return Batch(_to_device(b.ms, "cuda", torch.bfloat16), _to_device(b.ss, "cuda", torch.bfloat16),
                 b.season.cuda(), b.present.cuda(), b.target.cuda(), b.frac.cuda(), None)


def load_run(name):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ck["config"], f)
    cfg = load(f.name)
    task = TASKS[cfg.task]
    model = build_model(task, cfg).cuda().eval()
    return ck, cfg, task, model


@torch.no_grad()
def bench(name, bs, n_batches, reps, profile=False):
    ck, cfg, task, model = load_run(name)
    dl_val = gpu_loader(task, "val", bs, 8, False)
    b0 = next(iter(dl_val))
    sel = build_selector_for(task, cfg, model, (b0.ms, b0.ss))
    model.load_state_dict(ck["model"])
    nn.ModuleList(sel.modules()).load_state_dict(ck["selector"])
    for m in sel.modules():
        m.eval()
    set_attn(model, cfg.model.attention)
    val_threshold(model, sel, dl_val)
    batches = []
    for i, b in enumerate(gpu_loader(task, "test", bs, 8, False)):
        if i >= n_batches:
            break
        if b.target.shape[0] == bs:
            batches.append(on_gpu(b))
    n = sum(b.target.shape[0] for b in batches)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    held = torch.cuda.memory_allocated()     # weights + the cached test batches
    gm = {}

    def run():
        crops = 0
        for b in batches:
            crops += step(model, sel, b)[2]
        return crops
    run()                                    # warm-up (autotune, caches)
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        crops = run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    best = min(times)
    STAGES.clear()
    TIMING[0] = True
    t = time.perf_counter()
    run()
    torch.cuda.synchronize()
    staged = time.perf_counter() - t
    TIMING[0] = False
    st = dict(STAGES)
    # select = what step spends outside the model's stages (selector + glue)
    st["select+glue"] = staged - sum(st.values())
    if profile:
        from torch.profiler import ProfilerActivity, profile as prof
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            run()
            torch.cuda.synchronize()
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    mem = torch.cuda.max_memory_allocated() / 2**30
    row = {"run": name, "task": cfg.task, "batch": bs, "samples": n, "samples_s": round(n / best, 1),
           "ms_per_batch": round(1000 * best / len(batches), 2),
           "crops_per_sample": round(crops / n, 1), "peak_mem_gb": round(mem, 2),
           "peak_mem_run_gb": round(mem - held / 2**30, 2),     # above the weights + data held
           **{f"ms_{k}": round(1000 * v / len(batches), 2) for k, v in st.items()},
           "gpu": torch.cuda.get_device_name(), "fast": __import__("mistr.core").core.FAST, **gm}
    del model, sel, batches
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs_file", "--runs", dest="runs_file", default="")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no_save", action="store_true")
    a = ap.parse_args()
    instrument()
    runs = list(a.runs) + ([r for r in Path(a.runs_file).read_text().split() if r]
                           if a.runs_file else [])
    done = set()
    if a.skip_done:
        import pandas as pd
        for f in Path("results").glob("*/throughput.csv"):
            d = pd.read_csv(f)
            done |= set(d[d.batch == a.batch].run)
    for r in runs:
        if r in done:
            continue
        try:
            row = bench(r, a.batch, a.batches, a.reps, a.profile)
        except Exception as e:          # (one bad run does not end the sweep)
            print(f"FAILED {r}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            continue
        print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        if not a.no_save:
            _append(Path(f"results/{row['task']}/throughput.csv"), row)

if __name__ == "__main__":
    main()
