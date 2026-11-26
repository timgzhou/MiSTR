"""Per-sample compute spread of trained MiSTR runs on the test split (appendix: efficiency).

MiSTR's selection adapts to each sample, so its cost per SITS varies. Each run is evaluated
as at test time (threshold refit on val, exp.mistr.train.val_threshold) over the WHOLE test
split, and the crops each sample receives are counted (one crop per (window, step) as in
exp.mistr.train.step: per season under crop_time=season). Per-sample GMACs follow from the
run's recorded cost as in exp.viz.round1_main.mistr_cost: crop-free pass + per-crop GMACs x
that sample's crops (the figure script does that conversion; here only counts are stored).

    python -m exp.bench.mistr_crop_spread [--runs <file>] [--skip_done] <run> ...
    -> results/<task>/crop_spread.csv (one row per run: mean / std / percentiles of crops per
       sample) + results/<task>/crop_spread/<run>.pt (the per-sample counts, test order)
"""
from exp.common import olmo_bootstrap  # noqa: F401
import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.bench.mistr_throughput import load_run  # noqa: E402
from exp.mistr.train import set_attn, val_threshold  # noqa: E402
from exp.common.runlog import _append  # noqa: E402
from mistr_paper.baselines import build_selector_for  # noqa: E402
from mistr_paper.core.data import gpu_loader  # noqa: E402
from mistr_paper.core.selection import CoarsePass  # noqa: E402

PCTS = (0, 5, 25, 50, 75, 95, 100)


def per_sample(plan, B: int, season, crop_time: str) -> torch.Tensor:
    """(B,) crops per sample of one batch's plan (a Windows or a list of them)."""
    plans = plan if isinstance(plan, list) else [plan]
    n = torch.zeros(B, dtype=torch.long, device=season.device)
    for p in plans:
        if crop_time == "season":
            p = p.per_season(season.to(p.b.device))
        if len(p):
            n += torch.bincount(p.b.long(), minlength=B).to(n.device)
    return n


@torch.no_grad()
def spread(name, bs):
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
    counts = []
    for b in gpu_loader(task, "test", bs, 8, False):
        s = sel(CoarsePass.from_batch(model, b))
        counts.append(per_sample(s.plan, b.target.shape[0], b.season, model.cfg.crop_time).cpu())
    c = torch.cat(counts)
    out = Path(f"results/{cfg.task}/crop_spread")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(c, out / f"{name}.pt")
    q = torch.quantile(c.float(), torch.tensor([p / 100 for p in PCTS]))
    row = {"run": name, "task": cfg.task, "samples": len(c), "mean": round(c.float().mean().item(), 2),
           "std": round(c.float().std().item(), 2), "zero_share": round((c == 0).float().mean().item(), 4),
           **{f"p{p}": round(v.item(), 1) for p, v in zip(PCTS, q)}}
    del model, sel
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs_file", "--runs", dest="runs_file", default="")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    runs = list(a.runs) + ([r for r in Path(a.runs_file).read_text().split() if r.startswith("mistr_")]
                           if a.runs_file else [])
    for r in runs:
        if a.skip_done and any(Path("results").glob(f"*/crop_spread/{r}.pt")):
            continue
        try:
            row = spread(r, a.batch)
        except Exception as e:          # (one bad run does not end the sweep)
            print(f"FAILED {r}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
        _append(Path(f"results/{row['task']}/crop_spread.csv"), row)


if __name__ == "__main__":
    main()
