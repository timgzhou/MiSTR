"""Released checkpoints and test samples, for the notebooks (notebooks/).

    task, cfg, model, selector = load_pretrained("checkpoints/<run>.pt")      # MiSTR / fixed / baseline
    batch = load_samples("demo/pastis_test_samples.npz")                       # exact model inputs
    out = predict(model, selector, batch)      # logits, windows picked, high-res tokens, GMACs

The selector state saved with a checkpoint is the one the paper's test numbers used (its
threshold table refit on the validation split at the end of training), so no data is needed
to reproduce a checkpoint's test predictions on the shipped samples.

    python -m exp.demo export --task pastis --ids 40304 ... --out demo/pastis_test_samples.npz
        writes the test samples (as the test-time loader composites them) from the full data
    python -m exp.demo export_geoid --ids EMSR417-1-5_q2 EMSR456-5-2_q0 --out demo/geoid_test_series.npz
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from exp.mistr.train import n_crops, set_attn  # noqa: E402
from mistr.baselines import build_model, build_selector_for  # noqa: E402
from mistr.core.config import load  # noqa: E402
from mistr.core.data_cpu import Batch  # noqa: E402
from mistr.core.selection import CoarsePass  # noqa: E402


def load_pretrained(path, device="cuda"):
    """-> (task, cfg, model, selector), in eval mode, from a checkpoint alone (no data)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load(ck["config"])
    task = TASKS[cfg.task]
    model = build_model(task, cfg, pretrained=False).to(device)
    sel = build_selector_for(task, cfg, model, None)
    model.load_state_dict(ck["model"])
    nn.ModuleList(sel.modules()).load_state_dict(ck["selector"])
    set_attn(model, cfg.model.attention)
    model.eval()
    for m in sel.modules():
        m.eval()
    return task, cfg, model, sel


def _sample_to(s, fn):
    return s._replace(**{k: fn(v) for k, v in s.as_dict().items() if torch.is_tensor(v)})


# bfloat16 has no numpy dtype: stored as its int16 bit pattern; masks (a few values) as uint8.
def _pack(t: torch.Tensor):
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        u = t.unique()
        if len(u) <= 255 and (u == u.round()).all() and u.min() >= 0:
            return t.to(torch.uint8).numpy(), "bf16_small"
        return t.view(torch.int16).numpy(), "bf16"
    return t.numpy(), str(t.dtype)


def _unpack(a, kind) -> torch.Tensor:
    t = torch.from_numpy(a)
    return {"bf16_small": lambda: t.to(torch.bfloat16), "bf16": lambda: t.view(torch.bfloat16)}.get(kind, lambda: t)()


def save_samples(batch: Batch, ids: list[str], path):
    """Exact, compressed (.npz): the batch's tensors and the sample ids."""
    arrays, kinds = {}, {}
    for part in ("ms", "ss"):
        for k, v in getattr(batch, part).as_dict().items():
            if torch.is_tensor(v):
                arrays[f"{part}.{k}"], kinds[f"{part}.{k}"] = _pack(v)
    for k in ("season", "present", "target", "frac"):
        arrays[k], kinds[k] = _pack(getattr(batch, k))
    import numpy as np
    np.savez_compressed(path, ids=np.array(ids), kinds=np.array(json.dumps(kinds)), **arrays)


def load_samples(path, device="cuda") -> Batch:
    """The saved test samples as the model's input batch (+ .ids, the dataset's sample ids)."""
    import numpy as np
    z = np.load(path)
    kinds = json.loads(str(z["kinds"]))
    t = {k: _unpack(z[k], kinds[k]).to(device) for k in kinds}
    part = lambda p: MaskedOlmoEarthSample(**{k[len(p) + 1:]: v for k, v in t.items() if k.startswith(p + ".")})
    b = Batch(part("ms"), part("ss"), *(t[k] for k in ("season", "present", "target", "frac")))
    b.ids = list(z["ids"])
    return b


@torch.no_grad()
def predict(model, selector, batch) -> dict:
    """-> logits (B, C, H, W), the selection plan (Windows: sample, month, top-left px, size),
    high-res tokens per sample, and the coarse pass's own logits. One coarse pass, reused by the
    second pass as at test time (exp.mistr.train.step)."""
    ctx = CoarsePass.from_batch(model, batch)
    sel = selector(ctx)
    plan = sel.plan
    if ctx.coarse_tokens is not None and n_crops(plan) == 0:
        logits = ctx.out0.prediction
    else:
        logits = model(batch.ms, batch.ss, batch.season, batch.present, plan, sel.logw, batch.dates,
                       ctx.coarse_tokens, ctx.coarse_cache).prediction
    if model.cfg.crop_time == "season":
        plan = ([p.per_season(batch.season.to(p.b.device)) for p in plan] if isinstance(plan, list)
                else plan.per_season(batch.season.to(plan.b.device)))
    B = batch.target.shape[0]
    per = torch.zeros(B, dtype=torch.long)
    for p in (plan if isinstance(plan, list) else [plan]):
        if len(p):
            per += torch.bincount(p.b.cpu(), minlength=B)
    return {"logits": logits.float(), "coarse_logits": ctx.out0.prediction.float(), "plan": plan,
            "windows_per_sample": per, "crops": n_crops(plan)}


def export(task_name, ids, out, split="test"):
    """Composite the given test samples exactly as the test-time loader does, and save them."""
    from torch.utils.data import DataLoader, Subset

    from mistr.core.data import CompositeOnGPU, RawCollate, RawPaired
    task = TASKS[task_name]
    ds = RawPaired(task, split)
    base = ds.ds if hasattr(ds, "ds") else ds
    index = {base.sample_id(i): i for i in range(len(base))}
    missing = [i for i in ids if i not in index]
    if missing:
        raise SystemExit(f"not in {task_name} {split}: {missing}")
    dl = DataLoader(Subset(ds, [index[i] for i in ids]), batch_size=len(ids), collate_fn=RawCollate(task, ds.group))
    batch = CompositeOnGPU(task, ds.group, ds.parts)(next(iter(dl)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_samples(batch, ids, out)
    print(f"wrote {out}: {len(ids)} samples")


# --- GEOID-Flood flood tracking (exp.geoidchange.tiletrack; not RunConfig runs) -------------------
GEOID_ROOT = "data/GEOID-Flood-tileseries-long2/cache"


def load_geoid(path, lr_ps, crop=32, device="cuda"):
    """An EventTrack checkpoint (bare state dict) with the paper's round-O settings: no reference
    pass (head on f_t), causal, crop budget = a share of all (T - 1) x W^2 high-res slots. Fixed
    models: any crop size (it is unused at q = 0)."""
    from exp.geoidchange.tiletrack import EventTrack
    net = EventTrack(lr_ps, crop or 32, "self", 3).to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device))
    net.ref_n, net.causal, net.q_unit, net.budget_scope = 0, True, "series", "sample"
    return net


def export_geoid(recs, out, k_post=16):
    """Test series (dataset items, event at pass k_post of the T = 32 window) -> one .npz."""
    import numpy as np

    from exp.geoidchange.tiletrack import LongTrack
    LongTrack.root = GEOID_ROOT
    ds = LongTrack("test", 32)
    ds.k_post = k_post
    names = [f"{r['key']}_q{r['quad']}" for r in ds.recs]
    arrays = {}
    for n, rec in enumerate(recs):
        for name, a in zip(("x", "days", "valid", "k", "y", "rday", "yt"), ds[names.index(rec)]):
            arrays[f"{n}.{name}"] = np.asarray(a)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, recs=np.array(recs), **arrays)
    print(f"wrote {out}: {len(recs)} series")


def load_geoid_samples(path):
    """-> (names, [dataset items]) as exp.geoidchange.tiletrack.LongTrack returns them."""
    import numpy as np
    z = np.load(path)
    names = list(z["recs"])
    items = [tuple(z[f"{n}.{k}"] if z[f"{n}.{k}"].ndim else z[f"{n}.{k}"].item()
                   for k in ("x", "days", "valid", "k", "y", "rday", "yt")) for n in range(len(names))]
    return names, items


@torch.no_grad()
def predict_geoid(net, item, q):
    """One series -> per-pass class map (T, H, W) (0 dry, 1 permanent water, 2 flood), the windows
    re-encoded at high resolution (pass, y, x), and the tokens the model encoded."""
    from exp.geoidchange.tiletrack import collate
    s, d, v, post, y, _, rd, _ = collate([item])
    o, _, info = net(s, d, v, post, "joint", q)
    boxes = {k: info[k].cpu().numpy() for k in ("sel_t", "win_y", "win_x")} if q > 0 else None
    crops = info.get("crops", 0) if q > 0 else 0
    return {"pred": o.argmax(2)[0].cpu().numpy(), "event": int(post[0]), "label": y[0].numpy(),
            "days": rd[0].numpy(), "valid": v[0].numpy(), "windows": boxes,
            "tokens": int(v[0].sum()) * (512 // net.lr_ps) ** 2 + int(crops)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "export_geoid"))
    ap.add_argument("--task", default="pastis")
    ap.add_argument("--ids", nargs="+", required=True, help="sample ids (export) / series names (export_geoid)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "export_geoid":
        export_geoid(a.ids, a.out)
    else:
        export(a.task, a.ids, a.out, a.split)


if __name__ == "__main__":
    main()
