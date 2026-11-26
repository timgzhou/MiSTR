"""Every training run behind the paper, as code: `runs()` -> [RunConfig], parents (warm starts) first.

Each config's run name ends in a hash of every field set away from its default, and every results row
(results/<task>/mistr.csv) is keyed by that name, so `check` proves the grids below are exactly the
runs the paper's numbers come from:

    python -m exp.mistr.paper_configs check        # every generated name has its results row, and back
    python -m exp.mistr.paper_configs list         # the run names

Training: python -m exp.mistr.paper_runs (list / show / train / slurm), which reads runs() from here.

Run names: `fixed_` fixed-resolution models (`S<patch>T<sea|mon>`); `mistr_` MiSTR
(`S<coarse>Tsea+S<fine>Tmon_q<budget %>_woracle<warm-up epochs>`; "woracle" = the paper's guided
warm-up); `apt_`, `tldr_`, `upsample_` the compared methods (`_warm`: warm-started from
`train.init_from`); `_terramind_tiny` / `_base`: TerraMind 1.0 Tiny / OlmoEarth Base backbones;
`_year` tasks: one year-long composite (App. A). "Round 1" in the figure scripts (round1_main,
baselines_r1, selector_tables_r1, ...) is this main protocol: OlmoEarth Tiny initialization, guided
warm-up for the first half of training, 100 epochs (BioMassters 20). GEOID-Flood tracking runs are
not RunConfigs: python -m exp.paper_evals --part geoid.
"""
import math
import sys
from dataclasses import asdict

from mistr.core.config import RunConfig, from_dict

TASKS = ("pastis", "planteur", "biomassters", "yieldsat_raw")
EPOCHS = {"pastis": 100, "planteur": 100, "biomassters": 20, "yieldsat_raw": 100}
SEEDS = (0, 1, 2)
QS = (0.125, 0.25, 0.5)                       # MiSTR budget: share of the candidate high-res points
# MiSTR pairs (main text): coarse seasonal patch -> (window px, fine patch; 0 = the window itself)
PAIRS = {16: (8, 0), 14: (14, 7), 12: (12, 6), 10: (10, 5), 8: (4, 0), 6: (6, 3)}
TM_PAIRS = (16, 12, 8, 6, 4)                 # TerraMind: the same pairs + S4 & S2
FIXED_SEA, FIXED_MON = (16, 14, 12, 10, 8, 6, 4), (16, 14, 12, 10, 8, 6)
BASELINE_LR = 2.5                             # baselines: train.max_hours (chained 2.5 h jobs)


def cfg(task, **kv) -> RunConfig:
    """A RunConfig from dotted overrides, e.g. cfg("pastis", **{"model.coarse_patch": 8})."""
    d = asdict(RunConfig())
    d["task"] = task
    for k, v in kv.items():
        node = d
        *path, leaf = k.split(".")
        for p in path:
            if node.get(p) is None:
                node[p] = {}
            node = node[p]
        node[leaf] = v
    return from_dict(RunConfig, d)


def fixed(task, p, time="season", seed=0, backbone="tiny", **kv):
    ep = EPOCHS[task.replace("_year", "")]
    return cfg(task, fixed=True, **{"model.coarse_patch": p, "model.coarse_time": time, "model.backbone": backbone,
                                    "train.epochs": ep, "train.seed": seed, **kv})


def windows(q, w, tile=64):
    """High-res candidate points of one tile at window w, times the budget (the guided warm-up's count)."""
    return int(q * math.ceil(tile / w) ** 2)


def mistr(task, cp, q, seed=0, crop_time="month", coarse_time="season", backbone="tiny", warmup=True, **kv):
    w, fp = PAIRS.get(cp, (cp // 2, 0)) if cp not in (4,) else (4, 2)
    ep = EPOCHS[task]
    d = {"model.coarse_patch": cp, "model.window_px": w, "model.fine_patch": fp, "model.crop_time": crop_time,
         "model.coarse_time": coarse_time, "model.backbone": backbone, "select.percentage": q,
         "train.epochs": ep, "train.seed": seed}
    if cp == 6 and q == 0.5 and crop_time == "month" and backbone == "tiny":
        d["train.batch_size"] = 16                 # S6+S3 at q 50 %: batch 16 (GPU memory)
    if warmup:
        d.update({"train.warmup": "oracle", "train.warmup_epochs": ep // 2, "train.warmup_windows": windows(q, w)})
    d.update(kv)
    return cfg(task, **d)


def runs() -> list[RunConfig]:
    out = []
    add = out.append
    for t in TASKS:
        for s in SEEDS:
            for p in FIXED_SEA:
                add(fixed(t, p, "season", s))
            for p in FIXED_MON:
                add(fixed(t, p, "month", s))
            add(fixed(t + "_year", 16, "season", s))
            for cp in PAIRS:
                for q in QS:
                    add(mistr(t, cp, q, s))
    # TerraMind 1.0 Tiny backbone (App. H): fixed seasonal (+ monthly on PASTIS), MiSTR with seasonal fine
    # tokens (space only); PASTIS also monthly coarse + fine, seasonal coarse + monthly fine, and the context
    # variants of the second pass
    tm = "terramind_tiny"
    for t in TASKS:
        for s in SEEDS:
            for p in (16, 12, 8, 6, 4, 2) if t == "pastis" else (16, 8, 6, 4, 2):
                add(fixed(t, p, "season", s, tm))
            for cp in TM_PAIRS:
                for q in QS if t == "pastis" else QS[1:]:
                    add(mistr(t, cp, q, s, crop_time="season", backbone=tm))
    for p in (16, 12, 8, 6, 4, 2):
        add(fixed("pastis", p, "month", 0, tm, **({"train.batch_size": 16} if p == 2 else {})))
    for time in ("season", "month"):
        add(fixed("pastis", 8, time, 0, "terramind_tiny_joint"))
    for cp in TM_PAIRS:
        for q in QS:
            add(mistr("pastis", cp, q, coarse_time="month", backbone=tm))
    for cp in (16, 8):
        for q in QS:
            add(mistr("pastis", cp, q, backbone=tm))
        for bb in ("terramind_tiny_ctxall", "terramind_tiny_ctxnb"):
            for q in (0.25, 0.5) if cp == 16 else (0.25,):
                add(mistr("pastis", cp, q, backbone=bb))
    add(mistr("pastis", 16, 0.25, backbone="terramind_tiny_joint"))

    # --- PASTIS ablations on S16 & S8 and S8 & S4 (App. B, G): a 50-epoch guided warm-up model per pair,
    # budget and seed, then 50 epochs of each variant from it
    for cp in (16, 8):
        for s in SEEDS:
            add(mistr("pastis", cp, 0.25, s, warmup=False))                  # no warm-up (App. G)
            for q in QS:
                stage1 = mistr("pastis", cp, q, s, **{"train.epochs": 50})
                add(stage1)
                cont = {"train.epochs": 50, "train.init_from": stage1.name()}
                add(mistr("pastis", cp, q, s, warmup=False, **cont))
                for k, v in (("model.attention", "joint"), ("model.attention", "fine_reads_coarse"),
                             ("select.where", "grid"), ("select.where", "random")):
                    add(mistr("pastis", cp, q, s, warmup=False, **cont, **{k: v}))
                if q == 0.25:
                    for when in ("entropy", "grid", "random"):
                        add(mistr("pastis", cp, q, s, warmup=False, **cont, **{"select.when": when}))
        for q in QS:                                                         # space only (seasonal fine tokens)
            add(mistr("pastis", cp, q, crop_time="season"))

    # flexible time budget (App. C): one model for 0 / 0.5 / 1 extra seasons, picked by disagreement
    for cp in PAIRS:
        for q in QS:
            for s in SEEDS:
                add(mistr("pastis", cp, q, s, **{"select.extra_steps_choices": "0,0.5,1",
                                                 "select.extra_pick": "disagree",
                                                 "train.eval_percentages": "q:0.5,q:1"}))

    # warm-up attention (App. G): one-way during the guided warm-up, or selector-driven warm-up
    for cp in (16, 12):
        add(mistr("pastis", cp, 0.25, **{"train.warmup_attention": "one_way"}))
        sel = mistr("pastis", cp, 0.25, warmup=False, **{"train.epochs": 50, "train.warmup": "select",
                                                         "train.warmup_epochs": 50,
                                                         "train.warmup_attention": "one_way"})
        add(sel)
        for q in QS:
            add(mistr("pastis", cp, q, warmup=False, **{"train.epochs": 50, "train.init_from": sel.name()}))

    # 50 more epochs at lr 3e-3 from round-1 models (training-matched controls, App. B / D)
    lr3 = {"train.epochs": 50, "train.lr": 3e-3, "train.max_hours": BASELINE_LR}
    r1 = {q: mistr("pastis", 8, q).name() for q in QS}
    for q, src in ((0.125, 0.25), (0.25, 0.25), (0.5, 0.5)):
        add(mistr("pastis", 8, q, warmup=False, **lr3, **{"train.init_from": r1[src]}))
    for q in (0.25, 0.5):
        add(mistr("pastis", 6, q, warmup=False, **lr3, **{"train.init_from": mistr("pastis", 6, q).name()}))
    add(fixed("pastis", 8, "month", **lr3, **{"train.init_from": fixed("pastis", 8, "month").name()}))
    add(mistr("pastis", 6, 0.5, **{"model.coarse_head": False}))          # head at 16 px (App. A)

    # OlmoEarth Base (App. E / H): PASTIS, seed 0
    base = "base"
    for p in (16, 12, 10, 8, 6, 4):
        add(fixed("pastis", p, "season", 0, base, **({"train.batch_size": 16} if p == 4 else {})))
    for p in (16, 12, 10, 8, 6):
        add(fixed("pastis", p, "month", 0, base, **({"train.batch_size": 16} if p == 6 else {})))
    for cp in (16, 12, 10):
        for q in QS:
            add(mistr("pastis", cp, q, backbone=base))
    w10 = {"train.warmup_epochs": 10, "train.batch_size": 16}               # finer pairs: 10-epoch warm-up
    for cp in (8, 6):
        for q in QS[:2]:
            add(mistr("pastis", cp, q, backbone=base, **w10))
    for q in QS[:2]:
        add(mistr("pastis", 6, q, backbone=base, **w10, **{"select.extra_steps": 1.0, "select.extra_pick": "disagree"}))
        add(mistr("pastis", 4, q, backbone=base, **{**w10, **({"train.batch_size": 8} if q == 0.25 else {})}))

    out += baselines()
    return out


# --- compared methods (Sec. 4, App. D), warm-started from the seed-0 fixed model at the same patch size
# (fine-tuned: 50 epochs, BioMassters 10). Learning rate: swept on S12 (SWEEP), the picked one (LR) for all.
BL_EPOCHS = {"pastis": 50, "biomassters": 10, "yieldsat_raw": 50}
BL_BS = {"pastis": 8, "biomassters": 64, "yieldsat_raw": 8}
LR = {("apt", "pastis"): 3e-3, ("apt", "biomassters"): 3e-4, ("apt", "yieldsat_raw"): 3e-3,
      ("tldr", "pastis"): 3e-3, ("tldr", "biomassters"): 1e-3,
      ("anyup", "pastis"): 3e-3, ("anyup", "biomassters"): 3e-3, ("anyup", "yieldsat_raw"): 3e-3}
SWEEP = {  # (method, task) -> [(lr, batch size)] on top of the picked one, at S12 (+ default settings)
    ("apt", "pastis"): [(3e-4, 8), (1e-3, 8), (1e-2, 8)],
    ("apt", "biomassters"): [(1e-4, 64), (1e-4, 8), (3e-4, 8), (1e-3, 64), (1e-3, 8), (3e-3, 64), (3e-3, 8),
                             (1e-2, 64)],
    ("tldr", "pastis"): [(3e-4, 8), (1e-3, 8), (1e-2, 8)],
    ("tldr", "biomassters"): [(3e-4, 64), (3e-4, 8), (1e-3, 8), (3e-3, 64), (3e-3, 8), (1e-2, 64)],
    ("anyup", "pastis"): [(3e-4, 8), (1e-3, 8), (1e-2, 8)],
    ("anyup", "biomassters"): [(1e-4, 8), (3e-4, 64), (3e-4, 8), (1e-3, 64), (1e-3, 8), (3e-3, 8), (1e-2, 64)],
}
TLDR_R = ("0.1", "0.21", "0.4,0.1", "0.25,0.085,0.043", "0.19,0.1,0.085", "0.1,0.065,0.065,0.065")
LP_LRS = (1e-4, 1e-3, 1e-2)                   # the bilinear probe (head only): every lr at every patch size
LP_PICK = {"pastis": {16: 1e-2, 12: 1e-2, 8: 1e-3}, "biomassters": {16: 1e-3, 12: 1e-2, 8: 1e-3},
           "yieldsat_raw": {16: 1e-2, 12: 1e-2, 8: 1e-2}}


def baseline(task, method, p, time, init, **kv):
    return cfg(task, **{"model.coarse_patch": p, "model.attention": "joint", "baseline.method": method,
                        "baseline.tokens": {"season": "seasons", "month": "months"}[time],
                        "train.init_from": init, "train.max_hours": BASELINE_LR, **kv})


def baselines() -> list[RunConfig]:
    out = []
    add = out.append
    for t in ("pastis", "biomassters", "yieldsat_raw"):
        ep, bs = BL_EPOCHS[t], BL_BS[t]
        ft = lambda m, lr=None, b=None: {"train.epochs": ep, "train.lr": lr or LR[(m, t)], "train.batch_size": b or bs}
        # APT: S16 / S12 tokens (seasonal or monthly), keep 25 / 50 / 75 %, entropy or Laplacian scorer
        for p in (16, 12):
            for time in ("season", "month"):
                init = fixed(t, p, time).name()
                for keep in (0.25, 0.5, 0.75):
                    for sc in ("entropy",) if t == "yieldsat_raw" else ("entropy", "laplacian"):
                        a = {"model.window_px": p // 2, "baseline.keep": keep, "baseline.scorer": sc,
                             "train.compute_split": "test", "train.compute_batches": 16}
                        add(baseline(t, "apt", p, time, init, **a, **ft("apt")))
                        if p == 12 and time == "season" and keep == 0.5 and sc == "entropy":
                            for lr, b in SWEEP.get(("apt", t), []):
                                add(baseline(t, "apt", p, time, init, **a, **ft("apt", lr, b)))
        for p in (16, 12, 8):
            init_m, init_s = fixed(t, p, "month").name(), fixed(t, p, "season").name()
            # vid-TLDR: token merging on monthly tokens, training-free (frozen) or fine-tuned
            for r in TLDR_R:
                tr = {"baseline.r": r}
                add(baseline(t, "tldr", p, "month", init_m, **tr, **{"train.epochs": 0, "train.freeze_model": True,
                                                                    "train.batch_size": 8}))
                if t != "yieldsat_raw":
                    add(baseline(t, "tldr", p, "month", init_m, **tr, **ft("tldr")))
                    if p == 12 and r == "0.25,0.085,0.043":
                        for lr, b in SWEEP[("tldr", t)]:
                            add(baseline(t, "tldr", p, "month", init_m, **tr, **ft("tldr", lr, b)))
            # AnyUp, fine-tuned (seasonal tokens)
            add(baseline(t, "upsample", p, "season", init_s, **ft("anyup")))
            if p == 12:
                for lr, b in SWEEP.get(("anyup", t), []):
                    add(baseline(t, "upsample", p, "season", init_s, **ft("anyup", lr, b)))
            # bilinear upsampling + linear probe, then Upsample Anything (test-time) on its features
            lp = {"baseline.upsampler": "bilinear", "train.head_only": True, "train.epochs": ep, "train.batch_size": 8}
            for lr in LP_LRS if t != "yieldsat_raw" else (1e-2,):
                add(baseline(t, "upsample", p, "season", init_s, **lp, **{"train.lr": lr}))
            lr = LP_PICK[t][p]
            probe = baseline(t, "upsample", p, "season", init_s, **lp, **{"train.lr": lr}).name()
            frozen = {"baseline.upsampler": "upa", "train.epochs": 0, "train.freeze_model": True}
            add(baseline(t, "upsample", p, "season", probe, **frozen, **{"train.lr": lr, "train.batch_size": 8}))
            if t == "biomassters":
                add(baseline(t, "upsample", p, "season", probe, **frozen, **{"train.lr": lr, "train.batch_size": 8,
                                                                            "baseline.space": "predictions"}))
            else:
                add(baseline(t, "upsample", p, "season", init_s, **frozen, **{"train.batch_size": 32,
                                                                             "baseline.space": "predictions"}))
    return out


def by_name() -> dict[str, RunConfig]:
    return {c.name(): c for c in runs()}


def check():
    import csv
    from pathlib import Path
    want = {}
    for f in Path("results").glob("*/mistr.csv"):
        for r in csv.DictReader(open(f)):
            want[r["run"]] = f.parent.name
    have = [c.name() for c in runs()]
    dup = len(have) - len(set(have))
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    print(f"{len(set(have))} generated, {len(want)} in results: {len(missing)} not generated, "
          f"{len(extra)} generated without a results row, {dup} duplicates")
    return missing, extra


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "list":
        for c in runs():
            print(c.name())
    else:
        m, e = check()
        sys.exit(1 if m or e else 0)
