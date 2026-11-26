"""Every command behind the paper's results files except training the RunConfig runs
(exp.mistr.paper_runs), derived from the shipped results themselves, in pipeline order.

    python -m exp.paper_evals [--part data|evals|tessera|geoid] > steps.sh

data     MiSTR composite caches (optional: training reads raw data without them, slower)
evals    on trained checkpoints: test cost, attention probes, throughput, crop spread, the
         one-way attention cost, APT calibration (needed BEFORE training the APT runs)
tessera  frozen Tessera v2: embeddings, pool tuning, heads (App. H)
geoid    GEOID-Flood tracking (Sec. 5, App. K): series fetch + cache, training, rescoring, costs
Each line is one independent command (run them in order within a part; GPU for all but data).
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

R = Path("results")
TASKS = ("pastis", "planteur", "biomassters", "yieldsat_raw")


def col(path, name):
    if not path.exists():
        return []
    return [r[name] for r in csv.DictReader(open(path)) if r.get(name)]


def uniq(xs):
    return list(dict.fromkeys(xs))


def data():
    print("# --- data: MiSTR composite caches (docs/DATA.md for downloads and tile stores)")
    print(f"python -m exp.mistr.build_cache --tasks {' '.join(TASKS)}")


def evals():
    print("# --- APT calibration (results/<task>/apt.json, read when an APT run is built)")
    for t in TASKS:
        if any(r.startswith("apt_") for r in col(R / t / "mistr.csv", "run")):
            print(f"python -u -m exp.baselines.calibrate_apt --task {t}")
    print("# --- GMACs over the whole test split, runs trained before train.py measured it")
    for t in TASKS:
        runs = uniq(col(R / t / "test_cost.csv", "run"))
        if runs:
            print(f"python -m exp.mistr.measure_test_cost --task {t} --match '^({'|'.join(map(re.escape, runs))})$'")
    print("# --- attention groups / masking / decoder probes (App. G)")
    for t in TASKS:
        runs = uniq(col(R / t / "mech_attn.csv", "run"))
        if runs:
            print("python -u -m exp.mistr.mech_probe " + " ".join(runs))
    print("# --- throughput (App. E, Fig 21): every run at each batch size it was measured at")
    for t in TASKS:
        by = defaultdict(list)
        for r in csv.DictReader(open(R / t / "throughput.csv")) if (R / t / "throughput.csv").exists() else []:
            by[r["batch"]].append(r["run"])
        for b, runs in sorted(by.items(), key=lambda x: int(x[0])):
            print(f"python -m exp.bench.mistr_throughput --batch {b} " + " ".join(uniq(runs)))
    print("# --- high-res tokens per test sample (Fig 26)")
    for p in sorted(R.glob("*/crop_spread/*.pt")):
        print(f"python -m exp.bench.mistr_crop_spread {p.stem}")
    print("# --- second-pass attention cost (table oneway-cost)")
    for run in uniq(col(R / "pastis" / "oneway_cost.csv", "run")):
        print(f"python -u -m exp.mistr.oneway_cost {run}")


def tessera():
    print("# --- Tessera v2 (weights: pretrained/tessera/student_small.pt), PASTIS, App. H")
    rows = list(csv.DictReader(open(R / "tessera" / "pastis_frozen.csv")))
    cfgs = uniq(r["config"] for r in rows)
    for c in uniq(c.split(">")[0] for c in cfgs if c.endswith("-tattn")):
        print(f"python -m exp.tessera.pooltune --coarse {c}")
    for c in cfgs:
        print(f"python -u -m exp.tessera.embed --config '{c}'")
        for s in sorted({int(r['seed']) for r in rows if r['config'] == c}):
            print(f"python -u -m exp.tessera.head --config '{c}' --heads unet --seed {s}")


GEOID = ("python -W ignore -u -m exp.geoidchange.tiletrack train eval_every=2 classes=3 data=long "
         "long_root=data/GEOID-Flood-tileseries-long2/cache T=32 ref_n=0 near_days=0 gap_days=28 min_hist=30 "
         "mix_p=1 mix_frac=0.1,0.4 mix_ref=0 mix_flood=1 q_unit=series budget=batch ks_eval=8,12,16,20,24 "
         "resume=1 fast_prep=1")
QS = "0.01,0.02,0.03,0.04,0.06,0.08"


def geoid():
    ck = "checkpoints/geoidchange"
    print("# --- GEOID-Flood tracking: 512 px same-scene tiles, then the dense same-orbit S1 RTC series")
    print("python -m exp.geoidchange.tilecache workers=8")
    print("python -u -m exp.geoidchange.tileseries_fetch --before 24 --after 24 "
          "--out data/GEOID-Flood-tileseries-long2 --threads 8")
    print("python -W ignore -u -m exp.geoidchange.tileseries_cache ser=data/GEOID-Flood-tileseries-long2 "
          "splits=test,val,train")
    recs = [json.loads(l) for l in open(R / "geoidchange" / "trackN_rescore.jsonl")]
    cp4 = [json.loads(l) for l in open(R / "geoidchange" / "trackN_rescore_cp4.jsonl")]
    runs = uniq((r["name"], r.get("seed", 0), r["lr_ps"], r["crop"]) for r in recs + cp4)
    # + the fixed 20-epoch model every _cont / budget-range MiSTR run starts from (same seed)
    runs = uniq(runs + [(f"trkO_fix{lr}", s, lr, 0) for n, s, lr, cp in runs
                        if (cp or n.endswith("_cont")) and not n.endswith("_qr_or")])
    print("# --- training (fixed first: the budget-range MiSTR runs start from them)")
    for name, s, lr, cp in sorted(runs, key=lambda x: (x[3] > 0, x[0], x[1])):
        out = f"name={name} save={ck}/{name}_s{s}.pt seed={s} lr_ps={lr}"
        attn = " attn=chunk" if cp == 16 or (cp == 0 and lr == 16) else ""
        if cp == 0 and not name.endswith("_cont"):
            print(f"{GEOID} {out} q=0 epochs=20{attn}")
        elif cp == 0:
            print(f"{GEOID} {out} q=0 epochs=8 init={ck}/{name[:-5]}_s{s}.pt")
        elif name.endswith("_qr_or"):
            print(f"{GEOID} {out} crop={cp} q=0.04 q_set=0.01,0.02,0.04,0.08 epochs=28 warmup=14 "
                  f"train_sel=joint sels=joint qs={QS}{attn}")
        else:
            print(f"{GEOID} {out} crop={cp} q=0.04 q_set=0.01,0.02,0.04,0.08 epochs=8 "
                  f"init={ck}/trkO_fix{lr}_s{s}.pt train_sel=joint sels=joint qs={QS}{attn}")
    print("# --- rescoring per event position (results/geoidchange/trackN_rescore.jsonl)")
    for r in uniq((r["name"], r.get("seed", 0), r["lr_ps"], r["crop"], r["trunc"], r["q_unit"]) for r in recs):
        name, s, lr, cp, trunc, qu = r
        qs = "" if cp == 0 else f" qs={QS}"
        print(f"python -W ignore -u -m exp.geoidchange.trackN_rescore {name} {lr} {cp} seed={s} "
              f"q_unit={qu} trunc={int(trunc)}{qs}")
    print("# --- Fig 12 (App. K): the same 1,266 test series with 31 passes of history, cut at the labeled date")
    print("python -u -m exp.geoidchange.tileseries_fetch --before 31 --after 7 --splits test "
          "--out data/GEOID-Flood-tileseries-long3 --threads 8")
    print("python -W ignore -u -m exp.geoidchange.tileseries_cache ser=data/GEOID-Flood-tileseries-long3 splits=test")
    for name, s, lr, cp, qu in uniq((r["name"], r.get("seed", 0), r["lr_ps"], r["crop"], r["q_unit"]) for r in cp4):
        qs = "" if cp == 0 else f" qs={QS}"
        print(f"python -W ignore -u -m exp.geoidchange.trackN_rescore {name} {lr} {cp} seed={s} q_unit={qu} trunc=1"
              f" root=data/GEOID-Flood-tileseries-long3/cache ks=4,8,12,16,20,24,28,31 min_hist=0 common_prefix=1"
              f" out=results/geoidchange/trackN_rescore_cp4.jsonl{qs}")
    l3 = ("python -W ignore -u -m exp.geoidchange.tiletrack_flops long long_root=data/GEOID-Flood-tileseries-long3/cache "
          f"ref_ns=0 budget=sample q_unit=series qs={QS} combos=128c64,128c32,64c32,64c16 fixed=32,64,128")
    print(f"{l3} Ts=9,13,17,21,25,29,32 out=tiletrack_O_prefix_flops_long3")
    print(f"{l3} Ts=5 min_hist=0 out=tiletrack_O_prefix_flops_long3_t5")
    print("# --- causal-prefix costs (results/geoidchange/tiletrack_*_prefix_*flops.csv)")
    base = "python -W ignore -u -m exp.geoidchange.tiletrack_flops long long_root=data/GEOID-Flood-tileseries-long2/cache ref_ns=0 budget=sample"
    print(f"{base} Ts=9,13,17,21,25 q_unit=series qs={QS} combos=128c32,128c64,64c32,64c16 fixed= "
          "out=tiletrack_O_prefix_flops")
    print(f"{base} Ts=9,13,17,21,25 combos=128c32,128c64,64c32,64c16 "
          "qs=0.04,0.0625,0.08,0.125,0.16,0.25,0.32 out=tiletrack_N_prefix_flops")
    print(f"{base} Ts={','.join(map(str, range(2, 14)))} combos= min_hist=0 out=tiletrack_N_prefix_fixed_flops")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("data", "evals", "tessera", "geoid"), nargs="*")
    a = ap.parse_args()
    print("#!/bin/bash\n# generated by python -m exp.paper_evals; run from the repo root\nset -e")
    for part in a.part or ("data", "evals", "tessera", "geoid"):
        globals()[part]()


if __name__ == "__main__":
    main()
