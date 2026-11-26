"""Eval-only rescoring of a round-N checkpoint on the test series (event at passes 8-24, batch budget), saving the
per-class scores per event position (per_k iou_cls / biou_cls; the headline bIoU is the FLOOD class). All budgets
in one pass over the data (tiletrack.evaluate_multi).

Strided fixed res is scored at every stride phase: lag j = the last kept pass is j passes before the labelled date
(lag 0 = anchored on the event, an oracle); per_k keys there are the KEPT index, mapped back to the event pass.

trunc=1 (CAUSAL PREFIX): every series is cut to its passes 0..k, the event the LAST pass (T = k + 1), for each
event position k separately -- the compute a real-time model spends to predict the event date. Same (series, k)
samples as the T = 32 evaluation. Fixed resolution scores are unchanged by the cut (causal attention), MiSTR's are
not: its batch-shared crop budget can no longer go to passes after the event. Rows carry trunc = true and only
per_k (one entry per k).

    python -m exp.geoidchange.trackN_rescore NAME LR CROP [stride=1 qs=0.0625,0.125,0.25 seed=0 attn=sdpa trunc=0 n=0 q_unit=abs
                                                          ks=8,12,16,20,24 root=data/GEOID-Flood-tileseries-long2/cache out=...]
    ks / root: late event positions (App. K): root=data/GEOID-Flood-tileseries-long3/cache (31 passes of
    history, test only) ks=24,28,31
      -> results/geoidchange/trackN_rescore.jsonl   (one row per (q, lag))
"""
import json
import sys
import time

import torch

from exp.geoidchange import tiletrack as tt

KS = [8, 12, 16, 20, 24]


def main(a):
    name, lr, crop = a[0], int(a[1]), int(a[2])
    kw = dict(x.split("=", 1) for x in a[3:])
    seed, st, n = int(kw.get("seed", 0)), int(kw.get("stride", 1)), int(kw.get("n", 0))
    trunc = kw.get("trunc", "0") == "1"
    qs = [float(x) for x in kw.get("qs", "0").split(",")]
    tt.ATTN = kw.get("attn", "sdpa")
    ks = [int(x) for x in kw.get("ks", ",".join(map(str, KS))).split(",")]
    root = kw.get("root", "data/GEOID-Flood-tileseries-long2/cache")
    tt.FAST_PREP = kw.get("fast_prep", "1") == "1"   # raw dB from the workers, normalized on the GPU (bit-identical)
    L = tt.LongTrack
    L.root, L.stride, L.min_hist, L.near_days, L.ref_n = root, st, float(kw.get("min_hist", 30)), 30.0, 0
    # min_hist=0: eval the event at t = 4 (24 d of history at a 6-day revisit < the 30 d training rule)
    L.mix_p = 0.0
    net = tt.EventTrack(lr, crop if crop else 32, "self", 3).cuda().eval()
    net.load_state_dict(torch.load(f"checkpoints/geoidchange/{name}_s{seed}.pt", map_location="cuda"))
    net.q_unit, net.ref_n = kw.get("q_unit", "abs"), 0          # round O: q_unit=series (share of all slots)
    net.budget_scope = "batch" if max(qs) > 0 else "sample"
    combos = [("joint", q) if q > 0 else ("none", 0.0) for q in qs]
    out = kw.get("out", "results/geoidchange/trackN_rescore.jsonl")

    def write(row):
        r = row["test"]
        pk = {k: round(v["biou_cls"][2], 4) for k, v in r["per_k"].items()}
        print(name, row["q"], "lag", row["lag"], "trunc" if trunc else "", "flood biou per k", pk, flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")

    base = {"name": name, "seed": seed, "stride": st, "lr_ps": lr, "crop": crop, "trunc": trunc, "n": n,
            "q_unit": net.q_unit, "root": root}
    ref = L("test", 32, n=n, ks=ks)
    if kw.get("common", "0") == "1":     # only series feasible at EVERY k (same series per position: no sample-set
        cnt = {}                         # effect across positions; rows carry common = true)
        for r in ref.recs:
            cnt[(r["key"], r["quad"])] = cnt.get((r["key"], r["quad"]), 0) + 1
        ref.recs = [r for r in ref.recs if cnt[(r["key"], r["quad"])] == len(ks)]
        base["common"] = True
        print(f"common series: {len(ref.recs) // len(ks)} x {len(ks)} positions", flush=True)
    t0 = time.time()
    if trunc:
        assert st == 1, "strided runs: the cut changes nothing (causal), use the T = 32 rows"
        keep = {(r["key"], r["quad"], r["k_fix"]) for r in ref.recs}
        dss = {k: L("test", k + 1, ks=[k]) for k in ks}
        if kw.get("common_prefix", "0") == "1":
            # SAME series at every k, feasibility of the causal prefix only (k passes before the event, >= min_hist
            # days): a series with 31 passes of history can be cut to any earlier event position. No T = 32
            # window (passes after the event never matter when cut). Fixed models too: causal, = the T = 32 scores.
            com = set.intersection(*({(r["key"], r["quad"]) for r in d.recs} for d in dss.values()))
            keep = {(a, b, k) for a, b in com for k in ks}
            base["common_prefix"] = True
            print(f"common prefix series: {len(com)} x {len(ks)} positions", flush=True)
        per = {cq: {} for cq in combos}
        lt = {cq: [] for cq in combos}
        for k in ks:
            ds = dss[k]
            ds.recs = [r for r in ds.recs if (r["key"], r["quad"], k) in keep]
            res = tt.evaluate_multi(net, ds, combos)
            for cq, r in res.items():
                per[cq].update({str(kk): v for kk, v in r["per_k"].items()})
                lt[cq].append(dict(k=k, n=r["n"], lr_tokens=r["lr_tokens"], crop_tokens=r["crop_tokens"],
                                   crop_share=r["crop_share"], crops_by_flood=r["crops_by_flood"]))
            print(f"k {k}: {len(ds)} samples, {time.time() - t0:.0f}s", flush=True)
        for cq in combos:
            write({**base, "q": cq[1], "lag": 0, "test": {"per_k": per[cq], "per_k_info": lt[cq]}})
        return
    for lag in range(st):
        L.lag = lag
        res = tt.evaluate_multi(net, ref, combos)
        for cq, r in res.items():
            if st > 1:                                       # kept index -> event pass
                r["per_k"] = {str((int(kk) + (lag > 0)) * st): v for kk, v in r["per_k"].items()}
            write({**base, "q": cq[1], "lag": lag, "test": r})
        print(f"lag {lag}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
