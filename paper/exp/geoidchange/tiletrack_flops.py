"""Measured inference MACs per sample of every flood-tracking configuration (exp.geoidchange.tiletrack).

Forward passes of EventTrack on dense test series (T = 8 passes, 512 px), traced with
exp.bench.flop_trace.count_macs (linear / conv / einsum / SDPA + FlashAttention, MAC = multiply-add),
for each (lr_ps, crop, q, selector) of the monthly-trained (trkM_*) rows in results/geoidchange/tiletrack.jsonl.
Averaged over N_BATCH batches, because the crop count depends on the valid passes and on how many selected
windows share a crop. The selector scores are elementwise (not counted). Untrained weights: cost does not depend
on them.

    python -m exp.geoidchange.tiletrack_flops      -> results/geoidchange/tiletrack_flops.csv
    python -m exp.geoidchange.tiletrack_flops long [Ts=8,16,32 ref_ns=0,1]   -> tiletrack_long_flops.csv
    round N: ... long Ts=32 ref_ns=0 long_root=data/GEOID-Flood-tileseries-long2/cache budget=sample
             combos=128c32,128c64,64c32,64c16 qs=0.04,0.0625,0.08,0.125,0.16,0.25,0.32 out=tiletrack_N_flops
    causal prefix (x-axis of the round-N figures = the passes 0..k a real-time model needs for event k):
      long Ts=9,13,17,21,25 ref_ns=0 ... combos=... qs=... out=tiletrack_N_prefix_flops       (every pass / MiSTR)
      long Ts=2,3,...,13 ref_ns=0 ... combos= min_hist=0 out=tiletrack_N_prefix_fixed_flops   (strided: kept passes <= k)
        budget=sample for the cost table: with budget=batch the crops per series differ inside a batch and the
        crop sequences are PADDED to the longest, which count_macs counts; the per-series cost is ~linear in the
        crop count and the batch budget keeps the mean count = q x 4 x W^2, so the per-sample table = the mean cost
        (LongTrack test series of T passes, absolute budget q_unit=abs, joint selector; every trkL_ config)
"""
import sys
import csv
import json

import torch
from torch.utils.data import DataLoader

from exp.bench.flop_trace import count_macs
from exp.geoidchange.tiletrack import EventTrack, LongTrack, Track, collate

RES = "results/geoidchange"
B, N_BATCH = 4, 8


@torch.no_grad()
def main():
    cfgs = set()
    for line in open(f"{RES}/tiletrack.jsonl"):
        r = json.loads(line)
        if not r["name"].startswith("trkM_"):
            continue
        if r["q"] == 0:
            cfgs.add((r["lr_ps"], 0, 0.0, "none"))
        else:
            for sel in r["test"]:
                cfgs.add((r["lr_ps"], r["crop"], r["q"], sel))
    ds = Track("test", 8, three=True)
    batches = []
    for i, b in enumerate(DataLoader(ds, batch_size=B, shuffle=True, collate_fn=collate,
                                     generator=torch.Generator().manual_seed(0))):
        batches.append(b)
        if i + 1 == N_BATCH:
            break
    out = []
    for lr, cp, q, sel in sorted(cfgs):
        net = EventTrack(lr, cp or 32, "cat", 3).cuda().eval()
        net.q_unit = "total"
        s, d, v, post = batches[0][:4]
        net(s, d, v, post, sel if q else "joint", q)                                # warm-up
        mac = lt = ct = n = 0
        for b in batches:
            s, d, v, post = b[:4]
            with count_macs() as c:
                _, _, info = net(s, d, v, post, sel if q else "joint", q)
            mac, lt, ct, n = mac + c.total, lt + info["lr_tokens"], ct + info["crops"], n + len(post)
        out.append({"lr_ps": lr, "crop": cp, "q": q, "sel": sel, "gmac": round(mac / n / 1e9, 3),
                    "lr_tokens": round(lt / n, 1), "crop_tokens": round(ct / n, 1)})
        print(out[-1], flush=True)
    with open(f"{RES}/tiletrack_flops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)


@torch.no_grad()
def long_main(kw):
    Ts = [int(x) for x in kw.get("Ts", "8,16,32").split(",")]
    refs = [int(x) for x in kw.get("ref_ns", "0,1").split(",")]
    if kw.get("long_root"):
        LongTrack.root = kw["long_root"]
    if kw.get("min_hist"):              # cost only: short windows (T < ~4) have no series with 30 d of history
        LongTrack.min_hist = float(kw["min_hist"])
    qs = [float(x) for x in kw.get("qs", "0.0625,0.125,0.25").split(",")]
    combos = [tuple(int(v) for v in c.split("c")) for c in kw.get("combos", "64c32,128c64,128c32").split(",") if c]
    fixed = [int(x) for x in kw.get("fixed", "16,32,64,128").split(",") if x]
    cfgs = [(ps, 0, 0.0) for ps in fixed] + [(L, C, q) for L, C in combos for q in qs]
    budget = kw.get("budget", "sample")
    out = []
    for T in Ts:
        ds = LongTrack("test", T)
        batches = []
        for i, b in enumerate(DataLoader(ds, batch_size=B, shuffle=True, collate_fn=collate,
                                         generator=torch.Generator().manual_seed(0))):
            batches.append(b)
            if i + 1 == N_BATCH:
                break
        for R in refs:
            for lr, cp, q in cfgs:
                net = EventTrack(lr, cp or 32, "self" if R == 0 else "cat", 3).cuda().eval()
                net.q_unit, net.ref_n = kw.get("q_unit", "abs"), R      # round O: q_unit=series
                net.budget_scope = budget if R == 0 else "sample"
                s, d, v, post = batches[0][:4]
                net(s, d, v, post, "joint", q)                                          # warm-up
                mac = lt = ct = n = 0
                for b in batches:
                    s, d, v, post = b[:4]
                    with count_macs() as c:
                        _, _, info = net(s, d, v, post, "joint", q)
                    mac, lt, ct, n = mac + c.total, lt + info["lr_tokens"], ct + info["crops"], n + len(post)
                out.append({"T": T, "ref_n": R, "lr_ps": lr, "crop": cp, "q": q, "gmac": round(mac / n / 1e9, 3),
                            "lr_tokens": round(lt / n, 1), "crop_tokens": round(ct / n, 1)})
                print(out[-1], flush=True)
                del net
                torch.cuda.empty_cache()
    with open(f"{RES}/{kw.get('out', 'tiletrack_long_flops')}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["long"]:
        long_main(dict(a.split("=", 1) for a in sys.argv[2:]))
    else:
        main()
