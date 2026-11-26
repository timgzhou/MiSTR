"""Pack the fetched same-orbit RTC series (exp.geoidchange.tileseries_fetch) into per-split memmaps
aligned with the 512 px same-scene tile-pair records (exp.geoidchange.tilecache), for tileseq.

    <dst>/<split>.x      int16 memmap (sum_i A_i, 2, 512, 512): every record's passes, back to back
    <dst>/<split>.json   [{rec (index into tilecache <split>.json / .y.npy), key, quad, off, A,
                           post (index of the post pass), time [iso...], shift}] for records whose
                          tile was fetched (missing tiles / quads are skipped and counted)

    python -m exp.geoidchange.tileseries_cache        -> data/GEOID-Flood-tileseries/cache/
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path("data/GEOID-Flood-tiles512-samescene")
SER = Path("data/GEOID-Flood-tileseries")
S = 512
MIN_NCC, SHIFT_BOUND = 0.7, 4


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    global SER
    SER = Path(kw.get("ser", SER))                # e.g. ser=data/GEOID-Flood-tileseries-monthly
    dst = Path(kw.get("dst", SER / "cache"))
    dst.mkdir(parents=True, exist_ok=True)
    stats = {}
    for sp in kw.get("splits", "train,val,test").split(","):
        recs = json.load(open(SRC / f"{sp}.json"))
        by = defaultdict(list)
        for i, r in enumerate(recs):
            by[r["key"]].append(i)
        out, off, miss = [], 0, defaultdict(int)
        with open(dst / f"{sp}.x", "wb") as f:
            for key, ids in by.items():
                p = SER / key.rsplit("-", 1)[0] / f"{key}.npz"
                if not p.exists():
                    miss["no_post" if p.with_suffix(".missing").exists() else "not_fetched"] += len(ids)
                    continue
                z = np.load(p)
                # unreliable PC <-> GEOID alignment: post-scene NCC < MIN_NCC, or the shift hit
                # the search bound (the true offset may be larger)
                if "post_ncc" in z and (float(z["post_ncc"][1]) < MIN_NCC or
                                        int(np.abs(z["shift"]).max()) >= SHIFT_BOUND):
                    miss["misaligned"] += len(ids)
                    continue
                quads = list(z["quads"])
                data = z["data"]                                     # (A, 2, Q, S, S)
                for i in ids:
                    q = recs[i]["quad"]
                    if q not in quads:
                        miss["quad"] += 1
                        continue
                    x = np.ascontiguousarray(data[:, :, quads.index(q)])
                    f.write(x.tobytes())
                    out.append({"rec": i, "key": key, "quad": q, "off": off, "A": int(len(x)),
                                "post": int(z["post_idx"]), "time": [str(t) for t in z["time"]],
                                "shift": z["shift"].tolist() if "shift" in z else [0, 0]})
                    off += len(x)
        json.dump(out, open(dst / f"{sp}.json", "w"))
        A = np.array([o["A"] for o in out]) if out else np.zeros(1)
        P = np.array([o["post"] for o in out]) if out else np.zeros(1)
        stats[sp] = {"records": len(out), "missing": dict(miss), "passes_median": float(np.median(A)),
                     "post_median": float(np.median(P)), "frames": off}
        print(sp, stats[sp], flush=True)
    json.dump(stats, open(dst / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
