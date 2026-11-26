"""App. K, training-matched comparison: the numbers quoted in its text, from results/geoidchange/.

Fixed S32 at 20 epochs vs. 8 more epochs (_cont), mean +- s.e. over seeds, and the paired change;
MiSTR S64 & S16 points; all averaged over the event positions of Fig 05 (t = 12, 16, 20, 24).

    python -m exp.viz.geoid_k_numbers
"""
import json

import numpy as np

import exp.viz.geoid_trackN as g

RES = "results/geoidchange/trackN_rescore.jsonl"
KS = tuple(str(k) for k in g.KS_MAIN)


def fixed_seeds(name):
    """{seed: (mIoU, flood bIoU)} of a fixed model, 0-100, averaged over KS."""
    out = {}
    for line in open(RES):
        r = json.loads(line)
        if r["name"] == name and r.get("stride", 1) == 1 and r["q"] == 0 and all(k in r["test"]["per_k"] for k in KS):
            pk = r["test"]["per_k"]
            out[r.get("seed", 0)] = np.array([100 * np.mean([pk[k]["miou"] for k in KS]),
                                              100 * np.mean([pk[k]["biou_cls"][2] for k in KS])])
    return out


def mean_se(a):
    a = np.asarray(a)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


def main():
    f20, f28 = fixed_seeds("trkO_fix32"), fixed_seeds("trkO_fix32_cont")
    seeds = sorted(set(f20) & set(f28))
    for label, d in (("S32, 20 epochs", f20), ("S32, 28 epochs", f28)):
        m, s = mean_se([d[k] for k in seeds])
        print(f"{label}: {m[0]:.1f} +- {s[0]:.1f} mIoU, {m[1]:.1f} +- {s[1]:.1f} flood bIoU "
              f"({len(seeds)} seeds; seed 0: {d[0][0]:.1f}, {d[0][1]:.1f})")
    m, s = mean_se([f28[k] - f20[k] for k in seeds])
    print(f"8 more epochs: {m[0]:+.1f} +- {s[0]:.1f} mIoU, {m[1]:+.1f} +- {s[1]:.1f} flood bIoU (paired)")

    def pick(d):
        if not all(k in d for k in g.KS_MAIN):
            return None
        return {m: float(np.mean([d[k][m] for k in g.KS_MAIN])) for m in ("miou", "fbiou", "gmac")}
    g.FIXED = "cont"
    ser = g.series(g.load(), pick)
    s32 = next(gm for gm, _, L in ser["fixed"] if L == 32)
    for gm, e, q in ser[("range", 64, 16)]:
        print(f"MiSTR S64 & S16, q {100 * q:g} %: {e['miou']:.1f} mIoU, {e['fbiou']:.1f} flood bIoU at "
              f"{gm:.1f} GMACs ({s32 / gm:.1f}x less than S32's {s32:.1f})")


if __name__ == "__main__":
    main()
