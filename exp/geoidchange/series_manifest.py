"""The exact Sentinel-1 passes of every GEOID-Flood tracking series (exp.geoidchange.tileseries_fetch):
per tile, the relative orbit, orbit state, post-pass index, the PC-vs-GEOID pixel shift, and every
pass's acquisition time (microseconds; identifies the Planetary Computer `sentinel-1-rtc` scenes).

    python -m exp.geoidchange.series_manifest write ser=data/GEOID-Flood-tileseries-long2 \
        out=manifests/geoid_series_long2.json.gz
    python -m exp.geoidchange.series_manifest check ser=<your fetch> manifest=manifests/geoid_series_long2.json.gz

check lists tiles missing from the fetch, extra ones, and tiles whose passes or shift differ (a
reprocessed or withdrawn scene on Planetary Computer).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def read(ser):
    out = {}
    for f in sorted(Path(ser).glob("*/*.npz")):
        z = np.load(f)
        out[f.stem] = {"time": [str(t) for t in z["time"]], "post_idx": int(z["post_idx"]),
                       "rel_orbit": int(z["rel_orbit"]), "orbit": str(z["orbit"]),
                       "shift": [int(s) for s in z["shift"]], "quads": [int(q) for q in z["quads"]]}
    return out


def main(args):
    cmd, kw = args[0], dict(a.split("=", 1) for a in args[1:])
    if cmd == "write":
        m = read(kw["ser"])
        Path(kw["out"]).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(kw["out"], "wt") as f:
            json.dump({"source": "planetary-computer sentinel-1-rtc", "series": m}, f, sort_keys=True)
        print(f"wrote {kw['out']}: {len(m)} tiles, {sum(len(v['time']) for v in m.values())} passes")
    elif cmd == "check":
        want = json.load(gzip.open(kw["manifest"], "rt"))["series"]
        have = read(kw["ser"])
        missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
        diff = sorted(k for k in set(want) & set(have) if want[k] != have[k])
        print(f"{len(want)} tiles in the manifest: {len(missing)} missing, {len(extra)} extra, {len(diff)} differ")
        for name, ks in (("missing", missing), ("extra", extra), ("differ", diff)):
            if ks:
                print(f"  {name}: {' '.join(ks[:20])}{' ...' if len(ks) > 20 else ''}")
        sys.exit(1 if missing or diff else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
