"""Same-scene subset of the GRD-series cache: keep only chips whose tile CEMS delineated on the SAME
Sentinel-1 scene as GEOID's post image (|S1 post acquisition - delineation_time_post| < 1 h, from
tile_catalog.parquet). Labels there were drawn on the very image the model sees; on test such
chips score ~.06 IoU higher at every patch size, and chips whose S1 post precedes the delineation
score .37-.40. Tiles absent from the catalog (no delineation time) are dropped.

Frames are not copied: <split>.x is symlinked, recs keep their offsets; only recs / y / geoid are
subset.
    python -m exp.geoidchange.samescene   -> data/GEOID-Flood-series-grd/cache_samescene/
"""
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd

SRC = "data/GEOID-Flood-series-grd/cache"
DST = "data/GEOID-Flood-series-grd/cache_samescene"
CAT = "data/GEOID-Flood-labels/geoid-flood/tile_catalog.parquet"
MAX_GAP_S = 3600


def s1_time(p):
    return dt.datetime.strptime(re.search(r"_(\d{8}T\d{6})_", p).group(1), "%Y%m%dT%H%M%S")


def same_scene_keys():
    t = pd.read_parquet(CAT)
    keys = {}
    for r in t.itertuples():
        dl = dt.datetime.fromisoformat(r.delineation_time_post)
        gap = min(abs((s1_time(p) - dl).total_seconds()) for p in r.s1_post_products)
        keys[f"{r.event_id}-{r.tile_id}-{r.bbox_id}"] = gap
    return keys


def main():
    os.makedirs(DST, exist_ok=True)
    gaps = same_scene_keys()
    stats = {}
    for split in ("train", "val", "test"):
        recs = json.load(open(f"{SRC}/{split}.json"))
        keep = [i for i, r in enumerate(recs) if gaps.get(r["chip"].split("@")[0], 1e18) < MAX_GAP_S]
        json.dump([recs[i] for i in keep], open(f"{DST}/{split}.json", "w"))
        for k in ("y", "geoid"):
            a = np.load(f"{SRC}/{split}.{k}.npy", mmap_mode="r")
            np.save(f"{DST}/{split}.{k}.npy", np.ascontiguousarray(a[keep]))
        x = f"{DST}/{split}.x"
        if not os.path.lexists(x):
            os.symlink(os.path.abspath(f"{SRC}/{split}.x"), x)
        ev = {recs[i]["chip"].split("-")[0] for i in keep}
        stats[split] = {"chips": len(keep), "of": len(recs), "events": len(ev),
                        "flood_frac": round(float((np.load(f"{DST}/{split}.y.npy") == 2).mean()), 4),
                        "water_frac": round(float(np.isin(np.load(f"{DST}/{split}.y.npy"), [1, 2]).mean()), 4)}
        print(split, stats[split], flush=True)
    json.dump(stats, open(f"{DST}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
