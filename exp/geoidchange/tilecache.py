"""Same-scene FULL-TILE pair cache: GEOID's own (pre, post) S1 GRD pair on 512 px crops.

Why: the series caches are 64 px chips, all selected to contain flood (22-29% flood pixels), so
ps32 is a 2 x 2 token grid with almost no context. Here: every tile whose CEMS delineation used
GEOID's S1 post scene (|gap| < 1 h, tile_catalog.parquet), cut into its four 512 px quadrants,
kept when >= 50% of the quadrant is mapped (label != 255). No flood requirement: flood cover is
whatever the tile has. Split = the dataset's per-tile split (data_tiles_s256_st128.csv).

Stored per split: <split>.x int16 memmap (N, 2 dates [pre, post], 2 bands [VV, VH], 512, 512) in
centi-dB, -32768 = nodata (linear <= 0); <split>.y.npy uint8 (N, 512, 512) GEOID label
(0 / 1 permanent water / 2 flood / 255); <split>.json [{key, quad, row, col, days [pre, post]}].

    python -m exp.geoidchange.tilecache [workers=8]   -> data/GEOID-Flood-tiles512-samescene/
    python -m exp.geoidchange.tilecache subset=other splits=test   -> data/GEOID-Flood-tiles512-other/
        the complement (CEMS label NOT drawn on GEOID's S1 post scene: noisier labels; recs carry gap_s)
"""
import csv
import datetime as dt
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from exp.geoidchange.samescene import same_scene_keys
from mistr.data.datasets.geoidflood import DEFAULT_ROOT, SPLIT_CSV, _tile_files

DST = Path("data/GEOID-Flood-tiles512-samescene")
S = 512
NODATA = -32768
_DATE = re.compile(r"_(\d{8})T\d{6}")


def days_of(path):
    d = dt.datetime.strptime(_DATE.search(Path(path).name).group(1), "%Y%m%d").date()
    return (d - dt.date(1970, 1, 1)).days


def read_tile(stem):
    """stem -> list of (quad, row, col, x (2, 2, S, S) int16, y (S, S) uint8, days) kept crops."""
    event, tile = stem.rsplit("-", 1)
    files = _tile_files(Path(DEFAULT_ROOT), event, tile)
    if files is None:
        return stem, []
    try:
        fr = []
        for pas in ("pre", "post"):
            with rasterio.open(files[pas]) as r:
                a = r.read().astype(np.float32)                    # (2, H, W) linear sigma0
            ok = np.isfinite(a) & (a > 0)
            db = np.full(a.shape, NODATA, np.int16)
            db[ok] = np.clip(np.round(1000 * np.log10(a[ok])), -5000, 5000).astype(np.int16)
            fr.append(db)
        with rasterio.open(files["label"]) as r:
            lab = r.read(1).astype(np.uint8)
    except Exception as e:                                         # corrupt / truncated COG
        return stem, [("error", str(e))]
    days = [days_of(files["pre"]), days_of(files["post"])]
    out = []
    H, W = lab.shape
    for q, (r0, c0) in enumerate([(0, 0), (0, S), (S, 0), (S, S)]):
        if r0 + S > H or c0 + S > W:
            continue
        y = lab[r0:r0 + S, c0:c0 + S]
        if (y != 255).mean() < 0.5:
            continue
        x = np.stack([f[:, r0:r0 + S, c0:c0 + S] for f in fr])
        out.append((q, r0, c0, x, y, days))
    return stem, out


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    other = kw.get("subset", "same") == "other"     # the complement: label NOT drawn on GEOID's S1 post scene
    dst = Path(kw.get("dst", "data/GEOID-Flood-tiles512-other" if other else DST))
    dst.mkdir(parents=True, exist_ok=True)
    gaps = same_scene_keys()
    split_of = {}
    with open(Path(DEFAULT_ROOT) / SPLIT_CSV) as f:
        for row in csv.DictReader(f):
            if row["label_id"].endswith("_label"):
                split_of[row["label_id"][:-len("_label")]] = row["split"]
    if other:                                       # gap >= 1 h, or no delineation time in the catalog
        stems = sorted(s for s in split_of if gaps.get(s, 1e18) >= 3600)
    else:
        stems = sorted(s for s in split_of if gaps.get(s, 1e18) < 3600)
    print(f"{len(stems)} {'other' if other else 'same-scene'} tiles with a split", flush=True)
    by = {sp: [] for sp in kw.get("splits", "train,val,test").split(",")}
    for s in stems:
        if split_of[s] in by:
            by[split_of[s]].append(s)
    stats = {}
    with ProcessPoolExecutor(int(kw.get("workers", 8))) as ex:
        for sp, ss in by.items():
            xf = open(dst / f"{sp}.x", "wb")
            ys, recs, errs = [], [], []
            for stem, crops in ex.map(read_tile, ss, chunksize=4):
                for c in crops:
                    if c[0] == "error":
                        errs.append([stem, c[1]])
                        continue
                    q, r0, c0, x, y, days = c
                    xf.write(np.ascontiguousarray(x).tobytes())
                    ys.append(y)
                    recs.append({"key": stem, "quad": q, "row": r0, "col": c0, "days": days,
                                 "gap_s": gaps.get(stem)})          # S1 post vs delineation (None: no time)
            xf.close()
            np.save(dst / f"{sp}.y.npy", np.stack(ys) if ys else np.zeros((0, S, S), np.uint8))
            json.dump(recs, open(dst / f"{sp}.json", "w"))
            Y = np.stack(ys) if ys else np.zeros((1, 1), np.uint8)
            m = Y != 255
            stats[sp] = {"tiles": len(ss), "crops": len(recs), "errors": len(errs),
                         "events": len({r["key"].split("-")[0] for r in recs}),
                         "flood_frac": round(float((Y == 2).sum() / max(m.sum(), 1)), 4),
                         "water_frac": round(float(((Y == 1) | (Y == 2)).sum() / max(m.sum(), 1)), 4),
                         "crops_with_flood": round(float(np.mean([(y == 2).any() for y in ys])), 3) if ys else 0,
                         "error_list": errs[:20]}
            print(sp, {k: v for k, v in stats[sp].items() if k != "error_list"}, flush=True)
    json.dump(stats, open(dst / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
