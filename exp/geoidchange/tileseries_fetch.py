"""Same-orbit Sentinel-1 RTC series on the 512 px same-scene GEOID quadrants (exp.geoidchange.tilecache).

For temporal MiSTR ("find the event"): per same-scene tile, the Planetary Computer `sentinel-1-rtc`
passes of the RELATIVE ORBIT OF GEOID'S POST SCENE (one viewing geometry, 12-day repeat, 6-day
while S1A + S1B flew) around the post scene: up to BEFORE passes before it, the post pass itself
(PC's copy of the scene CEMS delineated on, matched within 2 min), up to AFTER passes after it. Read
onto the tile's own UTM grid (bilinear, as exp.geoidflood.fetch_s1_series), cut to the quadrants
kept by tilecache. A tile whose post scene Planetary Computer lacks is recorded and skipped.

    <out>/<event>/<tile>.npz
        quads   (Q,) int          tilecache quadrant ids (0..3) stored, in order
        data    (A, 2, Q, 512, 512) int16   VV, VH centi-dB, -32768 = no data
        time    (A,) str          pass start (ISO UTC), ascending
        post_idx                  index of the post pass in time
        rel_orbit, orbit          the post scene's relative orbit / orbit state

--before 24 --after 7 --out data/GEOID-Flood-tileseries-long: long dense series (T up to 32; exp.geoidchange.tiletrack
LongTrack).

--monthly: instead of consecutive passes, the post pass plus the same-orbit pass nearest each of post +- 30 k days
(k = 1..MONTHS, within +-15 d), i.e. ~monthly gaps over a year: training series where the dates other than the
event are (almost always) flood-free (temporal CutMix in exp.geoidchange.tiletrack). Output dir default
data/GEOID-Flood-tileseries-monthly.

Resumable (existing files skipped); shard over tiles:
    python -u -m exp.geoidchange.tileseries_fetch --shard 3 --num_shards 32 --threads 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from exp.geoidflood.fetch_s1_series import (NODATA, ROOT, SCENE_TOL, Token, read_acquisition,
                                            scene_time, search)

SRC = Path("data/GEOID-Flood-tiles512-samescene")
OUT = Path("data/GEOID-Flood-tileseries")
S = 512
BEFORE, AFTER = 7, 6
SPAN = 13 * (BEFORE + 1)          # days searched each side (12-day repeat, + slack)
MONTHS, GAP, TOL = 6, 30, 15      # --monthly: post +- GAP * k, k = 1..MONTHS, nearest pass within TOL days


def quads_by_tile(src=None, splits=("train", "val", "test")) -> dict[str, list[tuple[int, int, int]]]:
    out = defaultdict(list)
    for sp in splits:
        p = Path(src or SRC) / f"{sp}.json"
        if not p.exists():
            continue
        for r in json.load(open(p)):
            out[r["key"]].append((r["quad"], r["row"], r["col"]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def to_db(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(x)
    return np.where(np.isfinite(db), np.clip(np.round(db * 100), -32767, 32767), NODATA).astype(np.int16)


def shift_of(a, b, r=4):
    """Whole-px (dy, dx) with b[y, x] ~ a[y - dy, x - dx]; NCC unshifted, NCC at the best shift."""
    ok = (a > NODATA) & (b > NODATA)
    def ncc(dy, dx):
        h, w = a.shape
        A = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        B = b[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        m = ok[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] & \
            ok[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        if m.sum() < 1000:
            return -1.0
        return float(np.corrcoef(A[m][::5], B[m][::5])[0, 1])
    best = max((ncc(dy, dx), dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1))
    return best[1], best[2], ncc(0, 0), best[0]


def fetch_tile(tile: str, quads, token: Token, threads: int, monthly: bool = False, before: int = BEFORE,
               after: int = AFTER) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    post = scene_time(tile, "post")
    event = tile.rsplit("-", 1)[0]
    ref = sorted((ROOT / event / "s1grd").glob(f"{tile}_s1grd_post_*.tif"))[0]
    with rasterio.open(ref) as g:
        grid = (g.crs, g.transform, g.width, g.height)
        bbox = list(transform_bounds(g.crs, "EPSG:4326", *g.bounds))
    span = GAP * MONTHS + TOL + 1 if monthly else 13 * (max(before, after) + 1)
    items = search(bbox, post - dt.timedelta(days=span), post + dt.timedelta(days=span))
    passes = defaultdict(list)
    for it in items:
        p = it["properties"]
        passes[(p.get("platform"), p.get("sat:absolute_orbit"), p["datetime"][:10])].append(it)
    t_of = {k: dt.datetime.fromisoformat(min(i["properties"]["datetime"] for i in v)
                                         .replace("Z", "+00:00")).replace(tzinfo=None)
            for k, v in passes.items()}
    hit = [k for k in passes if abs(t_of[k] - post) <= SCENE_TOL]
    if not hit:
        return {"error": "post scene not on Planetary Computer"}
    pk = hit[0]
    rel = passes[pk][0]["properties"].get("sat:relative_orbit")
    keys = sorted((k for k in passes if passes[k][0]["properties"].get("sat:relative_orbit") == rel),
                  key=t_of.get)
    j = keys.index(pk)
    if monthly:                    # post + the pass nearest each month offset (none if > TOL days away)
        pick = {pk}
        for k in [m for m in range(-MONTHS, MONTHS + 1) if m]:
            tgt = post + dt.timedelta(days=GAP * k)
            c = min(keys, key=lambda q: abs(t_of[q] - tgt))
            if abs(t_of[c] - tgt) <= dt.timedelta(days=TOL):
                pick.add(c)
        keys = sorted(pick, key=t_of.get)
    else:
        keys = keys[max(0, j - before):j + after + 1]
    r0 = min(r for _, r, _ in quads)
    c0 = min(c for _, _, c in quads)
    win = Window(c0, r0, max(c for _, _, c in quads) + S - c0, max(r for _, r, _ in quads) + S - r0)

    # PC vs GEOID geolocation: a constant whole-pixel offset per tile (1-2 px, same orbit = same
    # offset); estimated on the post pass against GEOID's own post raster, applied to every pass
    # by reading a shifted window
    with rasterio.open(ref) as g:
        geo = to_db(g.read(window=win, boundless=True, fill_value=0).astype(np.float32))[0].astype(np.float32)
    pc = to_db(read_acquisition(passes[pk], grid, win, token))[0].astype(np.float32)
    dy, dx, ncc0, ncc = shift_of(pc, geo)
    # new pixel (y, x) <- old (y - dy, x - dx): translate the grid, the window stays in range
    grid = (grid[0], grid[1] * Affine.translation(-dx, -dy), grid[2], grid[3])

    def one(k):
        x = read_acquisition(passes[k], grid, win, token)                    # (2, h, w) linear
        return np.stack([to_db(x[:, r - r0:r - r0 + S, c - c0:c - c0 + S]) for _, r, c in quads], 1)

    with ThreadPoolExecutor(threads) as ex:
        data = list(ex.map(one, keys))
    keep = [i for i, d in enumerate(data) if (d != NODATA).mean() > 0.05 or keys[i] == pk]
    keys = [keys[i] for i in keep]
    return dict(quads=np.array([q for q, _, _ in quads], np.int32),
                data=np.stack([data[i] for i in keep]),
                time=np.array([t_of[k].isoformat() for k in keys]),
                post_idx=np.array(keys.index(pk)), shift=np.array([dy, dx], np.int32),
                post_ncc=np.array([ncc0, ncc], np.float32), rel_orbit=np.array(rel if rel is not None else -1),
                orbit=np.array(passes[pk][0]["properties"].get("sat:orbit_state", "")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=2.8)
    ap.add_argument("--out", default="")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--before", type=int, default=BEFORE, help="consecutive same-orbit passes before post")
    ap.add_argument("--after", type=int, default=AFTER)
    ap.add_argument("--src", default="", help="tile cache (default the same-scene one; e.g. "
                    "data/GEOID-Flood-tiles512-other)")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    tiles = list(quads_by_tile(a.src or None, a.splits.split(",")).items())[a.shard::a.num_shards]
    if a.limit:
        tiles = tiles[:a.limit]
    token, t_start, done, failed = Token(), time.time(), 0, 0
    out = Path(a.out or (str(OUT) + "-monthly" if a.monthly else OUT))
    print(f"shard {a.shard}/{a.num_shards}: {len(tiles)} tiles", flush=True)
    for tile, quads in tiles:
        path = out / tile.rsplit("-", 1)[0] / f"{tile}.npz"
        bad = path.with_suffix(".missing")
        if path.exists() or bad.exists():
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print(f"STOPPED (--max_hours) after {done} tiles; rerun to resume", flush=True)
            return
        t0 = time.time()
        try:
            res = fetch_tile(tile, quads, token, a.threads, a.monthly, a.before, a.after)
        except Exception as e:
            failed += 1
            print(f"FAILED {tile}: {type(e).__name__}: {e}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if "error" in res:
            bad.write_text(res["error"])
            print(f"{tile}: {res['error']}", flush=True)
            continue
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **res)
        tmp.replace(path)
        done += 1
        print(f"{tile}: {len(quads)} quads, {len(res['time'])} passes, post_idx {int(res['post_idx'])}, "
              f"shift {res['shift'].tolist()} ncc {res['post_ncc'].round(3).tolist()}, "
              f"{res['data'].nbytes / 2**20:.0f} MB, {time.time() - t0:.0f}s", flush=True)
    print(f"shard {a.shard} done: {done} fetched, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
