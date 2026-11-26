"""Paper material, datasets: the dataset table and one example (input time series, label) per
dataset.

    source env_setup/env_olmo.sh && python -m exp.viz.paper.datasets [--only table|examples]

Writes to figures/paper/datasets/:
    dataset_stats.json            every number in the table, as counted here from the data
    dataset_table.md / .tex       one row per modality of each release; the modalities the
                                  models read are in bold
    ../14_dataset-overview.pdf    the five examples in one figure (a full appendix page)

Timestep counts are per sample: PASTIS and PLANTEUR from metadata.geojson (all 2,433 / 2,750
patches), BioMassters
from the tortilla index (every --bio_every-th chip), GEOID from the packed series' metadata
(test + val chips' tiles, the fixed 6-month evaluation window), YieldSAT from the packed index
(all 2,173 fields).

Rendering (as dataset_visualization/poster_timeseries.py): S2 true colour B04/B03/B02, S1 as
R = VV, G = VH, B = VV - VH (dB); one stretch per series, so brightness changes are real.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from mistr_paper.data import All  # noqa: E402
from mistr_paper.data.datasets.biomassters import MONTHS, BiomasstersDataset  # noqa: E402
from mistr_paper.data.datasets.pastis import CLASS_NAMES, PastisDataset  # noqa: E402
from mistr_paper.data.datasets.planteur import CLASS_NAMES as PLANTEUR_CLASSES  # noqa: E402
from mistr_paper.data.datasets.planteur import DEFAULT_ROOT as PLANTEUR_ROOT  # noqa: E402
from mistr_paper.data.datasets.planteur import FOLDS as PLANTEUR_FOLDS  # noqa: E402
from mistr_paper.data.datasets.planteur import PlanteurDataset  # noqa: E402
from mistr_paper.data.datasets.yieldsat import CLEAR_SCL, PACKED, load_index  # noqa: E402
from mistr_paper.data.time import from_days  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "figures" / "paper" / "datasets"
from exp.viz.paper import poster_timeseries as poster  # noqa: E402

plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 7})
INK, MUTED, RULE = "#111111", "#555555", "#bbbbbb"
GEOID_COLORS = {0: (0.969, 0.969, 0.961), 1: (0.361, 0.784, 0.722), 2: (0.165, 0.471, 0.839),
                255: (0.78, 0.78, 0.76)}     # as the flood-tracking figures (exp.viz.geoid_track_example)
GEOID_NAMES = {0: "dry", 1: "permanent water", 2: "flood", 255: "unlabelled (ignored)"}
# the release's own class colours (documentation/colormap.txt); void white
def planteur_colors() -> dict:
    return {**{int(r[0]): tuple(int(v) / 255 for v in r[1:]) for r in
               (ln.split() for ln in open(PLANTEUR_ROOT / "documentation/colormap.txt")) if r},
            255: (1.0, 1.0, 1.0)}


def rng3(v):
    """min / median / max of a list of counts, as ints."""
    v = np.asarray(v)
    return [int(v.min()), int(np.median(v)), int(v.max())]


def span(r):
    return f"{r[0]}" if r[0] == r[2] else f"{r[0]}–{r[2]} (median {r[1]})"


# ---------------------------------------------------------------- statistics
def stats(bio_every: int = 10) -> dict:
    out = {}
    # PASTIS-R
    feats = json.load(open(REPO / "data/PASTIS-R/metadata.geojson"))["features"]
    n = {m: [len(f["properties"][f"dates-{m}"]) for f in feats] for m in ("S2", "S1A", "S1D")}
    out["pastis"] = {"patches": len(feats), "steps": {m: rng3(v) for m, v in n.items()},
                     "tiles": {s: len(TASKS["pastis"].dataset(s)) for s in ("train", "val", "test")}}
    # PLANTEUR
    feats = json.load(open(PLANTEUR_ROOT / "metadata.geojson"))["features"]
    n = {m: [len(f["properties"][f"dates-{m}"]) for f in feats] for m in ("S2", "S1", "LS")}
    out["planteur"] = {"patches": len(feats), "steps": {m: rng3(v) for m, v in n.items()},
                       "tiles": {s: len(TASKS["planteur"].dataset(s))
                                 for s in ("train", "val", "test")}}
    # BioMassters: months with a frame, per chip
    counts = {"S1": [], "S2": []}
    sizes = {}
    for split in ("train", "val", "test"):
        ds = BiomasstersDataset(split, {"S1": All(), "S2": All()}, tiling=None)
        sizes[split] = len(ds)
        for chip, pos in list(ds._pos.items())[::bio_every]:
            inner = ds._shards.read(pos)
            for m in counts:
                counts[m].append(int((inner["modality"] == m).sum()))
    out["biomassters"] = {"chips": sizes, "steps": {m: rng3(v) for m, v in counts.items()},
                          "chips_counted": len(counts["S1"]),
                          "tiles": {s: len(TASKS["biomassters"].dataset(s))
                                    for s in ("train", "val", "test")}}
    # GEOID-Flood flood tracking (exp.geoidchange.tiletrack): 512 px quadrants with >= 8 passes;
    # passes stored per quadrant and the revisit, for the monthly (train) and dense (test) series
    from exp.geoidchange.tiletrack import MonthlyTrack, Track
    out["geoid"] = {}
    for name, mk in (("monthly", lambda sp: MonthlyTrack(sp, 8)),
                     ("dense", lambda sp: Track(sp, 8, three=True))):
        d = {}
        for sp in ("train", "val", "test"):
            ds = mk(sp)
            d[sp] = len(ds.recs)
            if sp == "test":
                d["scenes_test"] = len({r["key"] for r in ds.recs})
                d["passes"] = rng3([len(r["idx"]) for r in ds.recs])
                d["gap_days"] = float(np.median(np.concatenate([np.diff(r["rday"]) for r in ds.recs])))
        out["geoid"][name] = d
    # YieldSAT
    idx = load_index(str(PACKED))
    raw = [len(f["raw_days"]) for f in idx["fields"].values()]
    ml = [sum(d >= 0 for d in f["ml24_days"]) for f in idx["fields"].values()]
    out["yieldsat"] = {"fields": len(raw), "steps": {"S2_raw": rng3(raw), "S2_ml24_filled": rng3(ml)},
                       "tiles": {s: len(TASKS["yieldsat_raw"].dataset(s))
                                 for s in ("train", "val", "test")}}
    return out


# ---------------------------------------------------------------- table
def table_rows(st: dict) -> list[dict]:
    """One row per modality. used: the models read it. `steps`: timesteps per sample in the
    release; `ours`: what the mixed-resolution model is given."""
    p, b, g, y = st["pastis"], st["biomassters"], st["geoid"], st["yieldsat"]
    pl = st["planteur"]
    tiles = lambda t: f"{t['train']:,} / {t['val']:,} / {t['test']:,}"  # noqa: E731
    return [
        dict(ds="PASTIS-R", task="crop type, 19 classes (mIoU)",
             size=f"{p['patches']:,} patches, 128 px; {tiles(p['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(p['steps']['S2'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means → single dates"),
        dict(mod="Sentinel-1 ascending", bands="VV, VH, VV−VH", used=True,
             steps=f"{span(p['steps']['S1A'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means → single dates"),
        dict(mod="Sentinel-1 descending", bands="VV, VH, VV−VH", used=False,
             steps=f"{span(p['steps']['S1D'])} dates", cadence="irregular, own dates", ours="—"),
        dict(ds="PLANTEUR", task="tropical crop type, 8 classes (mIoU)",
             size=f"{pl['patches']:,} patches, 128 px; {tiles(pl['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(pl['steps']['S2'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means (2019) → single dates"),
        dict(mod="Sentinel-1 RTC, both orbits", bands="VV, VH (linear → dB)", used=True,
             steps=f"{span(pl['steps']['S1'])} dates", cadence="irregular, own dates",
             ours="4 seasons → 12 monthly means (2019) → single dates"),
        dict(mod="Landsat-8 L1 (+ panchromatic)", bands="10 bands at 20 m (+ B8 at 10 m)",
             used=False, steps=f"{span(pl['steps']['LS'])} dates", cadence="irregular, 16-day",
             ours="—"),
        dict(mod="S2 / S1 / Landsat monthly mosaics", bands="as above", used=False,
             steps="12 months", cadence="least-cloudy scene per 30 days, 2019", ours="—"),
        dict(mod="Embeddings (AlphaEarth, Tessera, OlmoEarth)", bands="64–128 dims", used=False,
             steps="static", cadence="annual, 2019", ours="—"),
        dict(ds="BioMassters", task="above-ground biomass, t/ha (RMSE)",
             size=f"{sum(b['chips'].values()):,} chips, 256 px; {tiles(b['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A", bands="10 bands", used=True,
             steps=f"{span(b['steps']['S2'])} months", cadence="monthly composites, Sep–Aug",
             ours="4 seasons → 12 months (no dates in the release)"),
        dict(mod="Sentinel-1 ascending", bands="VV, VH", used=True,
             steps=f"{span(b['steps']['S1'])} months", cadence="monthly composites, Sep–Aug",
             ours="4 seasons → 12 months"),
        dict(mod="Sentinel-1 descending", bands="VV, VH", used=False,
             steps="≤ 12 months", cadence="monthly composites", ours="—"),
        dict(mod="S2 cloud probability", bands="1", used=False,
             steps="as Sentinel-2", cadence="monthly", ours="—"),
        dict(ds="GEOID-Flood", task="flood tracking: dry / permanent water / flood (3-class mIoU and "
             "mean boundary IoU on the labelled date)",
             size=f"{tiles(g['dense'])} quadrants of 512 px (dense series)",
             mod="Sentinel-1 RTC same-orbit series (ours, Planetary Computer)", bands="VV, VH",
             used=True,
             steps=f"monthly (train): {span(g['monthly']['passes'])} passes, every "
                   f"~{g['monthly']['gap_days']:.0f} d; dense (test): {span(g['dense']['passes'])} "
                   f"passes, every ~{g['dense']['gap_days']:.0f} d; windows of 8",
             cadence="passes of the post scene's relative orbit",
             ours="coarse S64 / S128 on every timestep → selected S32 / S64 crops"),
        dict(mod="Sentinel-1 GRD", bands="VV, VH", used=False,
             steps="2 (pre, post)", cadence="event pair",
             ours="— (earlier pre/post baseline only)"),
        dict(mod="Sentinel-1 RTC", bands="VV, VH", used=False, steps="2 (pre, post)",
             cadence="event pair", ours="—"),
        dict(mod="Sentinel-2 L2A", bands="multispectral", used=False, steps="1 (pre-event)",
             cadence="composite", ours="—"),
        dict(mod="Copernicus GLO-30 DEM", bands="1", used=False, steps="static", cadence="—",
             ours="—"),
        dict(ds="YieldSAT", task="crop yield, t/ha (pixel RMSE)",
             size=f"{y['fields']:,} field-seasons; {tiles(y['tiles'])} tiles of 64 px",
             mod="Sentinel-2 L2A, every acquisition", bands="12 bands (+ SCL)", used=True,
             steps=f"{span(y['steps']['S2_raw'])} dates", cadence="irregular, seeding → harvest",
             ours="8 seasons → 24 monthly clear-pixel means → single dates"),
        dict(mod="Sentinel-2 L2A, ML-ready", bands="12 bands", used=False,
             steps=f"24 month slots, {span(y['steps']['S2_ml24_filled'])} filled",
             cadence="one date per month slot", ours="— (fixed-resolution baselines only)"),
        dict(mod="Soil properties", bands="8 layers", used=False, steps="static", cadence="—",
             ours="—"),
        dict(mod="DEM + terrain derivatives", bands="5 layers", used=False, steps="static",
             cadence="—", ours="—"),
        dict(mod="Weather (per field)", bands="4 variables", used=False, steps="daily",
             cadence="daily", ours="—"),
    ]


HEAD = ("Dataset", "Label (metric)", "Samples (train / val / test)", "Modality", "Bands",
        "Timesteps per sample", "Cadence", "Given to the model as")


def write_table(st: dict):
    rows = table_rows(st)
    md = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    tex = [r"\begin{tabular}{lllllll}", r"\toprule",
           r"Dataset & Modality & Bands & Timesteps per sample & Cadence & Given to the model as \\"]
    for r in rows:
        bold = (lambda s: f"**{s}**") if r["used"] else (lambda s: s)
        md.append("| " + " | ".join([
            f"**{r['ds']}**" if "ds" in r else "", r.get("task", ""), r.get("size", ""),
            bold(r["mod"]), r["bands"], r["steps"], r["cadence"], r["ours"]]) + " |")
        esc = lambda s: (s.replace("%", r"\%").replace("→", r"$\rightarrow$")      # noqa: E731
                         .replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("−", "$-$")
                         .replace("–", "--").replace("—", "---"))
        if "ds" in r:
            tex.append(r"\midrule")
            tex.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{r['ds']}}} --- {esc(r['task'])}; "
                       rf"{esc(r['size'])}}} \\")
        m = rf"\textbf{{{esc(r['mod'])}}}" if r["used"] else esc(r["mod"])
        tex.append(f" & {m} & {esc(r['bands'])} & {esc(r['steps'])} & {esc(r['cadence'])} & "
                   f"{esc(r['ours'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    note = ("\n\nBold: the modalities the models read. Timesteps are per sample, counted from "
            "the data (minimum–maximum, median). \"Given to the model as\": the coarse ps16 "
            "level → the finer levels crops are taken from.\n")
    (OUT / "dataset_table.md").write_text("\n".join(md) + note)
    (OUT / "dataset_table.tex").write_text("\n".join(tex) + "\n")
    print("wrote", OUT / "dataset_table.md")


# ---------------------------------------------------------------- examples
def evenly(n: int, k: int) -> list[int]:
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))


def fmt(day) -> str:
    return from_days(int(day)).strftime("%Y-%m-%d")


def pastis_example(pid: str, ncols: int) -> dict:
    ds = PastisDataset("test", {"S2": All(), "S1A": All()}, tiling=None)
    s = ds[ds.index_of(pid)]
    s2, s1 = s.modalities["S2"], s.modalities["S1A"]
    f2 = poster.s2_rgb_series(s2.data.numpy(), [2, 1, 0])
    f1 = poster.s1_rgb_series(s1.data[:, 0].numpy(), s1.data[:, 1].numpy())
    # S2: clear dates only (display test: < 20% of pixels near white after the series stretch)
    good = [t for t in range(len(f2)) if (np.asarray(f2[t]).min(-1) > 0.9).mean() < 0.2]
    i2 = [good[i] for i in evenly(len(good), ncols)]
    i1 = evenly(len(f1), ncols)
    label = s.target.numpy().astype(int)
    present = [c for c in np.unique(label)]
    return dict(
        name="pastis", title=f"PASTIS-R · patch {pid}",
        sub=f"crop type · 10 m · 128 × 128 px · median {ST['pastis']['steps']['S2'][1]} S2 / "
            f"{ST['pastis']['steps']['S1A'][1]} S1 dates",
        rows=[(f"S2\n({len(i2)} of {len(f2)})",
               [(f2[i], fmt(s2.time.start[i])) for i in i2]),
              (f"S1\n({len(i1)} of {len(f1)})",
               [(f1[i], fmt(s1.time.start[i])) for i in i1])],
        label=poster.colorize(label, poster.PASTIS_COLORS), label_title="crop type",
        legend=[Patch(fc=poster.PASTIS_COLORS[c], ec="0.5", lw=0.4, label=CLASS_NAMES[c])
                for c in present])


PLANTEUR_CLEAR_B02 = 1000     # display only: brighter in blue (B02) = cloud / haze
PLANTEUR_LIT_B08 = 1500       # darker in NIR (B08) = cloud shadow (the patches are mostly green)


def planteur_clear(s2: np.ndarray) -> np.ndarray:
    """(T, C, H, W) -> (T,) share of pixels with data, B02 below PLANTEUR_CLEAR_B02 and B08 above
    PLANTEUR_LIT_B08. PLANTEUR ships no cloud mask; this only chooses which dates the figure
    shows (B02 alone let thin cloud and shadowed dates through)."""
    b2, b8 = s2[:, 0], s2[:, 6]
    return ((b2 > 0) & (b2 < PLANTEUR_CLEAR_B02) & (b8 > PLANTEUR_LIT_B08)).reshape(
        len(s2), -1).mean(1)


def planteur_example(pid: str, ncols: int) -> dict:
    PLANTEUR_COLORS = planteur_colors()
    for split in ("test", "val", "train"):
        ds = PlanteurDataset(split, {"S2": All(), "S1": All()}, tiling=None)
        if ds._meta[pid]["Fold"] in PLANTEUR_FOLDS[split]:
            break
    s = ds[ds.index_of(pid)]
    s2, s1 = s.modalities["S2"], s.modalities["S1"]
    year = lambda md: [t for t, d in enumerate(md.time.start.tolist())  # noqa: E731
                       if from_days(d).year == 2019]
    x2 = s2.data.numpy()
    good = [t for t in year(s2) if planteur_clear(x2[t:t + 1])[0] >= 0.9]
    i2 = [good[i] for i in evenly(len(good), ncols)]
    f2 = poster.s2_rgb_series(x2[i2], [2, 1, 0])           # one stretch over the shown dates
    y1 = year(s1)
    i1 = [y1[i] for i in evenly(len(y1), ncols)]
    f1 = poster.s1_rgb_series(s1.data[i1, 0].numpy(), s1.data[i1, 1].numpy(),
                              s1.pixel_valid[i1].numpy())
    label = s.target.numpy().astype(int)
    return dict(
        name="planteur", title=f"PLANTEUR · patch {pid} ({s.meta['island']}, {split})",
        sub=f"crop type · 10 m · 128 × 128 px · median {ST['planteur']['steps']['S2'][1]} S2 / "
            f"{ST['planteur']['steps']['S1'][1]} S1 dates",
        # S2 shown: clear 2019 dates (display only: >= 90% of pixels with B02 < PLANTEUR_CLEAR_B02
        # and B08 > PLANTEUR_LIT_B08)
        rows=[(f"S2\n({len(i2)} of {s2.num_frames})",
               [(f, fmt(s2.time.start[t])) for f, t in zip(f2, i2)]),
              (f"S1\n({len(i1)} of {s1.num_frames})",
               [(f, fmt(s1.time.start[t])) for f, t in zip(f1, i1)])],
        label=poster.colorize(label, PLANTEUR_COLORS), label_title="crop type",
        legend=[Patch(fc=PLANTEUR_COLORS[c], ec="0.5", lw=0.4,
                      label=PLANTEUR_CLASSES[c] if c != 255 else "void (ignored)")
                for c in np.unique(label)])


def biomassters_example(chip: str, ncols: int) -> dict:
    for split in ("test", "val", "train"):
        ds = BiomasstersDataset(split, {"S1": All(), "S2": All()}, tiling=None)
        if chip in ds._pos:
            break
    s = ds[ds.index_of(chip)]
    s2, s1 = s.modalities["S2"], s.modalities["S1"]
    f2 = poster.s2_rgb_series(s2.data.numpy(), [2, 1, 0], mask_fill=True)
    ok = (s1.data[:, :2] != 0).any(1).numpy()     # the ascending pass's own nodata
    f1 = poster.s1_rgb_series(s1.data[:, 0].numpy(), s1.data[:, 1].numpy(), ok)
    mon = [m[:3] for m in MONTHS]
    sh = evenly(12, ncols)
    agbm = s.target.numpy()
    vmax = float(np.percentile(agbm, 99.5))
    return dict(
        name="biomassters", title=f"BioMassters · chip {chip} ({split})",
        sub="above-ground biomass · 10 m · 256 × 256 px · 12 monthly composites",
        rows=[(f"S2\n({len(sh)} of 12)",
               [(f2[i] if bool(s2.frame_valid[i]) else None, mon[i]) for i in sh]),
              (f"S1\n({len(sh)} of 12)",
               [(f1[i] if bool(s1.frame_valid[i]) else None, mon[i]) for i in sh])],
        label=plt.get_cmap("YlGn")(np.clip(agbm / vmax, 0, 1))[..., :3],
        label_title="biomass (t/ha)", cbar=("YlGn", 0, vmax, "t/ha"))


ST = json.loads((OUT / "dataset_stats.json").read_text()) if (OUT / "dataset_stats.json").exists() \
    else None                    # dataset-wide series lengths for the subtitles (refreshed in main)
GEOID_REC = "EMSR275-1-4_q1"     # the flood-tracking paper example (exp.viz.geoid_track_example)


def geoid_example(chip: str | None, ncols: int) -> dict:
    """The data the flood-tracking models read (round O, App. K: exp.geoidchange.tiletrack data=long,
    long2 root): one 512 px quadrant of a GEOID-Flood scene as its DENSE same-orbit S1 series (up to
    24 passes before and after the event scene; models see T = 32), shown as the ncols consecutive
    passes around the labelled date (event at position 16 of 32, as the main-text example). Label:
    GEOID-Flood's 3-class map, which exists for that date only."""
    from exp.geoidchange.tiletrack import LongTrack
    want = chip or GEOID_REC
    LongTrack.root = "data/GEOID-Flood-tileseries-long2/cache"
    ds = LongTrack("test", 32)
    ds.k_post = 16
    i = next(i for i, rr in enumerate(ds.recs) if f"{rr['key']}_q{rr['quad']}" == want)
    x, _, valid, k, y, rday = ds[i][:6]
    k = int(k)
    lo = min(max(0, k - ncols // 2), len(x) - ncols)
    f = poster.gray_series(x[lo:lo + ncols, 0])
    row = [(f[t] if valid[lo + t] else None, "labelled date" if lo + t == k else
            f"{int(round(rday[lo + t])):+d} d") for t in range(ncols)]
    gap = float(np.median(np.concatenate([np.diff(r["rday"]) for r in ds.recs])))
    passes = int(np.median([len(r["idx"]) for r in ds.recs]))
    label = y.astype(int).copy()
    label[label < 0] = 255
    frac = {c: (label == c).mean() for c in GEOID_NAMES}
    return dict(
        name="geoid", title=f"GEOID-Flood S1 series · {want.replace('_q', ' quadrant ')} (test)",
        sub=f"flood tracking · 10 m · 512 × 512 px · dense same-orbit S1, a pass every ~{gap:.0f} d "
            f"(median {passes} passes per series, windows of 32)",
        rows=[(f"S1 VV\n({ncols} of 32)", row)], mark=[(0, k - lo)], mark_text="GEOID-Flood label",
        label=poster.colorize(label, GEOID_COLORS), label_title="3-class flood map",
        legend=[Patch(fc=GEOID_COLORS[c], ec="0.5", lw=0.4, label=GEOID_NAMES[c])
                for c in GEOID_NAMES if frac[c] > 0])


def yieldsat_example(field: str, ncols: int) -> dict:
    idx = load_index(str(PACKED))
    fields = idx["fields"]
    f = fields[field]
    s2 = np.load(PACKED / "s2" / f["country"] / f"{field}.npy", mmap_mode="r")
    scl = np.load(PACKED / "scl" / f["country"] / f"{field}.npy", mmap_mode="r")
    target = np.load(PACKED / "target" / f["country"] / f"{field}.npy")
    clear = np.isin(scl, CLEAR_SCL).reshape(len(scl), -1).mean(1)
    inside = (np.asarray(s2[:, 1:4]) != 0).any(1)               # the field polygon
    # shown: dates whose field pixels are >= 80% clear (SCL vegetation / bare soil), evenly
    # spaced in that list; one stretch over them
    px = max(int(inside.any(0).sum()), 1)
    clear_in = (np.isin(scl, CLEAR_SCL) & inside).reshape(len(scl), -1).sum(1) / px
    good = np.nonzero(clear_in >= 0.8)[0]
    shown = [int(good[i]) for i in evenly(len(good), ncols)]
    rgb = np.asarray(s2[shown][:, [3, 2, 1]]).astype(np.float32)          # (n, 3, H, W)
    lo, hi = np.percentile(rgb.transpose(0, 2, 3, 1)[inside[shown]], [1, 99])   # one for R, G, B
    frames = {t: poster._stretch(x.transpose(1, 2, 0), lo, hi, 0.8) for t, x in zip(shown, rgb)}
    # second row: false colour NIR / red / green (B08, B04, B03), same dates, its own stretch
    fc = np.asarray(s2[shown][:, [7, 3, 2]]).astype(np.float32)
    lo2, hi2 = np.percentile(fc.transpose(0, 2, 3, 1)[inside[shown]], [1, 99], axis=0)   # per band
    frames2 = {t: poster._stretch(x.transpose(1, 2, 0), lo2, hi2, 0.8) for t, x in zip(shown, fc)}
    imgs, imgs2 = [], []
    for t in shown:
        for fr, out in ((frames, imgs), (frames2, imgs2)):
            img = fr[t].copy()
            img[~inside[t]] = 1.0                               # outside the polygon: white
            out.append((img, fmt(f["raw_days"][t])))
    vmin, vmax = np.nanpercentile(target, [1, 99])
    lab = plt.get_cmap("viridis")(np.clip((target - vmin) / (vmax - vmin), 0, 1))[..., :3]
    lab[~np.isfinite(target)] = 1.0
    return dict(
        name="yieldsat", title=f"YieldSAT · {f['country']}, {f['crop']} {f['year']}",
        sub=f"crop yield · 10 m · {f['H']} × {f['W']} px field box · median "
            f"{ST['yieldsat']['steps']['S2_raw'][1]} S2 dates",
        rows=[(f"S2\n({len(shown)} of {len(clear)})", imgs),
              (f"S2 NIR, R, G\n({len(shown)} of {len(clear)})", imgs2)],
        label=lab, label_title="yield", cbar=("viridis", float(vmin), float(vmax), "t/ha"),
        field=field)


# ---------------------------------------------------------------- layout (inches, at print size)
PAGE_W = 6.875                 # full text width; the figure is a full appendix page with ~6 caption lines
LEFT, GAP, DATE_H = 0.8, 0.03, 0.12        # row-name margin, gap between frames, date line above frames
LAB_GAP, LEG_W = 0.12, 1.08                # label panel offset from the frames, legend / colour bar room
TITLE_H, SUB_LINE, BLOCK_GAP = 0.16, 0.115, 0.1
FS = dict(title=8, sub=6.5, row=6.5, date=5, label=6.5, legend=5.3, mark=5.5)


def cell_size(ncols: int) -> float:
    """Frame side: ncols frames + a label panel two frames (+ one date line) wide fill the page."""
    return (PAGE_W - LEFT - ncols * GAP - LAB_GAP - DATE_H - LEG_W) / (ncols + 2)


def sub_lines(ex: dict, ncols: int) -> list[str]:
    import textwrap
    width = int((PAGE_W - LEFT) / (0.0062 * FS["sub"]))         # characters per line
    return textwrap.wrap(ex["sub"], width)


def block_height(ex: dict, cell: float, ncols: int) -> float:
    return (TITLE_H + len(sub_lines(ex, ncols)) * SUB_LINE + 0.06
            + len(ex["rows"]) * (cell + DATE_H) + BLOCK_GAP + (0.09 if ex.get("mark") else 0))


def draw_block(fig, ex: dict, y_top: float, W: float, H: float, cell: float, ncols: int):
    """One dataset: title, wrapped subtitle, its modality rows (time runs left to right), the label
    on the right with its legend / colour bar. Coordinates in inches from the top-left."""
    def at(x, y):
        return x / W, 1 - y / H
    fig.text(*at(LEFT, y_top + TITLE_H / 2), ex["title"], fontsize=FS["title"], weight="bold",
             color=INK, va="center")
    y = y_top + TITLE_H
    for ln in sub_lines(ex, ncols):
        fig.text(*at(LEFT, y + SUB_LINE / 2), ln, fontsize=FS["sub"], color=MUTED, va="center")
        y += SUB_LINE
    y += 0.06
    y_rows = y
    marks = ex.get("mark")
    marks = marks if isinstance(marks, list) else [marks]
    for r, (name, frames) in enumerate(ex["rows"]):
        fig.text(*at(LEFT - 0.05, y + DATE_H + cell / 2), name, ha="right", va="center",
                 fontsize=FS["row"], color=INK, linespacing=1.1)
        for c, (img, date) in enumerate(frames):
            x0 = LEFT + c * (cell + GAP)
            fig.text(*at(x0 + cell / 2, y + DATE_H - 0.02), date, ha="center", va="bottom",
                     fontsize=FS["date"], color=MUTED)
            if img is None:                     # no frame in this slot
                fig.patches.append(plt.Rectangle(
                    at(x0, y + DATE_H + cell), cell / W, cell / H, fill=False, ec=RULE, lw=0.4,
                    ls=(0, (2, 2)), transform=fig.transFigure))
                continue
            ax = fig.add_axes([*at(x0, y + DATE_H + cell), cell / W, cell / H])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            marked = (r, c) in marks
            for sp in ax.spines.values():
                sp.set_linewidth(1.2 if marked else 0.3)
                sp.set_color("#eb6834" if marked else RULE)
            if marked and r == len(ex["rows"]) - 1:
                fig.text(*at(x0 + cell / 2, y + DATE_H + cell + 0.025), ex["mark_text"],
                         ha="center", va="top", fontsize=FS["mark"], color="#eb6834")
        y += cell + DATE_H
    # the label: as tall as the rows together (at most 2 frames + a date line), right of the grid
    rows_h = len(ex["rows"]) * (cell + DATE_H) - DATE_H
    lab = min(rows_h, 2 * cell + DATE_H)
    lx = LEFT + max(len(f) for _, f in ex["rows"]) * (cell + GAP) - GAP + LAB_GAP
    ly = y_rows + DATE_H + (rows_h - lab) / 2
    h, w = ex["label"].shape[:2]
    lw = lab * w / h
    ax = fig.add_axes([*at(lx, ly + lab), lw / W, lab / H])
    ax.imshow(ex["label"], interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3), sp.set_color(RULE)
    fig.text(*at(lx + lw / 2, ly - 0.02), "label: " + ex["label_title"], ha="center",
             va="bottom", fontsize=FS["label"], color=INK)
    kx = lx + lw + 0.06
    if ex.get("legend"):
        fig.legend(handles=ex["legend"], loc="upper left", frameon=False, fontsize=FS["legend"],
                   ncol=1, handlelength=0.9, handleheight=0.9,
                   handletextpad=0.35, columnspacing=0.5, labelspacing=0.18, borderaxespad=0,
                   borderpad=0, bbox_to_anchor=at(kx, ly))
    if ex.get("cbar"):
        cmap, vmin, vmax, unit = ex["cbar"]
        cax = fig.add_axes([*at(kx, ly + lab), 0.06 / W, lab / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), cmap), cax=cax)
        cb.set_label(unit, fontsize=FS["legend"])
        cb.ax.tick_params(labelsize=FS["legend"], length=1.5, width=0.4, pad=1)
        cb.outline.set_linewidth(0.3)


def save_examples(examples: list[dict], ncols: int):
    W, cell = PAGE_W, cell_size(ncols)

    def render(exs, name):
        H = sum(block_height(e, cell, ncols) for e in exs) - BLOCK_GAP + 0.04
        fig = plt.figure(figsize=(W, H), facecolor="white")
        y = 0.02
        for e in exs:
            draw_block(fig, e, y, W, H, cell, ncols)
            y += block_height(e, cell, ncols)
        fig.savefig(OUT.parent / f"{name}.pdf", dpi=400)       # figures/paper/ (Overleaf name)
        plt.close(fig)
        print("wrote", OUT.parent / f"{name}.pdf", f"{W:.2f} x {H:.2f} in, frames {cell:.2f} in")

    render(examples, "14_dataset-overview")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["table", "examples"])
    ap.add_argument("--ncols", type=int, default=8, help="timesteps shown per row")
    # the fixed example of each dataset in Fig 20
    ap.add_argument("--pastis_id", default="40091")
    ap.add_argument("--planteur_id", default="1212")
    ap.add_argument("--biomassters_chip", default="196a1012")
    ap.add_argument("--geoid_chip", default=GEOID_REC)
    ap.add_argument("--yieldsat_field", default="Argentina_DUP1_farm44_field461_wheat_2021")
    ap.add_argument("--bio_every", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only != "examples":
        path = OUT / "dataset_stats.json"
        global ST
        st = ST = stats(a.bio_every)
        path.write_text(json.dumps(st, indent=1))
        write_table(st)
    if a.only != "table":
        ex = [pastis_example(a.pastis_id, a.ncols), planteur_example(a.planteur_id, a.ncols),
              biomassters_example(a.biomassters_chip, a.ncols),
              geoid_example(a.geoid_chip, a.ncols), yieldsat_example(a.yieldsat_field, a.ncols)]
        (OUT / "examples.json").write_text(json.dumps(
            {e["name"]: e["title"] for e in ex}, indent=1))
        save_examples(ex, a.ncols)


if __name__ == "__main__":
    main()
