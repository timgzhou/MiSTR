"""PLANTEUR (Wolters et al. 2026, allenai/planteur-benchmark on Hugging Face): crop-type semantic
segmentation in France's overseas territories (Guadeloupe, Martinique, Guyane, Reunion, Mayotte,
a few Saint-Martin patches) from S2 + S1 time series. Packaged in the PASTIS file layout.

Raw layout (data/PLANTEUR): DATA_S2/S2_<id>.npy (T, 10, 128, 128) uint16 reflectance*1e4,
DATA_S1/S1_<id>.npy (T, 2, 128, 128) float32 LINEAR backscatter (VV, VH; 0 = no observation),
ANNOTATIONS/TARGET_<id>.npy (1, 128, 128) uint8 (8 classes, 255 void),
ANNOTATIONS_GEOTIFF/TARGET_<id>.tif (each patch's exact CRS + transform; five UTM zones),
per-modality acquisition dates in metadata.geojson.

Facts the spec encodes (checked by tests/test_planteur.py):
  - S2 and S1 have their own dates, 40-150 frames each, within 2018-09-01..2019-12-31; S1 is
    one series with both orbit directions in date order (orbits-S1; same-day A + D averaged).
  - S1 is returned in dB (10 log10), the unit OlmoEarth's S1 normalization expects (PASTIS-R's
    S1A is dB too). Its zeros (0.3% of pixels, swath edges) are pixel_valid False, as are S2
    pixels that are 0 in every band (no data); frames with no valid pixel are frame_valid False.
    S2 has no cloud mask: cloudy pixels stay valid.
  - Splits are the release's default spatial-block split: train = Fold 1, val = 4, test = 5
    (1896 / 411 / 443 patches), enumerated in metadata order.
  - Labels are for 2019; the monthly window the tasks use is calendar 2019 (YEAR_2019), the
    window of the paper's monthly mosaics.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio
import torch

from mistr_paper.data.dataset import MTMMDataset
from mistr_paper.data.geometry import Geometry
from mistr_paper.data.sample import ModalityData, MTMMSample
from mistr_paper.data.spatial import Tiling
from mistr_paper.data.spec import DatasetSpec, ModalitySpec, TargetSpec
from mistr_paper.data.temporal import TemporalPolicy
from mistr_paper.data.time import TimeAxis, yyyymmdd_to_days

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "PLANTEUR"
FOLDS = {"train": (1,), "val": (4,), "test": (5,)}
CLASS_NAMES = ("Background", "Meadow", "Fruits, vegetables, flowers", "Orchard", "Sugarcane",
               "Banana", "Pineapple", "Tropical tuber")
VOID = 255
YEAR_2019 = ("2019-01-01", "2019-12-31")
DB_FLOOR = -50.0                  # filler for S1 pixels with no observation (pixel_valid False)


class PlanteurDataset(MTMMDataset):
    SPEC = DatasetSpec(
        name="planteur",
        modalities={
            "S2": ModalitySpec("S2", ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                                      "B11", "B12"),
                               "reflectance*1e4", 10.0, "irregular", (40, 150), "day",
                               pool_reducer="mean", nodata=0.0),
            "S1": ModalitySpec("S1", ("VV", "VH"), "dB", 10.0, "irregular", (40, 150), "day",
                               pool_reducer="mean", nodata=DB_FLOOR),
        },
        target=TargetSpec("segmentation", "class", CLASS_NAMES, ignore_index=VOID),
        native_shape=(128, 128),
        split_sizes={"train": 1896, "val": 411, "test": 443},
        georeferenced=True,
        notes=("S1 converted from linear to dB; S1 mixes ascending and descending orbits",
               "S2 has no cloud mask", "Landsat, monthly mosaics and embeddings not read"))

    def __init__(self, split: str, modalities: dict[str, TemporalPolicy],
                 tiling: Tiling | None = Tiling(64, "col-major"), validate: bool = True,
                 root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        super().__init__(split, modalities, tiling, validate)

    @cached_property
    def _meta(self) -> dict[str, dict]:
        feats = json.load(open(self.root / "metadata.geojson"))["features"]
        return {str(f["properties"]["ID_PATCH"]): f["properties"] for f in feats}

    def _native_ids(self, split):
        return [pid for fold in FOLDS[split] for pid, p in self._meta.items() if p["Fold"] == fold]

    def _geometry(self, pid: str) -> Geometry:
        with rasterio.open(self.root / "ANNOTATIONS_GEOTIFF" / f"TARGET_{pid}.tif") as r:
            return Geometry(str(r.crs), tuple(r.transform)[:6], (r.height, r.width))

    def _load_target(self, pid):
        return torch.from_numpy(np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
                                .astype(np.int64))

    def _load(self, pid, modalities):
        props, geo = self._meta[pid], self._geometry(pid)
        out = {}
        for m in modalities:
            dates = props[f"dates-{m}"]
            days = [yyyymmdd_to_days(dates[k]) for k in sorted(dates, key=int)]
            x = np.load(self.root / f"DATA_{m}" / f"{m}_{pid}.npy")
            if x.shape[0] != len(days):
                raise ValueError(f"planteur/{pid}/{m}: {x.shape[0]} frames, {len(days)} dates")
            if m == "S1":
                ok = (x > 0).all(1)                                           # (T, H, W)
                x = np.where(ok[:, None], 10 * np.log10(np.maximum(x, 1e-10)), DB_FLOOR)
            else:
                ok = (x != 0).any(1)
            out[m] = ModalityData(torch.from_numpy(x.astype(np.float32)),
                                  self.SPEC.modalities[m].bands, TimeAxis.instants(days),
                                  torch.from_numpy(ok.reshape(len(days), -1).any(1)), geo,
                                  pixel_valid=torch.from_numpy(ok))
        return MTMMSample(pid, self.split, out, self._load_target(pid), geo,
                          {"fold": props["Fold"], "island": props["Island"]})


# --- fast tile reads (mistr_paper.core.data) ---------------------------------------------------
FRAME_VALID_CACHE = Path(__file__).resolve().parents[3] / "cache" / "planteur_frame_valid.npz"


def _ok(m: str, x: np.ndarray) -> np.ndarray:
    """(T, H, W) the pixels with an observation, as _load decides them."""
    return (x > 0).all(1) if m == "S1" else (x != 0).any(1)


def build_frame_valid(root: str | Path = DEFAULT_ROOT, out: Path = FRAME_VALID_CACHE) -> Path:
    """Every native patch's per-frame validity ("any valid pixel in the PATCH", as _load) for
    S2 and S1, so a tile can be read without its whole patch (~0.7 MB)."""
    root = Path(root)
    feats = json.load(open(root / "metadata.geojson"))["features"]
    arrs = {}
    for f in feats:
        pid = str(f["properties"]["ID_PATCH"])
        for m in ("S2", "S1"):
            x = np.load(root / f"DATA_{m}" / f"{m}_{pid}.npy", mmap_mode="r")
            arrs[f"{pid}/{m}"] = _ok(m, np.asarray(x)).reshape(x.shape[0], -1).any(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs)
    return out


def _tile_fast(self: PlanteurDataset, i: int) -> MTMMSample:
    """self[i] under All() for every modality, without validation: the per-tile work (S1 dB,
    masks) on the tile's window only, frame validity from the precomputed cache, geometry
    cached: the same sample."""
    if not hasattr(self, "_fv"):
        if not FRAME_VALID_CACHE.exists():
            raise FileNotFoundError(f"{FRAME_VALID_CACHE}: run planteur.build_frame_valid()")
        z = np.load(FRAME_VALID_CACHE)
        self._fv = {k: z[k] for k in z.files}
        self._geo = {}
    pid, off = self._index[i]
    row, col = off if off is not None else (0, 0)
    t = self.tiling.size if self.tiling is not None else self.SPEC.native_shape[0]
    if pid not in self._geo:
        self._geo[pid] = self._geometry(pid)
    geo = self._geo[pid].crop(row, col, t, t)
    props = self._meta[pid]
    out = {}
    for m in self.modalities:
        dates = props[f"dates-{m}"]
        days = [yyyymmdd_to_days(dates[k]) for k in sorted(dates, key=int)]
        # (a sequential read of the whole file, then the window: on Lustre ~4x faster than a
        # memory-mapped window's many small strided reads, 77 vs 315 ms per tile cold)
        x = np.ascontiguousarray(np.load(self.root / f"DATA_{m}" / f"{m}_{pid}.npy")
                                 [:, :, row:row + t, col:col + t])
        ok = _ok(m, x)
        if m == "S1":
            x = np.where(ok[:, None], 10 * np.log10(np.maximum(x, 1e-10)), DB_FLOOR)
        out[m] = ModalityData(torch.from_numpy(x.astype(np.float32)),
                              self.SPEC.modalities[m].bands, TimeAxis.instants(days),
                              torch.from_numpy(self._fv[f"{pid}/{m}"].copy()), geo,
                              pixel_valid=torch.from_numpy(ok))
    tgt = np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy", mmap_mode="r")
    target = torch.from_numpy(np.array(tgt[0, row:row + t, col:col + t]).astype(np.int64))
    sid = Tiling.tile_id(pid, row, col) if off is not None else pid
    return MTMMSample(sid, self.split, out, target, geo,
                      {"fold": props["Fold"], "island": props["Island"]})


PlanteurDataset.tile_fast = _tile_fast
