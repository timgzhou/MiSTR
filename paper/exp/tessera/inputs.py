"""PASTIS-R patches as Tessera time tokens, per pixel, at several temporal resolutions.

Raw data (data/PASTIS-R, mistr_paper.data.datasets.pastis): S2 (T, 10, 128, 128) int16 reflectance
x 1e4 in B02..B12 order, S1A (T, 3, 128, 128) float16 10 log10(power) dB, own dates per
modality. Restricted to the agri year 2018-10-01..2019-09-30 (AGRI_YEAR, as every PASTIS run).

Conversions to Tessera v2's training inputs (third_party/tessera/model.py constants):
  S2   bands reordered to B04 B02 B03 B08 B8A B05 B06 B07 B11 B12; raw units
  S1   Tessera stores (20 log10(amplitude) + 50) x 200 with amplitude = the RTC power value,
       i.e. (2 dB + 50) x 200 in power dB; VV, VH only (drop VV-VH); S1A statistics
  DOY  raw day of year 1..365; in the Oct..Sep agri year every date has its own DOY
  invalid  S2 pixel-frames at the -10000 nodata value (PASTIS May frames, see the
       pastis_clean task); no cloud mask (a B02 > 0.2 mask did not help in the frozen probe)

Temporal levels (a token is one time step of one modality at one pixel):
  raw    every acquisition, its own DOY
  semi   half-month composites (1st-15th, 16th-end), DOY = the 8th / 23rd
  mon    calendar-month composites, DOY = the 15th
  bimon  2-month composites Oct-Nov, Dec-Jan, Feb-Mar, Apr-May, Jun-Jul, Aug-Sep, DOY = the
         1st of the 2nd month (they do not nest in seasons: pairs bimon > mon / semi / raw)
  sea    3-month composites Oct-Dec, Jan-Mar, Apr-Jun, Jul-Sep (Task.season = 3), DOY =
         the 15th of the middle month
  uniK   K evenly spaced acquisitions per modality (Tessera's own linspace subsample)
Composites: S2 per-pixel median of valid frames, S1 mean (the spec reducers). A window with
no frame for the patch has no token; a pixel with no valid frame in a window has an invalid
token there.
Every token also carries the index of its window at each coarser level (Tokens.index: season
0..3, bimon 0..5, month 0..11, half 0..23; -1 where a composite spans several), so a finer
level's tokens can be matched to the coarse window they refine.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from exp.tessera.model import S1A_MEAN, S1A_STD, S2_MEAN, S2_STD

ROOT = Path(__file__).resolve().parents[2] / "data" / "PASTIS-R"
WINDOW = ("20181001", "20190930")
FOLDS = {"train": (1, 2, 3), "val": (4,), "test": (5,)}
S2_ORDER = [2, 0, 1, 6, 7, 3, 4, 5, 8, 9]          # PASTIS B02..B12 -> Tessera order
MONTHS = [(2018, m) for m in (10, 11, 12)] + [(2019, m) for m in range(1, 10)]
NODATA = -10000


def _doy(y, m, d) -> int:
    return dt.date(y, m, d).timetuple().tm_yday


# the window index of a token at each composite level (-1: the token spans several windows)
WINDOW_KEYS = ("season", "bimon", "month", "half")
LEVEL_KEY = {"sea": "season", "bimon": "bimon", "mon": "month", "semi": "half"}
LEVEL_W = {"sea": 4, "bimon": 6, "mon": 12, "semi": 24}     # windows per 12 months


def level_w(level: str, n_months: int = 12) -> int:
    return LEVEL_W[level] * n_months // 12


@dataclass
class Tokens:
    """one modality at one level: x (P, T, C) standardized, doy (T,), valid (P, T), and index
    {season, bimon, month, half} -> (T,) the window each token lies in at that level."""
    x: torch.Tensor
    doy: torch.Tensor
    valid: torch.Tensor
    index: dict

    @property
    def T(self):
        return self.x.shape[1]

    @property
    def month(self):
        return self.index["month"]

    @property
    def season(self):
        return self.index["season"]

    def window(self, level: str) -> torch.Tensor:
        """(T,) index of each token's window at composite level `level` (sea/bimon/mon/semi)."""
        return self.index[LEVEL_KEY[level]]

    def take(self, i):
        return Tokens(self.x[:, i], self.doy[i], self.valid[:, i], {k: v[i] for k, v in self.index.items()})


class Patch:
    """One sample (PASTIS patch / BioMassters tile block) on a device; levels built lazily.

    s2 (P, T2, 10) Tessera band order, raw units; s1 (P, T1, 2) Tessera S1 units mapped to the
    S1A statistics (descending frames are affinely remapped from S1D to S1A space, so one
    standardization serves both orbits); *_valid (P, T); dates: datetime.date per frame;
    s1_orbit (T1,) 0 asc / 1 desc: composites keep the orbits as separate tokens (Tessera
    treats asc + desc as one merged stream of frames); months: the dataset's 12 (year, month)
    of its year, in order (windows are counted from its first month)."""

    def __init__(self, s2, s2_valid, s2_dates, s1, s1_valid, s1_dates, s1_orbit, hw, target, device,
                 months=MONTHS, day0: dt.date | None = None):
        """day0: None = Tessera's raw day of year; a date = days since day0 + 1 (windows longer
        than a year, YieldSAT's 24 months, where calendar days repeat)."""
        self.device, self.hw, self.target, self.months = device, hw, target, list(months)
        self.day0 = day0
        self.s2, self.s2_valid, self.d2 = s2, s2_valid, s2_dates
        self.s1, self.s1_valid, self.d1 = torch.nan_to_num(s1), s1_valid, s1_dates
        self.s1_orbit = np.asarray(s1_orbit)

    @classmethod
    def from_pastis(cls, s2, s2_dates, s1, s1_dates, target, device):
        # S2 (T, 10, H, W) -> (P, T, 10) Tessera order, raw units; nodata mask
        s2 = torch.as_tensor(s2, device=device).float()
        T2, _, H, W = s2.shape
        s2 = s2[:, S2_ORDER].reshape(T2, 10, H * W).permute(2, 0, 1)
        s1 = torch.as_tensor(s1, device=device).float()[:, :2]
        T1 = s1.shape[0]
        s1 = (2 * s1.reshape(T1, 2, H * W).permute(2, 0, 1) + 50) * 200
        return cls(s2, ~(s2 <= NODATA + 1).all(-1), s2_dates, s1, torch.isfinite(s1).all(-1), s1_dates,
                   np.zeros(T1, int), (H, W), target, device)

    # -- raw and windows ------------------------------------------------------------------
    def _day(self, y, m, d) -> int:
        return _doy(y, m, d) if self.day0 is None else (dt.date(y, m, d) - self.day0).days + 1

    def _date_idx(self, dates):
        mon = np.array([self.months.index((d.year, d.month)) for d in dates])
        return mon, mon // 3

    def _date_index(self, dates) -> dict:
        mon = np.array([self.months.index((d.year, d.month)) for d in dates], dtype=np.int64)
        half = 2 * mon + (np.array([d.day for d in dates]) > 15)
        return {"season": mon // 3, "bimon": mon // 2, "month": mon, "half": half}

    def _std(self, mod, x):
        mean, std = (S2_MEAN, S2_STD) if mod == "S2" else (S1A_MEAN, S1A_STD)
        return (x - torch.as_tensor(mean, device=x.device)) / torch.as_tensor(std + 1e-9, device=x.device)

    def _raw(self, mod) -> Tokens:
        x, v, dates = (self.s2, self.s2_valid, self.d2) if mod == "S2" else (self.s1, self.s1_valid, self.d1)
        doy = [self._day(d.year, d.month, d.day) for d in dates]
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)
        return Tokens(self._std(mod, x), t(doy).float(), v,
                      {k: t(a) for k, a in self._date_index(dates).items()})

    def _composite(self, mod, groups) -> Tokens:
        """groups: [(frame indices, doy, {window key: index})] -> one token per non-empty group."""
        x, v = (self.s2, self.s2_valid) if mod == "S2" else (self.s1, self.s1_valid)
        if mod != "S2" and len(set(self.s1_orbit.tolist())) > 1:     # one token per orbit
            groups = [(idx[self.s1_orbit[idx] == o], doy, ind) for idx, doy, ind in groups
                      for o in sorted(set(self.s1_orbit.tolist()))]
        xs, vs, doys, inds = [], [], [], []
        for idx, doy, ind in groups:
            if len(idx) == 0:
                continue
            g = x[:, idx]
            gv = v[:, idx]
            if mod == "S2":
                c = torch.where(gv[..., None], g, torch.nan).nanmedian(1).values
            else:
                c = (g * gv[..., None]).sum(1) / gv.sum(1, keepdim=True).clamp(min=1)
            xs.append(torch.nan_to_num(c)); vs.append(gv.any(1))
            doys.append(doy); inds.append(ind)
        t = lambda a: torch.as_tensor(np.asarray(a), device=self.device)
        if not xs:                              # no frame at all (YieldSAT has no S1)
            P = x.shape[0]
            e = torch.zeros(0, dtype=torch.long, device=self.device)
            return Tokens(x.new_zeros(P, 0, x.shape[-1]), e.float(), v.new_zeros(P, 0),
                          {k: e for k in WINDOW_KEYS})
        return Tokens(self._std(mod, torch.stack(xs, 1)), t(doys).float(), torch.stack(vs, 1),
                      {k: t([d.get(k, -1) for d in inds]) for k in WINDOW_KEYS})

    def level(self, name: str, mod: str) -> Tokens:
        key = (name, mod)
        cache = self.__dict__.setdefault("_levels", {})
        if key in cache:
            return cache[key]
        dates = self.d2 if mod == "S2" else self.d1
        mon, _ = self._date_idx(dates) if dates else (np.zeros(0, int), None)
        day = np.array([d.day for d in dates], dtype=int)
        n = len(self.months)
        if name == "raw":
            tok = self._raw(mod)
        elif name.startswith("uni"):
            k = int(name[3:])
            r = self._raw(mod)
            tok = r.take(torch.as_tensor(np.linspace(0, r.T - 1, min(k, r.T)).astype(int), device=self.device,
                                         dtype=torch.long))
        elif name == "mon":
            tok = self._composite(mod, [(np.where(mon == j)[0], self._day(*self.months[j], 15),
                                         {"month": j, "season": j // 3, "bimon": j // 2}) for j in range(n)])
        elif name == "semi":
            g = []
            for j in range(n):
                for half, (lo, hi, dd) in enumerate(((1, 15, 8), (16, 31, 23))):
                    g.append((np.where((mon == j) & (day >= lo) & (day <= hi))[0], self._day(*self.months[j], dd),
                              {"half": 2 * j + half, "month": j, "season": j // 3, "bimon": j // 2}))
            tok = self._composite(mod, g)
        elif name == "bimon":       # Oct-Nov, Dec-Jan, ..., Aug-Sep; DOY = the 1st of the 2nd month
            tok = self._composite(mod, [(np.where(mon // 2 == b)[0], self._day(*self.months[2 * b + 1], 1),
                                         {"bimon": b}) for b in range(n // 2)])
        elif name == "sea":
            tok = self._composite(mod, [(np.where(mon // 3 == s)[0], self._day(*self.months[3 * s + 1], 15),
                                         {"season": s}) for s in range(n // 3)])
        else:
            raise ValueError(name)
        cache[key] = tok
        return tok


class PastisPatches(torch.utils.data.Dataset):
    """split -> raw numpy arrays of each patch (CPU, for a DataLoader); Patch(...) on GPU."""

    def __init__(self, split: str, root: Path = ROOT, limit: int | None = None):
        feats = json.load(open(root / "metadata.geojson"))["features"]
        # benchmark folds, enumerated fold by fold in metadata order (as PastisDataset)
        props = [f["properties"] for f in feats]
        self.items = [p for fold in FOLDS[split] for p in props if p["Fold"] == fold]
        if limit:
            self.items = self.items[:limit]
        self.root = root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        pid = p["ID_PATCH"]
        out = {"id": int(pid)}
        for m in ("S2", "S1A"):
            dd = [str(v) for _, v in sorted(p[f"dates-{m}"].items(), key=lambda x: int(x[0]))]
            keep = [j for j, d in enumerate(dd) if WINDOW[0] <= d <= WINDOW[1]]
            x = np.load(self.root / f"DATA_{m}" / f"{m}_{pid}.npy", mmap_mode="r")
            out[m] = np.ascontiguousarray(x[keep]).astype(np.float32)
            out[m + "_dates"] = [dt.date(int(dd[j][:4]), int(dd[j][4:6]), int(dd[j][6:])) for j in keep]
        out["target"] = np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].astype(np.int64)
        return out


def to_patch(item, device) -> Patch:
    if item.get("dataset") == "biomassters":
        return bio_patch(item, device)
    if item.get("dataset") == "planteur":
        return planteur_patch(item, device)
    if item.get("dataset") == "yieldsat":
        return yieldsat_patch(item, device)
    return Patch.from_pastis(item["S2"], item["S2_dates"], item["S1A"], item["S1A_dates"], item["target"], device)


# ---- BioMassters ----------------------------------------------------------------------------
# Natively monthly (one S2 + one asc + one desc S1 image per month, Sep..Aug of one unknown
# year, placeholder 2000/2001): levels sea (Sep-Nov, ...), bimon, mon; no semi / raw.
# S2: L2A B02..B12 like PASTIS (same reorder); invalid = frame / pixel invalid (MaskInvalid flat,
# constant-fill dates dropped) or cloud probability >= 50 (0-100).
# S1: dB (10 log10 power), VV/VH asc + desc -> (2 dB + 50) x 200; desc remapped S1D -> S1A
# space. Note: BioMassters S1 is ~5-6 dB lower than PASTIS (VV -16.4 vs -10.9 dB on average).
# Samples: the 4 centre 64 px tiles of each 256 px chip (rows / cols 64 and 128) from the
# biomassters_tile64 store, i.e. the chip's central 128 x 128 px, as one 128 px sample.
BIO_MONTHS = [(2000, m) for m in (9, 10, 11, 12)] + [(2001, m) for m in range(1, 9)]
BIO_CENTRE = ((64, 64), (64, 128), (128, 64), (128, 128))
CLOUD_MAX = 50


class BioChips(torch.utils.data.Dataset):
    def __init__(self, split: str, limit: int | None = None):
        from exp.common.tasks import Biomassters
        from mistr_paper.data import All, MaskInvalid
        self.ds = Biomassters().open(split, {"S1": All(), "S2": MaskInvalid(flat=True)})
        pos = {sid: i for i, sid in enumerate(self.ds.store.ids)}
        chips = sorted({sid.split("@")[0] for sid in self.ds.store.ids})
        self.items = [(c, [pos[f"{c}@r{r}c{cc}"] for r, cc in BIO_CENTRE]) for c in chips
                      if all(f"{c}@r{r}c{cc}" in pos for r, cc in BIO_CENTRE)]
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        chip, tiles = self.items[i]
        ss = [self.ds[j] for j in tiles]

        def stitch(get):            # 4 tiles (.., 64, 64) -> (.., 128, 128): r64c64 r64c128 / r128c64 r128c128
            a = [get(t) for t in ss]
            return np.concatenate([np.concatenate(a[:2], -1), np.concatenate(a[2:], -1)], -2)
        s2, s1 = ss[0].modalities["S2"], ss[0].modalities["S1"]
        return {"dataset": "biomassters", "id": chip,
                "S2": stitch(lambda t: t.modalities["S2"].data.numpy()),
                "S2_valid": stitch(lambda t: (t.modalities["S2"].pixel_valid
                                               & t.modalities["S2"].frame_valid[:, None, None]).numpy()
                                   & (t.modalities["S2"].aux["cloud_prob"].numpy() < CLOUD_MAX)),
                "S1": stitch(lambda t: t.modalities["S1"].data.numpy()),
                "S1_valid": stitch(lambda t: (t.modalities["S1"].pixel_valid
                                               & t.modalities["S1"].frame_valid[:, None, None]).numpy()),
                "S2_start": s2.time.start.numpy(), "S1_start": s1.time.start.numpy(),
                "target": stitch(lambda t: t.target.numpy()).astype(np.float32)}


def _day_to_date(d: int) -> dt.date:
    return dt.date(1970, 1, 1) + dt.timedelta(days=int(d) + 14)      # mid-month of a month slot


def bio_patch(item, device) -> Patch:
    from exp.tessera.model import S1A_MEAN, S1A_STD
    import exp.tessera.model as tmod
    S1D_MEAN, S1D_STD = tmod._tm.S1D_BAND_MEAN, tmod._tm.S1D_BAND_STD
    t = lambda a: torch.as_tensor(np.asarray(a), device=device)
    s2 = t(item["S2"]).float()                                          # (12, 10, H, W)
    T2, _, H, W = s2.shape
    s2 = s2[:, S2_ORDER].reshape(T2, 10, H * W).permute(2, 0, 1)
    v2 = t(item["S2_valid"]).reshape(T2, H * W).T
    s1 = t(item["S1"]).float()                                          # (12, 4, H, W) dB
    v1 = t(item["S1_valid"]).reshape(12, H * W).T
    asc = (2 * s1[:, :2].reshape(12, 2, H * W).permute(2, 0, 1) + 50) * 200
    desc = (2 * s1[:, 2:].reshape(12, 2, H * W).permute(2, 0, 1) + 50) * 200
    desc = (desc - t(S1D_MEAN)) / t(S1D_STD) * t(S1A_STD) + t(S1A_MEAN)
    d2 = [_day_to_date(d) for d in item["S2_start"]]
    d1 = [_day_to_date(d) for d in item["S1_start"]]
    return Patch(s2, v2, d2, torch.cat([asc, desc], 1), torch.cat([v1, v1], 1), d1 + d1,
                 np.r_[np.zeros(12, int), np.ones(12, int)], (H, W), item["target"], device, BIO_MONTHS)


# ---- PLANTEUR -------------------------------------------------------------------------------
# PASTIS file layout (mistr_paper.data.datasets.planteur): S2 (T, 10, 128, 128) uint16 B02..B12, no
# cloud mask (all-zero = no data); S1 (T, 2, 128, 128) LINEAR backscatter, 0 = no observation,
# asc + desc in one date-ordered series (orbits-S1 per frame) -> dB = 10 log10 -> Tessera units,
# desc remapped to S1A stats; calendar 2019 (the labels' year, the tasks' window); folds
# train 1 / val 4 / test 5; target uint8, 255 void.
PLANTEUR_ROOT = Path(__file__).resolve().parents[2] / "data" / "PLANTEUR"
PLANTEUR_MONTHS = [(2019, m) for m in range(1, 13)]
PLANTEUR_FOLDS = {"train": ("1",), "val": ("4",), "test": ("5",)}


class PlanteurPatches(torch.utils.data.Dataset):
    def __init__(self, split: str, limit: int | None = None, root: Path = PLANTEUR_ROOT):
        props = [f["properties"] for f in json.load(open(root / "metadata.geojson"))["features"]]
        self.items = [p for fold in PLANTEUR_FOLDS[split] for p in props if str(p["Fold"]) == fold]
        if limit:
            self.items = self.items[:limit]
        self.root = root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        pid = p["ID_PATCH"]
        out = {"dataset": "planteur", "id": str(pid)}
        for m in ("S2", "S1"):
            dd = [str(v) for _, v in sorted(p[f"dates-{m}"].items(), key=lambda x: int(x[0]))]
            keep = [j for j, d in enumerate(dd) if "20190101" <= d <= "20191231"]
            x = np.load(self.root / f"DATA_{m}" / f"{m}_{pid}.npy", mmap_mode="r")
            out[m] = np.ascontiguousarray(x[keep]).astype(np.float32)
            out[m + "_dates"] = [dt.date(int(dd[j][:4]), int(dd[j][4:6]), int(dd[j][6:])) for j in keep]
            if m == "S1":
                orb = [v for _, v in sorted(p["orbits-S1"].items(), key=lambda x: int(x[0]))]
                out["S1_orbit"] = np.array([orb[j] == "D" for j in keep], dtype=np.int64)
        out["target"] = np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].astype(np.int64)
        return out


def _s1_desc_to_asc(x, device):
    import exp.tessera.model as tmod
    t = lambda a: torch.as_tensor(a, device=device)
    return (x - t(tmod._tm.S1D_BAND_MEAN)) / t(tmod._tm.S1D_BAND_STD) * t(S1A_STD) + t(S1A_MEAN)


def planteur_patch(item, device) -> Patch:
    s2 = torch.as_tensor(item["S2"], device=device).float()
    T2, _, H, W = s2.shape
    s2 = s2[:, S2_ORDER].reshape(T2, 10, H * W).permute(2, 0, 1)
    s1 = torch.as_tensor(item["S1"], device=device).float()               # (T1, 2, H, W) linear
    T1 = s1.shape[0]
    s1 = s1.reshape(T1, 2, H * W).permute(2, 0, 1)
    v1 = (s1 > 0).all(-1)
    s1 = (2 * 10 * torch.log10(s1.clamp(min=1e-6)) + 50) * 200
    orb = torch.as_tensor(item["S1_orbit"], device=device).bool()
    s1 = torch.where(orb[None, :, None], _s1_desc_to_asc(s1, device), s1)
    return Patch(s2, (s2 != 0).any(-1), item["S2_dates"], s1, v1, item["S1_dates"], item["S1_orbit"],
                 (H, W), item["target"], device, PLANTEUR_MONTHS)


# ---- YieldSAT -------------------------------------------------------------------------------
# The raw release (mistr_paper.data.datasets.yieldsat.YieldSatDataset, every acquisition): 64 px tiles
# of a field's box, S2 only (12 bands -> Tessera's 10 by name), valid = SCL 4 / 5 (pixel_valid);
# NO S1 (Tessera's pool returns zeros for an empty modality, and so do we). Window: the 24
# calendar months of the two years ending with the year of the last frame (PoolYears(2), as
# yieldsat_raw), so levels have 8 seasons / 12 bimon / 24 months / 48 half-months; the day
# input is days since 1 Jan of the first year + 1 (1..730: calendar days repeat over 2 years).
# Target: t/ha, NaN = unlabelled.
TESS_BANDS = ("B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12")


class YieldTiles(torch.utils.data.Dataset):
    def __init__(self, split: str, limit: int | None = None):
        from mistr_paper.data import All
        from mistr_paper.data.datasets.yieldsat import YieldSatDataset
        self.ds = YieldSatDataset(split, {"S2": All()})
        self.n = min(len(self.ds), limit) if limit else len(self.ds)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        smp = self.ds[i]
        m = smp.modalities["S2"]
        sel = [m.bands.index(b) for b in TESS_BANDS]
        dates = [dt.date(1970, 1, 1) + dt.timedelta(days=int(d)) for d in m.time.start]
        return {"dataset": "yieldsat", "id": smp.id, "S2": m.data[:, sel].numpy().astype(np.float32),
                "S2_valid": (m.pixel_valid & m.frame_valid[:, None, None]).numpy(), "S2_dates": dates,
                "target": smp.target.numpy().astype(np.float32)}


def yieldsat_patch(item, device) -> Patch:
    s2 = torch.as_tensor(item["S2"], device=device).float()               # Tessera band order already
    T2, _, H, W = s2.shape
    s2 = s2.reshape(T2, 10, H * W).permute(2, 0, 1)
    v2 = torch.as_tensor(item["S2_valid"], device=device).reshape(T2, H * W).T
    last = item["S2_dates"][-1].year
    months = [(y, mm) for y in (last - 1, last) for mm in range(1, 13)]
    P = H * W
    return Patch(s2, v2, item["S2_dates"], s2.new_zeros(P, 0, 2), v2.new_zeros(P, 0), [], np.zeros(0, int),
                 (H, W), item["target"], device, months, day0=dt.date(last - 1, 1, 1))


DATASETS = {"pastis": {"items": lambda split, limit=None: PastisPatches(split, limit=limit), "hw": 128,
                       "n_months": 12, "levels": ("sea", "bimon", "mon", "semi", "raw")},
            "planteur": {"items": lambda split, limit=None: PlanteurPatches(split, limit=limit), "hw": 128,
                         "n_months": 12, "levels": ("sea", "bimon", "mon", "semi", "raw")},
            "biomassters": {"items": lambda split, limit=None: BioChips(split, limit=limit), "hw": 128,
                            "n_months": 12, "levels": ("sea", "bimon", "mon")},
            "yieldsat": {"items": lambda split, limit=None: YieldTiles(split, limit=limit), "hw": 64,
                         "n_months": 24, "levels": ("sea", "bimon", "mon", "semi", "raw")}}
