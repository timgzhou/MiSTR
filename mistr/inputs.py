"""Raw Sentinel-2 / Sentinel-1 time series -> MiSTR's inputs.

    x = prepare(s2, s2_dates, s1, s1_dates, start="2018-10")

Each modality's acquisitions are averaged per calendar month over a 12-month window (invalid
frames and pixels left out; a month without any valid pixel is marked missing), the months are
averaged into four 3-month seasons, both are normalized with OlmoEarth's pretraining statistics,
and the image is cut into 64 x 64 px tiles (padded at the bottom / right edge to a multiple of
64). The model reads the seasons as coarse tokens and cuts its fine tokens from the months.
This is what the paper's training pipeline does (mistr.core.data.CompositeOnGPU in paper/).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

TILE = 64
MONTHS, SEASON = 12, 3
# the bands prepare() reads, in this order (Sentinel-2 L2A as in PASTIS: surface reflectance x 1e4,
# no BOA offset; Sentinel-1 backscatter in dB)
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S1_BANDS = ("VV", "VH")
S2_IMPUTE = {"B01": "B02", "B09": "B8A"}      # OlmoEarth bands the input lacks: copied neighbours


@dataclass
class Inputs:
    """A batch of 64 px tiles, as the model reads them.
    months, seasons   MaskedOlmoEarthSample: (B, 64, 64, T, bands) normalized, bf16, + masks
    season            (B, 12) long: each month's season (0..3)
    present           (B, 12) bool: every modality has data that month
    grid, size        tiles per column / row of the image, and its size before padding"""
    months: MaskedOlmoEarthSample
    seasons: MaskedOlmoEarthSample
    season: torch.Tensor
    present: torch.Tensor
    grid: tuple[int, int] = (1, 1)
    size: tuple[int, int] = (TILE, TILE)

    def __len__(self) -> int:
        return self.season.shape[0]

    def tiles(self, i: int, j: int) -> "Inputs":
        """Tiles i:j (a batch of them)."""
        cut = lambda s: s._replace(**{k: v[i:j] for k, v in s.as_dict().items() if v is not None})  # noqa: E731
        return Inputs(cut(self.months), cut(self.seasons), self.season[i:j], self.present[i:j])

    def to(self, device) -> "Inputs":
        mv = lambda s: s._replace(**{k: v.to(device) for k, v in s.as_dict().items() if v is not None})  # noqa: E731
        return Inputs(mv(self.months), mv(self.seasons), self.season.to(device),
                      self.present.to(device), self.grid, self.size)


def _date(d) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(np.datetime64(d, "D")))


def _month_starts(start) -> list[dt.date]:
    s = _date(f"{start}-01" if len(str(start)) == 7 else start)
    return [dt.date(s.year + (s.month - 1 + i) // 12, (s.month - 1 + i) % 12 + 1, 1)
            for i in range(MONTHS + 1)]


def _stamp(d: dt.date) -> list[int]:
    return [d.day, d.month - 1, d.year]           # OlmoEarth's timestamp: day, month 0-11, year


def _masked_mean(x, valid, onehot):
    """x (F, C, H, W), valid (F, H, W), onehot (T, F) -> (T, C, H, W) mean of the valid entries
    per step (0 where none), (T, H, W) any valid."""
    w, v = onehot.float(), valid.float()
    cnt = torch.einsum("tf,fhw->thw", w, v)
    s = torch.einsum("tf,fchw->tchw", w, x * v[:, None])
    mean = torch.where(cnt[:, None] > 0, s / cnt.clamp(min=1)[:, None], torch.zeros((), device=x.device))
    return mean, cnt > 0


class _Normalize:
    """OlmoEarth's pretraining ("computed") normalization of one modality, input bands mapped to
    its band order (bands it lacks imputed)."""

    def __init__(self, name: str, bands: tuple[str, ...], impute: dict[str, str], device):
        om = Modality.get(name)
        norm = Normalizer(Strategy.COMPUTED)
        cfg, k = norm.norm_config[om.name], norm.std_multiplier
        have = {b.upper(): i for i, b in enumerate(bands)}
        self.idx = torch.tensor([have[impute.get(b, b).upper()] for b in om.band_order], device=device)
        mean = torch.tensor([cfg[b]["mean"] for b in om.band_order], dtype=torch.float64)
        std = torch.tensor([cfg[b]["std"] for b in om.band_order], dtype=torch.float64)
        lo, span = mean - k * std, 2 * k * std
        self.lo, self.inv, self.name = lo.float().to(device), (1 / span).float().to(device), om.name

    def __call__(self, x):          # (T, C, H, W) -> (H, W, T, C')
        y = (x[:, self.idx] - self.lo[:, None, None]) * self.inv[:, None, None]
        return y.permute(2, 3, 0, 1)


def _composite(frames, dates, valid, starts, device):
    """Frames (F, C, H, W) on dates -> monthly means (12, C, H, W), month valid (12,), season means
    (4, C, H, W), season valid (4,)."""
    x = torch.as_tensor(np.asarray(frames, np.float32), device=device)
    F_ = x.shape[0]
    if len(dates) != F_:
        raise ValueError(f"{F_} frames but {len(dates)} dates")
    pv = (torch.ones(x.shape[:1] + x.shape[2:], dtype=torch.bool, device=device) if valid is None
          else torch.as_tensor(np.asarray(valid, bool), device=device))
    pv = pv & torch.isfinite(x).all(1)
    x = torch.nan_to_num(x)
    month = torch.tensor([sum(_date(d) >= s for s in starts) - 1 for d in dates], device=device)
    onehot = month[None, :] == torch.arange(MONTHS, device=device)[:, None]       # (12, F)
    mx, mpv = _masked_mean(x, pv, onehot)
    mfv = mpv.flatten(1).any(1)
    g = torch.arange(MONTHS, device=device)[None, :] // SEASON == torch.arange(MONTHS // SEASON, device=device)[:, None]
    sx, _ = _masked_mean(mx, mfv[:, None, None] & mpv, g)
    return mx, mfv, sx, mfv.view(-1, SEASON).any(1)


def _tiles(a: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(H, W, ...) -> (n tiles, 64, 64, ...), row-major, the image zero-padded at the bottom and
    right to whole tiles."""
    nh, nw = -(-H // TILE), -(-W // TILE)
    pad = a.new_zeros((nh * TILE, nw * TILE) + a.shape[2:])
    pad[:H, :W] = a
    return pad.view(nh, TILE, nw, TILE, *a.shape[2:]).transpose(1, 2).reshape(nh * nw, TILE, TILE, *a.shape[2:])


@torch.no_grad()
def prepare(s2, s2_dates, s1, s1_dates, start, s2_valid=None, s1_valid=None,
            device="cuda") -> Inputs:
    """One image's time series -> its 64 px tiles as MiSTR inputs.

    s2          (F2, 10, H, W) Sentinel-2 L2A, bands S2_BANDS, reflectance x 1e4
    s1          (F1, 2, H, W) Sentinel-1 VV, VH backscatter in dB (the model was trained on
                ascending passes)
    *_dates     one date per frame (datetime.date, "YYYY-MM-DD" or numpy datetime64)
    start       the first of the 12 months ("2018-10"; frames outside the 12 months are ignored)
    *_valid     optional (F, H, W) bool: False at pixels to leave out of the monthly means
                (nodata, clouds). Non-finite values are always left out. The released model was
                trained on PASTIS's frames as given (no cloud mask; its -10000 nodata dates
                included), so the default reproduces its training inputs.
    """
    starts = _month_starts(start)
    s2 = np.asarray(s2, np.float32)
    if s2.ndim != 4 or s2.shape[1] != len(S2_BANDS):
        raise ValueError(f"s2: want (frames, {len(S2_BANDS)}, H, W), got {s2.shape}")
    s1 = np.asarray(s1, np.float32)
    if s1.ndim != 4 or s1.shape[1] != len(S1_BANDS):
        raise ValueError(f"s1: want (frames, {len(S1_BANDS)}, H, W), got {s1.shape}")
    H, W = s2.shape[2:]
    if s1.shape[2:] != (H, W):
        raise ValueError(f"s1 is {s1.shape[2:]} px, s2 {(H, W)}")
    parts = {"sentinel2_l2a": (s2, s2_dates, s2_valid, S2_BANDS, S2_IMPUTE),
             "sentinel1": (s1, s1_dates, s1_valid, S1_BANDS, {})}
    ts_m = torch.tensor([_stamp(d) for d in starts[:MONTHS]])
    ts_s = []
    for i in range(0, MONTHS, SEASON):          # a season's date: the middle day of its 3 months
        a, b = starts[i], starts[i + SEASON] - dt.timedelta(days=1)
        ts_s.append(_stamp(a + dt.timedelta(days=((b - a).days) // 2)))
    ts_s = torch.tensor(ts_s)
    months, seasons, every, s_any = {}, {}, None, None
    for name, (x, dates, valid, bands, impute) in parts.items():
        norm = _Normalize(name, bands, impute, device)
        mx, mfv, sx, sfv = _composite(x, dates, valid, starts, device)
        for out, v, fv in ((months, mx, mfv), (seasons, sx, sfv)):
            y = _tiles(norm(v), H, W).to(torch.bfloat16)                         # (n, 64, 64, T, C)
            mask = torch.full_like(y, MaskValue.ONLINE_ENCODER.value)
            mask.masked_fill_((~fv)[None, None, None, :, None], MaskValue.MISSING.value)
            out[name], out[MaskedOlmoEarthSample.get_masked_modality_name(name)] = y, mask
        every = mfv if every is None else every & mfv
        s_any = sfv if s_any is None else s_any | sfv
    n = next(iter(months.values())).shape[0]
    season = (torch.arange(MONTHS, device=device) // SEASON).expand(n, -1).contiguous()
    present = (every & s_any[season[0]]).expand(n, -1).contiguous()
    return Inputs(MaskedOlmoEarthSample(timestamps=ts_m.expand(n, -1, -1).to(device), **months),
                  MaskedOlmoEarthSample(timestamps=ts_s.expand(n, -1, -1).to(device), **seasons),
                  season, present, (-(-H // TILE), -(-W // TILE)), (H, W))
