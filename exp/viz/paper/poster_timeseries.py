"""Poster figures: one sample per dataset, its full multimodal time series (date + image per
frame) and its label.

    (a helper of exp.viz.paper.datasets, which renders Fig 14 with it)

Standalone, writes to dataset_visualization/poster/:
    <dataset>_timeseries.{png,pdf}   overview figure (300 dpi PNG; PDF with embedded rasters)
    <dataset>_frames/                every frame and the label as its own PNG, nearest-neighbour
                                     upscaled to >= 1024 px, for laying out on the poster by hand

Rendering choices (the same across datasets):
  - S2 true colour = B04/B03/B02, BioMassters also gets NIR false colour B08/B04/B03.
  - S1 composite = R: VV, G: VH, B: VV-VH (dB). Water is dark, vegetation green, urban pink.
  - Stretches are fixed per SERIES, not per frame, so brightness changes over time are real.
    S2 bounds are medians of per-frame percentiles, so a few cloudy frames do not wash out the rest.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# PASTIS's own colour scheme (garnot's utae-paps repo): tab20, black background, white void.
_tab = plt.get_cmap("tab20").colors
PASTIS_COLORS = [(0, 0, 0)] + [_tab[i] for i in range(1, 19)] + [(1, 1, 1)]


# ---------------------------------------------------------------- rendering helpers
def _stretch(x, lo, hi, gamma=1.0):
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1) ** gamma


NODATA_GRAY = 0.82      # nodata pixels in frames that are shown
MIN_VALID = 0.5         # frames with less valid coverage are invalid dates: hidden (None)


def _finish(imgs, m):
    out = []
    for img, mm in zip(imgs, m):
        if mm.mean() < MIN_VALID:
            out.append(None)
            continue
        img[~mm] = NODATA_GRAY
        out.append(img)
    return out


def s2_valid(frames, mask_fill=False):
    """(T, C, H, W) -> (T, H, W): not all-zero; with mask_fill also not inside a constant fill
    block (BioMassters pads swath edges with blocks of one value in every band: a pixel equal to
    a horizontal AND a vertical neighbour in all bands; real imagery essentially never is)."""
    m = (frames != 0).any(1)
    if mask_fill:
        eq = lambda a, b: (a == b).all(1)
        h, v = np.zeros_like(m), np.zeros_like(m)
        e = eq(frames[..., :, 1:], frames[..., :, :-1])
        h[..., :, 1:] |= e; h[..., :, :-1] |= e
        e = eq(frames[..., 1:, :], frames[..., :-1, :])
        v[..., 1:, :] |= e; v[..., :-1, :] |= e
        m &= ~(h & v)
    return m


def s2_rgb_series(frames, idx, gamma=0.85, mask_fill=False):
    """frames (T, C, H, W) reflectance*1e4 -> list of (H, W, 3) in [0, 1] with one stretch.
    Invalid pixels (see s2_valid) are grey; bounds come from frames >= 50 % valid."""
    x = frames[:, idx].astype(np.float32)                       # (T, 3, H, W)
    m = s2_valid(frames, mask_fill)                              # (T, H, W)
    good = [t for t in range(len(x)) if m[t].mean() >= 0.5]
    lo = np.median([np.percentile(x[t][:, m[t]], 1, axis=1) for t in good], axis=0)
    hi = np.median([np.percentile(x[t][:, m[t]], 99, axis=1) for t in good], axis=0)
    return _finish([_stretch(f.transpose(1, 2, 0), lo, hi, gamma) for f in x], m)


def s1_rgb_series(vv, vh, valid=None):
    """vv, vh (T, H, W) dB -> R=VV, G=VH, B=VV-VH, one 2-98 % stretch over valid pixels."""
    chans = np.stack([vv, vh, vv - vh], 1).astype(np.float32)   # (T, 3, H, W)
    m = np.ones(vv.shape, bool) if valid is None else valid
    lo = np.array([np.percentile(chans[:, c][m], 2) for c in range(3)])
    hi = np.array([np.percentile(chans[:, c][m], 98) for c in range(3)])
    return _finish([_stretch(f.transpose(1, 2, 0), lo, hi) for f in chans], m)


def gray_series(x, valid=None):
    m = np.ones(x.shape, bool) if valid is None else valid
    lo, hi = np.percentile(x[m], [2, 98])
    return _finish([np.repeat(_stretch(f, lo, hi)[..., None], 3, -1) for f in x], m)


def colorize(label, colors: dict | list):
    lut = colors if isinstance(colors, dict) else dict(enumerate(colors))
    out = np.zeros(label.shape + (3,), np.float32)
    for k, c in lut.items():
        out[label == k] = c
    return out
