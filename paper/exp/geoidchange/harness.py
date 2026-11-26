"""Smoke-test harness for GEOID change: one model (OlmoEarth tiny + MixedResOE, pair-diff head),
sequences of frames, per-pair labels, one evaluation (where: flood IoU; when: date error).

Sequence modes (cfg.mode):
  bi       GEOID's own (pre, post) GRD pair, label on it
  pair     RTC: two frames. Train = (ref, post) with the label + (neg_p) a consecutive pair
           entirely before the event with an all-zero map. Eval = every consecutive pair, alone
  series   RTC: all dates of the window in one sequence, the same pair labels (ignored pairs
           after post). Eval = the window's sequence
  month    as series with one frame per calendar month (cfg.pool: mean / median / min / max)
  ctx      pair + the K dates before it in one sequence (only the last pair decoded)
"""

import numpy as np
import torch
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from exp.common import olmo_bootstrap
olmo_bootstrap.apply()
from mistr_paper.encoders.olmoearth import OlmoEarthInput, _timestamps  # noqa: E402
OUT = "results/geoidchange"
INP = OlmoEarthInput({"S1": "sentinel1"})
BAND = INP._band_index("S1", ("VV", "VH"))


def to_sample(x, days, valid):
    arr = np.transpose(x[:, :, BAND], (0, 3, 4, 1, 2))                    # (B, H, W, T, C)
    s1 = torch.from_numpy(INP.normalizer.normalize(INP.olmo["S1"], arr)).float()
    mask = torch.full_like(s1, MaskValue.ONLINE_ENCODER.value)
    b, t = np.nonzero(~valid)
    mask[b, :, :, t] = MaskValue.MISSING.value
    return MaskedOlmoEarthSample(sentinel1=s1, sentinel1_mask=mask,
                                 timestamps=_timestamps(torch.from_numpy(days)))


class IoU:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, p, y):
        ok = y >= 0
        self.tp += int(((p == 1) & (y == 1) & ok).sum())
        self.fp += int(((p == 1) & (y == 0) & ok).sum())
        self.fn += int(((p == 0) & (y == 1) & ok).sum())

    def get(self):
        return self.tp / max(self.tp + self.fp + self.fn, 1)
