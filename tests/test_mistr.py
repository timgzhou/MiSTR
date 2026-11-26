"""prepare() on synthetic series (CPU); the model on them if a GPU, FlashAttention and the
Hugging Face model are available."""
import datetime as dt

import numpy as np
import pytest
import torch

from mistr import prepare
from mistr.inputs import MONTHS, TILE


def series(H=70, W=130, seed=0):
    rng = np.random.default_rng(seed)
    d2 = [dt.date(2018, 10, 3) + dt.timedelta(days=10 * i) for i in range(36)]
    d1 = [dt.date(2018, 10, 1) + dt.timedelta(days=6 * i) for i in range(60)]
    s2 = rng.uniform(0, 4000, (len(d2), 10, H, W)).astype(np.float32)
    s1 = rng.uniform(-25, -5, (len(d1), 2, H, W)).astype(np.float32)
    return s2, d2, s1, d1


def test_prepare_tiles_and_months():
    s2, d2, s1, d1 = series()
    x = prepare(s2, d2, s1, d1, "2018-10", device="cpu")
    assert len(x) == 2 * 3 and x.grid == (2, 3) and x.size == (70, 130)
    assert x.months.sentinel2_l2a.shape == (6, TILE, TILE, MONTHS, 12)     # B01 / B09 imputed
    assert x.seasons.sentinel1.shape == (6, TILE, TILE, 4, 2)
    assert x.season[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert x.present.all()
    assert x.months.timestamps[0, 0].tolist() == [1, 9, 2018]             # 1 Oct 2018
    assert x.seasons.timestamps[0, 1, 1] == 1                               # Jan-Mar: February


def test_prepare_missing_month_and_valid_mask():
    s2, d2, s1, d1 = series(64, 64)
    keep = [i for i, d in enumerate(d2) if d.month != 2]                   # no S2 in February
    valid = np.ones((len(keep), 64, 64), bool)
    valid[:, :10] = False                                                   # top rows masked out
    x = prepare(s2[keep], [d2[i] for i in keep], s1, d1, "2018-10", s2_valid=valid, device="cpu")
    assert not x.present[0, 4] and x.present[0].sum() == 11
    assert (x.months.sentinel2_l2a_mask[0, :, :, 4] == 3).all()            # MISSING
    m = x.months.sentinel2_l2a[0, :10, :, 0].float()                       # (rows, cols, bands)
    assert (m == m[:1, :1]).all()                       # no valid pixel: raw 0, the same per band
    assert not (x.months.sentinel2_l2a[0, 10:, :, 0].float() == m[:1, :1]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_predict():
    pytest.importorskip("flash_attn")
    from mistr import MiSTR
    try:
        model = MiSTR.from_pretrained("TimZhou/mistr-olmoearth-tiny-pastis")
    except Exception as e:                                                  # offline
        pytest.skip(f"model not available: {e}")
    s2, d2, s1, d1 = series()
    out = model.predict(prepare(s2, d2, s1, d1, "2018-10"), budget=0.25)
    assert out.logits.shape == (19, 70, 130) and out.classes.shape == (70, 130)
    lo = model.predict(prepare(s2, d2, s1, d1, "2018-10"), budget=0.05)
    assert lo.fine_tokens <= out.fine_tokens
    assert (out.windows[:, 1] < 70).all() and (out.windows[:, 2] < 130).all()
