"""MiSTR: adaptively mixing spatiotemporal resolution for satellite image time series.

    from mistr import MiSTR, prepare
    model = MiSTR.from_pretrained("TimZhou/mistr-olmoearth-tiny-pastis")
    out = model.predict(prepare(s2, s2_dates, s1, s1_dates, start="2018-10"), budget=0.5)
"""
from mistr.inputs import S1_BANDS, S2_BANDS, Inputs, prepare
from mistr.model import MiSTR, Prediction

__all__ = ["MiSTR", "Prediction", "prepare", "Inputs", "S2_BANDS", "S1_BANDS"]
__version__ = "0.1.0"
