"""Published methods compared with MiSTR, each behind RunConfig.baseline and trained, evaluated
and traced by exp.mistr.train like any mixed-resolution run (docs/baselines.md):

    upsample   a fixed model's final tokens through AnyUp / Upsample Anything / bilinear
    tldr       vid-TLDR token merging inside a trained fixed model, training-free
    apt        APT adaptive patches
"""
from mistr.baselines.apt import AptModel, AptSelector
from mistr.baselines.base import NoCrops
from mistr.baselines.upsample import UpsampleModel
from mistr.baselines.vid_tldr import TldrModel
from mistr.core.model import MiSTR
from mistr.core.selection import build_selector

MODELS = {"upsample": UpsampleModel, "tldr": TldrModel, "apt": AptModel}


def build_model(task, cfg, pretrained: bool = True) -> MiSTR:
    """cfg: RunConfig."""
    if cfg.baseline is None:
        return MiSTR(task, cfg.model, pretrained)
    if task.per_step:
        raise ValueError("baselines: one map per time series (not per-step tasks)")
    return MODELS[cfg.baseline.method](task, cfg, pretrained)


def build_selector_for(task, cfg, model, example):
    if cfg.fixed:
        return NoCrops()
    if cfg.baseline is not None:
        return AptSelector(task, cfg, model) if cfg.baseline.method == "apt" else NoCrops()
    if task.per_step:
        raise ValueError("per-step tasks are not supported")
    return build_selector(cfg.select, cfg.model.window_px, model)
