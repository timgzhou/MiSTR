"""Per-dataset task specs for the frozen Tessera pipeline (embed / pooltune / head).

    pastis       19-class segmentation, void 19 ignored; SegMeter mIoU; uncertainty = entropy
    planteur     8-class segmentation, void 255 ignored; SegMeter mIoU; uncertainty = entropy
    yieldsat     pixel yield regression (t/ha; NaN = unlabelled, skipped by loss and meter),
                 standardized by yieldsat_raw's train-pixel mean / std; pooled pixel RMSE
                 (the yieldsat_raw task's primary); heads as biomassters
    biomassters  AGBM regression (t/ha), standardized as exp.common.tasks.Biomassters; the
                 head predicts (mean, log-variance): mean by MSE (exactly the usual loss), the
                 log-variance by Gaussian NLL with the mean detached, so it never changes the
                 mean; RegMeter RMSE; uncertainty = predicted log-variance (the per-pixel
                 uncertainty the fractional window budgets read; OlmoEarth MiSTR used the
                 variance of its season tokens' predictions instead, which Tessera lacks)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.common.paths import FEATURES

ROOT = FEATURES / "tessera"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "tessera"


class _MaskedReg:
    """RegMeter over the labelled (finite-target) pixels only (YieldSAT: NaN = unlabelled)."""

    def __init__(self, scale):
        from exp.common.metrics import RegMeter
        self.m = RegMeter(scale)

    def update(self, pred, target):
        keep = torch.isfinite(target)
        if keep.any():
            self.m.update(pred[:, :1].movedim(1, -1)[keep].view(-1, 1), target[keep])

    def compute(self):
        return self.m.compute()


def _norm(name):
    """(mean, std) of the regression target."""
    if name == "biomassters":
        from exp.common.tasks import AGBM_MEAN, AGBM_STD
        return AGBM_MEAN, AGBM_STD
    from exp.common.tasks import TASKS
    idx = TASKS["yieldsat_raw"].index
    return idx["target_mean"], idx["target_std"]


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str                   # classification | regression
    n_out: int
    void: int = 19              # classification: the ignored label

    @property
    def out(self) -> Path:      # caches (PASTIS kept at the root for the runs made before)
        return ROOT if self.name == "pastis" else ROOT / self.name

    @property
    def higher_is_better(self) -> bool:
        return self.kind == "classification"

    def csv(self, head: str) -> Path:
        return RESULTS / (f"{self.name}_frozen" + ("" if head == "unet" else f"_{head}") + ".csv")

    def target(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "classification":
            return y.long()
        mean, std = _norm(self.name)
        return (y.float() - mean) / std

    def labelled(self, y: torch.Tensor) -> torch.Tensor:
        return y != self.void if self.kind == "classification" else torch.isfinite(y)

    def loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """out (B, n_out, ...) , y (B, ...) as returned by target()."""
        if self.kind == "classification":
            return F.cross_entropy(out.float(), y, ignore_index=self.void)
        mu, lv = out[:, 0].float(), out[:, 1].float().clamp(-10, 10)
        keep = torch.isfinite(y)                 # (YieldSAT: NaN = unlabelled)
        if not keep.any():
            return out.float().sum() * 0
        mu, lv, y = mu[keep], lv[keep], y[keep]
        nll = 0.5 * (lv + (y - mu.detach()) ** 2 * torch.exp(-lv))
        return F.mse_loss(mu, y) + nll.mean()

    def meter(self, boundary_px: int = 0):
        if self.kind == "classification":
            from exp.common.metrics import SegMeter
            if self.name == "planteur":
                from mistr_paper.data.datasets.planteur import CLASS_NAMES
            else:
                from mistr_paper.data.datasets.pastis import CLASS_NAMES
            return SegMeter(self.n_out, CLASS_NAMES[:self.n_out], ignore_label=self.void, boundary_px=boundary_px)
        return _MaskedReg(_norm(self.name)[1])

    @property
    def metric(self) -> str:
        return "miou" if self.kind == "classification" else "rmse"

    def uncertainty(self, out: torch.Tensor) -> torch.Tensor:
        if self.kind == "classification":
            p = out.float().softmax(1)
            return -(p * p.clamp(min=1e-8).log()).sum(1)
        return out[:, 1].float()


SPECS = {"pastis": Spec("pastis", "classification", 19),
         "planteur": Spec("planteur", "classification", 8, void=255),
         "biomassters": Spec("biomassters", "regression", 2),
         "yieldsat": Spec("yieldsat", "regression", 2)}
