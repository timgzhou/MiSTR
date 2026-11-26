"""The MiSTR model: tokenize -> encode -> decode, one encoder for every patch size.

    out0 = model.uniform(ms, ss, season, present)          coarse tokens only (the coarse pass)
    plan = selector(..., out0, ...)                        mistr.core.windows.Windows
    out  = model(ms, ss, season, present, plan)            coarse tokens + the plan's windows

Submodules: mix (the one-way encoder around the backbone), head, readout (the decoder).
load_weights() initializes a model from another run's checkpoint (e.g. MiSTR from the
fixed-resolution model it warm-starts from).
"""
from dataclasses import dataclass

import torch
from torch import nn

import mistr.core as _pkg     # (FAST, read at call time)
from exp.common.config import MODEL_SIZE_TO_ID
from exp.common.heads import Pa2PxHead
from exp.common.tasks import Task
from mistr.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _to_device, crop_patches
from mistr.encoders.terramind import MODELS as TERRAMIND, TerraMindOE
from mistr.core.encoder import OneWayEncoder, Packed
from mistr.core.config import ENCODER_ATTN, ModelConfig
from mistr.core.decoder import AttentionPoolDecoder
from mistr.core.windows import LOOKUP_PS, TILE, Windows

LR_PS = 16          # the head maps a token to a 16 x 16 px block (finer tokens: area-pooled)
@dataclass
class MixedPassOutput:
    """The model's output on coarse tokens + fine tokens (the second pass, MiSTR's result).
    prediction       (B, C, 64, 64) class logits / regression values per pixel
    coarse_features  (B, NS, h, w, D) the final coarse tokens
    fine_features    (N, D) the final fine tokens, in window order"""
    prediction: torch.Tensor
    coarse_features: torch.Tensor
    fine_features: torch.Tensor


@dataclass
class CoarsePassOutput(MixedPassOutput):
    """The coarse pass (coarse tokens only): what the selector reads; fine_features is empty.
    coarse_tokens    the embedded coarse tokens, reused by the second pass at inference
    coarse_cache     one-way attention: the coarse tokens entering every block and the final
                     ones (OneWayEncoder.lr_trace), the keys / values the fine tokens read"""
    coarse_tokens: Packed | None = None
    coarse_cache: tuple | None = None


class MiSTR(nn.Module):
    new_modules: tuple[str, ...] = ()     # state keys load_weights may not find (a baseline's own)

    def __init__(self, task: Task, cfg: ModelConfig, pretrained: bool = True):
        super().__init__()
        if task.per_step:
            raise ValueError("per-step tasks are not supported")
        if cfg.coarse_patch not in (2, 4, 6, 8, 10, 12, 14, 16):
            raise ValueError(f"coarse_patch {cfg.coarse_patch}: 16, 14, 12, 10, 8, 6, 4 or 2")
        self.task, self.cfg = task, cfg
        if cfg.backbone in TERRAMIND:     # TerraMind-1.0, per time step (mistr.encoders.terramind)
            encoder = TerraMindOE(cfg.backbone, pretrained)
        else:
            encoder = OlmoEarthEncoder(MODEL_SIZE_TO_ID[cfg.backbone],
                                       OlmoEarthInput(task.olmoearth.modalities,
                                                      task.olmoearth.impute),
                                       LR_PS, TILE, name=cfg.backbone,
                                       pretrained=pretrained).encoder
        self.mix = OneWayEncoder(encoder, 0, "center", cfg.coarse_patch,
                                 attn=ENCODER_ATTN[cfg.attention])
        self.mix.hr_type.requires_grad_(False)     # (a learned fine-token embedding: off, zero)
        D = encoder.embedding_size
        # the head writes a head_px x head_px map per token: 16, or a ps32 token's own 32
        # (coarse_head: the season token's own patch)
        self.head_px = cfg.coarse_patch if cfg.coarse_head else max(LR_PS, cfg.coarse_patch)
        self.head = Pa2PxHead(task.reduction, D, task.num_outputs, self.head_px)
        # ps6 seasons cover the tile padded to 66 px (OneWayEncoder.embed): the readout works on
        # 2 px cells of that frame, crops placed lo px in, the map cropped back to the tile
        self.lo = (-(-TILE // cfg.coarse_patch) * cfg.coarse_patch - TILE) // 2
        self.readout = AttentionPoolDecoder(D, lr_px=self.head_px, base_px=cfg.coarse_patch,
                                        **({"crop_px": 2, "out_px": TILE}
                                           if self.lo or cfg.coarse_patch < LOOKUP_PS else {}))

    @property
    def readout0(self) -> nn.Module:
        """The readout of the coarse pass (the readout)."""
        return self.readout

    @property
    def device(self):
        return self.head.linear.weight.device

    # -- tokenize ---------------------------------------------------------------------------
    def tokenize_seasons(self, ss, season=None) -> Packed:
        """The coarse composites at the coarse patch size."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.mix.head_blocks(self.mix.embed(ss, self.cfg.coarse_patch))

    def tokenize_crops(self, ms, plan: Windows, season, lr: Packed, ss=None) -> Packed:
        """The plan's windows of the monthly images, one single-patch sequence each, placed at
        their pixel position and the time slot of their coarse step."""
        if plan.src != "months":
            raise ValueError(f"windows of {plan.src}: only monthly windows are supported")
        t = plan.t if plan.src_t is None else plan.src_t
        if self.cfg.crop_time == "season":      # space-only: the coarse step's own composite
            ms, t = ss, season[plan.b, t]
        crops = crop_patches(ms, plan.b, t, plan.py, plan.px, plan.patch, pixels=True)
        slot = season[plan.b, plan.t] if plan.slot is None else plan.slot
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crops, plan.patch, slot=slot, py=plan.py + self.lo,
                                px=plan.px + self.lo)
            return self.mix.head_blocks(hr, lr, plan.b)

    # -- encode / decode --------------------------------------------------------------------
    def encode(self, lr: Packed, crops: list[tuple[Packed, Windows]], steps: int | None = None,
               lr_enc: tuple | None = None):
        """The transformer on every sample's coarse tokens and its fine tokens group by group
        (OneWayEncoder.tail_multi). -> (coarse tokens (B, NS, h, w, D), fine tokens (N, D) in
        group order). lr_enc: the coarse pass's OneWayEncoder.lr_trace (cross attn, inference)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr_x, outs = self.mix.tail_multi(lr, [(hr, p.b) for hr, p in crops], lr_enc)
        lr_F = lr.pooled(lr_x).float()
        hr_F = [hr.pooled(x).float()[:, 0, 0, 0] for (hr, _), x in zip(crops, outs)]
        return lr_F, (torch.cat(hr_F) if hr_F else lr_F.new_zeros(0, lr_F.shape[-1]))

    def decode(self, lr_F, hr_F, plans: list[Windows], season, present, logw=None):
        """Every token through the head at its own patch size, pooled per pixel over the
        tokens covering it (AttentionPoolDecoder). plans: the non-empty crop groups, hr_F's order.

        One group and several groups (APT) take separate code paths (footprints as ps4 cell
        counts; pixel positions only when a group is off the ps4 grid), which keeps both
        bit-exact with the checkpoints they were trained as."""
        return self._decode(lr_F, hr_F, plans, season, present, logw)

    def _decode(self, lr_F, hr_F, plans: list[Windows], season, present, logw=None):
        if not plans:
            z = torch.zeros(0, dtype=torch.long, device=lr_F.device)
            return self.readout(self.head.linear, lr_F, season, present, hr_F, (z,) * 4)
        if len(plans) == 1:
            p = plans[0]
            off = not p.on_grid or self.lo
            return self.readout(self.head.linear, lr_F, season, present, hr_F,
                                (p.b, p.t, p.y, p.x), logw,
                                None if _pkg.FAST else torch.ones_like(p.b),   # (None = all 1)
                                hr_px=(p.py + self.lo, p.px + self.lo) if off else None,
                                hr_fp=p.patch)
        if logw is not None:
            raise ValueError("crop gates with several crop groups: not ported")
        cat = lambda f: torch.cat([f(p) for p in plans])                         # noqa: E731
        idx = tuple(cat(lambda p, i=i: (p.b, p.t, p.y, p.x)[i]) for i in range(4))
        sizes = cat(lambda p: torch.full_like(p.b, p.cells))
        off = any(p.patch < LOOKUP_PS or p.offgrid for p in plans) or self.lo
        return self.readout(self.head.linear, lr_F, season, present, hr_F, idx, None, sizes,
                            hr_px=(cat(lambda p: p.py + self.lo), cat(lambda p: p.px + self.lo))
                            if off else None,
                            hr_fp=cat(lambda p: torch.full_like(p.b, p.patch)) if off
                            else plans[0].patch)

    # -- passes -----------------------------------------------------------------------------
    def _inputs(self, ms, ss, season, present):
        dev = self.device
        ms, season, present = _to_device(ms, dev, torch.bfloat16), season.to(dev), present.to(dev)
        if self.cfg.coarse_time == "month":       # every month its own slot, the monthly images as tokens
            B, T = present.shape
            return ms, ms, torch.arange(T, device=dev).expand(B, T).contiguous(), present
        return ms, _to_device(ss, dev, torch.bfloat16), season, present

    def uniform(self, ms, ss, season, present) -> CoarsePassOutput:
        """The season tokens alone (the plan with no crops): what a selector reads. Eval mode and
        bfloat16 throughout. Call it OUTSIDE
        any autocast region: autocast caches its bfloat16 weight copies until the outermost
        region exits, and copies made under no_grad would leave the next pass's weights without
        gradients."""
        if torch.is_autocast_enabled():
            raise RuntimeError("MiSTR.uniform inside an autocast region (see docstring)")
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        was = self.training
        if was:             # (eval() walks every module: skipped when already in eval mode)
            self.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lr = self.tokenize_seasons(ss, season)
                lr_enc = None
                if self.mix.attn != "joint":   # (= encode(lr, []), its trace kept)
                    lr_enc = self.mix.lr_trace(lr)
                    lr_F = lr.pooled(lr_enc[1]).float()
                    hr_F = lr_F.new_zeros(0, lr_F.shape[-1])
                else:
                    lr_F, hr_F = self.encode(lr, [], ss.timestamps.shape[1])
                logits = self.decode(lr_F, hr_F, [], season, present).float()
        finally:
            if was:
                self.train()
        return CoarsePassOutput(logits, lr_F, hr_F, lr, lr_enc)

    def split(self, plan: Windows | list[Windows], logw=None):
        """-> the non-empty crop groups, each window split into model.crop_patch sub-windows
        (the crop gate log-weights repeated with them)."""
        plans = [p for p in ([plan] if isinstance(plan, Windows) else plan) if len(p)]
        cp = self.cfg.fine_patch
        if not cp or not any(p.patch > cp for p in plans):
            return plans, logw
        if logw is not None:
            logw = logw.repeat_interleave((plans[0].patch // cp) ** 2)
        return [p.split(cp) if p.patch > cp else p for p in plans], logw

    def forward(self, ms, ss, season, present, plan: Windows | list[Windows], logw=None,
                dates=None, seasons: Packed | None = None, lr_enc: tuple | None = None) -> MixedPassOutput:
        """plan: one window group, or several (APT: two patch sizes). dates: must be None (date
        crops are not supported). seasons / lr_enc: the coarse pass's embedded tokens and trace, reused at
        inference (they carry no gradient: never in training)."""
        if dates is not None:
            raise ValueError("date crops are not supported")
        if ((seasons is not None or lr_enc is not None)
                and (self.training or torch.is_grad_enabled())):
            raise ValueError("forward(seasons=/lr_enc=...) reuses no-grad tokens: inference only")
        ms, ss, season, present = self._inputs(ms, ss, season, present)
        if self.cfg.crop_time == "season":
            if logw is not None:
                raise ValueError("crop gates with crop_time season: not ported")
            plan = ([p.per_season(season) for p in plan] if isinstance(plan, list)
                    else plan.per_season(season))
        plans, logw = self.split(plan, logw)
        lr = self.tokenize_seasons(ss, season) if seasons is None else seasons
        crops = [(self.tokenize_crops(ms, p, season, lr, ss), p) for p in plans]
        lr_F, hr_F = self.encode(lr, crops, ss.timestamps.shape[1], lr_enc)
        return MixedPassOutput(self.decode(lr_F, hr_F, plans, season, present, logw), lr_F, hr_F)


def load_weights(model: MiSTR, ckpt: str) -> list[str]:
    """Initialize `model` from checkpoints/<ckpt>.pt (every key the model has; a baseline's
    new_modules may be missing) -> the checkpoint's keys the model does not use."""
    sd = torch.load(f"checkpoints/{ckpt}.pt", map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    extra = [k for k in own if k not in sd and model.new_modules
             and k.startswith(model.new_modules)]
    missing = [k for k in own if k not in sd and k not in extra]
    if missing:
        raise KeyError(f"{ckpt}: missing {missing[:5]}")
    model.load_state_dict({k: sd[k] for k in own if k in sd}, strict=not extra)
    return sorted(k for k in sd if k not in own)
