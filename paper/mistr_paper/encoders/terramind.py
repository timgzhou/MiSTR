"""TerraMind-1.0 (IBM / ESA, arXiv:2504.11171) behind the encoder interface mistr_paper.core.encoder.OneWayEncoder
uses: a MULTIMODAL but SINGLE-TIME ViT, run per time step.

TerraMind's encoder on raw pixels: per modality a bias-free linear projection of each 16 x 16
patch, + a fixed 2D sin-cos of the patch's grid index, + one learned vector per modality; the S2
and S1 tokens of a scene form one sequence. Here:

  per step     (terramind_tiny) every time step is its own attention sequence: the coarse tokens
               of one season (S2 + S1), joined by the fine windows of that season's months
               (OneWayEncoder.per_step: sequences keyed by (sample, Packed.step)); the decoder
               pools across steps, as PANGAEA's L-TAE does over per-step features (how TerraMind
               was evaluated on PASTIS). terramind_tiny_joint: one sequence per sample.
  time         TerraMind has none. A token's time t = its calendar month + 1/2 (a monthly
               composite is dated the 1st, a season composite mid its centre month), a cyclic
               code [sin, cos](2 pi k t / 12), k = 1, 2, 3 (12, 6 and 4 months; k = 1 is one turn
               per 4 seasons: a season at slot s, its months at s -+ 1/3), through a learned
               linear map that starts at ZERO: the pretrained model exactly, at first.
  patch size   PI-resize of the 16 px projection to the token's patch size (resize_matrix).
  positions    the 2D sin-cos at continuous coordinates: a token's CENTRE in ps16 grid units,
               (y + p / 2) / 16 - 1 / 2 (the grid index at p = 16). TerraMind's own code stretches
               its 14 x 14 grid onto any input size instead; physical positions are used here.
               Its layout (checked against the checkpoint's buffer in __init__): the first D / 2
               dims [sin | cos] of the ROW index, the last D / 2 of the column.
  bands        S2 L2A without B01 / B09 (OlmoEarth imputes them for PASTIS; their projection
               weights are dropped, as TerraTorch's band selection does); S1 GRD VV, VH in dB.
               OlmoEarth's computed normalization undone (affine per band), TerraMind's
               pretraining mean / std applied.

Blocks are OlmoEarth Blocks carrying TerraMind's weights (fused qkv split; LayerNorm eps 1e-6,
the checkpoint's norm biases loaded, GELU, no layer scale), so varlen FlashAttention and the
one-way attention paths run as for OlmoEarth. Output per step: the mean over S2 and S1 tokens of
a patch (Packed.pooled), TerraMind's merge_method "mean".
"""
from __future__ import annotations

import math
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.attention import Block
from torch import nn


ROOT = Path(__file__).resolve().parents[2] / "cache" / "pretrained"
# name -> (folder, weights, dim, heads, per-step attention, one-way crop context:
# OneWayEncoder.context)
BASE_PS = 16        # the patch size the pretrained projection was trained at


@lru_cache(maxsize=None)
def resize_matrix(p: int) -> torch.Tensor:
    """(p^2, 256) M with M[i, j] = (resize_{p -> 16} e_i)_j, bicubic + antialias as
    OlmoEarth's FlexiPatchEmbed resizes its input; the kernel for p is w @ M^T (PI-resize)."""
    basis = torch.eye(p * p, dtype=torch.float64).view(p * p, 1, p, p)
    if p == BASE_PS:
        return torch.eye(p * p, dtype=torch.float64)
    out = F.interpolate(basis, size=(BASE_PS, BASE_PS), mode="bicubic", antialias=True,
                        align_corners=False)
    return out.view(p * p, BASE_PS * BASE_PS)


_TINY = ("TerraMind-1.0-tiny", "TerraMind_v1_tiny.pt", 192, 3)
MODELS = {"terramind_tiny": (*_TINY, True, "season"),
          "terramind_tiny_ctxnb": (*_TINY, True, "neighbor"),
          "terramind_tiny_ctxall": (*_TINY, True, "all"),
          "terramind_tiny_joint": (*_TINY, False, "season")}
DEPTH = 12
# OlmoEarth modality -> (TerraMind modality, its pretrained band order, the bands used = OlmoEarth
# names in TerraMind's order, TerraMind's pretraining mean / std of the bands used)
TM_S2 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
MODS = {
    "sentinel2_l2a": ("untok_sen2l2a@224", TM_S2,
                      ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
                      (1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234,
                       3132.22, 2424.884, 1857.648),
                      (2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918,
                       1753.829, 1434.261, 1334.311)),
    "sentinel1": ("untok_sen1grd@224", ("vv", "vh"), ("vv", "vh"),
                  (-12.599, -20.293), (5.195, 5.890)),
}
HARMONICS = (1, 2, 3)


def sincos_2d(y: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    """TerraMind's build_2d_sincos_posemb (temperature 1e4) at continuous (row y, column x)
    in grid units -> (N, dim): [sin y, cos y, sin x, cos x], dim / 4 frequencies each."""
    q = dim // 4
    omega = 1.0 / 1e4 ** (torch.arange(q, device=y.device, dtype=torch.float64) / q)
    a, b = y.reshape(-1, 1).double() * omega, x.reshape(-1, 1).double() * omega
    return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], 1).float()


def month_time(timestamps: torch.Tensor) -> torch.Tensor:
    """(..., 3) [day, month 0-11, year] -> (...,) the composite's time in months (centre)."""
    return timestamps[..., 1].float() + 0.5


class TerraMindOE(nn.Module):
    """The encoder object OneWayEncoder wraps: blocks, norm, embedding_size, per_step, and
    mixres_embed (its tokenizer)."""

    def __init__(self, name: str = "terramind_tiny", pretrained: bool = True):
        super().__init__()
        folder, weights, D, heads, per_step, context = MODELS[name]
        self.embedding_size, self.name, self.per_step, self.context = D, name, per_step, context
        self.mods = list(MODS)
        self.proj_w = nn.ParameterDict()            # modality -> (D, bands, 256)
        self.mod_emb = nn.ParameterDict()           # modality -> (D,)
        norm = Normalizer(Strategy.COMPUTED)
        for m, (_, _, use, mean, std) in MODS.items():
            self.proj_w[m] = nn.Parameter(torch.empty(D, len(use), BASE_PS * BASE_PS))
            nn.init.xavier_uniform_(self.proj_w[m].view(D, -1))
            self.mod_emb[m] = nn.Parameter(torch.zeros(D))
            # OlmoEarth's computed normalization, undone: raw = (y - y0) / scale; then
            # x_tm = (raw - mean) / std = a y + b
            mo = Modality.get(m)
            idx = [mo.band_order.index(b) for b in use]
            y0 = norm.normalize(mo, np.zeros((1, len(mo.band_order))))[0, idx]
            y1 = norm.normalize(mo, np.full((1, len(mo.band_order)), 1000.0))[0, idx]
            scale = (y1 - y0) / 1000
            mean, std = np.array(mean), np.array(std)
            self.register_buffer(f"idx_{m}", torch.tensor(idx), persistent=False)
            self.register_buffer(f"a_{m}", torch.tensor(1 / (scale * std), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer(f"b_{m}", torch.tensor(-y0 / (scale * std) - mean / std,
                                                        dtype=torch.float32), persistent=False)
        # the time code: zero at first (TerraMind has none)
        self.time_proj = nn.Linear(2 * len(HARMONICS), D, bias=False)
        nn.init.zeros_(self.time_proj.weight)
        ln = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(Block(D, heads, 4.0, qkv_bias=True, norm_layer=ln,
                                          use_flash_attn=True) for _ in range(DEPTH))
        self.norm = ln(D)
        if pretrained:
            self.load_terramind(ROOT / folder / weights)

    def load_terramind(self, path: Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        own = {"norm.weight": sd["encoder_norm.weight"], "norm.bias": sd["encoder_norm.bias"]}
        for m, (tm, order, use, _, _) in MODS.items():
            e = f"encoder_embeddings.{tm}."
            w = sd[e + "proj.weight"]                                    # (D, 256 * C), C fastest
            w = w.view(w.shape[0], BASE_PS * BASE_PS, len(order)).permute(0, 2, 1)
            own[f"proj_w.{m}"] = w[:, [order.index(b) for b in use]]
            own[f"mod_emb.{m}"] = sd[e + "mod_emb"].reshape(-1)
            # the fixed positions: check the layout sincos_2d assumes
            pe = sd[e + "pos_emb"][0]                                    # (196, D), 14 x 14
            g = torch.arange(14)
            gy, gx = torch.meshgrid(g, g, indexing="ij")                 # token n = row * 14 + col
            err = (sincos_2d(gy.flatten(), gx.flatten(), pe.shape[1]) - pe.float()).abs().max()
            if err > 1e-4:
                raise ValueError(f"{path}: {tm} pos_emb is not the assumed sin-cos ({err:.2e})")
        for i in range(DEPTH):
            s, p = f"encoder.{i}.", f"blocks.{i}."
            q, k, v = sd[s + "attn.qkv.weight"].chunk(3)
            qb, kb, vb = sd[s + "attn.qkv.bias"].chunk(3)
            own.update({p + "attn.q.weight": q, p + "attn.k.weight": k, p + "attn.v.weight": v,
                        p + "attn.q.bias": qb, p + "attn.k.bias": kb, p + "attn.v.bias": vb})
            for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                      "attn.proj.weight", "attn.proj.bias", "mlp.fc1.weight", "mlp.fc1.bias",
                      "mlp.fc2.weight", "mlp.fc2.bias"):
                own[p + n] = sd[s + n]
        missing, unexpected = self.load_state_dict(own, strict=False)
        missing = [k for k in missing if k != "time_proj.weight"]
        if missing or unexpected:
            raise ValueError(f"{path}: missing {missing[:4]}, unexpected {unexpected[:4]}")

    def kernel(self, m: str, p: int) -> torch.Tensor:
        """(D, C, p^2) the PI-resized projection of modality m for patch p."""
        w = self.proj_w[m]
        if p == BASE_PS:
            return w
        return w.float() @ resize_matrix(p).to(w.device, torch.float32).T

    def time_encoding(self, timestamps: torch.Tensor) -> torch.Tensor:
        """(..., 3) -> (..., D): the learned map of the cyclic month code."""
        ang = 2 * math.pi * month_time(timestamps)[..., None] / 12 * torch.tensor(
            HARMONICS, device=timestamps.device, dtype=torch.float32)
        return self.time_proj(torch.cat([ang.sin(), ang.cos()], -1).to(self.time_proj.weight.dtype))

    def mixres_embed(self, sample: MaskedOlmoEarthSample, patch_size: int,
                     slot: torch.Tensor | None = None, py: torch.Tensor | None = None,
                     px: torch.Tensor | None = None):
        """-> (tokens (S, N, D), mask (S, N), dims, masks, step (S, N), image), the layout
        OneWayEncoder.embed takes from a non-OlmoEarth backbone, for S2 and S1 (each its own
        modality, one band set): + each token's attention step: its time index (crops: `slot`), and its
        image: the calendar month (0-11) of its step (crops: of their image; a season: its
        centre month, as dated)."""
        p, D = patch_size, self.embedding_size
        dev = self.proj_w[self.mods[0]].device
        ts = sample.timestamps.to(dev)
        toks, masks_out, tmasks, steps, images, dims = [], {}, [], [], [], {}
        c = lambda i: (i.float() + p / 2) / BASE_PS - 0.5                      # noqa: E731
        for m in self.mods:
            x = getattr(sample, m)
            if x is None:
                continue
            S, H, W, T, _ = x.shape
            if H % p or W % p:
                raise ValueError(f"{H} x {W} px is not a multiple of patch {p}")
            h, w = H // p, W // p
            x = x.index_select(-1, getattr(self, f"idx_{m}")).float() * getattr(self, f"a_{m}") \
                + getattr(self, f"b_{m}")
            x = x.view(S, h, p, w, p, T, -1).permute(0, 1, 3, 5, 6, 2, 4).reshape(S, h, w, T, -1, p * p)
            tok = torch.einsum("shwtcq,dcq->shwtd", x, self.kernel(m, p).float())
            if slot is None:
                gy, gx = torch.meshgrid(torch.arange(h, device=dev) * p,
                                        torch.arange(w, device=dev) * p, indexing="ij")
                pos = sincos_2d(c(gy), c(gx), D).view(h, w, 1, D)
                step = torch.arange(T, device=dev).view(1, 1, 1, T).expand(S, h, w, T)
                image = ts[:, :, 1].view(S, 1, 1, T).expand(S, h, w, T)       # its month
            else:
                if (h, w, T) != (1, 1, 1):
                    raise ValueError(f"crops are one patch of one step, got {(h, w, T)}")
                pos = sincos_2d(c(py.to(dev)), c(px.to(dev)), D).view(S, 1, 1, 1, D)
                step = slot.to(dev).view(S, 1, 1, 1)
                image = ts[:, 0, 1].view(S, 1, 1, 1)                           # its month
            tok = tok + pos + self.mod_emb[m].float() \
                + self.time_encoding(ts).float()[:, None, None]
            mask = getattr(sample, MaskedOlmoEarthSample.get_masked_modality_name(m))
            tmask = mask[:, ::p, ::p, :, 0]                                    # (S, h, w, T)
            toks.append(tok.reshape(S, -1, D))
            tmasks.append(tmask.reshape(S, -1))
            steps.append(step.reshape(S, -1))
            images.append(image.reshape(S, -1))
            dims[m] = (S, h, w, T, 1, D)
            masks_out[MaskedOlmoEarthSample.get_masked_modality_name(m)] = tmask[..., None]
        return (torch.cat(toks, 1), torch.cat(tmasks, 1), dims, masks_out,
                torch.cat(steps, 1).long(), torch.cat(images, 1).long())
