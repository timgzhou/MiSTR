"""Tessera v2 pixel student (third_party/tessera) with masked, two-pass forwards for MiSTR.

Tessera encodes every 10 m pixel on its own: per modality (S2, S1) a post-LN transformer over
that pixel's time tokens (band MLP + sinusoidal raw-DOY encoding), a softmax attention pool
over time (Linear(D, 1)), then concat + MLP -> 128-d. There is no spatial axis, so MiSTR on
Tessera selects only WHEN, per pixel.

    coarse_pass(bb, x, doy, valid)        coarse tokens only; returns final tokens, the tokens
                                          entering every block (keys / values for the fine
                                          pass) and the pool logits (the readout attention)
    fine_pass(bb, cp, xf, doyf, validf)   one-way: fine tokens read the coarse tokens (cached,
                                          never updated) and each other
    pool(bb, h, valid, logits)            masked attention pool
    fuse(z2, z1)                          Tessera's dim_reducer -> 128-d

Masks are per pixel and per token (True = real token); a pixel's sequence is the tokens it has.
The plain-model path (all valid) matches PixelStudent.encode (tests/test_tessera_model.py).

MACs (macs_*): counted from real token counts only, multiply-accumulates of every Linear plus
the two attention matmuls (q k^T, a v); norms, softmax, activations ignored, as in our
OlmoEarth counts.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "tessera"))
import model as _tm  # noqa: E402  (third_party/tessera/model.py, Tessera v2 student)

CKPT = REPO / "pretrained" / "tessera" / "student_small.pt"
S2_MEAN, S2_STD = _tm.S2_BAND_MEAN, _tm.S2_BAND_STD
S1A_MEAN, S1A_STD = _tm.S1A_BAND_MEAN, _tm.S1A_BAND_STD
NEG = -1e4


def load(device, ckpt: Path = CKPT) -> _tm.PixelStudent:
    m = _tm.load_model(str(ckpt), device)
    if m.enable_qk_norm:
        raise NotImplementedError("qk-norm students: not wired into the masked forward")
    return m.eval()


def doy_encoding(d_model: int, doy: torch.Tensor) -> torch.Tensor:
    """Tessera's sinusoidal encoding of the raw DOY, any leading shape."""
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=doy.device)
                    * -(math.log(10000.0) / d_model))
    pos = doy.float()[..., None] * div
    pe = torch.zeros(*doy.shape, d_model, device=doy.device)
    pe[..., 0::2], pe[..., 1::2] = torch.sin(pos), torch.cos(pos)
    return pe


def embed(bb, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """x (P, T, C) standardized bands, doy (T,) or (P, T) -> (P, T, D)."""
    h = bb.embedding(x)
    return h + doy_encoding(h.shape[-1], doy).to(h.dtype)


def _block(layer, h: torch.Tensor, kv: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
    """One nn.TransformerEncoderLayer (post-LN, ReLU) on queries h (P, Tq, D) reading keys /
    values from kv (P, Tk, D) (None: h itself); mask (P, Tk) True = a key to read."""
    sa = layer.self_attn
    P, Tq, D = h.shape
    H = sa.num_heads
    W, b = sa.in_proj_weight, sa.in_proj_bias
    src = h if kv is None else kv
    q = F.linear(h, W[:D], b[:D])
    k = F.linear(src, W[D:2 * D], b[D:2 * D])
    v = F.linear(src, W[2 * D:], b[2 * D:])
    sh = lambda t: t.view(P, t.shape[1], H, D // H).transpose(1, 2)
    o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask[:, None, None, :])
    o = sa.out_proj(o.transpose(1, 2).reshape(P, Tq, D))
    h = layer.norm1(h + o)
    return layer.norm2(h + layer.linear2(F.relu(layer.linear1(h))))


@dataclass
class Pass:
    h: torch.Tensor           # (P, T, D) final tokens
    trace: list               # tokens entering each block (keys / values of the fine pass)
    logits: torch.Tensor      # (P, T) attention-pool logits, NEG at missing tokens
    valid: torch.Tensor       # (P, T)


def _guard(valid: torch.Tensor) -> torch.Tensor:
    """a pixel with no token at all still needs one key (its output is then meaningless and
    such pixels are reported by the caller)."""
    v = valid.clone()
    v[~v.any(1), 0] = True
    return v


def coarse_pass(bb, x, doy, valid, keep_trace: bool = True) -> Pass:
    valid = _guard(valid)
    h = embed(bb, x, doy)
    trace = []
    for layer in bb.transformer_encoder.layers:
        if keep_trace:
            trace.append(h)
        h = _block(layer, h, None, valid)
    logits = bb.attn_pool.query(h)[..., 0].float().masked_fill(~valid, NEG)
    return Pass(h, trace, logits, valid)


def fine_pass(bb, cp: Pass, x, doy, valid) -> Pass:
    """one-way: fine queries read [coarse keys at the same depth, fine keys]."""
    h = embed(bb, x, doy)
    keym = torch.cat([cp.valid, valid], 1)
    for layer, hc in zip(bb.transformer_encoder.layers, cp.trace):
        h = _block(layer, h, torch.cat([hc, h], 1), keym)
    logits = bb.attn_pool.query(h)[..., 0].float().masked_fill(~valid, NEG)
    return Pass(h, [], logits, valid)


def pool(h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(logits, 1)
    return (w[..., None] * h.float()).sum(1)


def fuse(m, z2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    return m.dim_reducer(torch.cat([z2, z1], -1).to(next(m.dim_reducer.parameters()).dtype))


# ---- MACs ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Dims:
    D: int        # token width (4 latent_dim)
    L: int        # blocks
    F: int        # FFN width
    lat: int      # latent_dim

    @classmethod
    def of(cls, m) -> "Dims":
        l0 = m.s2_backbone.transformer_encoder.layers[0]
        return cls(l0.linear1.in_features, len(m.s2_backbone.transformer_encoder.layers),
                   l0.linear1.out_features, l0.linear1.in_features // 4)


def macs_tokens(d: Dims, C: int, n_q: torch.Tensor, n_k: torch.Tensor) -> torch.Tensor:
    """MACs of n_q query tokens (per pixel) through embedding + all blocks + pool, each query
    reading n_k keys; the keys' own k / v projections are counted with their own tokens."""
    D = d.D
    per_tok = C * D + D * D + d.L * (4 * D * D + 2 * D * d.F) + D
    return n_q.double() * per_tok + d.L * 2 * D * n_q.double() * n_k.double()


def macs_reducer(d: Dims) -> int:
    return 8 * d.lat * 16 * d.lat + 16 * d.lat * 128


def to_tensor(a, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=device)
