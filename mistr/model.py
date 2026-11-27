"""MiSTR on OlmoEarth: a coarse pass over seasonal composites, a selection of where and when the
prediction needs more detail, and a sparse fine pass over monthly images at those points.

    model = MiSTR.from_pretrained("TimZhou/mistr-olmoearth-tiny-pastis-s6s3")
    out = model.predict(prepare(...), budget=0.5, seasons=2)

Two released configurations: coarse 8 px / fine 4 px tokens (windows = fine tokens), and coarse
6 px / 6 px windows of four 3 px fine tokens (64 px tiles padded to 66 px, the decoder on 2 px
cells), trained with 1 to 2 seasons per window.

1. Coarse pass: every season's composite as coarse_patch px tokens (OlmoEarth's patch embedding
   and encodings), through the backbone; each block's input tokens are kept.
2. Where: the coarse prediction's per-pixel entropy, averaged over each window_px window; a
   window is picked when its mean is above a global threshold, the (1 - budget) quantile of the
   validation distribution (a table saved with the weights).
3. When: per coarse cell, the season the decoder's attention score ranks highest; a picked
   window takes that season's months. Models with a season range add a second season, the one
   whose own coarse map disagrees most with the first (Jensen-Shannon divergence), at the picked
   windows whose entropy passes a second table (seasons=2: all of them, 1.5: the more uncertain).
4. Fine pass: each picked (window, month) as fine_patch px tokens, placed at their pixel position
   and their season's time slot. Fine tokens read their tile's coarse tokens at the same depth and
   the tile's other fine tokens (one-way attention: coarse tokens never read fine ones, so the
   coarse pass is reused, not recomputed).
5. Decoder: a linear head maps every token to a small logit map; each cell (or, for fine tokens
   off the cell grid, each pixel) pools the maps of the tokens covering it (its coarse token in
   every season, and its fine tokens), weighted by a softmax over a learned per-token score.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from olmoearth_pretrain.data.constants import BASE_GSD
from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks
from olmoearth_pretrain.nn.encodings import get_1d_sincos_pos_encoding
from torch import nn

from mistr.inputs import TILE, Inputs

CELL = 4                    # the decoder's and the selector's grid: 4 px cells
LEVELS = 101                # quantile levels of the threshold table


# -- tokens of one pass, packed (varlen: missing tokens dropped, no padding) --------------------
@dataclass
class Packed:
    x: torch.Tensor             # (N_valid, D)
    seqlens: torch.Tensor       # (S,) tokens per sequence (a tile)
    keep: torch.Tensor          # (S, N) which token slots are valid
    dims: dict                  # modality -> token shape (S, h, w, T, band sets, D)
    masks: dict                 # mask name -> (S, h, w, T, band sets)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """(S, T, h, w, D): per step, the mean over band sets, then over the modalities present."""
        S, N = self.keep.shape
        full = x.new_zeros(S * N, x.shape[-1])
        full[self.keep.flatten()] = x
        full = full.view(S, N, -1)
        per, i = {}, 0
        for m, shape in self.dims.items():
            n = 1
            for s in shape[1:-1]:
                n *= s
            per[m] = full[:, i:i + n].reshape(*shape[:-1], -1)
            i += n
        tam = TokensAndMasks(**per, **self.masks)
        T = next(iter(per.values())).shape[3]
        return torch.stack([_pool_step(tam, t) for t in range(T)], 1)


def _pool_step(tam, t: int) -> torch.Tensor:
    feats, weights = [], []
    for m in tam.modalities:
        x = getattr(tam, m)[:, :, :, t:t + 1]                     # (S, h, w, 1, bandsets, D)
        mask = getattr(tam, tam.get_masked_modality_name(m))[:, :, :, t:t + 1]
        online = mask == MaskValue.ONLINE_ENCODER.value
        if online.all():
            feats.append(torch.mean(x, dim=(-2, -3)))
            weights.append(torch.ones(x.shape[:3], device=x.device, dtype=x.dtype))
        else:
            w = online.unsqueeze(-1).to(x.dtype)
            cnt = w.sum(dim=(-2, -3))
            feats.append((x * w).sum(dim=(-2, -3)) / cnt.clamp(min=1))
            weights.append((cnt[..., 0] > 0).to(x.dtype))
    f, w = torch.stack(feats, -1), torch.stack(weights, -1)
    if (w == 1).all():
        return f.mean(dim=-1)
    return (f * w.unsqueeze(-2)).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1)


def _cu(seqlens: torch.Tensor) -> torch.Tensor:
    return F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))


def flash_available(device) -> bool:
    """FlashAttention 2: installed and an Ampere or newer GPU."""
    if torch.device(device).type != "cuda":
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.get_device_capability(device) >= (8, 0)


def varlen_attention(q, k, v, cu_q, cu_k, max_q: int, max_k: int, scale: float, flash: bool):
    """Attention over packed sequences: q (Nq, heads, d), k / v (Nk, heads, d), sequence i
    = rows cu[i]:cu[i + 1] -> (Nq, heads, d). flash: FlashAttention's varlen kernel (how the
    paper ran); else PyTorch SDPA on the sequences padded, padding masked out of the keys."""
    if flash:
        import flash_attn
        return flash_attn.flash_attn_varlen_func(q, k.to(q.dtype), v.to(q.dtype), cu_q, cu_k,
                                                 max_q, max_k, softmax_scale=scale)
    cu_q, cu_k = cu_q.long(), cu_k.long()
    S, dev = len(cu_q) - 1, q.device
    n_q, n_k = cu_q.diff(), cu_k.diff()

    def pad(x, cu, n, L):
        seq = torch.repeat_interleave(torch.arange(S, device=dev), n)
        pos = torch.arange(len(x), device=dev) - cu[seq]
        out = x.new_zeros((S, L) + x.shape[1:])
        out[seq, pos] = x
        return out.transpose(1, 2), seq, pos                      # (S, heads, L, d)

    Q, sq, pq = pad(q, cu_q, n_q, max_q)
    K, _, _ = pad(k.to(q.dtype), cu_k, n_k, max_k)
    V, _, _ = pad(v.to(q.dtype), cu_k, n_k, max_k)
    mask = (torch.arange(max_k, device=dev)[None] < n_k[:, None])[:, None, None]   # (S, 1, 1, Lk)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=scale)
    return out.transpose(1, 2)[sq, pq]


def _frame_lo(px: int) -> int:
    """Padding before the tile when px-px patches tile it padded to whole patches."""
    return (-(-TILE // px) * px - TILE) // 2


def _pad(v: torch.Tensor, size: int, dims=(1, 2)) -> torch.Tensor:
    """Replicate-pad dims of v (64 px), centred, to size px."""
    i = (torch.arange(size, device=v.device) - (size - TILE) // 2).clamp(0, TILE - 1)
    return v.index_select(dims[0], i).index_select(dims[1], i)


def _sincos_2d(py, px, n):
    return torch.cat([get_1d_sincos_pos_encoding(px.float(), n // 2),
                      get_1d_sincos_pos_encoding(py.float(), n // 2)], 1)


@dataclass
class Prediction:
    """logits            (C, H, W) MiSTR's prediction (image) or (B, C, 64, 64) (tiles)
    coarse_logits       the coarse pass's own prediction, same shape
    windows             (N, 3) long: (month 0-11, y, x) of each re-encoded window (window_px
                        px), top-left pixel in the image (tiles: (N, 4), the tile first)
    coarse_tokens, fine_tokens   tokens encoded
    window_px           window side in px"""
    logits: torch.Tensor
    coarse_logits: torch.Tensor
    windows: torch.Tensor
    coarse_tokens: int
    fine_tokens: int
    window_px: int = 4

    @property
    def classes(self) -> torch.Tensor:
        return self.logits.argmax(-3)


class MiSTR(nn.Module):
    def __init__(self, config: dict, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.coarse_patch, self.fine_patch = config["coarse_patch"], config["fine_patch"]
        # selected windows (window px, split into fine_patch px tokens when larger)
        self.window_px = config.get("window_px", self.fine_patch)
        if (self.coarse_patch, self.window_px, self.fine_patch) not in ((8, 4, 4), (6, 6, 3)):
            raise NotImplementedError("released configurations: coarse 8 / fine 4 px, "
                                      "coarse 6 / windows 6 / fine 3 px")
        # a patch that does not divide the 64 px tile (6 px) tiles it padded, centred, to whole
        # patches (66 px): coarse tokens, windows and the decoder share that frame
        self.lo = _frame_lo(self.coarse_patch)
        self.cell = CELL if not self.lo else 2          # the decoder's cells (px)
        self.extra = config.get("extra_pick") is not None   # trained with 1 / 1.5 / 2 seasons
        self.backbone = backbone
        for m in backbone.modules():
            if hasattr(m, "use_flash_attn"):
                m.use_flash_attn = True
        self.attention = "auto"     # "flash" | "sdpa" | "auto": flash where it runs (see flash_available)
        D = backbone.embedding_size
        self.num_classes = config["num_classes"]
        self.head = nn.Linear(D, self.num_classes * self.coarse_patch ** 2)
        self.score = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.register_buffer("quantiles", torch.zeros(LEVELS))   # window entropy (val split)
        if self.extra:      # a second season's: entropy of the windows with a candidate season
            self.register_buffer("quantiles_x", torch.zeros(LEVELS))
            self.register_buffer("ratio_x", torch.ones(()))      # candidates per picked window

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device=None, attention: str = "auto") -> "MiSTR":
        """A Hugging Face model repo or a local folder with config.json, backbone/config.json
        (OlmoEarth's architecture) and model.safetensors. device: default a GPU if there is one.
        attention: "flash" (FlashAttention 2, Ampere or newer GPUs; how the paper ran), "sdpa"
        (PyTorch's attention: any GPU, or CPU) or "auto" (flash where available)."""
        from olmoearth_pretrain.model_loader import load_model_from_path
        from safetensors.torch import load_file
        d = Path(repo_or_dir)
        if not d.is_dir():
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(repo_or_dir))
        config = json.loads((d / "config.json").read_text())
        backbone = load_model_from_path(str(d / "backbone"), load_weights=False)
        model = cls(config, getattr(backbone, "encoder", backbone))
        model.load_state_dict(load_file(str(d / "model.safetensors")))
        if attention not in ("auto", "flash", "sdpa"):
            raise ValueError(f"attention {attention!r}: auto | flash | sdpa")
        model.attention = attention
        return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()

    @property
    def device(self):
        return self.head.weight.device

    @property
    def flash(self) -> bool:
        if self.attention == "sdpa":
            return False
        ok = flash_available(self.device)
        if self.attention == "flash" and not ok:
            raise RuntimeError("attention='flash' needs flash-attn and an Ampere or newer GPU")
        return ok

    def _amp(self):
        """bf16 autocast on a GPU (as trained); float32 on CPU (OlmoEarth's patch resizing has no
        bf16 CPU kernel)."""
        if self.device.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def _block(self, blk, x, cu, mx: int, flash: bool):
        """One backbone block on packed sequences (the block's own forward under flash)."""
        if flash:
            return blk(x=x, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
        a = blk.attn
        heads = lambda t: t.view(len(t), a.num_heads, -1)                      # noqa: E731
        n = blk.norm1(x)
        o = varlen_attention(a.q_norm(heads(a.q(n))), a.k_norm(heads(a.k(n))), heads(a.v(n)),
                             cu, cu, mx, mx, a.scale, False)
        x = x + blk.ls1(a.proj(o.reshape(len(o), -1)))
        return x + blk.ls2(blk.mlp(blk.norm2(x)))

    # -- tokens ---------------------------------------------------------------------------
    def _embed(self, sample, patch: int, slot=None, py=None, px=None) -> Packed:
        """Patch embedding + OlmoEarth's encodings, packed. Fine tokens (one patch, T = 1): time
        slot 0 moved to `slot`, position (0, 0) to their centre in the coarse grid's frame."""
        enc = self.backbone
        if slot is None and TILE % patch:          # (6 px seasons: the tile padded to 66 px)
            size = -(-TILE // patch) * patch
            sample = sample._replace(**{k: v if k == "timestamps" else _pad(v, size)
                                        for k, v in sample.as_dict().items() if v is not None})
        tokens, masks, dims = enc.split_tokens_masks_and_dims(enc.patch_embeddings(sample, patch))
        tokens = enc.composite_encodings(tokens, sample.timestamps, patch, BASE_GSD)
        if slot is not None:
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            off = (patch - self.coarse_patch) / 2
            zero = torch.zeros_like(py).float()
            d_time = get_1d_sincos_pos_encoding(slot.float(), n) - get_1d_sincos_pos_encoding(zero, n)
            d_pos = _sincos_2d(py.float() + off, px.float() + off, n) - _sincos_2d(zero, zero, n)
            for m, t in tokens.items():
                delta = torch.zeros_like(t[:, 0, 0, 0, 0])
                delta[:, n:2 * n] = d_time                    # (S2 and S1: spatial, multitemporal)
                delta[:, 3 * n:4 * n] = d_pos
                tokens[m] = t + delta[:, None, None, None, None].to(t.dtype)
        x, mask = enc.collapse_and_combine_hwtc({**tokens, **masks})
        keep = mask == MaskValue.ONLINE_ENCODER.value
        return Packed(x[keep], keep.sum(1), keep, dims, masks)

    def _coarse(self, x: Inputs):
        """Coarse pass -> (coarse tokens, every block's input tokens, final coarse features
        (B, 4, h, w, D))."""
        lr = self._embed(x.seasons, self.coarse_patch)
        cu, mx, trace, h = _cu(lr.seqlens), int(lr.seqlens.max()), [], lr.x
        flash = self.flash
        for blk in self.backbone.blocks:
            trace.append(h)
            h = self._block(blk, h, cu, mx, flash)
        return lr, trace, lr.pooled(self.backbone.norm(h)).float()

    def _fine(self, lr: Packed, trace, crops: Packed, owner: torch.Tensor) -> torch.Tensor:
        """The fine tokens through every block, reading their tile's coarse tokens at that block
        and the tile's fine tokens -> (N, D) normed."""
        S, dev, flash = len(lr.seqlens), lr.x.device, self.flash
        t_s = torch.repeat_interleave(owner, crops.seqlens)                     # token -> tile
        order = torch.argsort(t_s, stable=True)
        n_q = torch.bincount(t_s, minlength=S)
        n_k = lr.seqlens + n_q
        cu_q, cu_k = _cu(n_q), _cu(n_k)
        k_seq = torch.cat([torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens), t_s[order]])
        k_order = torch.argsort(k_seq, stable=True)        # keys: each tile's coarse, then fine tokens
        max_q, max_k = int(n_q.max()), int(n_k.max())
        x_c = crops.x
        for blk, x_lr in zip(self.backbone.blocks, trace):
            a = blk.attn
            heads = lambda v: v.view(len(v), a.num_heads, -1)                  # noqa: E731
            n_lr, n_c = blk.norm1(x_lr), blk.norm1(x_c[order])
            q = a.q_norm(heads(a.q(n_c)))
            k = torch.cat([a.k_norm(heads(a.k(n_lr))), a.k_norm(heads(a.k(n_c)))])[k_order]
            v = torch.cat([heads(a.v(n_lr)), heads(a.v(n_c))])[k_order]
            out = varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, a.scale, flash)
            y = torch.empty_like(x_c)
            y[order] = blk.ls1(a.proj(out.reshape(len(out), -1))).to(x_c.dtype)
            x_c = x_c + y
            x_c = x_c + blk.ls2(blk.mlp(blk.norm2(x_c)))
        return self.backbone.norm(x_c)

    # -- decoder --------------------------------------------------------------------------
    def _decode(self, lr_F, season, present, hr=None, b=None, y=None, x=None):
        """Per 4 px cell, the softmax(score)-weighted mean of the logit maps of the tokens
        covering it -> (B, C, 64, 64). lr_F (B, NS, h, w, D) coarse features; hr (N, D) fine
        features at cells (b, y, x) of the 16 x 16 cell grid (padded frames: top-left pixels,
        _decode_framed)."""
        if self.lo:
            return self._decode_framed(lr_F, season, present, hr, b, y, x)
        B, NS, h, w, D = lr_F.shape
        c, bp, C = CELL, self.coarse_patch, self.num_classes
        rs = bp // c
        H, W = h * rs, w * rs
        Q, dev = H * W, lr_F.device
        s_log = rearrange(self.head(lr_F.float()), "b s gh gw (k i j) -> b s k (gh i) (gw j)", k=C, i=bp, j=bp)
        s_log = rearrange(s_log, "b s k (y i) (x j) -> b (y x) s k (i j)", i=c, j=c)   # (B, Q, NS, C, 16)
        n_s = (F.one_hot(season.clamp(min=0), NS).float()
               * (present & (season >= 0)).float()[..., None]).sum(1)           # months per season
        s_a = self.score(lr_F.float())[..., 0].masked_fill((n_s == 0)[:, :, None, None], float("-inf"))
        s_a = s_a.repeat_interleave(rs, 2).repeat_interleave(rs, 3).flatten(2).transpose(1, 2)
        logs, acts = [s_log], [s_a]
        if hr is not None and len(hr):
            maps = self.head(hr.float()).view(len(hr), C, bp, bp)
            m = F.avg_pool2d(maps, bp // c)                                      # (N, C, 4, 4)
            crop = rearrange(m, "n k u v -> n k (u v)")
            c_a = self.score(hr.float())[..., 0]
            cell = b * Q + y * W + x
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.bincount(cell_s, minlength=B * Q)
            M = int(count.max())
            slot = torch.arange(len(cell), device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, slot] = order
            idx = idx.view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            logs.append(crop.index_select(0, safe.reshape(-1)).view(B, Q, M, C, c * c))
            acts.append(c_a.index_select(0, safe.reshape(-1)).view(B, Q, M).masked_fill(~ok, float("-inf")))
        wgt = torch.softmax(torch.cat(acts, 2), 2).nan_to_num(0.0)
        out = torch.einsum("bqk,bqkcp->bqcp", wgt, torch.cat(logs, 2))
        return rearrange(out, "b (y x) k (i j) -> b k (y i) (x j)", y=H, x=W, i=c, j=c)

    def _decode_framed(self, lr_F, season, present, hr=None, b=None, py=None, px=None):
        """_decode for coarse tokens on a padded frame (6 px: 66 px), on 2 px cells. Fine tokens
        (top-left pixel (py, px) in the tile) cover parts of several cells, so the pool runs per
        PIXEL over the tokens covering it; the map is cropped back to the tile."""
        B, NS, h, w, D = lr_F.shape
        c, bp, C, fp = self.cell, self.coarse_patch, self.num_classes, self.fine_patch
        rs = bp // c
        H, W = h * rs, w * rs
        Q, dev = H * W, lr_F.device
        s_log = rearrange(self.head(lr_F.float()), "b s gh gw (k i j) -> b s k (gh i) (gw j)", k=C, i=bp, j=bp)
        s_log = rearrange(s_log, "b s k (y i) (x j) -> b (y x) s k (i j)", i=c, j=c)
        n_s = (F.one_hot(season.clamp(min=0), NS).float()
               * (present & (season >= 0)).float()[..., None]).sum(1)
        s_a = self.score(lr_F.float())[..., 0].masked_fill((n_s == 0)[:, :, None, None], float("-inf"))
        s_a = s_a.repeat_interleave(rs, 2).repeat_interleave(rs, 3).flatten(2).transpose(1, 2)
        logs, acts, cover = [s_log], [s_a], None
        if hr is not None and len(hr):
            N = len(hr)
            py, px = py + self.lo, px + self.lo                       # in the padded frame
            maps = F.avg_pool2d(self.head(hr.float()).view(N, C, bp, bp), bp // fp)   # (N, C, fp, fp)
            c_a_tok = self.score(hr.float())[..., 0]
            ar_c, ar_n = torch.arange(c, device=dev), torch.arange(N, device=dev)
            k = fp // c + 1                     # cells a token overlaps per side, at most
            dd = torch.arange(k, device=dev)
            da, db = dd.repeat_interleave(k), dd.repeat(k)
            cy, cx = py[None] // c + da[:, None], px[None] // c + db[:, None]       # (k^2, N)
            ui = cy[..., None] * c + ar_c - py[None, :, None]                      # (k^2, N, c)
            uj = cx[..., None] * c + ar_c - px[None, :, None]
            cov = (((ui >= 0) & (ui < fp))[..., :, None] & ((uj >= 0) & (uj < fp))[..., None, :]
                   & ((cy < H) & (cx < W) & (cy >= 0) & (cx >= 0))[..., None, None])
            keep = cov.flatten(2).any(2)
            g = maps[ar_n[None, :, None, None], :, ui.clamp(0, fp - 1)[..., :, None],
                     uj.clamp(0, fp - 1)[..., None, :]]                            # (k^2, N, c, c, C)
            crop = g.permute(0, 1, 4, 2, 3).flatten(3).flatten(0, 1)               # entries
            cov_e = cov.flatten(2).flatten(0, 1)
            cell = torch.where(keep, b * Q + cy * W + cx, B * Q).flatten()          # (B Q: none)
            c_a = c_a_tok[ar_n.expand(k * k, -1).flatten()]
            E = len(cell)
            order = torch.argsort(cell, stable=True)
            cell_s = cell[order]
            count = torch.zeros(B * Q + 1, dtype=torch.long, device=dev).index_add_(
                0, cell_s, torch.ones_like(cell_s))
            M = int(count[:B * Q].max())
            slot = torch.arange(E, device=dev) - (torch.cumsum(count, 0) - count)[cell_s]
            idx = torch.full((B * Q + 1, M), -1, dtype=torch.long, device=dev)
            idx[cell_s, torch.where(cell_s < B * Q, slot, 0)] = order
            idx = idx[:B * Q].view(B, Q, M)
            ok = idx >= 0
            safe = idx.clamp(min=0)
            take = lambda t: t.index_select(0, safe.reshape(-1)).view(*safe.shape, *t.shape[1:])  # noqa: E731
            logs.append(take(crop) * ok[..., None, None])
            acts.append(take(c_a).masked_fill(~ok, float("-inf")))
            cover = torch.cat([torch.ones(B, Q, NS, c * c, dtype=torch.bool, device=dev),
                               take(cov_e) & ok[..., None]], 2)
        a, cand = torch.cat(acts, 2), torch.cat(logs, 2)
        if cover is not None:
            wgt = torch.softmax(a[..., None].masked_fill(~cover, float("-inf")), 2).nan_to_num(0.0)
            out = torch.einsum("bqkp,bqkcp->bqcp", wgt, cand)
        else:
            out = torch.einsum("bqk,bqkcp->bqcp", torch.softmax(a, 2).nan_to_num(0.0), cand)
        img = rearrange(out, "b (y x) k (i j) -> b k (y i) (x j)", y=H, x=W, i=c, j=c)
        return img[..., self.lo:self.lo + TILE, self.lo:self.lo + TILE]

    # -- selection ------------------------------------------------------------------------
    def threshold(self, budget: float, table: torch.Tensor | None = None) -> torch.Tensor:
        """The window entropy above which a share `budget` of the (validation) windows lie."""
        q = self.quantiles if table is None else table
        v = (1 - budget) * (LEVELS - 1)
        i = min(int(v), LEVELS - 2)
        return q[i] + (v - i) * (q[i + 1] - q[i])

    def _pool(self, u: torch.Tensor) -> torch.Tensor:
        """(B, 64, 64) -> the mean per window (B, G, G), on the padded frame if windows do not
        divide the tile."""
        w = self.window_px
        if TILE % w:
            u = _pad(u, -(-TILE // w) * w)
        return F.avg_pool2d(u[:, None], w)[:, 0]

    def _season_disagreement(self, lr_F, used) -> torch.Tensor:
        """(B, NS, NS, G, G) the Jensen-Shannon divergence of two seasons' own coarse maps,
        averaged per window; -inf on the diagonal and at unused seasons."""
        B, NS, gh, gw, _ = lr_F.shape
        C, bp = self.num_classes, self.coarse_patch
        with self._amp():
            m = self.head(lr_F).view(-1, C, bp, bp)
        m = rearrange(m, "(b s gh gw) c i j -> b s c (gh i) (gw j)", s=NS, gh=gh, gw=gw)
        m = m[..., self.lo:self.lo + TILE, self.lo:self.lo + TILE].float()
        p = m.softmax(2)
        h = -(p * p.clamp(min=1e-8).log()).sum(2)                                # (B, NS, 64, 64)
        G = -(-TILE // self.window_px)
        D = m.new_full((B, NS, NS, G, G), float("-inf"))
        for a in range(NS):
            for b in range(a + 1, NS):
                pm = (p[:, a] + p[:, b]) / 2
                js = -(pm * pm.clamp(min=1e-8).log()).sum(1) - (h[:, a] + h[:, b]) / 2
                D[:, a, b] = D[:, b, a] = self._pool(js)
        ok = used[:, :, None] & used[:, None, :]
        return D.masked_fill(~ok[..., None, None], float("-inf"))

    def _select(self, logits0, lr_F, season, present, budget, extra: float = 0.0):
        """-> (tile, month, top-left y, x) of every selected window (window_px px). extra:
        the seasons beyond the first, 0..1 per selected window (the most uncertain first)."""
        p = logits0.float().softmax(1)
        u = self._pool(-(p * p.clamp(min=1e-8).log()).sum(1))                  # (B, G, G)
        where = (u >= self.threshold(budget)) & (u >= 0)
        B, T = season.shape
        NS = lr_F.shape[1]
        valid = present & (season >= 0)
        months = (F.one_hot(season.clamp(min=0), NS).float() * valid.float()[..., None]).sum(1)
        r = self.score(lr_F.float())[..., 0].masked_fill((months == 0)[:, :, None, None], float("-inf"))
        gh = r.shape[-1]
        f = self.coarse_patch // self.window_px         # windows per coarse cell side
        up = lambda m: m.repeat_interleave(f, -2).repeat_interleave(f, -1)       # noqa: E731
        if not self.extra:          # the most attended season's months, per coarse cell
            top = (r.argsort(1, descending=True).argsort(1) < 1) & torch.isfinite(r)
            when = top.gather(1, season.clamp(min=0)[:, :, None, None].expand(B, T, gh, gh)) & valid[:, :, None, None]
            sel_t = where[:, None] & up(when)                                    # (B, T, G, G)
        else:                       # (window, season) pairs: the top season + extra by threshold
            r = up(r)
            G = r.shape[-1]
            used = torch.isfinite(r)
            sel = where[:, None] & (r.argsort(1, descending=True).argsort(1) == 0) & used
            share = min(1.0, extra / max(float(self.ratio_x), 1e-6))
            if share > 0:           # 2nd: the season disagreeing most with the top one
                D = self._season_disagreement(lr_F, used.any(-1).any(-1))
                top1 = r.argmax(1)
                row = D.gather(1, top1[:, None, None].expand(-1, 1, NS, -1, -1))[:, 0]
                v, j = row.max(1)
                second = F.one_hot(j, NS).permute(0, 3, 1, 2).bool() & torch.isfinite(v)[:, None]
                cand = where[:, None] & second & used
                sel = sel | cand & (u[:, None] >= self.threshold(share, self.quantiles_x))
            sg = season.clamp(min=0)[:, :, None, None].expand(B, T, G, G)
            sel_t = sel.gather(1, sg) & valid[:, :, None, None]
        b, t, gy, gx = sel_t.nonzero(as_tuple=True)
        w, lo = self.window_px, _frame_lo(self.window_px)
        py, px = ((g * w - lo).clamp(0, TILE - w) for g in (gy, gx))           # frame -> tile
        return b, t, py, px

    def _split(self, b, t, py, px):
        """Each window as its (window_px / fine_patch)^2 fine tokens, row-major, consecutive."""
        f = self.window_px // self.fine_patch
        if f == 1:
            return b, t, py, px
        k, n = f * f, len(b)
        o = torch.arange(f, device=b.device) * self.fine_patch
        dy, dx = o.repeat_interleave(f).repeat(n), o.repeat(f).repeat(n)
        r = lambda v: v.repeat_interleave(k)                                    # noqa: E731
        return r(b), r(t), r(py) + dy, r(px) + dx

    def _extra(self, seasons: float) -> float:
        if not 1 <= seasons <= 2:
            raise ValueError(f"seasons {seasons}: 1 to 2 per selected window")
        if seasons > 1 and not self.extra:
            raise ValueError("this model re-encodes one season per window (seasons=1)")
        return seasons - 1.0

    # -- inference ------------------------------------------------------------------------
    @torch.no_grad()
    def predict_tiles(self, x: Inputs, budget: float = 0.5, seasons: float = 1) -> Prediction:
        """A batch of tiles -> per-tile prediction. budget: the share of windows re-encoded,
        on the validation distribution (config["train_budget"] as trained; lower budgets cost
        less). seasons: models trained with a season range (config["seasons"]) re-encode 1 to
        2 seasons per selected window; 1.5: a second season for the most uncertain half."""
        extra = self._extra(seasons)
        x = x.to(self.device, torch.float32 if self.device.type == "cpu" else torch.bfloat16)
        season, present = x.season, x.present
        # bf16 autocast for the backbone and the coarse prediction; selection and the final
        # decoder in fp32 (as in training)
        with self._amp():
            lr, trace, lr_F = self._coarse(x)
            logits0 = self._decode(lr_F, season, present).float()
        wb, wt, wy, wx = self._select(logits0, lr_F, season, present, budget, extra)
        windows = torch.stack([wb, wt, wy, wx], 1).cpu()
        if len(wb) == 0:
            return Prediction(logits0, logits0, windows, int(len(lr.x)), 0, self.window_px)
        b, t, py, px = self._split(wb, wt, wy, wx)
        fp = self.fine_patch
        with self._amp():
            crops = _crop(x.months, b, t, py, px, fp)
            hr = self._embed(crops, fp, slot=season[b, t], py=py + self.lo, px=px + self.lo)
            hr_x = self._fine(lr, trace, hr, b)
        hr_F = hr.pooled(hr_x).float()[:, 0, 0, 0]
        if self.lo:     # (padded frame: tokens placed by pixel)
            logits = self._decode(lr_F, season, present, hr_F, b, py, px).float()
        else:           # (4 px tokens on the 4 px cells)
            logits = self._decode(lr_F, season, present, hr_F, b, py // CELL, px // CELL).float()
        return Prediction(logits, logits0, windows, int(len(lr.x)), int(len(hr.x)), self.window_px)

    @torch.no_grad()
    def predict(self, x: Inputs, budget: float = 0.5, seasons: float = 1,
                batch_size: int = 32) -> Prediction:
        """An image prepared by mistr.prepare -> its prediction (C, H, W), tiles stitched
        (budget, seasons: see predict_tiles)."""
        outs = [self.predict_tiles(x.tiles(i, i + batch_size), budget, seasons)
                for i in range(0, len(x), batch_size)]
        (nh, nw), (H, W) = x.grid, x.size

        def stitch(k):
            a = torch.cat([getattr(o, k) for o in outs])                         # (n, C, 64, 64)
            a = a.view(nh, nw, *a.shape[1:]).permute(2, 0, 3, 1, 4).reshape(a.shape[1], nh * TILE, nw * TILE)
            return a[:, :H, :W]

        w, base = [], 0
        for i, o in enumerate(outs):
            v = o.windows.clone()
            v[:, 0] += i * batch_size
            w.append(v)
        w = torch.cat(w)
        ty, tx = w[:, 0] // nw, w[:, 0] % nw
        win = torch.stack([w[:, 1], ty * TILE + w[:, 2], tx * TILE + w[:, 3]], 1)
        win = win[(win[:, 1] < H) & (win[:, 2] < W)]
        return Prediction(stitch("logits"), stitch("coarse_logits"), win,
                          sum(o.coarse_tokens for o in outs), sum(o.fine_tokens for o in outs),
                          self.window_px)


def _crop(sample, b, t, y, x, size):
    """The size x size px block at top-left pixel (y, x) of month t of tile b -> N one-month samples."""
    r = y[:, None] + torch.arange(size, device=y.device)
    c = x[:, None] + torch.arange(size, device=x.device)

    def crop(k, v):
        if k == "timestamps":
            return v[b, t][:, None]
        return v[b[:, None, None], r[:, :, None], c[:, None, :], t[:, None, None]][:, :, :, None]

    return sample._replace(**{k: crop(k, v) for k, v in sample.as_dict().items() if v is not None})
