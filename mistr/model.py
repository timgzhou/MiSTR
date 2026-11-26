"""MiSTR on OlmoEarth: a coarse pass over seasonal composites, a selection of where and when the
prediction needs more detail, and a sparse fine pass over monthly images at those points.

    model = MiSTR.from_pretrained("TimZhou/mistr-olmoearth-tiny-pastis")
    out = model.predict(prepare(...), budget=0.5)

1. Coarse pass: every season's composite as coarse_patch px tokens (OlmoEarth's patch embedding
   and encodings), through the backbone; each block's input tokens are kept.
2. Where: the coarse prediction's per-pixel entropy, averaged over each fine_patch px window;
   a window is picked when its mean is above a global threshold, the (1 - budget) quantile of the
   training distribution (a table saved with the weights).
3. When: per coarse cell, the season the decoder's attention score ranks highest; a picked
   window takes that season's months.
4. Fine pass: each picked (window, month) as one fine_patch px token, placed at its pixel position
   and its season's time slot. Fine tokens read their tile's coarse tokens at the same depth and
   the tile's other fine tokens (one-way attention: coarse tokens never read fine ones, so the
   coarse pass is reused, not recomputed).
5. Decoder: a linear head maps every token to a small logit map; each 4 px cell pools the maps
   of the tokens covering it (its coarse token in every season, and its fine tokens), weighted by
   a softmax over a learned per-token score.
"""
from __future__ import annotations

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


def _sincos_2d(py, px, n):
    return torch.cat([get_1d_sincos_pos_encoding(px.float(), n // 2),
                      get_1d_sincos_pos_encoding(py.float(), n // 2)], 1)


@dataclass
class Prediction:
    """logits            (C, H, W) MiSTR's prediction (image) or (B, C, 64, 64) (tiles)
    coarse_logits       the coarse pass's own prediction, same shape
    windows             (N, 3) long: (month 0-11, y, x) of each fine token, top-left pixel in the
                        image (tiles: (N, 4), the tile first)
    coarse_tokens, fine_tokens   tokens encoded"""
    logits: torch.Tensor
    coarse_logits: torch.Tensor
    windows: torch.Tensor
    coarse_tokens: int
    fine_tokens: int

    @property
    def classes(self) -> torch.Tensor:
        return self.logits.argmax(-3)


class MiSTR(nn.Module):
    def __init__(self, config: dict, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.coarse_patch, self.fine_patch = config["coarse_patch"], config["fine_patch"]
        if (self.coarse_patch, self.fine_patch) != (8, 4):
            raise NotImplementedError("this release: coarse 8 px, fine 4 px tokens")
        self.backbone = backbone
        for m in backbone.modules():
            if hasattr(m, "use_flash_attn"):
                m.use_flash_attn = True
        D = backbone.embedding_size
        self.num_classes = config["num_classes"]
        self.head = nn.Linear(D, self.num_classes * self.coarse_patch ** 2)
        self.score = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.register_buffer("quantiles", torch.zeros(LEVELS))   # window entropy, train set

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str = "cuda") -> "MiSTR":
        """A Hugging Face model repo or a local folder with config.json, backbone/config.json
        (OlmoEarth's architecture) and model.safetensors."""
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
        return model.to(device).eval()

    @property
    def device(self):
        return self.head.weight.device

    # -- tokens ---------------------------------------------------------------------------
    def _embed(self, sample, patch: int, slot=None, py=None, px=None) -> Packed:
        """Patch embedding + OlmoEarth's encodings, packed. Fine tokens (one patch, T = 1): time
        slot 0 moved to `slot`, position (0, 0) to their centre in the coarse grid's frame."""
        enc = self.backbone
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
        for blk in self.backbone.blocks:
            trace.append(h)
            h = blk(x=h, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
        return lr, trace, lr.pooled(self.backbone.norm(h)).float()

    def _fine(self, lr: Packed, trace, crops: Packed, owner: torch.Tensor) -> torch.Tensor:
        """The fine tokens through every block, reading their tile's coarse tokens at that block
        and the tile's fine tokens -> (N, D) normed."""
        import flash_attn
        S, dev = len(lr.seqlens), lr.x.device
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
            out = flash_attn.flash_attn_varlen_func(q, k.to(q.dtype), v.to(q.dtype), cu_q, cu_k,
                                                    max_q, max_k, softmax_scale=a.scale)
            y = torch.empty_like(x_c)
            y[order] = blk.ls1(a.proj(out.reshape(len(out), -1))).to(x_c.dtype)
            x_c = x_c + y
            x_c = x_c + blk.ls2(blk.mlp(blk.norm2(x_c)))
        return self.backbone.norm(x_c)

    # -- decoder --------------------------------------------------------------------------
    def _decode(self, lr_F, season, present, hr=None, b=None, y=None, x=None):
        """Per 4 px cell, the softmax(score)-weighted mean of the logit maps of the tokens
        covering it -> (B, C, 64, 64). lr_F (B, NS, h, w, D) coarse features; hr (N, D) fine
        features at cells (b, y, x) of the 16 x 16 cell grid."""
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

    # -- selection ------------------------------------------------------------------------
    def threshold(self, budget: float) -> torch.Tensor:
        """The window entropy above which a share `budget` of the training windows lie."""
        v = (1 - budget) * (LEVELS - 1)
        i = min(int(v), LEVELS - 2)
        return self.quantiles[i] + (v - i) * (self.quantiles[i + 1] - self.quantiles[i])

    def _select(self, logits0, lr_F, season, present, budget):
        """-> (tile, month, cell y, cell x) of every fine token."""
        p = logits0.float().softmax(1)
        u = F.avg_pool2d((-(p * p.clamp(min=1e-8).log()).sum(1))[:, None], self.fine_patch)[:, 0]
        where = (u >= self.threshold(budget)) & (u >= 0)                       # (B, 16, 16)
        B, T = season.shape
        NS = lr_F.shape[1]
        valid = present & (season >= 0)
        months = (F.one_hot(season.clamp(min=0), NS).float() * valid.float()[..., None]).sum(1)
        r = self.score(lr_F.float())[..., 0].masked_fill((months == 0)[:, :, None, None], float("-inf"))
        top = (r.argsort(1, descending=True).argsort(1) < 1) & torch.isfinite(r)  # (B, NS, h, w)
        gh = r.shape[-1]
        when = top.gather(1, season.clamp(min=0)[:, :, None, None].expand(B, T, gh, gh)) & valid[:, :, None, None]
        f = self.coarse_patch // self.fine_patch
        when = when.repeat_interleave(f, 2).repeat_interleave(f, 3)              # (B, T, 16, 16)
        return (where[:, None] & when).nonzero(as_tuple=True)

    # -- inference ------------------------------------------------------------------------
    @torch.no_grad()
    def predict_tiles(self, x: Inputs, budget: float = 0.5) -> Prediction:
        """A batch of tiles -> per-tile prediction. budget: the share of training windows that
        would be re-encoded (the model was trained at 0.5; lower budgets cost less)."""
        x = x.to(self.device)
        season, present = x.season, x.present
        # bf16 autocast for the backbone and the coarse prediction; selection and the final
        # decoder in fp32 (as in training)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr, trace, lr_F = self._coarse(x)
            logits0 = self._decode(lr_F, season, present).float()
        b, t, y, xx = self._select(logits0, lr_F, season, present, budget)
        if len(b) == 0:
            return Prediction(logits0, logits0, torch.zeros(0, 4, dtype=torch.long),
                              int(len(lr.x)), 0)
        fp = self.fine_patch
        py, px = y * fp, xx * fp
        with torch.autocast("cuda", dtype=torch.bfloat16):
            crops = _crop(x.months, b, t, py, px, fp)
            hr = self._embed(crops, fp, slot=season[b, t], py=py, px=px)
            hr_x = self._fine(lr, trace, hr, b)
        hr_F = hr.pooled(hr_x).float()[:, 0, 0, 0]
        # (fine cells on the 4 px grid: fine_patch == CELL)
        logits = self._decode(lr_F, season, present, hr_F, b, y, xx).float()
        return Prediction(logits, logits0, torch.stack([b, t, py, px], 1).cpu(),
                          int(len(lr.x)), int(len(hr.x)))

    @torch.no_grad()
    def predict(self, x: Inputs, budget: float = 0.5, batch_size: int = 32) -> Prediction:
        """An image prepared by mistr.prepare -> its prediction (C, H, W), tiles stitched."""
        outs = [self.predict_tiles(x.tiles(i, i + batch_size), budget)
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
                          sum(o.coarse_tokens for o in outs), sum(o.fine_tokens for o in outs))


def _crop(sample, b, t, y, x, size):
    """The size x size px block at top-left pixel (y, x) of month t of tile b -> N one-month samples."""
    r = y[:, None] + torch.arange(size, device=y.device)
    c = x[:, None] + torch.arange(size, device=x.device)

    def crop(k, v):
        if k == "timestamps":
            return v[b, t][:, None]
        return v[b[:, None, None], r[:, :, None], c[:, None, :], t[:, None, None]][:, :, :, None]

    return sample._replace(**{k: crop(k, v) for k, v in sample.as_dict().items() if v is not None})
