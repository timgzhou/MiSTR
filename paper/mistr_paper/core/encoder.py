"""MiSTR's encoder: OlmoEarth (or another backbone with mixres_embed) on a coarse token
sequence plus fine tokens of selected windows, with shared weights (paper Sec. 3.1, 3.3).

    embed       patch embedding + the backbone's encodings, packed (varlen, no padding). A
                fine token takes the time slot of the coarse step it belongs to, its own month,
                and the position of its CENTRE in the coarse frame (py + (ps - lr_ps) / 2), so
                a large patch sits at the mean position of the small patches that tile it
    tail_multi  the transformer blocks + final norm on the coarse tokens and the fine tokens:
                attn "joint"     one sequence per sample (fine and coarse tokens read each other)
                attn "cross"     the coarse sequence on its own; fine tokens read only their
                                 sample's coarse tokens at the same depth
                attn "cross_hr"  one-way attention: as cross, + every fine token of the sample
    In the cross modes the coarse tokens never read fine ones, so inference reuses the coarse
    pass (lr_trace) and pays only for the fine tokens. hr_type is a learned type embedding
    added to fine tokens (MiSTR keeps it at zero; the GEOID-Flood tracking models learn it).

Everything runs through varlen FlashAttention: sequences are packed back to back (cu_seqlens),
so tokens never cross samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue, TokensAndMasks
import olmoearth_pretrain.nn.flexi_vit as _fv
from olmoearth_pretrain.nn.encodings import (get_1d_sincos_pos_encoding,
                                             get_2d_sincos_pos_encoding_with_resolution)
from olmoearth_pretrain.nn.flexi_vit import PositionEncoding
from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths
from torch import nn

import mistr_paper.core as _pkg     # (FAST, read at call time)
from mistr_paper.encoders.olmoearth import _pad_space, _spatial_size, pool_timestep


def _cu(seqlens: torch.Tensor) -> torch.Tensor:
    """cu_seqlens (S + 1,) int32. FAST: without upstream's torch.tensor([0]) host -> device copy
    (a sync per call)."""
    if _pkg.FAST:
        return F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return get_cumulative_sequence_lengths(seqlens)


def pool_steps(tam: TokensAndMasks) -> torch.Tensor:
    """(B, T, gH, gW, D): pool_timestep for every step at once, its two branches per step (all
    tokens online / not) chosen on the GPU (torch.where) instead of by a sync each."""
    feats, weights = [], []
    for m in tam.modalities:
        x = getattr(tam, m)                                       # (B, gH, gW, T, S, D)
        mask = getattr(tam, tam.get_masked_modality_name(m))
        online = mask == MaskValue.ONLINE_ENCODER.value           # (B, gH, gW, T, S)
        all_t = online.movedim(3, 0).flatten(1).all(1)            # (T,)
        w = online.unsqueeze(-1).to(x.dtype)
        cnt = w.sum(dim=-2)                                       # (B, gH, gW, T, 1)
        general = (x * w).sum(dim=-2) / cnt.clamp(min=1)
        sel = all_t[:, None]                                      # (T, 1) -> (.., T, D)
        feats.append(torch.where(sel, torch.mean(x, dim=-2), general))
        weights.append(torch.where(all_t, torch.ones_like(cnt[..., 0]),
                                   (cnt[..., 0] > 0).to(x.dtype)))
    f, w = torch.stack(feats, -1), torch.stack(weights, -1)      # (B,gH,gW,T,D,M), (B,gH,gW,T,M)
    ones = (w == 1).movedim(3, 0).flatten(1).all(1)              # (T,)
    out = torch.where(ones[:, None], f.mean(dim=-1),
                      (f * w.unsqueeze(-2)).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1))
    return out.movedim(3, 1).contiguous()     # (the stacked layout: a strided view changes the
    #                                           readout GEMM's kernel, so its bf16 bits)


# -- inference: the input-independent encodings as cached tables (FAST) ---------------------------
# OlmoEarth's encodings (CompositeEncodings) and the crops' slot / position shifts are built from
# arange / sin / cos / einsum, dozens of tiny kernels per call; every piece is elementwise in its
# position (or a constant), so a table computed once, IN THE SAME AUTOCAST STATE (the sincos
# einsum runs in bf16 under autocast), and indexed gives the same bits. Eval + no_grad only.
_TABLES: dict = {}
POS_RANGE = 4096        # crop pixel positions in [-POS_RANGE, POS_RANGE) (tiles are <= a few 100 px)
SLOT_RANGE = 128        # time slots in [0, SLOT_RANGE)


def _cache_ok(module: nn.Module) -> bool:
    return _pkg.FAST and not torch.is_grad_enabled() and not module.training


def _amp_key(dev) -> tuple:
    return (str(dev), torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())


def composite_encodings(ce, tokens: dict, timestamps: torch.Tensor, patch_size: int,
                        input_res: int = BASE_GSD) -> dict:
    """ce.forward(tokens, timestamps, patch_size, input_res), bit-exact: a (h, w, t, band sets, D)
    table of the channel + slot + spatial parts per shape (cached), the month part per batch."""
    if not _cache_ok(ce):
        return ce.forward(tokens, timestamps, patch_size, input_res)
    out = {}
    names = _fv.get_modalities_to_process(_fv.return_modalities_from_dict(tokens),
                                          ce.supported_modality_names)
    for name in names:
        t = tokens[name]
        mod = Modality.get(name)
        if (t.ndim != 6 or not mod.is_multitemporal or not mod.is_spatial
                or ce.position_encoding != PositionEncoding.ABSOLUTE):
            out[name] = ce._apply_encodings_per_modality(name, t, timestamps=timestamps,
                                                         patch_size=patch_size, input_res=input_res)
            continue
        b, h, w, T, bs, D = t.shape
        dev, n = t.device, ce.embedding_dim_per_embedding_type
        chan = ce.per_modality_channel_embeddings[mod.name]
        # one table per shape, rebuilt when the (learned) channel embeddings change: keyed on the
        # tensor's version, only the newest kept (one per optimizer step accumulated on the GPU
        # in training's eval-mode coarse passes: ~0.8 GB / epoch)
        key = ("composite", name, h, w, T, bs, D, patch_size, input_res, _amp_key(dev))
        ver = (chan.data_ptr(), chan._version)
        got = _TABLES.get(key)
        st = got[1] if got is not None and got[0] == ver else None
        if st is None:      # (as _apply_encodings_per_modality: zeros, then += each part)
            if chan.shape[0] != bs:
                raise ValueError(f"{name}: {chan.shape[0]} channel embeddings, {bs} band sets")
            st = torch.zeros((h, w, T, bs, D), device=dev)
            st[..., :n] += chan.to(dev)
            st[..., n:2 * n] += get_1d_sincos_pos_encoding(torch.arange(T, device=dev), n)[:, None]
            sp = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=(h, w), res=torch.ones(1, device=dev) * ce.calculate_gsd_ratio(
                    input_res, patch_size), encoding_dim=n, device=dev)
            st[..., 3 * n:4 * n] += sp.reshape(h, w, n)[:, :, None, None]
            _TABLES[key] = (ver, st)
        o = t + st                                   # (tokens + 0 = tokens: the month slot
        o[..., 2 * n:3 * n] += ce.month_embed(timestamps[:, :, 1])[:, None, None, :, None]  # after)
        out[name] = o
    return out


def slot_shift(slot: torch.Tensor, n: int) -> torch.Tensor:
    """get_1d_sincos(slot) - get_1d_sincos(0) per crop (N, n), from a cached table."""
    key = ("slot", n, _amp_key(slot.device))
    tab = _TABLES.get(key)
    if tab is None:
        ar = torch.arange(SLOT_RANGE, device=slot.device).float()
        tab = _TABLES[key] = (get_1d_sincos_pos_encoding(ar, n)
                              - get_1d_sincos_pos_encoding(torch.zeros(SLOT_RANGE, device=slot.device), n))
    return tab[slot]


def pos_shift(py: torch.Tensor, px: torch.Tensor, off: float, n: int) -> torch.Tensor:
    """sincos_2d(py + off, px + off) - sincos_2d(0, 0) per crop (N, n) for integer pixel positions,
    from a cached 1D table (sincos_2d = [1D(x), 1D(y)], each elementwise)."""
    dev = py.device
    key = ("pos", off, n, _amp_key(dev))
    tab = _TABLES.get(key)
    if tab is None:
        ar = torch.arange(-POS_RANGE, POS_RANGE, device=dev).float() + off
        tab = _TABLES[key] = (get_1d_sincos_pos_encoding(ar, n // 2)
                              - get_1d_sincos_pos_encoding(torch.zeros(len(ar), device=dev), n // 2))
    return torch.cat([tab[px + POS_RANGE], tab[py + POS_RANGE]], 1)


@dataclass
class Packed:
    """Valid tokens of S sequences, back to back."""
    x: torch.Tensor             # (N_valid, D)
    seqlens: torch.Tensor       # (S,) long
    keep: torch.Tensor          # (S, N) bool: which of each sequence's N token slots are valid
    dims: dict                  # modality -> token shape (S, h, w, [t,] bandsets, D)
    masks: dict                 # masked-modality name -> (S, h, w, [t,] bandsets) mask values
    yx: torch.Tensor | None = None  # crops: (S, 2) centre pixel (y, x) of each crop
    trace: list | None = None       # lr: its tokens entering each block of [0, L) (head_attn)
    typed: bool = False             # crops: hr_type already added (head_attn)
    step: torch.Tensor | None = None  # (N_valid,) each token's time step (per-step backbones)
    image: torch.Tensor | None = None  # (N_valid,) crops: the image (month) it is cut from (")

    def subset(self, sel: torch.Tensor) -> "Packed":
        """The sequences sel (S,) bool, their tokens included (crops: a part of the picks)."""
        tok = torch.repeat_interleave(sel, self.seqlens)
        return Packed(self.x[tok], self.seqlens[sel], self.keep[sel],
                      {m: (int(sel.sum()),) + tuple(d[1:]) for m, d in self.dims.items()},
                      {m: v[sel] for m, v in self.masks.items()},
                      None if self.yx is None else self.yx[sel], None, self.typed,
                      None if self.step is None else self.step[tok],
                      None if self.image is None else self.image[tok])

    def pooled(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """(S, T, h, w, D): per step, mean over bandsets then over present modalities
        (mistr_paper.encoders.olmoearth.pool_timestep, i.e. OlmoEarthEncoder's output)."""
        x = self.x if x is None else x
        S, N = self.keep.shape
        full = x.new_zeros(S * N, x.shape[-1])
        if _pkg.FAST:       # (boolean index assignment syncs; masked_scatter does not)
            full.masked_scatter_(self.keep.flatten()[:, None], x)
        else:
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
        if _pkg.FAST and all(v.ndim == 6 for v in per.values()):
            return pool_steps(tam)
        return torch.stack([pool_timestep(tam, t) for t in range(T)], 1)


_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def sincos_2d(py: torch.Tensor, px: torch.Tensor, dim: int) -> torch.Tensor:
    """OlmoEarth's 2D spatial encoding at (possibly fractional) pixel positions -> (N, dim);
    equals get_2d_sincos_pos_encoding_with_resolution on a grid (x first, then y)."""
    return torch.cat([get_1d_sincos_pos_encoding(px.float(), dim // 2),
                      get_1d_sincos_pos_encoding(py.float(), dim // 2)], 1)


ATTN = ("joint", "cross", "cross_hr")
CONTEXT = ("season", "neighbor", "all")      # per_step one-way: what an image's crops read
MAX_SEQS = 32768            # sequences per FlashAttention varlen call (see OneWayEncoder.blocks)


class OneWayEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, inject_layer: int = 0, pos: str = "center",
                 lr_ps: int = 16, attn: str = "joint"):
        super().__init__()
        if inject_layer != 0 or pos != "center":
            raise ValueError("OneWayEncoder: only inject_layer 0 and pos 'center' are supported")
        if attn not in ATTN:
            raise ValueError(f"attn {attn!r} not in {ATTN}")
        if not encoder.blocks[0].attn.use_flash_attn:
            raise ValueError("OneWayEncoder packs sequences for varlen FlashAttention")
        self.attn = attn
        # per_step (a single-time backbone, mistr_paper.encoders.terramind): one attention sequence
        # per (sample, time step) instead of per sample. Crops: attn joint, they join their
        # (owner, slot) sequence; one-way, the crops of one image (owner, month) read each other
        # (cross_hr) and their context, the cached coarse tokens of: "season" their slot;
        # "neighbor" their slot + the adjacent slot on the side of the crop's month (none for
        # the slot's own month; no wrap at the series' ends); "all" every step of the sample
        self.per_step = bool(getattr(encoder, "per_step", False))
        self.context = getattr(encoder, "context", "season")
        if self.context not in CONTEXT:
            raise ValueError(f"context {self.context!r} not in {CONTEXT}")
        self.encoder, self.L = encoder, 0
        self.hr_type = nn.Parameter(torch.zeros(encoder.embedding_size))
        self.pos, self.lr_ps = pos, lr_ps

    # -- embedding ------------------------------------------------------------------------
    def embed(self, sample: MaskedOlmoEarthSample, patch_size: int, slot: torch.Tensor | None = None,
              py: torch.Tensor | None = None, px: torch.Tensor | None = None,
              times: torch.Tensor | None = None) -> Packed:
        """Patch embedding + OlmoEarth's encodings, packed. For crops (T = 1, one patch each),
        `slot` (S,) replaces time slot 0 and (py, px) (S,) top-left pixel positions replace
        (0, 0) (moved to the crop's centre in the lr frame under pos "center"). `times` (S, T)
        replaces the time slots 0 .. T - 1 of a whole sequence."""
        enc = self.encoder
        if times is not None and hasattr(enc, "mixres_embed"):
            raise ValueError(f"{type(enc).__name__}: no time override for the season tokens")
        if slot is None and _spatial_size(sample) % patch_size:   # ps12 seasons on 64 px: the
            hs = _spatial_size(sample)                            # tile padded (readout out_px)
            sample = _pad_space(sample, -(-hs // patch_size) * patch_size)
        if hasattr(enc, "mixres_embed"):    # a non-OlmoEarth backbone tokenizes itself, tokens
            # placed at their centres (mistr_paper.encoders.terramind)
            out = enc.mixres_embed(sample, patch_size, slot, py, px)
            x, mask, dims, masks = out[:4]
            keep = mask == MaskValue.ONLINE_ENCODER.value
            yx = (torch.stack([py, px], 1).float() + patch_size / 2) if slot is not None else None
            step = out[4][keep] if len(out) > 4 else None       # (mistr_paper.encoders.terramind)
            image = out[5][keep] if len(out) > 5 else None
            return Packed(x[keep], keep.sum(1), keep, dims, masks, yx, step=step, image=image)
        x = enc.patch_embeddings.forward(sample, patch_size)
        tokens, masks, dims = enc.split_tokens_masks_and_dims(x)
        tokens = composite_encodings(enc.composite_encodings, tokens, sample.timestamps,
                                     patch_size, BASE_GSD)
        if times is not None:
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            S, T = times.shape
            d_time = (get_1d_sincos_pos_encoding(times.reshape(-1).float(), n).view(S, T, n)
                      - get_1d_sincos_pos_encoding(torch.arange(T, device=times.device), n))
            for m, t in tokens.items():
                if not Modality.get(m).is_multitemporal:
                    continue
                delta = t.new_zeros(S, T, t.shape[-1])
                delta[:, :, n:2 * n] = d_time.to(t.dtype)
                lead = (1,) * (t.ndim - 4)                                     # h, w if spatial
                tokens[m] = t + delta.view((S,) + lead + (T, 1, t.shape[-1]))
        if slot is not None:
            n = enc.composite_encodings.embedding_dim_per_embedding_type
            dev = next(iter(tokens.values())).device
            zero = torch.zeros_like(py)
            yx = torch.stack([py, px], 1).float() + patch_size / 2             # true centre
            if (_cache_ok(enc) and not py.is_floating_point() and not px.is_floating_point()
                    and not slot.is_floating_point()):
                d_time = slot_shift(slot.to(dev), n)
                d_pos = pos_shift(py.to(dev), px.to(dev), (patch_size - self.lr_ps) / 2, n)
            else:
                py, px = (v.float() + (patch_size - self.lr_ps) / 2 for v in (py, px))  # centre
                d_time = (get_1d_sincos_pos_encoding(slot.float().to(dev), n)
                          - get_1d_sincos_pos_encoding(zero.float().to(dev), n))
                d_pos = (sincos_2d(py.to(dev), px.to(dev), n)
                         - sincos_2d(zero.to(dev), zero.to(dev), n))
            for m, t in tokens.items():
                if t.shape[1:4] != (1, 1, 1):
                    raise ValueError(f"crop tokens of {m} are {tuple(t.shape)}, want one patch")
                mod = Modality.get(m)
                delta = torch.zeros_like(t[:, 0, 0, 0, 0])                     # (S, D)
                if mod.is_multitemporal:
                    delta[:, n:2 * n] = d_time
                if mod.is_spatial:
                    delta[:, 3 * n:4 * n] = d_pos
                tokens[m] = t + delta[:, None, None, None, None].to(t.dtype)
        both = {**tokens, **masks}
        x, mask = enc.collapse_and_combine_hwtc(both)
        keep = mask == MaskValue.ONLINE_ENCODER.value
        return Packed(x[keep], keep.sum(1), keep, dims, masks, yx if slot is not None else None)

    # -- transformer ------------------------------------------------------------------------
    def blocks(self, x: torch.Tensor, seqlens: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        if lo >= hi:
            return x
        if len(seqlens) > MAX_SEQS:      # FlashAttention's varlen kernel fails ("invalid
            # configuration argument") past ~65k sequences in one call (GEOID series: 64 chips x
            # ~1000 one-token crops); sequences are independent, so chunks give the same result
            ends = torch.cumsum(seqlens, 0)
            out = []
            for i in range(0, len(seqlens), MAX_SEQS):
                a = int(ends[i - 1]) if i else 0
                b = int(ends[min(i + MAX_SEQS, len(seqlens)) - 1])
                out.append(self.blocks(x[a:b], seqlens[i:i + MAX_SEQS], lo, hi))
            return torch.cat(out)
        cu = _cu(seqlens)
        mx = int(seqlens.max())
        for blk in self.encoder.blocks[lo:hi]:
            x = blk(x=x, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
        return x

    def head_blocks(self, p: Packed, lr: Packed | None = None,
                    owner: torch.Tensor | None = None) -> Packed:
        """Blocks before the fine tokens join: none (they join at the first block)."""
        return p

    def _typed(self, p: Packed, dtype) -> torch.Tensor:
        return (p.x if p.typed else p.x + self.hr_type).to(dtype)

    def _lr_trace(self, x: torch.Tensor, seqlens: torch.Tensor, lo: int, hi: int
                  ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """blocks [lo, hi) on the lr sequences -> (their tokens entering each block, output)."""
        cu, mx, trace = _cu(seqlens), int(seqlens.max()), []
        self._kv = None
        kv = [] if _pkg.FAST and not torch.is_grad_enabled() and not self.training else None
        for blk in self.encoder.blocks[lo:hi]:
            trace.append(x)
            if kv is not None:      # (inference: the blocks' own k / v of these tokens, reused by
                got = self._kv_hooks(blk)                              # the one-way blocks)
                got["on"] = True
            x = blk(x=x, cu_seqlens=cu, max_seqlen=mx, attn_mask=None, rope_positions=None)
            if kv is not None:
                got["on"] = False
                k, v = got.pop("k", None), got.pop("v", None)
                kv = kv + [(k, v)] if k is not None and v is not None else None  # (no hook fired:
        self._kv = (trace[0], kv) if kv and trace else None              # recompute instead)
        return trace, x

    def _kv_hooks(self, blk) -> dict:
        """A dict the block's attention k / v projections write their outputs to while its "on"
        is set (forward hooks, installed once per block)."""
        a = blk.attn
        if not hasattr(a, "_mistr_kv"):
            d = a._mistr_kv = {"on": False}

            def hook(name):
                def f(m, i, o):
                    if d["on"]:
                        d[name] = o
                return f
            a.k.register_forward_hook(hook("k"))
            a.v.register_forward_hook(hook("v"))
        return a._mistr_kv

    def tail(self, lr: Packed, hr: Packed | None, owner: torch.Tensor | None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """blocks [L, n) + norm on each lr sequence joined by the hr crops it owns
        (owner (N_crops,) = lr sequence index). -> (lr tokens (N_lr, D), hr tokens)."""
        if hr is None or owner is None or len(owner) == 0:
            return self.tail_multi(lr, [])[0], None
        lr_x, (hr_x,) = self.tail_multi(lr, [(hr, owner)])
        return lr_x, hr_x

    def tail_multi(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]],
                   lr_enc: tuple[list[torch.Tensor], torch.Tensor] | None = None
                   ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tail with several crop sets (e.g. month crops, S2 date crops, S1 date crops), each
        (Packed, owner (N_set,)); every lr sequence is joined by the crops of every set it
        owns. -> (lr tokens (N_lr, D), [tokens of each set]). lr_enc: lr_trace(lr) of the
        crop-free pass, reused by the attn modes where lr never reads crops."""
        if self.per_step and self.attn != "joint" and any(len(o) for _, o in crop_sets):
            return self._oneway_by_image(lr, [(p, o) for p, o in crop_sets if len(o)], lr_enc)
        if self.per_step:
            lr_g, sets_g, order = self._by_step(lr, crop_sets)
            if lr_enc is not None:      # (lr_trace: its trace is in step order, output not)
                lr_enc = (lr_enc[0], lr_enc[1][order])
            y, outs = self._tail_multi(lr_g, sets_g, lr_enc)
            out = torch.empty_like(y)
            out[order] = y
            return out, outs
        return self._tail_multi(lr, crop_sets, lr_enc)

    STEPS = 64          # time steps per sequence at most (per_step keys: sequence * STEPS + step)

    def _by_step(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]] = ()
                 ) -> tuple[Packed, list[tuple[Packed, torch.Tensor]], torch.Tensor]:
        """per_step: lr's sequences split by time step -> (the lr tokens regrouped into one
        sequence per (sample, step) present among lr tokens and crops, sorted; each crop set
        with its owner remapped to its (owner, slot) group; the lr token order used)."""
        if lr.step is None:
            raise ValueError("per_step: the backbone gives no token steps (Packed.step)")
        dev, K = lr.x.device, self.STEPS
        S = len(lr.seqlens)
        k_lr = torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens) * K + lr.step
        k_c = [o.to(dev) * K + self._per_crop(p, p.step) for p, o in crop_sets]
        groups = torch.unique(torch.cat([k_lr] + k_c))                       # sorted
        g_lr = torch.searchsorted(groups, k_lr)
        order = torch.argsort(g_lr, stable=True)
        lr_g = Packed(lr.x[order], torch.bincount(g_lr, minlength=len(groups)), None, {}, {})
        self._groups = groups               # (the (sample, step) keys, for _oneway_by_image)
        return lr_g, [(p, torch.searchsorted(groups, k)) for (p, _), k in zip(crop_sets, k_c)], order

    @staticmethod
    def _per_crop(p: Packed, v: torch.Tensor) -> torch.Tensor:
        """A per-token value constant within each crop -> (C,) per crop (0: a crop with no
        tokens)."""
        out = torch.zeros(len(p.seqlens), dtype=torch.long, device=p.x.device)
        if len(p.x):
            out[torch.repeat_interleave(torch.arange(len(p.seqlens), device=p.x.device),
                                        p.seqlens)] = v
        return out

    def _oneway_by_image(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]],
                         lr_enc: tuple[list[torch.Tensor], torch.Tensor] | None
                         ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """per_step, one-way attention: the coarse tokens per (sample, step) (lr_trace), each
        image's crops (owner, month) reading their context's cached coarse tokens and (cross_hr)
        each other. -> as tail_multi."""
        lr_g, sets_g, order = self._by_step(lr, crop_sets)     # crop owners -> (owner, slot)
        if lr_enc is None:
            trace, x = self._lr_trace(lr_g.x, lr_g.seqlens, self.L, len(self.encoder.blocks))
            x_lr = self.encoder.norm(x)
        else:
            trace, x_lr = lr_enc[0], lr_enc[1][order]
        dev, M, K, keys = lr.x.device, 16, self.STEPS, self._groups
        k_img = [g * M + self._per_crop(p, p.image) for p, g in sets_g]       # (slot group, month)
        groups = torch.unique(torch.cat(k_img))
        own = groups // M                                       # its slot's group
        # context = the contiguous run of (sample, step) groups [lo, hi] (keys sorted)
        if self.context == "season":
            lo = hi = own
        elif self.context == "all":
            smp = keys[own] // K
            lo = torch.searchsorted(keys, smp * K)
            hi = torch.searchsorted(keys, (smp + 1) * K) - 1
        else:                               # neighbor: toward the crop month's side of its slot
            g_month = torch.zeros(len(keys), dtype=torch.long, device=dev)
            g_month[torch.repeat_interleave(torch.arange(len(keys), device=dev),
                                            lr_g.seqlens)] = lr.image[order]
            d = torch.sign((groups % M - g_month[own] + 6) % 12 - 6)          # -1, 0, +1
            nb_key = keys[own] + d
            nb = torch.searchsorted(keys, nb_key).clamp(max=len(keys) - 1)
            ok = (keys[nb] == nb_key) & (keys[own] // K == nb_key // K) & (keys[own] % K + d >= 0)
            nb = torch.where(ok, nb, own)
            lo, hi = torch.minimum(own, nb), torch.maximum(own, nb)
        ends = torch.cumsum(lr_g.seqlens, 0)
        start = ends[lo] - lr_g.seqlens[lo]
        n_ctx = ends[hi] - start
        off = torch.cumsum(n_ctx, 0) - n_ctx
        idx = (torch.repeat_interleave(start - off, n_ctx)
               + torch.arange(int(n_ctx.sum()), device=dev))   # the context tokens, group by group
        lr_c = Packed(trace[0], n_ctx, None, {}, {})    # (keys: trace[i][idx], projected once)
        outs = self._cross_blocks(lr_c, trace,
                                  [(p, torch.searchsorted(groups, k)) for (p, _), k in zip(sets_g, k_img)],
                                  self.L, self.attn == "cross_hr", kv_idx=idx)
        out = torch.empty_like(x_lr)
        out[order] = x_lr
        return out, [self.encoder.norm(o) for o in outs]

    def _tail_multi(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]],
                    lr_enc: tuple[list[torch.Tensor], torch.Tensor] | None = None
                    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        S = len(lr.seqlens)
        crop_sets = [(p, o) for p, o in crop_sets if o is not None and len(o)]
        if not crop_sets:
            if lr_enc is not None and self.attn != "joint":     # the crop-free pass itself
                return lr_enc[1], []
            return self.encoder.norm(self.blocks(lr.x, lr.seqlens, self.L,
                                                 len(self.encoder.blocks))), []
        if self.attn != "joint":
            return self._tail_oneway(lr, crop_sets, lr_enc)
        dev = lr.x.device
        seq = torch.cat([torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens)]
                        + [torch.repeat_interleave(o.to(dev), p.seqlens) for p, o in crop_sets])
        x = torch.cat([lr.x] + [self._typed(p, lr.x.dtype) for p, _ in crop_sets])
        order = torch.argsort(seq, stable=True)          # each sample: its lr, then its crops
        y = self.blocks(x[order], torch.bincount(seq, minlength=S), self.L,
                        len(self.encoder.blocks))
        out = torch.empty_like(y)
        out[order] = y
        out = self.encoder.norm(out)
        sizes = [len(lr.x)] + [len(p.x) for p, _ in crop_sets]
        parts = list(torch.split(out, sizes))
        return parts[0], parts[1:]

    def lr_trace(self, lr: Packed) -> tuple[list[torch.Tensor], torch.Tensor]:
        """blocks [L, n) + norm on the lr sequences alone -> (their tokens entering each block,
        the normed output): what tail_multi(lr_enc=...) reuses. per_step: the trace in step
        order (as tail_multi regroups lr), the output in lr's order."""
        if self.per_step:
            lr_g, _, order = self._by_step(lr)
            trace, x = self._lr_trace(lr_g.x, lr_g.seqlens, self.L, len(self.encoder.blocks))
            out = torch.empty_like(x)
            out[order] = x
            return trace, self.encoder.norm(out)
        return self._lr_trace_seq(lr)

    def _lr_trace_seq(self, lr: Packed) -> tuple[list[torch.Tensor], torch.Tensor]:
        trace, x = self._lr_trace(lr.x, lr.seqlens, self.L, len(self.encoder.blocks))
        return trace, self.encoder.norm(x)

    def _tail_oneway(self, lr: Packed, crop_sets: list[tuple[Packed, torch.Tensor]],
                     lr_enc: tuple[list[torch.Tensor], torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tail_multi for attn "cross" / "cross_hr": the coarse sequence on its own (or
        lr_enc, its earlier run), the fine tokens reading it."""
        trace, x_lr = self._lr_trace_seq(lr) if lr_enc is None else lr_enc
        outs = self._cross_blocks(lr, trace, crop_sets, self.L, self.attn == "cross_hr")
        return x_lr, [self.encoder.norm(o) for o in outs]

    def _cross_blocks(self, lr: Packed, trace: list[torch.Tensor],
                      crop_sets: list[tuple[Packed, torch.Tensor]], lo: int, with_hr: bool,
                      kv_idx: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Blocks lo, lo + 1, .. (one per entry of trace = the lr tokens entering that block)
        on the crop tokens alone, with the blocks' own weights: varlen FlashAttention with the
        crop tokens of each sample as queries and, as keys / values, that sample's lr tokens
        (with_hr: + all of its crop tokens). -> each set's tokens."""
        if _pkg.FAST:
            return self._cross_blocks_fast(lr, trace, crop_sets, lo, with_hr, kv_idx)
        import flash_attn                   # (looked up per call: exp.bench.flop_trace wraps it)
        S, dev = len(lr.seqlens), lr.x.device
        x_c = torch.cat([self._typed(p, lr.x.dtype) for p, _ in crop_sets])
        crop_len = torch.cat([p.seqlens for p, _ in crop_sets])                # (C,)
        owner = torch.cat([o.to(dev) for _, o in crop_sets])                   # (C,)
        t_s = torch.repeat_interleave(owner, crop_len)                         # token -> sample
        order = torch.argsort(t_s, stable=True)                                # queries by sample
        n_q = torch.bincount(t_s, minlength=S)
        n_k = lr.seqlens + (n_q if with_hr else 0)
        cu_q, cu_k = (_cu(n) for n in (n_q, n_k))
        if with_hr:                         # keys: each sample's lr tokens, then its crop tokens
            k_seq = torch.cat([torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens),
                               t_s[order]])
            k_order = torch.argsort(k_seq, stable=True)
        max_q, max_k = int(n_q.max()), int(n_k.max())     # (once: a GPU -> CPU sync each)
        for blk, x_lr in zip(self.encoder.blocks[lo:], trace):
            a = blk.attn
            if a.attn_drop.p or PositionEncoding.is_rope(a.position_encoding):
                raise ValueError("cross attention: no attention dropout, absolute positions")
            heads = lambda v: v.view(len(v), a.num_heads, -1)                  # noqa: E731
            n_lr, n_c = blk.norm1(x_lr), blk.norm1(x_c[order])
            q = a.q_norm(heads(a.q(n_c)))
            k, v = a.k_norm(heads(a.k(n_lr))), heads(a.v(n_lr))
            if kv_idx is not None:          # (keys shared by several sequences: projected once)
                k, v = k[kv_idx], v[kv_idx]
            if with_hr:
                k = torch.cat([k, a.k_norm(heads(a.k(n_c)))])[k_order]
                v = torch.cat([v, heads(a.v(n_c))])[k_order]
            out = flash_attn.flash_attn_varlen_func(
                q, k.to(q.dtype), v.to(q.dtype), cu_q, cu_k, max_q, max_k, softmax_scale=a.scale)
            y = torch.empty_like(x_c)
            y[order] = blk.ls1(a.proj_drop(a.proj(out.reshape(len(out), -1)))).to(x_c.dtype)
            x_c = x_c + blk.drop_path(y)
            x_c = x_c + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(x_c))))
        return list(torch.split(x_c, [len(p.x) for p, _ in crop_sets]))

    def _cross_blocks_fast(self, lr: Packed, trace: list[torch.Tensor],
                           crop_sets: list[tuple[Packed, torch.Tensor]], lo: int, with_hr: bool,
                           kv_idx: torch.Tensor | None = None) -> list[torch.Tensor]:
        """_cross_blocks, the same arithmetic (bit-exact): the crop tokens sorted by sample ONCE
        (not gathered and scattered back in every block; every op but attention is per token),
        counts without bincount / repeat_interleave syncs, the two max lengths in one sync."""
        import flash_attn                   # (looked up per call: exp.bench.flop_trace wraps it)
        S, dev = len(lr.seqlens), lr.x.device
        x_c = torch.cat([self._typed(p, lr.x.dtype) for p, _ in crop_sets])
        crop_len = torch.cat([p.seqlens for p, _ in crop_sets])                # (C,)
        owner = torch.cat([o.to(dev) for _, o in crop_sets])                   # (C,)
        t_s = torch.repeat_interleave(owner, crop_len, output_size=len(x_c))   # token -> sample
        order = torch.argsort(t_s, stable=True)                                # queries by sample
        n_q = torch.zeros(S, dtype=lr.seqlens.dtype, device=dev).index_add_(
            0, t_s, torch.ones_like(t_s, dtype=lr.seqlens.dtype))
        n_k = lr.seqlens + (n_q if with_hr else 0)
        cu_q, cu_k = (_cu(n) for n in (n_q, n_k))
        if with_hr:                         # keys: each sample's lr tokens, then its crop tokens
            n_lr = len(kv_idx) if kv_idx is not None else len(trace[0])
            k_seq = torch.cat([torch.repeat_interleave(torch.arange(S, device=dev), lr.seqlens,
                                                       output_size=n_lr), t_s[order]])
            k_order = torch.argsort(k_seq, stable=True)
            # each key's place: its position in the sorted (lr then crops, per sample) order
            pos = torch.empty_like(k_order)
            pos[k_order] = torch.arange(len(k_order), device=dev)
            pos_lr, pos_c = pos[:n_lr], pos[n_lr:]
        max_q, max_k = torch.stack([n_q.max(), n_k.max()]).tolist()      # (one GPU -> CPU sync)
        # the coarse pass's own k / v of the lr tokens (_lr_trace, inference), not re-projected
        kv = self._kv[1] if getattr(self, "_kv", None) and self._kv[0] is trace[0] else None
        self._kv = None                     # (read once: not held past this batch)
        xs = x_c[order]
        for i, (blk, x_lr) in enumerate(zip(self.encoder.blocks[lo:], trace)):
            a = blk.attn
            if a.attn_drop.p or PositionEncoding.is_rope(a.position_encoding):
                raise ValueError("cross attention: no attention dropout, absolute positions")
            heads = lambda v: v.view(len(v), a.num_heads, -1)                  # noqa: E731
            n_c = blk.norm1(xs)
            q = a.q_norm(heads(a.q(n_c)))
            if kv is not None:
                k, v = a.k_norm(heads(kv[i][0])), heads(kv[i][1])
            else:
                n_lr_x = blk.norm1(x_lr)
                k, v = a.k_norm(heads(a.k(n_lr_x))), heads(a.v(n_lr_x))
            if kv_idx is not None:          # (keys shared by several sequences: projected once)
                k, v = k[kv_idx], v[kv_idx]
            if with_hr:                     # (= cat([lr, crops])[k_order], one write each)
                k_c, v_c = a.k_norm(heads(a.k(n_c))), heads(a.v(n_c))
                kb = k.new_empty((len(k_order),) + k.shape[1:])
                vb = v.new_empty((len(k_order),) + v.shape[1:])
                kb[pos_lr], kb[pos_c] = k, k_c.to(k.dtype)
                vb[pos_lr], vb[pos_c] = v, v_c.to(v.dtype)
                k, v = kb, vb
            out = flash_attn.flash_attn_varlen_func(
                q, k.to(q.dtype), v.to(q.dtype), cu_q, cu_k, max_q, max_k, softmax_scale=a.scale)
            xs = xs + blk.drop_path(blk.ls1(a.proj_drop(a.proj(out.reshape(len(out), -1))))
                                    .to(xs.dtype))
            xs = xs + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(xs))))
        x_c = torch.empty_like(xs)
        x_c[order] = xs
        return list(torch.split(x_c, [len(p.x) for p, _ in crop_sets]))
