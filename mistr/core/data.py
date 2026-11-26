"""mistr.core.data_cpu's batches, composited on the GPU (train.gpu_data).

The CPU loader pools every tile's raw acquisitions into monthly composites, groups those into
seasons and normalizes both, again every epoch: ~130 ms of CPU per sample, so 8 workers feed a
PASTIS batch every ~460 ms while the GPU needs ~110 ms for a training step. Here the workers
only READ the raw frames (no temporal policy, no validation) and work out, per sample, the
small things that need dates (each frame's month slot from the task's own Pool, the month and
season bounds, timestamps, season index, frac, the target); the pixel work runs on the GPU:

    MaskInvalid -> monthly masked mean (Pool) or the frames as they are (All) -> seasons
    (GroupSteps) -> OlmoEarth normalization + imputation -> masks -> Batch

Same semantics as the CPU path (exp.mistr.check_gpu_data compares the two): a frame belongs to
the month its interval starts in; invalid frames and pixels never enter a mean, a slot with
none is 0 and invalid; a month is valid if any pixel is, a season if any of its months is.
Floating point differs only in summation order and fp32 vs float64 normalization (below bf16
resolution, which is what the model reads). The monthly inputs stay on the GPU for the crops.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from torch.utils.data import DataLoader, Dataset, RandomSampler

from exp.common.tasks import Task
from mistr.data import All, GroupSteps
from mistr.data.temporal import Compose, MaskInvalid, Pool
from mistr.encoders.olmoearth import OlmoEarthInput, _timestamps
from mistr.core.data_cpu import Batch, within
from mistr.core.seasons import season_index


def _parts(policy):
    """(MaskInvalid or None, Pool or All) of a task's per-modality policy."""
    ps = policy.policies if isinstance(policy, Compose) else [policy]
    mask = [p for p in ps if isinstance(p, MaskInvalid)]
    rest = [p for p in ps if not isinstance(p, MaskInvalid)] or [All()]   # (a bare mask: All)
    if len(rest) != 1 or not isinstance(rest[0], (Pool, All)) or len(mask) > 1:
        raise NotImplementedError(f"gpu_data: policy {policy!r} (MaskInvalid? + Pool | All)")
    return (mask[0] if mask else None), rest[0]


def _compact(x: torch.Tensor) -> torch.Tensor:
    """Integer-valued float frames (S2 reflectance) as int16 / uint16 for the trip to the GPU:
    lossless, half the bytes; anything else unchanged."""
    if not x.is_floating_point() or x.element_size() < 4 or not bool((x == x.round()).all()):
        return x
    lo, hi = float(x.min()), float(x.max())
    if -32768 <= lo and hi <= 32767:
        return x.to(torch.int16)
    if 0 <= lo and hi <= 65535:
        return x.to(torch.int32).to(torch.uint16)
    return x


class _Light:
    """What Task.target reads of an MTMMBatch."""

    def __init__(self, ids, target):
        self.ids, self.target = ids, target


class RawPaired(Dataset):
    """Per sample: the raw frames of each modality and the date bookkeeping (CPU, cheap)."""

    def __init__(self, task: Task, split: str):
        self.task, self.split = task, split
        lr = task.lr_policy()
        if not (isinstance(lr, GroupSteps) and lr.reducer == "mean"):
            raise NotImplementedError(f"gpu_data: lr policy {lr!r} (GroupSteps mean)")
        self.group = lr.size
        self.policies = task.policies()
        self.parts = {m: _parts(p) for m, p in self.policies.items()}
        self.ds = task.open(split, {m: All() for m in task.olmoearth.modalities})
        self.ds.validate = False
        self.spec = self.ds.SPEC.modalities
        for m, (_, pool) in self.parts.items():
            if isinstance(pool, Pool) and pool._reducer(self.spec[m]) != "mean":
                raise NotImplementedError(f"gpu_data: {m} pools with "
                                          f"{pool._reducer(self.spec[m])}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # (a raw dataset that can read one tile without its whole native sample: PLANTEUR)
        s = self.ds.tile_fast(i) if hasattr(self.ds, "tile_fast") else self.ds[i]
        mods = {}
        for m in self.task.olmoearth.modalities:
            md = s.modalities[m]
            _, pool = self.parts[m]
            F = md.data.shape[0]
            if isinstance(pool, Pool):
                bounds = pool._months(md)
                lo = torch.tensor([b[0] for b in bounds])
                hi = torch.tensor([b[1] for b in bounds])
                st = md.time.start
                inside = (st[:, None] >= lo[None]) & (st[:, None] <= hi[None])     # (F, T)
                slot = torch.where(inside.any(1), inside.long().argmax(1),
                                   torch.full((F,), -1, dtype=torch.long))
            else:                                   # All: every frame its own step
                lo, hi = md.time.start.clone(), md.time.end.clone()
                slot = torch.arange(F)
            mods[m] = dict(data=_compact(md.data), frame_valid=md.frame_valid.bool(),
                           pixel_valid=None if md.pixel_valid is None else md.pixel_valid.bool(),
                           slot=slot, t_start=lo, t_end=hi, bands=md.bands)
        return s.id, mods, s.target


@dataclass
class RawBatch:
    ids: list
    mods: dict          # modality -> dict of padded tensors (B, F, ...) + bounds (B, T)
    target: torch.Tensor
    ts_m: torch.Tensor          # (B, T, 3) month timestamps (start)
    ts_s: torch.Tensor          # (B, NS, 3) season timestamps (mid)
    season: torch.Tensor        # (B, T)
    frac: torch.Tensor          # (B, T)

    def pin_memory(self):
        pin = lambda x: x.pin_memory() if torch.is_tensor(x) else x          # noqa: E731
        return RawBatch(self.ids, {m: {k: pin(v) for k, v in d.items()}
                                   for m, d in self.mods.items()},
                        pin(self.target), pin(self.ts_m), pin(self.ts_s), pin(self.season),
                        pin(self.frac))


class RawCollate:
    def __init__(self, task: Task, group: int):
        self.task, self.group = task, group
        self.ref = next(iter(task.olmoearth.modalities))

    def __call__(self, items) -> RawBatch:
        ids = [it[0] for it in items]
        mods = {}
        for m in items[0][1]:
            ds = [it[1][m] for it in items]
            F = max(d["data"].shape[0] for d in ds)
            pad = lambda x, v: torch.cat([x, x.new_full((F - x.shape[0], *x.shape[1:]), v)])  # noqa: E731
            if len({d["data"].dtype for d in ds}) > 1:      # (_compact chose per sample)
                for d in ds:
                    d["data"] = d["data"].to(torch.int32).float() if d["data"].dtype == torch.uint16 \
                        else d["data"].float()
            out = {"data": torch.stack([pad(d["data"], 0) for d in ds]),
                   "frame_valid": torch.stack([pad(d["frame_valid"], False) for d in ds]),
                   "slot": torch.stack([pad(d["slot"], -1) for d in ds]),
                   "t_start": torch.stack([d["t_start"] for d in ds]),
                   "t_end": torch.stack([d["t_end"] for d in ds]),
                   "bands": ds[0]["bands"]}
            if ds[0]["pixel_valid"] is not None:
                out["pixel_valid"] = torch.stack([pad(d["pixel_valid"], False) for d in ds])
            mods[m] = out
        r = mods[self.ref]
        T = r["t_start"].shape[1]
        if T % self.group:
            raise ValueError(f"{T} steps do not split into groups of {self.group}")
        g = self.group
        s_start, s_end = r["t_start"][:, ::g], r["t_end"][:, g - 1::g]
        season = season_index(s_start, s_end, r["t_start"], r["t_end"])
        frac = within(r["t_start"], r["t_end"], s_start, s_end, season)
        mid = torch.where(s_start >= 0, (s_start + s_end) // 2, s_start)
        target = self.task.target(_Light(ids, torch.stack([it[2] for it in items])))
        return RawBatch(ids, mods, target, _timestamps(r["t_start"]), _timestamps(mid),
                        season, frac)


class CompositeOnGPU:
    """RawBatch (on the GPU) -> mistr.core.data_cpu.Batch."""

    def __init__(self, task: Task, group: int, parts: dict):
        m, imp = task.olmoearth.modalities, task.olmoearth.impute
        self.inp = OlmoEarthInput(m, imp, timestamp="start")
        self.mods, self.group, self.parts = list(m), group, parts
        self._norm = {}

    def _norm_params(self, m, bands, dev):
        key = (m, tuple(bands))
        if key not in self._norm:
            om = self.inp.olmo[m]
            idx = self.inp._band_index(m, bands)
            cfg = self.inp.normalizer.norm_config[om.name]
            k = self.inp.normalizer.std_multiplier
            if om.skip_normalization:
                lo = torch.zeros(len(idx), dtype=torch.float64)
                span = torch.ones(len(idx), dtype=torch.float64)
            else:
                mean = torch.tensor([cfg[b]["mean"] for b in om.band_order], dtype=torch.float64)
                std = torch.tensor([cfg[b]["std"] for b in om.band_order], dtype=torch.float64)
                lo, span = mean - k * std, 2 * k * std
            self._norm[key] = (torch.tensor(idx, device=dev), lo.float().to(dev),
                               (1 / span).float().to(dev), om.name)
        return self._norm[key]

    @staticmethod
    def _masked_mean(x, valid, onehot):
        """x (B, F, C, H, W), valid (B, F, H, W), onehot (B, T, F) -> (B, T, C, H, W) mean of the
        valid entries per step (0 where none), (B, T, H, W) any valid."""
        w = onehot.float()
        v = valid.float()
        cnt = torch.einsum("btf,bfhw->bthw", w, v)
        s = torch.einsum("btf,bfchw->btchw", w, x * v[:, :, None])
        mean = torch.where(cnt[:, :, None] > 0, s / cnt.clamp(min=1)[:, :, None],
                           torch.zeros((), device=x.device))
        return mean, cnt > 0

    def _pack(self, steps: dict, ts, dev):
        """{modality: (data (B, T, C, H, W), frame_valid (B, T), bands)} -> the sample, valid."""
        d = {"timestamps": ts}
        for m, (x, fv, bands) in steps.items():
            idx, lo, inv, name = self._norm_params(m, bands, dev)
            y = ((x[:, :, idx] - lo[:, None, None]) * inv[:, None, None])        # (B,T,C,H,W)
            y = y.permute(0, 3, 4, 1, 2).to(torch.bfloat16)                     # (B,H,W,T,C)
            mask = torch.full_like(y, MaskValue.ONLINE_ENCODER.value)
            mask.masked_fill_((~fv)[:, None, None, :, None], MaskValue.MISSING.value)
            d[name] = y
            d[MaskedOlmoEarthSample.get_masked_modality_name(name)] = mask
        valid = torch.stack([fv for _, fv, _ in steps.values()]).any(0)
        return MaskedOlmoEarthSample(**d), valid

    @torch.no_grad()
    def __call__(self, rb: RawBatch) -> Batch:
        dev = torch.device("cuda")
        months, seasons, g = {}, {}, self.group
        every = None
        for m in self.mods:
            r = rb.mods[m]
            x = r["data"].to(dev, non_blocking=True)                            # (B,F,C,H,W)
            x = (x.to(torch.int32) if x.dtype == torch.uint16 else x).float()
            fv = r["frame_valid"].to(dev, non_blocking=True)
            pv = (r["pixel_valid"].to(dev, non_blocking=True) if "pixel_valid" in r
                  else torch.ones(x.shape[:2] + x.shape[3:], dtype=torch.bool, device=dev))
            mask, pool = self.parts[m]
            if mask is not None:                                    # MaskInvalid, on the GPU
                bad = torch.zeros_like(pv)
                if mask.nodata_below is not None:
                    bad |= (x < mask.nodata_below).any(2)
                if mask.flat:
                    eq_x = (x[..., :, 1:] == x[..., :, :-1]).all(2)
                    eq_y = (x[..., 1:, :] == x[..., :-1, :]).all(2)
                    both = eq_x[..., :-1, :] & eq_y[..., :, :-1]
                    flat = torch.zeros_like(bad)
                    flat[..., :-1, :-1] |= both
                    flat[..., 1:, 1:] |= both
                    bad |= flat
                pv = pv & ~bad
                fv = fv & (bad.float().mean((2, 3)) <= mask.frame_frac)
            slot = r["slot"].to(dev, non_blocking=True)
            if isinstance(pool, Pool):
                T = r["t_start"].shape[1]
                onehot = (slot[:, None, :] == torch.arange(T, device=dev)[None, :, None])
                mx, mpv = self._masked_mean(x, fv[..., None, None] & pv, onehot)
                mfv = mpv.flatten(2).any(2)
            else:                                   # All: the frames as they are
                mx, mpv, mfv = x, pv, fv
            B, T = mfv.shape
            onehot = (torch.arange(T, device=dev)[None, :] // g
                      == torch.arange(T // g, device=dev)[:, None])[None].expand(B, -1, -1)
            sx, _ = self._masked_mean(mx, mfv[..., None, None] & mpv, onehot)
            sfv = mfv.view(B, T // g, g).any(2)
            months[m] = (mx, mfv, r["bands"])
            seasons[m] = (sx, sfv, r["bands"])
            every = mfv if every is None else every & mfv
        ms, _ = self._pack(months, rb.ts_m.to(dev, non_blocking=True), dev)
        ss, s_valid = self._pack(seasons, rb.ts_s.to(dev, non_blocking=True), dev)
        season = rb.season.to(dev, non_blocking=True)
        present = every & (season >= 0) & s_valid.gather(1, season.clamp(min=0))
        return Batch(ms, ss, season, present, rb.target.to(dev, non_blocking=True),
                     rb.frac.to(dev, non_blocking=True))


class GPULoader:
    """A DataLoader of RawBatches whose iteration yields GPU-composited Batches."""

    def __init__(self, dl: DataLoader, composer: CompositeOnGPU):
        self.dl, self.composer = dl, composer
        self.dataset = dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for rb in self.dl:
            yield self.composer(rb)


def gpu_loader(task: Task, split: str, batch_size: int, num_workers: int, shuffle: bool,
               epoch_samples: int = 0) -> GPULoader:
    """As mistr.core.data_cpu.loader (no date crops), composited on the GPU -- or read from
    the split's stored composites when built (mistr.core.cache; MISTR_NO_CACHE=1: live)."""
    import os
    from mistr.core import cache
    if (cache.exists(task.name, split) and not os.environ.get("MISTR_NO_CACHE")
            and not epoch_samples):
        print(f"data: stored composites {cache.cache_dir(task.name, split)}", flush=True)
        return cache.CachedLoader(task.name, split, batch_size, shuffle)
    # (raw batches are many large tensors: fd passing runs out of descriptors at 8 workers)
    torch.multiprocessing.set_sharing_strategy("file_system")
    ds = RawPaired(task, split)
    sampler = (RandomSampler(ds, num_samples=epoch_samples)
               if shuffle and 0 < epoch_samples < len(ds) else None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                    sampler=sampler, num_workers=num_workers,
                    collate_fn=RawCollate(task, ds.group),
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers else None,
                    drop_last=shuffle, pin_memory=True)
    return GPULoader(dl, CompositeOnGPU(task, ds.group, ds.parts))
