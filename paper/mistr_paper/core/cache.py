"""The GPU loader's composites, computed ONCE per split and stored (train.gpu_data reads them
when present).

mistr_paper.core.data re-reads every raw acquisition of a tile and averages it into months and
seasons every epoch, though nothing in that step is random. PLANTEUR (90 S2 + 127 S1 frames,
11.6 MB per sample) and YieldSAT (42 S2 frames) are loader-bound at ~90 / ~60 samples/s; their
composites are 6-10x smaller. Stored per split (directory <root>/<task>/<split>/):

    meta.json                      N, modality names, their OlmoEarth names, shapes, dtypes
    ms.<name>.npy  ss.<name>.npy   (N, H, W, T | NS, C) bfloat16 bits as int16 (normalized)
    mfv.<name>.npy sfv.<name>.npy  (N, T | NS) bool  step has data (the mask is MISSING where not)
    ts_m.npy ts_s.npy              (N, T | NS, 3) the timestamps
    season.npy present.npy frac.npy target.npy

Built with the live CompositeOnGPU over the split in order (exp.mistr.build_cache), so a cached
batch is the live one up to the summation order of the per-batch einsum (exp.mistr.check_cache).
Read: every array memory-mapped, the batch's rows gathered and moved to the GPU (no workers;
the page cache holds a split: PLANTEUR train ~13 GB).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

from mistr_paper.core.data_cpu import Batch

# stored composites; $MISTR_CACHE overrides (exp/common/paths.py)
ROOT = Path(os.environ.get("MISTR_CACHE", Path(__file__).resolve().parents[2] / "cache" / "mistr_cache"))
VERSION = 1
AHEAD = 4           # batches read ahead, in parallel


def cache_dir(task_name: str, split: str) -> Path:
    return ROOT / task_name / split


def exists(task_name: str, split: str) -> bool:
    return (cache_dir(task_name, split) / "meta.json").exists()


def _np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy()


def unpack(b: Batch) -> dict:
    """A live Batch -> {field: (B, ...) numpy} as stored, + the modality name map."""
    out, names = {}, []
    for key, s in (("ms", b.ms), ("ss", b.ss)):
        d = s.as_dict()
        for name, x in d.items():
            if x is None or name == "timestamps" or name.endswith("_mask"):
                continue
            m = d[MaskedOlmoEarthSample.get_masked_modality_name(name)]
            fv = (m != MaskValue.MISSING.value).any(4).any(2).any(1)            # (B, T)
            # the mask is exactly "MISSING where the step has no data", constant over H, W, C
            rebuilt = torch.full_like(m, MaskValue.ONLINE_ENCODER.value)
            rebuilt.masked_fill_((~fv)[:, None, None, :, None], MaskValue.MISSING.value)
            if not torch.equal(rebuilt, m):
                raise ValueError(f"{key}.{name}: mask is not per step")
            out[f"{key}.{name}"] = _np(x)
            out[f"{key[0]}fv.{name}"] = _np(fv)
            if key == "ms":
                names.append(name)
        out[f"ts_{key[0]}"] = _np(d["timestamps"])
    out["season"], out["present"], out["frac"] = _np(b.season), _np(b.present), _np(b.frac)
    out["target"] = _np(b.target)
    if b.dates is not None:
        raise ValueError("date crops are not cached")
    return out, names


def build(task, split: str, batch_size: int = 32, num_workers: int = 8) -> Path:
    """Run the live GPU loader over the split in order, write the arrays."""
    from mistr_paper.core.data import gpu_loader
    out = cache_dir(task.name, split)
    out.mkdir(parents=True, exist_ok=True)
    dl = gpu_loader(task, split, batch_size, num_workers, False)
    n = len(dl.dataset)
    mm, names, i = {}, None, 0
    for b in dl:
        arrs, names = unpack(b)
        k = len(next(iter(arrs.values())))
        for f, a in arrs.items():
            if f not in mm:
                mm[f] = np.lib.format.open_memmap(out / f"{f}.npy.tmp", "w+", a.dtype,
                                                  (n, *a.shape[1:]))
            mm[f][i:i + k] = a
        i += k
    if i != n:
        raise RuntimeError(f"{task.name}/{split}: wrote {i} of {n} samples")
    for f, a in mm.items():
        a.flush()
        del a
        os.replace(out / f"{f}.npy.tmp", out / f"{f}.npy")
    (out / "meta.json").write_text(json.dumps({"version": VERSION, "n": n, "names": names,
                                               "fields": sorted(mm)}))
    return out


class CachedLoader:
    """Batches of a built cache, as mistr_paper.core.data.GPULoader yields them. shuffle: a fresh
    permutation of the split from torch's global RNG each epoch (train.py seeds it per epoch);
    drop_last with shuffle, as the live loader."""

    def __init__(self, task_name: str, split: str, batch_size: int, shuffle: bool):
        d = cache_dir(task_name, split)
        meta = json.loads((d / "meta.json").read_text())
        if meta["version"] != VERSION:
            raise ValueError(f"{d}: cache version {meta['version']}, expected {VERSION}")
        self.n, self.names = meta["n"], meta["names"]
        self.a = {f: np.load(d / f"{f}.npy", mmap_mode="r") for f in meta["fields"]}
        self.bs, self.shuffle = batch_size, shuffle
        self.dataset = range(self.n)        # (len(loader.dataset), as the live loader)

    def __len__(self):
        return self.n // self.bs if self.shuffle else -(-self.n // self.bs)

    def _read(self, idx: np.ndarray) -> dict:
        """The batch's rows of every array, on the CPU (the prefetch thread: numpy copies
        release the GIL)."""
        idx = np.sort(idx)          # (sequential reads; the order inside a batch is irrelevant)
        return {f: torch.from_numpy(np.ascontiguousarray(a[idx])) for f, a in self.a.items()}

    def _sample(self, rows, key, dev):
        # (plain copies: pinning a fresh buffer per batch cost more than it saved)
        d = {"timestamps": rows[f"ts_{key[0]}"].to(dev)}
        for name in self.names:
            x = rows[f"{key}.{name}"].to(dev).view(torch.bfloat16)
            fv = rows[f"{key[0]}fv.{name}"].to(dev)
            m = torch.full_like(x, MaskValue.ONLINE_ENCODER.value)
            m.masked_fill_((~fv)[:, None, None, :, None], MaskValue.MISSING.value)
            d[name] = x
            d[MaskedOlmoEarthSample.get_masked_modality_name(name)] = m
        return MaskedOlmoEarthSample(**d)

    def batch(self, rows: dict) -> Batch:
        dev = torch.device("cuda")
        return Batch(self._sample(rows, "ms", dev), self._sample(rows, "ss", dev),
                     *(rows[f].to(dev) for f in ("season", "present", "target", "frac")))

    def __iter__(self):
        order = torch.randperm(self.n).numpy() if self.shuffle else np.arange(self.n)
        chunks = [order[k * self.bs:(k + 1) * self.bs] for k in range(len(self))]
        # read the next AHEAD batches in parallel while one trains (a cold cache comes over the
        # network file system: one reader ~400 MB/s; warm, the page cache serves it)
        with ThreadPoolExecutor(AHEAD) as ex:
            q = [ex.submit(self._read, c) for c in chunks[:AHEAD]]
            for k in range(len(chunks)):
                rows = q.pop(0).result()
                if k + AHEAD < len(chunks):
                    q.append(ex.submit(self._read, chunks[k + AHEAD]))
                yield self.batch(rows)
