"""Measured inference cost (multiply-accumulates per sample) of a forward pass.

`count_macs()` is a context manager wrapped around a no-grad forward pass (exp.mistr.train,
exp.geoidchange.tiletrack_flops), so the count covers everything a model does at inference:
the coarse pass, the selector, the fine tokens of the selected windows, and the decoder.

What is counted
---------------
torch's FlopCounterMode counts the matmul-like ops it sees dispatched (linear, conv, bmm,
einsum, SDPA) at 2 FLOPs per multiply-accumulate (one multiply, one add), so its total is
halved here to give MACs -- the same convention as exp/bench/olmo_throughput.py, and as the
analytic estimate 12 D^2 per token and block + 2 n^2 D per attention sequence and block.

FlashAttention is a custom CUDA kernel the counter cannot see, so `count_macs` wraps
flash_attn's two entry points and adds the attention arithmetic itself:

    per (query sequence of n_q tokens, key sequence of n_k tokens), nh heads of width d:
        Q K^T      : n_q * n_k * nh * d  MACs
        attn @ V   : n_q * n_k * nh * d  MACs

Elementwise ops (norms, activations, softmax, additions) are not counted by either part.
"""
from exp.common import olmo_bootstrap  # noqa: F401  (OlmoEarth import shims; see module)

import csv
import os
from contextlib import contextmanager
from pathlib import Path

from torch.utils.flop_counter import FlopCounterMode


class MacCount:
    """Filled in by count_macs. All totals are for the whole traced batch."""

    def __init__(self):
        self.attn = 0.0          # MACs inside FlashAttention kernels (added analytically)
        self.traced = 0.0        # MACs FlopCounterMode saw (linear / conv / einsum / SDPA)
        self.token_blocks = 0    # query tokens summed over every FlashAttention call
        self.max_seq = 0         # longest key sequence any FlashAttention call saw
        self.analytic = 0.0      # MACs reported by add_macs (elementwise kernels, see there)

    @property
    def total(self) -> float:
        return self.attn + self.traced + self.analytic


_ACTIVE = None      # the count_macs in progress


def add_macs(n: float) -> None:
    """Report MACs the tracer cannot see, computed by the caller: elementwise arithmetic that IS
    the method (Upsample Anything's per-image fit). No-op outside count_macs."""
    if _ACTIVE is not None:
        _ACTIVE.analytic += n


@contextmanager
def count_macs():
    """with count_macs() as c: model(batch)  ->  c.total MACs for that batch."""
    import flash_attn

    global _ACTIVE
    count = _ACTIVE = MacCount()
    real_varlen, real_dense = flash_attn.flash_attn_varlen_func, flash_attn.flash_attn_func

    def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kw):
        # q (N_q, nh, d), k (N_k, nh, d): tokens of every sequence, packed back to back.
        # cu_seqlens_* (S + 1,): cumulative sequence lengths.
        len_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).double()        # (S,)
        len_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).double()        # (S,)
        heads_x_width = q.shape[1] * q.shape[2]
        count.attn += 2.0 * float((len_q * len_k).sum()) * heads_x_width
        count.token_blocks += int(len_q.sum())
        count.max_seq = max(count.max_seq, int(len_k.max()))
        return real_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           *args, **kw)

    def dense(q, k, v, *args, **kw):
        # q (B, n_q, nh, d), k (B, n_k, nh, d): padded batches, every row the same length
        B, n_q, nh, d = q.shape
        n_k = k.shape[1]
        count.attn += 2.0 * B * n_q * n_k * nh * d
        count.token_blocks += B * n_q
        count.max_seq = max(count.max_seq, n_k)
        return real_dense(q, k, v, *args, **kw)

    flash_attn.flash_attn_varlen_func, flash_attn.flash_attn_func = varlen, dense
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            yield count
    finally:
        flash_attn.flash_attn_varlen_func, flash_attn.flash_attn_func = real_varlen, real_dense
        _ACTIVE = None
        count.traced = float(counter.get_total_flops()) / 2.0     # FLOPs -> MACs


def append_row(path: Path, row: dict) -> None:
    """Append one row; the header is written when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def flop_row(run: str, task: str, model_size: str, count: MacCount, n_samples: int,
             n_blocks: int, crops_per_sample: float | str = "", samples_s: float | str = "",
             peak_mem_gb: float | str = "") -> dict:
    """The CSV row both scripts write. gmacs_* are per sample."""
    return {
        "job": os.environ.get("SLURM_JOB_ID", ""), "run": run, "task": task,
        "model_size": model_size, "samples_traced": n_samples,
        "gmacs": round(count.total / n_samples / 1e9, 3),
        "gmacs_attention": round(count.attn / n_samples / 1e9, 3),
        "gmacs_other": round(count.traced / n_samples / 1e9, 3),
        # tokens that went through a transformer block, averaged over the blocks
        "tokens_per_block": round(count.token_blocks / n_samples / n_blocks, 1),
        "longest_sequence": count.max_seq,
        "crops_per_sample": crops_per_sample,
        "samples_s": round(samples_s, 2) if samples_s != "" else "",
        "peak_mem_gb": round(peak_mem_gb, 2) if peak_mem_gb != "" else "",
    }
