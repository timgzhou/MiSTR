"""MiSTR: mixed spatiotemporal resolution transformers (coarse pass -> window selection ->
fine tokens of the selected windows -> decoder)."""
# Inference fast path (fewer GPU -> CPU syncs and copies; bit-exact with the plain path, checked
# by exp.bench.check_exact). False: the plain path, kept for that check.
FAST = True

from mistr.core.model import CoarsePassOutput, MiSTR, MixedPassOutput, ModelConfig, load_weights
from mistr.core.windows import Windows

__all__ = ["MiSTR", "ModelConfig", "CoarsePassOutput", "MixedPassOutput", "Windows", "load_weights"]
