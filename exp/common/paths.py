"""Where the repo's data, caches, checkpoints and results live.

Every entry point runs from the repo root (`python -m exp...`), so data paths elsewhere stay
relative ("data/...", "checkpoints/...", "results/..."); make those directories symlinks to put
them on another disk. The large derived stores have their own roots, overridable separately:

    $MISTR_TILESTORE   64 px tile stores of the raw datasets (mistr/data/tile_store.py), default cache/tilestore
    $MISTR_CACHE       MiSTR's stored monthly / seasonal composites (mistr/core/cache.py), default cache/mistr_cache
    $MISTR_FEATURES    frozen-encoder embeddings (Tessera, exp/tessera), default cache/features
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CACHE = REPO / "cache"
FEATURES = Path(os.environ.get("MISTR_FEATURES", CACHE / "features")).expanduser()
TILESTORE = Path(os.environ.get("MISTR_TILESTORE", CACHE / "tilestore")).expanduser()
ANYUP_REPO = REPO / "third_party" / "anyup"
