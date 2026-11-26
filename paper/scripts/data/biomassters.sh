#!/bin/bash
# BioMassters as GeoBench-v2 tortilla shards (HF dataset aialliance/biomassters, CC BY 4.0) -> data/BioMassters
# (7 shards, 129 GB), then the loader's index.   default revision = the one the paper used (2026-01-14).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("aialliance/biomassters", repo_type="dataset", revision="${REV:-5ae1d147b53c71eb9b7f7ab1796a084866493e44}",
                  allow_patterns=["*.tortilla"], local_dir="data/BioMassters")
PY
python -m mistr_paper.data.datasets.biomassters --build_index
