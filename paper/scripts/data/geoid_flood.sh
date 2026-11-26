#!/bin/bash
# GEOID-Flood (HF dataset links-ads/geoid-flood, CC BY 4.0; contains modified Copernicus data): the layers
# the code reads (s1grd, label, validity; ~95 GB of 584 GB) with the release's own downloader
# (get_data.py from the dataset repo) -> data/GEOID-Flood-full. Needs huggingface_hub only.
#
# The paper uses GEOID-Flood v1.0 (revision 86840746, 2026-08-04). v1.1 (2026-10-29) removed the
# v1.0 files from `main`, so both get_data.py and every file it fetches are pinned to that revision
# (get_data.py has no revision option: its hf_hub_download calls are pinned here).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REV=${REV:-868407460bf3db492f50730a57585916baa71dc6}
mkdir -p data/GEOID-Flood
[ -f data/GEOID-Flood/get_data.py ] || python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download("links-ads/geoid-flood", "get_data.py", repo_type="dataset", revision="$REV", local_dir="data/GEOID-Flood")
PY
HF_HUB_DISABLE_XET=1 python -u - --dest data/GEOID-Flood-full --layer s1grd label validity --workers 4 <<PY
import functools, runpy, sys
import huggingface_hub
huggingface_hub.hf_hub_download = functools.partial(huggingface_hub.hf_hub_download, revision="$REV")
sys.argv[0] = "data/GEOID-Flood/get_data.py"
runpy.run_path(sys.argv[0], run_name="__main__")
PY
