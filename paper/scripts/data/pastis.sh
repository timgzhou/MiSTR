#!/bin/bash
# PASTIS-R (Garnot et al. 2022, Zenodo record 5735646, CC BY 4.0) -> data/PASTIS-R (51 GB unpacked).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p data && cd data
[ -f PASTIS-R.zip ] || curl -fL --retry 10 -o PASTIS-R.zip "https://zenodo.org/records/5735646/files/PASTIS-R.zip?download=1"
unzip -q -n PASTIS-R.zip
ls PASTIS-R/DATA_S2 | wc -l      # 2433 patches
