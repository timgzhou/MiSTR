# Data

Everything is read from `data/` (relative to the repo root; make it a symlink to put the data on
another disk). Nothing here is redistributed: each dataset is downloaded from its publisher, then
indexed / packed by the scripts below. Derived stores are optional speed-ups and go to `cache/`
(`$MISTR_TILESTORE`, `$MISTR_CACHE`; see `exp/common/paths.py`).

| Dataset | Source (licence) | On disk | Download | Prepare |
|---|---|---|---|---|
| PASTIS-R | Zenodo 5735646 (CC BY 4.0) | 51 GB, `data/PASTIS-R` | `bash scripts/data/pastis.sh` | none |
| PLANTEUR | HF `allenai/planteur-benchmark` @ `51b74a91` (ODC-BY) | ~120 GB, `data/PLANTEUR` | `bash scripts/data/planteur.sh` | none |
| BioMassters | HF `aialliance/biomassters` @ `5ae1d147`, GeoBench-v2 tortilla shards (CC BY 4.0) | 129 GB, `data/BioMassters` | `bash scripts/data/biomassters.sh` | index (in the script) |
| YieldSAT | yieldsat.github.io (`Both.zip`, login required; see the site for terms) | 34.5 GB packed, `data/YieldSAT` | manual, below | `exp.yieldsat.prepare` |
| GEOID-Flood | HF `links-ads/geoid-flood` **v1.0** @ `86840746` (CC BY 4.0; modified Copernicus data) | 167 GB download (s1grd, label, validity), `data/GEOID-Flood-full` | `bash scripts/data/geoid_flood.sh` | tiles + S1 series, below |

The scripts pin the revisions the paper used (`REV=<commit>` overrides). GEOID-Flood v1.1
(2026-10-29) removed the v1.0 files from `main`; the script fetches v1.0 at its commit.

Splits are the benchmarks' own (PASTIS folds 1-3 / 4 / 5; PLANTEUR's spatial-block folds 1 / 4 / 5;
BioMassters as GeoBench-v2; GEOID-Flood's per-tile split). YieldSAT has no fixed split: its
field-grouped 10-fold CV is made fixed (md5 order within each country x crop; test fold 0, val fold
1) in `mistr/data/datasets/yieldsat.py`. Every task is defined in `exp/common/tasks.py` (inputs,
time steps, label, metric); samples are 64 x 64 px tiles of each patch.

Check the downloads with the tests (each skips when its data is missing):

    python -m pytest -q tests/test_pastis.py tests/test_planteur.py tests/test_biomassters.py \
        tests/test_yieldsat.py tests/test_geoidflood.py

## YieldSAT

Download `Both.zip` (23.6 GB) from the YieldSAT site (it needs a login), then

    cd data/YieldSAT && unzip Both.zip && unzip Preprocessed.zip -d preprocessed-24-ts \
        && unzip Raw.zip -d original-preprocessed && cd -
    python -u -m exp.yieldsat.prepare index          # 40 s
    python -u -m exp.yieldsat.prepare pack --workers 32   # 2.5 min, 34.5 GB
    python -u -m exp.yieldsat.prepare verify --workers 4  # the ML-ready view equals the NetCDF at every labelled pixel

## GEOID-Flood tracking series (Sec. 5, App. K)

The tracking experiments use dense same-orbit Sentinel-1 RTC series around each GEOID-Flood test
tile, fetched from Microsoft Planetary Computer (`sentinel-1-rtc`, free, no account), not shipped:

    python -m exp.geoidchange.tilecache workers=8                       # 512 px same-scene tiles (from data/GEOID-Flood-full)
    python -u -m exp.geoidchange.tileseries_fetch --before 24 --after 24 \
        --out data/GEOID-Flood-tileseries-long2 --threads 8             # resumable; --shard i --num_shards n to split
    python -u -m exp.geoidchange.tileseries_cache ser=data/GEOID-Flood-tileseries-long2 splits=test,val,train

`python -m exp.paper_evals --part geoid` prints these with the training and scoring commands.
Planetary Computer reprocesses items now and then; `manifests/geoid_series_long2.json.gz` (and
`geoid_series_long3.json.gz`, the Fig 12 test series) list every series the paper used (per tile: relative orbit, orbit state, geolocation shift and every
pass's acquisition time). Compare a fetch with it:

    python -m exp.geoidchange.series_manifest check ser=data/GEOID-Flood-tileseries-long2 \
        manifest=manifests/geoid_series_long2.json.gz

## Optional derived stores

    python -m exp.common.build_tile_store --task pastis --num_workers 8    # 64 px tile stores (any task)
    python -m exp.mistr.build_cache --tasks pastis planteur biomassters yieldsat_raw   # MiSTR composites

Training reads the raw data when they are missing (slower; same samples, same order). Build them
with this repo's code: a store records the dataset class that built it by module path.

## Pretrained weights

OlmoEarth v1 (Tiny / Base) is downloaded by `olmoearth-pretrain` on first use (OlmoEarth Artifact
License). TerraMind 1.0 Tiny (Apache-2.0) is read from
`cache/pretrained/TerraMind-1.0-tiny/TerraMind_v1_tiny.pt`:

    python -c "from huggingface_hub import hf_hub_download as d; d('ibm-esa-geospatial/TerraMind-1.0-tiny', \
        'TerraMind_v1_tiny.pt', local_dir='cache/pretrained/TerraMind-1.0-tiny')"

Tessera v2 small (`ucam-eo/tessera`; code MIT, weights CC0 for v1.1, check v2's terms) is read
from `pretrained/tessera/student_small.pt`.
