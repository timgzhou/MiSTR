# MiSTR: Adaptively Mixing Spatiotemporal Resolution for Satellite Image Time Series

Tim G. Zhou, Anthony Fuller, Avery Ma, Geoff Pleiss, Evan Shelhamer
(University of British Columbia, Vector Institute, RBC Borealis)

Code, run configurations and results for the paper.

MiSTR (Mixed Spatiotemporal Resolution) makes a remote-sensing transformer spend its compute where and
when the task needs detail. A full **coarse pass** encodes the time series at low resolution (large
patches, seasonal composites) and predicts; from it, the model **selects** the spatiotemporal points
that need more detail (where the coarse prediction is uncertain, and when, by the decoder's
attention). A sparse **fine pass** adds high-resolution tokens (small patches, single months) at those
points only; they read the cached coarse tokens (one-way attention), so the coarse pass is not
recomputed, and the decoder pools both. The backbone is OlmoEarth (also TerraMind, Tessera in the
appendix), fine-tuned end to end.

```
mistr/            the library
  core/           the model (model, one-way encoder, decoder), window selection, run config, data batches
  data/           multimodal, multitemporal datasets: PASTIS, PLANTEUR, BioMassters, YieldSAT, GEOID-Flood
  encoders/       OlmoEarth, TerraMind
  baselines/      compared methods: APT, vid-TLDR, AnyUp / Upsample Anything
exp/              experiments, run as `python -m exp.<pkg>.<module>` from the repo root
  mistr/          training (train; paper_configs = every run of the paper, paper_runs), probes, costs
  geoidchange/    GEOID-Flood flood tracking (tiletrack) and its S1 series
  tessera/        frozen Tessera v2 (App. H)
  viz/            every figure and table of the paper
results/          the results files every figure and table is drawn from
manifests/        the exact Sentinel-1 passes of the GEOID-Flood tracking series
notebooks/        two demos on released checkpoints (PASTIS, GEOID-Flood)
```

## Install

Python 3.12 and a CUDA GPU. FlashAttention is required: the datasets have missing timesteps, and
OlmoEarth's SDPA path cannot mask missing tokens (it refuses such batches).

```shell
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt                  # the exact versions of the paper's runs
pip install wheel packaging ninja psutil
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation flash-attn==2.8.3   # prebuilt wheel
pip install --no-deps -e .
python -m pytest -q                                   # data tests skip until the data is downloaded
```

Tested from scratch in the `python:3.12` Docker image (PyPI wheels only, CUDA 12 driver).
`pip install -e ".[flash]"` instead of the lock file takes the declared ranges.

## Demo

Two notebooks run released checkpoints on a few test samples (a GPU, about a minute each; no dataset
download): `notebooks/01_pastis_mistr.ipynb` (crop types; MiSTR S8 seasonal + S4 monthly at a 50 %
budget: predictions, the points it re-encodes, compute per tile) and
`notebooks/02_geoid_flood_tracking.ipynb` (32-pass Sentinel-1 series; fixed S128, MiSTR S128 & S32,
fixed S32). They read the files listed in `demo/MANIFEST.json` (checkpoints in `checkpoints/`, samples
in `demo/`, with sha256 sums). The checkpoints and demo samples are not hosted yet; they will be
linked here soon.

## Data

`docs/DATA.md`: where each dataset comes from, how to download and prepare it, and the
GEOID-Flood Sentinel-1 series.

## Reproduce the paper

Every figure and table of the paper (numbers as printed), from the shipped `results/`. "data" = the
datasets (`docs/DATA.md`); "ckpt" = trained checkpoints `checkpoints/<run>.pt` (train them with
`exp.mistr.paper_runs`; released with the public code).

| Paper | Command (`python -m ...`) | Needs |
|---|---|---|
| Fig 3 | `exp.viz.round1_main --paper` | |
| Fig 4 | `exp.viz.baselines_r1 --side` | |
| Fig 5 | `exp.viz.geoid_trackN tradeoff_col` | |
| Fig 6 | `exp.viz.geoid_tracklong_paper rec=EMSR275-1-4_q1 combo=128c32 q=0.08 compact=1 out=figures/paper/06_GEOID-sample.pdf` | data, ckpt |
| Figs 7, 8 | `exp.viz.space_time_ablation` | |
| Fig 9 | `exp.viz.season_range_tradeoff` | |
| Fig 10 | `exp.viz.round1_throughput` | |
| Fig 11 | `exp.viz.selector_gallery_r1 --paper` | data, ckpt |
| Fig 12 | `exp.viz.per_sample_compute`, `exp.viz.per_sample_compute --task biomassters` | data, ckpt |
| Fig 13, Table 5 | `exp.viz.oneway_attn_trained --paper` | |
| Fig 14 | `exp.viz.terramind_tradeoff paper` | |
| Fig 15 | `exp.viz.tessera_frozen_tradeoff paper` | |
| Fig 16, Tables 9, 10 | `exp.viz.round1_main_std --paper` | |
| Fig 17 | `exp.viz.geoid_cutmix_long_example k=16 rec=EMSR352-1-282_q2 out=figures/paper/11_GEOID-Cutmix-sample` | data |
| Fig 18 | `exp.viz.geoid_tracklong_samples out=figures/paper/13_GEOID-sample-full.pdf` | data, ckpt |
| Fig 19 | `exp.viz.geoid_trackN per_position fixed=cont cp=1 ks=4,8,12,16,20,24,28,31 out=figures/paper/12_GEOID-per-position.pdf` | data |
| Fig 20 | `exp.viz.paper.datasets` | data |
| Table 1 | `exp.viz.token_counts_table` | |
| Tables 2, 3 | `exp.viz.selector_tables_r1` | |
| Table 4 | `exp.viz.oneway_cost_table` | |
| Tables 6, 7, 8 | `exp.viz.oneway_probe_tables` | |
| Table 11 | `exp.viz.dataset_splits_table` | data |
| App. K numbers | `exp.viz.geoid_k_numbers` | |

Outputs go to `figures/paper/`. Regenerated from `results/`, they match the paper's figures pixel
for pixel and its tables byte for byte.

The results themselves:

```shell
python -m exp.mistr.paper_configs check                     # the 1,088 generated runs = the results rows
python -m exp.mistr.paper_runs list --tasks pastis          # the training runs, in order
python -m exp.mistr.paper_runs show <run name>              # one run's full config
python -m exp.mistr.paper_runs train --tasks pastis --match '^fixed_'
python -m exp.mistr.paper_runs slurm --tasks pastis > submit.sh   # or one job per run
python -m exp.paper_evals > evals.sh                         # probes, costs, throughput, Tessera, GEOID-Flood
```

Each training run appends its row to `results/<task>/mistr.csv` and saves `checkpoints/<run>.pt`.
Expect about 1-3 GPU-hours per Tiny run on one L40S (PLANTEUR and YieldSAT read faster with the
composite caches; `docs/DATA.md`).

## Licence

Code: Apache-2.0 (`LICENSE`). Released checkpoints are fine-tuned from OlmoEarth weights and are
distributed under the OlmoEarth Artifact License, including its use restrictions (`MODEL_LICENSE.md`).
Vendored code and datasets keep their own licences (`NOTICE`).

## Citation

```bibtex
@misc{zhou2026mistr,
  title  = {{MiSTR}: Adaptively Mixing Spatiotemporal Resolution for Satellite Image Time Series},
  author = {Zhou, Tim G. and Fuller, Anthony and Ma, Avery and Pleiss, Geoff and Shelhamer, Evan},
  year   = {2026},
}
```
