# MiSTR: Adaptively Mixing Spatiotemporal Resolution for Satellite Image Time Series

Tim G. Zhou, Anthony Fuller, Avery Ma, Geoff Pleiss, Evan Shelhamer
(University of British Columbia, Vector Institute, RBC Borealis)

MiSTR lets a remote-sensing transformer spend its compute where and when the task needs detail.
A **coarse pass** encodes the whole time series at low resolution (8 px patches of seasonal
composites) and predicts. From that prediction, MiSTR **selects** the points that need more detail:
*where* the coarse prediction is uncertain, and *when*, by the decoder's attention. A sparse
**fine pass** adds 4 px tokens of single months at those points only. Fine tokens read the
cached coarse tokens (one-way attention), so the coarse pass is never recomputed, and the
decoder combines both. The backbone is [OlmoEarth](https://github.com/allenai/olmoearth_pretrain).

This repository has two parts:

- `mistr/`: the model, ready to use (about 800 lines: `model.py`, `inputs.py`).
- `paper/`: everything to reproduce the paper (training, baselines, every figure and table); see
  [`paper/README.md`](paper/README.md).

## Install

Python 3.12; a GPU is optional. Try it in Colab first:
<a href="https://colab.research.google.com/github/timgzhou/MiSTR/blob/main/notebooks/mistr_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
(`notebooks/mistr_quickstart.ipynb`).

```shell
pip install "mistr @ git+https://github.com/timgzhou/MiSTR"
```

That runs on CPU or any GPU with PyTorch's attention. On Ampere or newer GPUs, installing
FlashAttention 2 makes the model use it automatically, which is how the paper ran (bit-identical to the
paper's numbers; PyTorch's attention agrees with it on 99.7 % of pixels):

```shell
pip install wheel packaging ninja psutil
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation flash-attn==2.8.3   # prebuilt wheel, torch 2.9
```

## Quickstart

```python
from mistr import MiSTR, prepare

model = MiSTR.from_pretrained("TimZhou/mistr-olmoearth-tiny-pastis-s6s3")   # ~26 MB; GPU if there is one

x = prepare(s2, s2_dates, s1, s1_dates, start="2018-10")   # one image's time series, any size
out = model.predict(x, budget=0.5, seasons=2)

out.classes        # (H, W) crop type per pixel (model.config["class_names"])
out.logits         # (19, H, W)
out.windows        # (N, 3) the re-encoded (month, y, x): windows of out.window_px px, top-left pixel
out.coarse_logits  # the coarse pass's own prediction
```

**Inputs** (`prepare`):

| argument | shape | content |
|---|---|---|
| `s2` | (frames, 10, H, W) | Sentinel-2 L2A bands `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`, surface reflectance x 10,000 (no BOA offset) |
| `s1` | (frames, 2, H, W) | Sentinel-1 `VV VH` backscatter in dB (the model was trained on ascending passes) |
| `s2_dates`, `s1_dates` | one per frame | `datetime.date`, `"YYYY-MM-DD"` or `numpy.datetime64` |
| `start` | | the first of 12 months; frames outside them are ignored |
| `s2_valid`, `s1_valid` | (frames, H, W) bool, optional | `False` at pixels to leave out of the monthly means |

Pixels are at 10 m. `prepare` averages each modality's frames per calendar month, averages the
months into four 3-month seasons, and normalizes both with OlmoEarth's statistics. It then cuts the
image into 64 x 64 px tiles, padding the bottom and right edges. `predict` runs the tiles in
batches and stitches the result back to H x W. A month with no valid frame in a modality is
treated as missing.

**Budget and seasons.** `budget` is the share of windows that get fine tokens, measured on the
validation distribution. A fixed threshold on the coarse prediction's uncertainty, shipped with the
weights, decides which windows; so an image with more uncertain areas gets more fine tokens. Both
models were trained at 0.5, and lower budgets trade accuracy for compute. `seasons` (S6+S3 model
only) re-encodes each selected window in 1 or 2 seasons. At 1.5, only the more uncertain windows
get a second season; at budgets of 0.25 or less every selected window does, so 1.5 and 2 coincide.

## The released models

Both are OlmoEarth-v1-Tiny, fine-tuned for crop-type segmentation on
[PASTIS-R](https://github.com/VSainteuf/pastis-benchmark): 19 classes, agricultural parcels in
France, October 2018 to September 2019, trained on folds 1-3, tested on fold 5 (1,984 tiles).

| model | coarse / fine tokens | coarse tokens / tile |
|---|---|---|
| [`TimZhou/mistr-olmoearth-tiny-pastis-s6s3`](https://huggingface.co/TimZhou/mistr-olmoearth-tiny-pastis-s6s3) | 6 px seasonal / 3 px monthly (6 px windows), 1-2 seasons per window | 1,936 |
| [`TimZhou/mistr-olmoearth-tiny-pastis`](https://huggingface.co/TimZhou/mistr-olmoearth-tiny-pastis) | 8 px seasonal / 4 px monthly, 1 season per window | 1,024 |

Test mIoU and fine tokens per 64 px tile:

| model | seasons | budget | test mIoU | fine tokens / tile |
|---|---|---|---|---|
| S6+S3 | 2 | 0.5 (trained) | **61.21** | 5,666 |
| S6+S3 | 1.5 | 0.5 | 60.87 | 4,279 |
| S6+S3 | 2 or 1.5 | 0.25 | 59.61 | 2,881 |
| S6+S3 | 1 | 0.5 | 58.96 | 2,843 |
| S6+S3 | 2 or 1.5 | 0.1 | 57.27 | 1,239 |
| S6+S3 | any | 0 (coarse pass only) | 52.96 | 0 |
| S8+S4 | 1 | 0.5 (trained) | 57.29 | 1,517 |
| S8+S4 | 1 | 0.25 | 56.06 | 770 |
| S8+S4 | 1 | 0.1 | 53.84 | 318 |
| S8+S4 | 1 | 0 (coarse pass only) | 50.60 | 0 |

A coarse token covers 4 seasons x one patch x 4 token groups (3 Sentinel-2 band groups and
Sentinel-1). A selected window adds, per month of each selected season, 4 tokens per fine patch
(S6+S3: four 3 px patches per 6 px window). In the paper these are S6 seasonal + S3 monthly (its
flexible temporal budget appendix, 60.9 ± 0.2 mIoU over three seeds with 2 seasons; this is seed 1)
and S8 seasonal + S4 monthly. On other regions, years or crop types, expect to fine-tune: see
`paper/` for the training code.

Results can differ in the last bf16 bits with the batch size (`predict(..., batch_size=)`), which
flips the class of very few pixels.

## Reproducing the paper

[`paper/README.md`](paper/README.md): installation, datasets, every training run as code,
and the command behind every figure and table.

## Citation

```bibtex
@misc{zhou2026mistr,
  title  = {{MiSTR}: Adaptively Mixing Spatiotemporal Resolution for Satellite Image Time Series},
  author = {Zhou, Tim G. and Fuller, Anthony and Ma, Avery and Pleiss, Geoff and Shelhamer, Evan},
  year   = {2026},
}
```

## Licence

Code: Apache-2.0 (`LICENSE`). The released weights are fine-tuned from OlmoEarth and are
distributed under the OlmoEarth Artifact License, including its use restrictions
(`MODEL_LICENSE.md`). Third-party code and data keep their own licences (`NOTICE`).
