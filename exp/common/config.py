"""OlmoEarth checkpoints by model size (the `backbone` / `model_size` of a run)."""
# model_size -> olmoearth_pretrain ModelID member. A bare size is OlmoEarth v1 (the paper's
# models); later releases are keyed <version>_<size> (ModelID names: olmoearth_pretrain/model_loader.py).
MODEL_SIZE_TO_ID = {
    "nano": "OLMOEARTH_V1_NANO",
    "tiny": "OLMOEARTH_V1_TINY",
    "base": "OLMOEARTH_V1_BASE",
    "large": "OLMOEARTH_V1_LARGE",
    "v1_1_nano": "OLMOEARTH_V1_1_NANO",
    "v1_1_tiny": "OLMOEARTH_V1_1_TINY",
    "v1_1_base": "OLMOEARTH_V1_1_BASE",
    "v1_2_nano": "OLMOEARTH_V1_2_NANO",
    "v1_2_tiny": "OLMOEARTH_V1_2_TINY",
    "v1_2_small": "OLMOEARTH_V1_2_SMALL",
    "v1_2_base": "OLMOEARTH_V1_2_BASE",
}
