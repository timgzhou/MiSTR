"""A paper checkpoint -> a from_pretrained folder (config.json, backbone/config.json,
model.safetensors).

    python -m tools.export_pretrained checkpoints/<run>.pt OUT_DIR
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

RENAME = {"mix.encoder.": "backbone.", "head.linear.": "head.", "readout.score.": "score."}
DROP = ("mix.hr_type",)        # a fine-token type embedding, zero and frozen in MiSTR


def export(ckpt: str, out: str, class_names: list[str]):
    from olmoearth_pretrain.model_loader import ModelID, _resolve_artifact_path
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m, s = cfg["model"], cfg["select"]
    want = dict(backbone="tiny", coarse_patch=8, coarse_time="season", window_px=4, fine_patch=0,
                attention="one_way", coarse_head=True)
    if any(m[k] != v for k, v in want.items()) or (s["where"], s["when"]) != ("uncertainty", "attention") \
            or s["extra_steps"]:
        raise ValueError(f"{ckpt}: not the released configuration ({m}, {s})")
    sd = {}
    for k, v in ck["model"].items():
        if k in DROP:
            if v.abs().max() != 0:
                raise ValueError(f"{k} is not zero")
            continue
        new = next((n + k[len(o):] for o, n in RENAME.items() if k.startswith(o)), None)
        if new is None:
            raise KeyError(f"unmapped key {k}")
        sd[new] = v.contiguous()
    sd["quantiles"] = ck["selector"]["0.tab_s"].contiguous()
    o = Path(out)
    (o / "backbone").mkdir(parents=True, exist_ok=True)
    shutil.copy(_resolve_artifact_path(ModelID.OLMOEARTH_V1_TINY, "config.json"), o / "backbone/config.json")
    save_file(sd, str(o / "model.safetensors"))
    config = {"backbone": "OlmoEarth-v1-Tiny", "coarse_patch": 8, "fine_patch": 4,
              "train_budget": s["percentage"], "num_classes": len(class_names),
              "class_names": class_names, "task": cfg["task"], "source_run": Path(ckpt).stem}
    (o / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    return o


if __name__ == "__main__":
    from mistr_paper.data.datasets.pastis import CLASS_NAMES
    print(export(sys.argv[1], sys.argv[2], list(CLASS_NAMES[:19])))
