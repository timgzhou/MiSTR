"""App. L, table datasets: train / val / test sizes, counted from the data (patches for PASTIS and
PLANTEUR, chips for BioMassters, field-seasons for YieldSAT: the sample id before '@'; series for
GEOID-Flood, from the tracking cache). Needs the datasets (docs/DATA.md).

    python -m exp.viz.dataset_splits_table      # -> figures/paper/dataset-splits.tex (the tabular rows)
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from mistr.core.data import RawPaired  # noqa: E402

SPLITS = ("train", "val", "test")
ROWS = (("PASTIS", "pastis", "segmentation", "S2 + S1 (4)", "18 crops + bg."),
        ("PLANTEUR", "planteur", "segmentation", "S2 + S1 (4)", "7 crops + bg."),
        ("BioMassters", "biomassters", "regression", "S2 + S1 (4)", "biomass (t/ha)"),
        ("YieldSAT", "yieldsat_raw", "regression", "S2 (3)", "yield (t/ha)"),
        ("GEOID-Flood", None, "tracking", "S1 (1)", "dry / water / flood"))
GEOID_CACHE = Path("data/GEOID-Flood-tileseries-long2/cache")
OUT = Path("figures/paper/dataset-splits.tex")


def units(task, split):
    if task is None:
        return len(json.load(open(GEOID_CACHE / f"{split}.json")))
    ds = RawPaired(TASKS[task], split)
    base = ds.ds if hasattr(ds, "ds") else ds
    return len({str(base.sample_id(i)).split("@")[0] for i in range(len(base))})


def main():
    lines = []
    for name, task, kind, inp, target in ROWS:
        n = " / ".join(f"{units(task, s):,}".replace(",", "{,}") for s in SPLITS)
        lines.append(f"{name} & {kind} & {inp} & {target} & {n} \\\\")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text(), end="")


if __name__ == "__main__":
    main()
