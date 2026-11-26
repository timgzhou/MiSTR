"""Training constants shared with OlmoEarth's fine-tuning recipe, and the results-CSV / log helpers."""
import csv
import fcntl
from pathlib import Path

# OlmoEarth's fine-tuning constants (olmoearth_pretrain/evals/finetune/constants.py, 0.1.2)
FREEZE_EPOCH_FRACTION = 0.2
UNFREEZE_LR_FACTOR = 0.1
SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_COOLDOWN = 0.2, 2, 10


def _brief(metrics: dict) -> str:
    """Metrics for the log; per-class IoUs only when there are a few (all go to the CSV)."""
    def per_class(prefix):   # prefix_<class name>, excluding the prefix_headline summary key
        ks = [k for k in metrics if k.startswith(prefix) and k != f"{prefix}headline"]
        return ks if len(ks) <= 3 else []
    hide = (set(k for k in metrics if k.startswith(("iou_", "biou_")))
            - set(per_class("iou_")) - set(per_class("biou_")))
    keep = [k for k in metrics if k != "primary" and k not in hide]
    return f"primary {metrics['primary']:.4f} | " + " ".join(f"{k} {metrics[k]:.4f}" for k in keep)


def _append(path: Path, row: dict) -> None:
    """Append a row; a row with columns the file lacks widens its header (older rows get
    empty cells), so a new field never misaligns the CSV. Locked: jobs append concurrently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, fields = [], []
        if path.exists():
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                fields, rows = list(r.fieldnames or []), list(r)
        extra = [k for k in row if k not in fields]
        if not extra:
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            return
        fields += extra
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows + [row])
        tmp.replace(path)
