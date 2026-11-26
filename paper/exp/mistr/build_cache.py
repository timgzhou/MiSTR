"""Build the stored composites of mistr_paper.core.cache (train.gpu_data reads them afterwards).

    python -m exp.mistr.build_cache --tasks pastis planteur yieldsat_raw [--splits train val test]

--link_months (the *_year tasks): after a split is built, every array equal to the base
task's (the task name minus "_year": the month composites, timestamps, target) is replaced by a
symlink to the base task's file, so only the coarse-step arrays take new disk.
"""
from exp.common import olmo_bootstrap  # noqa: F401
import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from mistr_paper.core import cache  # noqa: E402


def link_months(task: str, split: str) -> list[str]:
    """Replace each array of task's cache that equals the base task's by a symlink to it."""
    base = cache.cache_dir(task.removesuffix("_year"), split)
    d, linked = cache.cache_dir(task, split), []
    if not (base / "meta.json").exists():
        return linked
    for f in sorted(d.glob("*.npy")):
        b = base / f.name
        if f.is_symlink() or not b.exists():
            continue
        x, y = np.load(f, mmap_mode="r"), np.load(b, mmap_mode="r")
        if x.shape != y.shape or x.dtype != y.dtype:
            continue
        step = max(1, (1 << 28) // max(1, x[:1].nbytes))      # (~256 MB chunks)
        if all(np.array_equal(x[i:i + step], y[i:i + step]) for i in range(0, len(x), step)):
            del x
            f.unlink()
            f.symlink_to(b.resolve())
            linked.append(f.name)
    return linked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--link_months", action="store_true",
                    help="symlink the arrays equal to the base task's cache (*_year tasks)")
    a = ap.parse_args()
    os.environ["MISTR_NO_CACHE"] = "1"          # (build from the live loader)
    for t in a.tasks:
        for s in a.splits:
            t0 = time.time()
            d = cache.build(TASKS[t], s, num_workers=a.workers)
            linked = link_months(t, s) if a.link_months else []
            gb = sum(f.stat().st_size for f in d.glob("*.npy") if not f.is_symlink()) / 1e9
            print(f"{t}/{s}: {d} {gb:.1f} GB new in {(time.time() - t0) / 60:.1f} min"
                  + (f", linked {' '.join(linked)}" if a.link_months else ""), flush=True)


if __name__ == "__main__":
    main()
