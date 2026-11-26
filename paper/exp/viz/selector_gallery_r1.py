"""Fig 11 (App. F): where and when MiSTR S8Tsea & S4Tmon (q 50 %, guided warm-up, seed 0, its val-refit
threshold, as at test time) re-encodes high-resolution tokens on one fixed test tile per dataset, drawn
in exp.viz.selector_samples' layout (inputs | coarse loss & uncertainty | label & coarse prediction ||
MiSTR prediction & change). Needs the round-1 checkpoints and the test data.

    python -m exp.viz.selector_gallery_r1 --paper     -> figures/paper/16_MiSTR-selection-samples.pdf
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import pathlib  # noqa: E402

import torch  # noqa: E402

from exp.viz import selector_samples as S  # noqa: E402
from mistr_paper.core.data_cpu import Collate, Paired  # noqa: E402

# the fixed test tile per dataset (YieldSAT: with select.input_mask on at test time, val threshold refit --
# the trained model, no retraining)
PAPER = {"pastis": ("40304@r0c0", False), "planteur": ("1013@r64c64", False),
         "biomassters": ("e7d705ff@r64c0", False),
         "yieldsat_raw": ("Argentina_DUP1_farm22_field215_soybean_2024@r52c-15", True)}
PAPER_FILE = "16_MiSTR-selection-samples"     # one figure (was 16-19, one each)
# rotated label at the left of each block (no title on top; the config is in the LaTeX caption)
SIDE = {"pastis": ("PASTIS", "crop-type segmentation"),
        "planteur": ("PLANTEUR", "crop-type segmentation"),
        "biomassters": ("BioMassters", "above-ground biomass regression"),
        "yieldsat_raw": ("YieldSAT", "crop yield regression")}


def paper_one(tname, a):
    """-> the draw() arguments for the chosen tile of tname (runs the model)."""
    from dataclasses import replace
    from exp.mistr.train import val_threshold
    from mistr_paper.core.data import gpu_loader
    tile, mask = PAPER[tname]
    run = S.checkpoint(tname, a.q, a.pair)
    task, cfg, model, sel = S.build(run)
    if mask:                    # never pick windows without S2 input; refit as at test time
        sel.cfg = replace(sel.cfg, input_mask=True)
        val_threshold(model, sel, gpu_loader(task, "val", cfg.train.batch_size or task.batch_size[16],
                                             8, False))
    ds = Paired(task, "test")
    S._SPEC[task.name] = ds.spec
    item = ds[ds.ds.index_of(tile)]
    r = S.compute(task, model, sel, Collate(task)([item]))
    out = dict(task=task, r=r, tile=tile, seasons_s2=item[1].modalities["S2"], cp=sel.window_px,
               lo=sel.frame_lo(sel.window_px), coarse=cfg.model.coarse_patch)
    print(f"{tname}: {tile}, {run}", flush=True)
    del model, sel
    torch.cuda.empty_cache()
    return out


def paper(tasks, a):
    """figures/paper/16_MiSTR-selection-samples.pdf: one figure, one two-row block per dataset
    (rotated dataset label at the left; YieldSAT's 3 seasons keep the other columns aligned)."""
    import matplotlib.pyplot as plt
    blocks = [(t, paper_one(t, a)) for t in PAPER if t in tasks]
    with plt.rc_context({"pdf.fonttype": 42}):
        fig = plt.figure(figsize=(8.6, 2.65 * len(blocks)))
        for k, (sf, (t, b)) in enumerate(zip(fig.subfigures(len(blocks), 1, hspace=0.0), blocks)):
            S.draw(title="", path=None, only_used=True, stats_in_title=False, side=SIDE[t],
                   fig=sf, slots=4, key=k == 0, **b)
        out = pathlib.Path(f"figures/paper/{PAPER_FILE}.pdf")
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", action="store_true", help="(the only mode; kept for the README command)")
    ap.add_argument("--tasks", nargs="+", default=list(PAPER))
    ap.add_argument("--q", default="50")
    ap.add_argument("--pair", default="S8Tsea+S4Tmon")
    a = ap.parse_args()
    paper(a.tasks, a)


if __name__ == "__main__":
    main()
