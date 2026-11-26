"""Appendix figure: MiSTR spends its compute unevenly over samples.

    [density of high-resolution tokens per test sample] [4 token bins x 3 test samples: S2 | label]

Per dataset (--task): round-1 MiSTR S8Tsea+S4Tmon q 50 % (Fig 16's pair), seed 0, threshold refit
on val as at test time. Counts: exp.bench.mistr_crop_spread (results/<task>/crop_spread/<run>.pt, the whole
test split); high-resolution tokens = crops x the fine tokens one crop adds (from the run's own
trace: (tokens_per_block - fixed S8Tsea's) / crops: 4). Samples: 3 fixed test tiles per bin
(PAPER_TILES; bins 50 / 1000 / 2000 / 2900 +- 50 tokens), shaded on the histogram and named above
each column (originally a seeded draw among the bin's tiles >= 70 % labelled). Each picked
sample's count is recomputed here and must match the stored one within 5 % (bf16 differs across processes and batch sizes).
Composite = per-pixel median of the season mean composites the model sees (RGB, one stretch per
tile); label in Fig 16's palette (grey: void); regression: t/ha, one viridis scale per figure.

    python -m exp.viz.per_sample_compute [--task pastis|planteur|biomassters|yieldsat_raw]
      -> figures/paper/26_MiSTR-per-sample-compute[-<task>].pdf (PASTIS without the suffix)
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exp.viz import common as bp  # noqa: E402
from exp.viz import common as rp  # noqa: E402
from exp.viz import selector_samples as ss  # noqa: E402
from mistr_paper.data.guidance import S2_RGB  # noqa: E402
from mistr_paper.core.data_cpu import Collate, Paired  # noqa: E402

PAIR, Q = "S8Tsea+S4Tmon", "50"
# high-resolution tokens per sample: bin centres
DEFAULT_BINS = (50, 1000, 2000, 2900)
HALF = 50                          # +- half width
N_PER_BIN = 3
# Fig 12's test tiles per bin (PASTIS, BioMassters)
PAPER_TILES = {
    "pastis": {50: ("30516@r0c0", "20502@r64c64", "40345@r0c0"),
               1000: ("10031@r0c0", "10029@r64c0", "10073@r64c0"),
               2000: ("40327@r0c0", "30177@r0c64", "20290@r0c64"),
               2900: ("30306@r64c0", "20637@r64c64", "30427@r0c64")},
    "biomassters": {50: ("7bd46c72@r128c0", "61b88adf@r128c128", "b6bd866c@r192c128"),
                    1000: ("f2909ba6@r64c192", "eb9292ef@r192c64", "e3da49d2@r192c192"),
                    2000: ("cc21b584@r128c192", "92b6606b@r64c128", "6b7fd04f@r128c0"),
                    2900: ("8d093961@r192c192", "71e5c718@r0c192", "9a4e7b4f@r192c128")}}
NAMES = {"pastis": "PASTIS", "planteur": "PLANTEUR", "biomassters": "BioMassters",
         "yieldsat_raw": "YieldSAT"}


def out_paths(task):
    sfx = "" if task == "pastis" else "-" + NAMES[task]
    return (pathlib.Path(f"figures/paper/26_MiSTR-per-sample-compute{sfx}.pdf"),
            pathlib.Path(f"figures/per_sample_compute/info{sfx}.json"))     # picks (not a figure)


def tokens_per_crop(TASK: str, run: str) -> float:
    d = pd.read_csv(f"results/{TASK}/mistr.csv", low_memory=False).drop_duplicates("run", keep="last")
    r = d[d.run == run].iloc[-1]
    f = d[d.run.str.match(rf"fixed_{TASK}_S{r['model.coarse_patch']:.0f}Tsea_ep\d+_seed0_[0-9a-f]+$")]
    return (r.tokens_per_block - f.tokens_per_block.iloc[-1]) / r.compute_crops_per_sample


def composite(task, seasons_s2):
    bands = list(ss.Paired_spec(task)["S2"].bands)
    x = seasons_s2.data.float().numpy()[:, [bands.index(b) for b in S2_RGB]]     # (NS, 3, H, W)
    x = np.where((x < -100) | (x == 0).all(1, keepdims=True), np.nan, x)
    x[~seasons_s2.frame_valid.numpy().astype(bool)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(x, 0).transpose(1, 2, 0)                                # (H, W, 3)
    lo, hi = np.nanpercentile(m, 1), np.nanpercentile(m, 99.5) * 1.15
    v = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1) ** 0.9
    v[np.isnan(v).any(-1)] = ss.VOID
    return v


def label_rgb(target):
    tab = plt.get_cmap("tab20").colors
    pal = np.array([(0, 0, 0)] + [tab[i] for i in range(1, 19)] + [ss.VOID])
    t = np.asarray(target)
    return pal[np.where((t < 0) | (t > 18), 19, t)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="pastis", choices=list(PAPER_TILES))
    a = ap.parse_args()
    TASK = a.task
    OUT, INFO = out_paths(TASK)
    bins_c = DEFAULT_BINS
    # round 1's run: warm-up = half the epochs (ss.checkpoint also matches stage-1-only _ep50 ones)
    import glob
    import re
    c = []
    for p in glob.glob(f"checkpoints/mistr_{TASK}_{PAIR}_q{Q}_woracle*_ep*_seed0_*.pt"):
        m = re.search(r"_woracle(\d+)_ep(\d+)_seed0_[0-9a-f]+\.pt$", p)
        if m and 2 * int(m.group(1)) == int(m.group(2)):
            c.append(pathlib.Path(p).stem)
    if len(c) != 1:
        raise SystemExit(f"round-1 checkpoint: {c}")
    run = c[0]
    counts = torch.load(f"results/{TASK}/crop_spread/{run}.pt").numpy()
    k = tokens_per_crop(TASK, run)
    tok = counts * k
    task, cfg, model, sel = ss.build(run)
    cls = task.kind == "classification"
    ds = Paired(task, "test")
    ss._SPEC[task.name] = ds.spec
    if len(ds) != len(counts):
        raise SystemExit(f"test split {len(ds)} != stored counts {len(counts)}")
    col = Collate(task)
    void = ds.ds.SPEC.target.ignore_index if hasattr(ds.ds, "SPEC") else -1
    labelled = {}

    def lab_share(i):       # labelled share of a tile (classes: target not void; regression: finite)
        if i not in labelled:
            y = col([ds[i]]).target[0].float().numpy()
            labelled[i] = float((((y >= 0) & (y != void)) if cls else np.isfinite(y)).mean())
        return labelled[i]

    picks = []                              # [(bin centre, half width, [sample indices])]
    for c in bins_c:
        tiles = PAPER_TILES[a.task][c]
        picks.append((c, HALF, sorted((ds.ds.index_of(t) for t in tiles), key=lambda i: tok[i])))
    info = []
    samples = {}
    targets = {}
    for c, half, idx in picks:
        for i in idx:
            item = ds[i]
            r = ss.compute(task, model, sel, col([item]))
            # (bf16 at batch 1 vs the stored batch-32 pass, and across processes: a few windows at
            # the threshold may flip; +-3 crops seen on the same sample between runs)
            if abs(r["crops"] - int(counts[i])) > max(5, 0.05 * counts[i]):
                raise SystemExit(f"sample {i}: recomputed {r['crops']} crops != stored {int(counts[i])}")
            targets[i] = r["target"] if cls else ss.denorm(task)(r["target"])
            samples[i] = composite(task, item[1].modalities["S2"])
            info.append({"bin": c, "index": i, "tile": ds.ds.sample_id(i), "crops": int(counts[i]),
                         "crops_recomputed": int(r["crops"]), "hr_tokens": float(tok[i]),
                         "labelled": round(lab_share(i), 3),
                         "quantile": round(float((tok < tok[i]).mean()), 3)})
            print(json.dumps(info[-1]), flush=True)

    if cls:
        labels = {i: label_rgb(t) for i, t in targets.items()}
    else:                   # one t/ha scale over the figure's samples
        v = np.concatenate([t[np.isfinite(t)] for t in targets.values()])
        vlo, vhi = np.percentile(v, 1), np.percentile(v, 99)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vlo, vhi)

        def reg_rgb(t):
            im = cmap(norm(np.nan_to_num(t, nan=vlo)))[..., :3]
            im[~np.isfinite(t)] = ss.VOID
            return im
        labels = {i: reg_rgb(t) for i, t in targets.items()}
    samples = {i: (samples[i], labels[i]) for i in samples}

    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "font.family": ["Liberation Serif", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(6.875, 2.75), facecolor=rp.SURFACE)
        widths = [3.4, .3]
        for j in range(len(bins_c)):
            widths += [1, 1] + ([.32] if j < len(bins_c) - 1 else [])
        gs = fig.add_gridspec(N_PER_BIN, len(widths), width_ratios=widths, wspace=.03,
                              hspace=.05, left=.05, right=.995, bottom=.12, top=.84)
        ax = fig.add_subplot(gs[:, 0])
        edges = np.linspace(0, np.percentile(tok, 99.5), 31)
        ax.hist(np.clip(tok, 0, edges[-1]), bins=edges, density=True, color=bp.OURS, alpha=.85,
                edgecolor=rp.SURFACE, linewidth=.4, zorder=2)
        for c, half, _ in picks:    # the sampled bins
            ax.axvspan(max(0, c - half), c + half, color=rp.INK, alpha=.18, lw=0, zorder=1)
        ax.axvline(tok.mean(), color=rp.INK, lw=.9, ls="--", zorder=3)
        ax.text(tok.mean(), ax.get_ylim()[1] * .97, " mean", va="top", ha="left", fontsize=7)
        ax.set_xlabel("high-resolution tokens per sample")
        ax.set_ylabel("density")
        ax.set_facecolor(rp.SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([])
        ax.set_title(f"{NAMES[TASK]} test split", loc="left", fontweight="bold", pad=6)
        col0 = 2
        for j, (c, half, idx) in enumerate(picks):
            for row, i in enumerate(idx):
                for k_, im in enumerate(samples[i]):
                    a_ = fig.add_subplot(gs[row, col0 + k_])
                    a_.imshow(im, interpolation="nearest")
                    a_.set_xticks([])
                    a_.set_yticks([])
                    for sp in a_.spines.values():
                        sp.set_linewidth(.4)
                    if row == 0:        # sub-column heads: input | label
                        a_.set_title(("input (S2)", "label")[k_], fontsize=6.5, pad=2)
                        if k_ == 0:
                            left = a_.get_position().x0
                        else:
                            right = a_.get_position().x1
            # the bin's head over its two sub-columns, underlined across both
            fig.text((left + right) / 2, .905, f"{max(c - half, 0):,}–{c + half:,} tokens", ha="center",  # range, not +-
                     va="bottom", fontsize=7.5, fontweight="bold")
            fig.add_artist(plt.Line2D([left, right], [.9, .9], color=rp.INK, lw=.6))
            if j < len(picks) - 1:  # a divider in the gap before the next bin
                nxt = gs[0, col0 + 3].get_position(fig).x0
                xm = (right + nxt) / 2
                fig.add_artist(plt.Line2D([xm, xm], [.12, .93], color=rp.INK, lw=.5, alpha=.5))
            col0 += 3
        x0, x1 = fig.axes[1].get_position().x0, fig.axes[-1].get_position().x1
        fig.text((x0 + x1) / 2, .975, r"fewer high-resolution tokens  $\longrightarrow$  more",
                 ha="center", va="bottom")
        if not cls:         # the label scale, under the right-most bin
            cax = fig.add_axes([right - .12, .035, .12, .02])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation="horizontal")
            cb.set_label("label (t/ha)", fontsize=6.5, labelpad=1)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
            cb.outline.set_linewidth(.4)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, facecolor=rp.SURFACE, bbox_inches="tight", pad_inches=.02)
    INFO.parent.mkdir(parents=True, exist_ok=True)
    INFO.write_text(json.dumps(
        {"run": run, "tokens_per_crop": k, "mean_tokens": float(tok.mean()), "samples": info},
        indent=1))
    print(f"wrote {OUT}  (tokens per crop {k:.2f}, mean {tok.mean():.0f})")


if __name__ == "__main__":
    main()
