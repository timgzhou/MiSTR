"""Selector sample drawing (Figs 11 and 12: exp.viz.selector_gallery_r1, exp.viz.per_sample_compute): how a
MiSTR checkpoint picks windows (where: coarse-pass uncertainty above the val threshold) and seasons (when:
the most-attended season token) on a test tile, and what its second pass changes. Layout:

    <season> input   x NS | S_LR T_season loss        | label                        || MiSTR prediction
    <season> token loss   | S_LR T_season uncertainty | S_LR T_season prediction     || MiSTR change

Classification (PASTIS, PLANTEUR): loss = per-pixel cross-entropy, uncertainty = entropy of the
coarse prediction, change = fixed (coarse wrong -> MiSTR right) / broken (the reverse).
Regression (BioMassters, YieldSAT): loss = squared error in the loss's standardized units,
uncertainty = variance of the season tokens' own maps over the used seasons (what the selector
thresholds), change = |MiSTR error| - |coarse error| in t/ha (blue: MiSTR closer).
Season inputs: the model's own season MEAN composites of S2 (PASTIS: nodata frames included, as
the model sees them), one stretch per tile. Token loss: each season token's own head map
(selection.season_maps) vs the label, averaged per window; striped = picked windows, in a
season panel the windows whose picked season it is.
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import AGBM_MEAN, AGBM_STD, TASKS  # noqa: E402
from exp.mistr.train import set_attn, val_threshold  # noqa: E402
from mistr_paper.data.guidance import S2_RGB  # noqa: E402
from mistr_paper.data.time import from_days  # noqa: E402
from mistr_paper.baselines import build_model, build_selector_for  # noqa: E402
from mistr_paper.core.config import load  # noqa: E402
from mistr_paper.core.data import gpu_loader  # noqa: E402
from mistr_paper.core.selection import CoarsePass, season_maps, spatial_uncertainty  # noqa: E402
from mistr_paper.core.windows import TILE  # noqa: E402

NAMES = {"pastis": "PASTIS", "planteur": "PLANTEUR", "biomassters": "BioMassters",
         "yieldsat_raw": "YieldSAT"}
VOID = (0.82, 0.82, 0.82)
SEL = "#00e5ff"
SEL_STRIPE = (0.0, 0.898, 1.0, 0.6)
FIXED, BROKEN = (0.13, 0.4, 0.67), (0.8, 0.2, 0.15)
S_LR = r"$S_{\mathrm{LR}}T_{\mathrm{season}}$"
MON = "JFMAMJJASOND"


def checkpoint(task: str, q: str, pair: str = "S6Tsea+S3Tmon-w6") -> str:
    c = sorted(glob.glob(f"checkpoints/mistr_{task}_{pair}_q{q}_woracle*_seed0_*.pt"))
    # round 1 only: no _state, head16 or backbone variants (terramind_tiny, base, ...)
    c = [p for p in c if re.fullmatch(r"checkpoints/mistr_.+_woracle\d+_ep\d+_seed0_[0-9a-f]+\.pt", p)]
    # round 1 = warm-up for half the epochs (drops stage-1-only woracle50_ep50 runs)
    c = [p for p in c if (m := re.search(r"_woracle(\d+)_ep(\d+)_", p)) and 2 * int(m[1]) == int(m[2])]
    if len(c) != 1:
        raise FileNotFoundError(f"{task} q{q}: {c}")
    return pathlib.Path(c[0]).stem


def build(name: str):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ck["config"], f)
    cfg = load(f.name)
    task = TASKS[cfg.task]
    model = build_model(task, cfg, pretrained=False).cuda()
    bs = cfg.train.batch_size or task.batch_size[16]
    val = gpu_loader(task, "val", bs, 8, False)
    b0 = next(iter(val))
    sel = build_selector_for(task, cfg, model, (b0.ms, b0.ss))
    model.load_state_dict(ck["model"])
    nn.ModuleList(sel.modules()).load_state_dict(ck["selector"])
    set_attn(model, cfg.model.attention)
    model.eval()
    for m in sel.modules():
        m.eval()
    val_threshold(model, sel, val)               # as at test time
    del val
    return task, cfg, model, sel


def denorm(task):
    """standardized regression output -> t/ha"""
    if task.name == "biomassters":
        return lambda z: z * AGBM_STD + AGBM_MEAN
    idx = task.index
    return lambda z: z * idx["target_std"] + idx["target_mean"]


@torch.no_grad()
def compute(task, model, sel, batch):
    cls = task.kind == "classification"
    ctx = CoarsePass.from_batch(model, batch)
    plan = sel(ctx).plan
    out = model(batch.ms, batch.ss, batch.season, batch.present, plan, None, None,
                ctx.coarse_tokens, ctx.coarse_cache)
    p0, p1 = ctx.out0.prediction.float()[0], out.prediction.float()[0]          # (C, 64, 64)
    tgt = batch.target.cuda()
    if cls:
        y = tgt[0].long()
        ok = y >= 0
        lossf = lambda lg: F.cross_entropy(lg, y.clamp(min=0)[None].expand(lg.shape[0], -1, -1),  # noqa: E731
                                           reduction="none")
    else:
        y = (tgt[0, 0] if tgt.ndim == 4 else tgt[0]).float()
        ok = torch.isfinite(y)
        lossf = lambda lg: (lg[:, 0] - y.nan_to_num(0)) ** 2                    # noqa: E731
    loss0, loss1 = lossf(p0[None])[0], lossf(p1[None])[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sm = season_maps(model, ctx.out0.coarse_features).float()                # (NS, C, 64, 64)
    loss_s = lossf(sm)
    cp = sel.window_px
    okf = ok.float()
    lab = sel.pool(okf[None], cp)[0]
    win_s = sel.pool(loss_s * okf, cp) / lab.clamp(min=1e-6)
    win_s = win_s.masked_fill(lab[None] < 0.25, float("nan"))
    # the selector's picks, replayed: where (window threshold) x when (top-attended season)
    u_s = sel.window_uncertainty(ctx)
    hs = sel.picked_windows(ctx, u_s, sel.cfg.percentage)[0]
    r, _ = sel.attention_ranking(ctx)
    top = sel._up(r.shape[-1], r.device)(r)[0].argmax(0)
    chosen = torch.where(hs, top, torch.full_like(top, -1))
    n_plan_windows = len({(int(a), int(b)) for a, b in zip(plan.py.tolist(), plan.px.tolist())})
    nan = lambda v: torch.where(ok, v, torch.nan).cpu().numpy()                 # noqa: E731
    d = dict(target=(y.cpu().numpy() if cls else nan(y)), ok=ok.cpu().numpy(),
             loss0=nan(loss0), loss1=nan(loss1), win_s=win_s.cpu().numpy(),
             unc=spatial_uncertainty(ctx)[0].float().cpu().numpy(),
             chosen=chosen.cpu().numpy(), used=ctx.used[0].cpu().numpy(),
             crops=len(plan), windows=int(hs.sum()), plan_windows=n_plan_windows)
    if cls:
        d["pred0"], d["pred1"] = p0.argmax(0).cpu().numpy(), p1.argmax(0).cpu().numpy()
    else:
        d["pred0"], d["pred1"] = p0[0].cpu().numpy(), p1[0].cpu().numpy()
    return d


def season_labels(seasons_s2, NS):
    out = []
    for k in range(NS):
        a, b = from_days(int(seasons_s2.time.start[k])), from_days(int(seasons_s2.time.end[k]))
        s = f"{a.strftime('%b')}–{b.strftime('%b')}"
        out.append(s + (f" ’{b.strftime('%y')}" if NS > 4 else ""))
    return out


def rgb_fn(task, seasons_s2):
    bands = list(Paired_spec(task)["S2"].bands)
    idx = [bands.index(b) for b in S2_RGB]
    x = seasons_s2.data.float().numpy()[:, idx]                                 # (NS, 3, H, W)
    # nodata: PASTIS -10000 fill (< -100), YieldSAT no clear pixel / outside the field (all 0),
    # whole steps not frame_valid
    x = np.where((x < -100) | (x == 0).all(1, keepdims=True), np.nan, x)
    x[~seasons_s2.frame_valid.numpy().astype(bool)] = np.nan
    # one stretch per tile set by the per-pixel median over the seasons (method_parts.s2_stretch):
    # a snowy / cloudy season saturates to white instead of darkening the others
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x, 0)
    if np.isnan(med).all():
        med = x
    lo, hi = np.nanpercentile(med, 1), np.nanpercentile(med, 99.5) * 1.15
    # seasons far apart in brightness (BioMassters: snow vs dark forest, > 2.5x in median): each
    # season its own stretch (display only; noted in the caption)
    b = [float(np.nanmedian(x[k])) for k in range(len(x)) if not np.isnan(x[k]).all()]
    per_season = len(b) > 1 and min(b) > 0 and max(b) / min(b) > 2.5

    def f(k):
        if np.isnan(x[k]).all():
            return np.full(x.shape[2:] + (3,), VOID)
        l_, h_ = ((np.nanpercentile(x[k], 1), np.nanpercentile(x[k], 99.5) * 1.15) if per_season
                  else (lo, hi))
        v = np.clip((x[k].transpose(1, 2, 0) - l_) / (h_ - l_ + 1e-9), 0, 1) ** 0.9
        v[np.isnan(v).any(-1)] = VOID
        return v
    f.per_season = per_season
    return f


_SPEC = {}


def Paired_spec(task):
    return _SPEC[task.name]


def stripes(ax, mask, cp, lo, lw=0.8):
    """picked windows (G x G on the padded frame, offset lo): stripes + outer edges"""
    G = mask.shape[0]
    o = -lo - 0.5
    for y, x in zip(*np.nonzero(mask)):
        ax.add_patch(Rectangle((x * cp + o, y * cp + o), cp, cp, fill=False, hatch="////",
                               edgecolor=SEL_STRIPE, linewidth=0))
        for dy, dx, seg in ((-1, 0, ([x, x + 1], [y, y])), (1, 0, ([x, x + 1], [y + 1, y + 1])),
                            (0, -1, ([x, x], [y, y + 1])), (0, 1, ([x + 1, x + 1], [y, y + 1]))):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < G and 0 <= xx < G and mask[yy, xx]):
                ax.plot(np.array(seg[0]) * cp + o, np.array(seg[1]) * cp + o, color=SEL,
                        lw=lw, solid_capstyle="butt")
    ax.set_xlim(-0.5, TILE - 0.5)
    ax.set_ylim(TILE - 0.5, -0.5)


def draw(task, r, tile, seasons_s2, cp, lo, title, path, only_used=False, coarse=None,
         stats_in_title=True, side=None, fig=None, slots=None, key=True):
    """only_used: season columns for the seasons with data only (YieldSAT: ~3 of 8).
    coarse: the coarse pass's patch size, for the paper's labels ($S_8T_{sea}$; None: S_LR);
    stats_in_title: append the metric change and the window / crop counts to the title.
    side: (name, description) rotated at the left instead of the title on top.
    fig: draw into this (sub)figure and do not save (path ignored); slots: season columns to
    reserve, so blocks with fewer seasons keep the right-hand panels aligned (blank at the gap);
    key: draw the "selected point" key (one per combined figure)."""
    LR = S_LR if coarse is None else rf"$S_{{{coarse}}}T_{{sea}}$"
    cls = task.kind == "classification"
    NS = len(r["used"])
    names = season_labels(seasons_s2, NS)
    rgb = rgb_fn(task, seasons_s2)
    lcmap = plt.get_cmap("magma").copy()
    lcmap.set_bad(VOID)
    vmax = float(np.nanpercentile(np.concatenate([r["loss0"].ravel(), r["win_s"].ravel()]), 98))
    picked = r["chosen"] >= 0
    ok = r["ok"]
    if cls:
        tab = plt.get_cmap("tab20").colors
        pal = np.array([(0, 0, 0)] + [tab[i] for i in range(1, 19)] + [VOID])
        lab = lambda v: pal[np.where(v < 0, 19, v)]                             # noqa: E731
        lab_t = lambda v: lab(np.where(ok, v, -1))                              # noqa: E731
        right0, right1 = (r["pred0"] == r["target"]) & ok, (r["pred1"] == r["target"]) & ok
        change = np.ones(ok.shape + (3,))      # unlabelled: white, as unchanged (can never change)
        change[right1 & ~right0] = FIXED
        change[right0 & ~right1] = BROKEN
        m0, m1 = right0.sum() / ok.sum(), right1.sum() / ok.sum()
        metric = f"pixel acc {m0:.3f} → {m1:.3f}"
    else:
        dn = denorm(task)
        tv, p0v, p1v = dn(r["target"]), dn(r["pred0"]), dn(r["pred1"])
        vcmap = plt.get_cmap("viridis").copy()
        vcmap.set_bad(VOID)
        vlo, vhi = np.nanpercentile(tv, 1), np.nanpercentile(tv, 99)
        e0, e1 = np.abs(p0v - tv), np.abs(p1v - tv)
        dE = np.where(ok, e1 - e0, np.nan)
        dmax = float(np.nanpercentile(np.abs(dE), 98)) or 1.0
        dcmap = plt.get_cmap("RdBu_r").copy()
        dcmap.set_bad("white")                 # unlabelled: white, as no change
        m0, m1 = np.sqrt(np.nanmean(e0[ok] ** 2)), np.sqrt(np.nanmean(e1[ok] ** 2))
        metric = f"pixel RMSE {m0:.2f} → {m1:.2f} t/ha"

    def img(ax, x, **kw):
        ax.imshow(x, interpolation="nearest", **kw)

    P = {}
    raw = seasons_s2.data.float().numpy()
    shown = [s for s in range(NS) if r["used"][s] or not only_used]
    for s in shown:
        # constant fill (BioMassters): share of the panel at its most common nonzero B04 value
        b = raw[s, list(Paired_spec(task)["S2"].bands).index("B04")].ravel()
        b = b[b != 0]
        fill = np.unique(b, return_counts=True)[1].max() / raw[s, 0].size if b.size else 0
        tt = f"{names[s]} input" + (f"\n({fill:.0%} one constant value)" if fill > 0.5 else "")
        P[f"in{s}"] = (lambda ax, s=s: img(ax, rgb(s)), tt)

        def sl(ax, s=s):
            if r["used"][s]:
                img(ax, np.kron(r["win_s"][s], np.ones((cp, cp)))[lo:lo + TILE, lo:lo + TILE],
                    cmap=lcmap, vmin=0, vmax=vmax)
                stripes(ax, r["chosen"] == s, cp, lo)
            else:
                img(ax, np.full((TILE, TILE, 3), VOID))
                ax.text(TILE / 2, TILE / 2, "no data", ha="center", va="center", fontsize=6)
        P[f"sl{s}"] = (sl, (f"{names[s]} token loss" if coarse is None
                            else f"{names[s]} {LR}\nprediction loss"))

    def lrloss(ax):
        img(ax, r["loss0"], cmap=lcmap, vmin=0, vmax=vmax)
        stripes(ax, picked, cp, lo)
    P["lrloss"] = (lrloss, LR + (" loss" if coarse is None else " prediction loss"))

    def unc(ax):
        img(ax, r["unc"], cmap="viridis")
        stripes(ax, picked, cp, lo)
    P["unc"] = (unc, LR + (" entropy" if cls else " season variance"))
    if cls:
        P["label"] = (lambda ax: img(ax, lab_t(r["target"])), "label")
        P["pred0"] = (lambda ax: img(ax, lab_t(r["pred0"])), LR + " prediction")
        P["pred1"] = (lambda ax: img(ax, lab_t(r["pred1"])), "MiSTR prediction")

        def chg(ax):
            img(ax, change)
            stripes(ax, picked, cp, lo)
        P["chg"] = (chg, "MiSTR accuracy change")
    else:
        kw = dict(cmap=vcmap, vmin=vlo, vmax=vhi)
        P["label"] = (lambda ax: img(ax, tv, **kw), "label")
        P["pred0"] = (lambda ax: img(ax, np.where(ok, p0v, np.nan), **kw), LR + " prediction")
        P["pred1"] = (lambda ax: img(ax, np.where(ok, p1v, np.nan), **kw), "MiSTR prediction")

        def chg(ax):
            img(ax, dE, cmap=dcmap, vmin=-dmax, vmax=dmax)
            stripes(ax, picked, cp, lo)
        P["chg"] = (chg, "MiSTR |error| change")

    cols = [(f"in{s}", f"sl{s}") for s in shown] + [("lrloss", "unc"), ("label", "pred0"),
                                                         ("pred1", "chg")]
    nc = len(cols)
    gap = max(0, (slots or len(shown)) - len(shown))      # blank season columns
    own = fig is None
    with plt.rc_context({"font.family": "serif", "font.serif": ["Liberation Serif", "Times"],
                         "mathtext.fontset": "stix", "font.size": 7, "hatch.linewidth": 0.6}):
        if own:
            fig = plt.figure(figsize=(1.2 * (nc + gap) + 0.2, 3.3))
        gs = fig.add_gridspec(2, nc + gap + 1, width_ratios=[1] * (nc + gap - 1) + [0.12, 1],
                              left=0.01, right=0.99, top=0.84 if own else 0.93, bottom=0.2 if own else 0.19,
                              wspace=0.06, hspace=0.32)
        axs = {}
        for j, (a_, b_) in enumerate(cols):
            c = (j if j < len(shown) else j + gap) if j < nc - 1 else nc + gap
            for i, name in enumerate((a_, b_)):
                ax = fig.add_subplot(gs[i, c])
                P[name][0](ax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(P[name][1], fontsize=6.5, pad=2)
                axs[name] = ax
        x0, x1 = axs["label"].get_position().x1, axs["pred1"].get_position().x0
        fig.add_artist(plt.Line2D([(x0 + x1) / 2] * 2, [0.06, 0.9], color="0.35", lw=0.8,
                                  transform=fig.transSubfigure))
        if side:        # paper: (name, description) rotated at the left, no title on top
            fig.text(-0.012, 0.52, side[0], rotation=90, ha="right", va="center", fontsize=8,
                     fontweight="bold")
            if side[1]:
                fig.text(0.004, 0.52, side[1], rotation=90, ha="right", va="center", fontsize=6.5)
        else:
            fig.suptitle(title + (f"   |   {metric}   |   {r['windows']} windows, {r['crops']} crops"
                                  if stats_in_title else ""),
                         fontsize=8, y=0.98)

        def cbar(cmap, norm, under, label):
            p0_, p1_ = axs[under[0]].get_position(), axs[under[-1]].get_position()
            cax = fig.add_axes([p0_.x0, 0.105 if own else 0.1, p1_.x1 - p0_.x0, 0.022 if own else 0.025])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax,
                              orientation="horizontal")
            cb.set_label(label, fontsize=6, labelpad=1.5)
            cax.tick_params(labelsize=5.5, length=2, pad=1)

        cbar(lcmap, plt.Normalize(0, vmax), [f"sl{shown[0]}", f"sl{shown[-1]}"],
             ("cross-entropy" if cls else "squared error (standardized)")
             + (" (token losses and " + LR + " loss)" if coarse is None
                else " (" + LR + " prediction losses)"))
        cbar(plt.get_cmap("viridis"), plt.Normalize(np.nanmin(r["unc"]), np.nanmax(r["unc"])),
             ["unc"], "entropy" if cls else "variance over seasons")
        if cls:
            p = axs["chg"].get_position()
            fig.legend([Patch(facecolor=FIXED), Patch(facecolor=BROKEN),
                        Patch(facecolor="white", edgecolor="0.6", linewidth=0.5)],
                       [f"fixed: {int((right1 & ~right0).sum())} px",
                        f"broken: {int((right0 & ~right1).sum())} px", "unchanged"],
                       loc="upper left", bbox_to_anchor=(p.x0 - 0.005, 0.16), frameon=False,
                       fontsize=5.5, handlelength=1.2, handleheight=1.0, borderaxespad=0,
                       labelspacing=0.3)
        else:
            cbar(vcmap, plt.Normalize(vlo, vhi), ["pred0"], "t/ha")
            cbar(dcmap, plt.Normalize(-dmax, dmax), ["chg"], "Δ|error| t/ha (blue: better)")
        h = Patch(facecolor="0.15", edgecolor=SEL, hatch="////", linewidth=0.9)
        p = axs["label"].get_position()
        key and fig.legend([h], ["selected point"], loc="upper left",
                   bbox_to_anchor=(p.x0 - 0.005, 0.16 if cls else 0.0),
                   frameon=False, fontsize=5.5, handlelength=1.6, handleheight=1.6,
                   borderaxespad=0)
        if own:
            fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
    return float(m0), float(m1)
