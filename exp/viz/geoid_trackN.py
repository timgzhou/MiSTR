"""Round N figures (GEOID-Flood flood tracking, T = 32 dense series, event at ANY position, no reference,
batch-shared crop budget, seed 0). Scores: results/geoidchange/trackN_rescore.jsonl (exp.geoidchange.trackN_rescore,
per-class scores per event position): 3-class mIoU and FLOOD-class boundary IoU on the labelled date.

x = CAUSAL-PREFIX cost: GMACs of the forward over passes 0..k, all a real-time model spends to predict event pass k
(causal attention: later passes cannot change it). results/geoidchange/tiletrack_N_prefix_flops.csv (every pass and
MiSTR at T = k + 1) and tiletrack_N_prefix_fixed_flops.csv (strided: the kept passes <= k, T = 2..13).
  fixed / strided: scores of the T = 32 evaluation (= the prefix ones: causal);
  MiSTR = the BUDGET-RANGE models (one model per resolution pair, q drawn per batch from 4/8/16/32 % in training;
    the method), scored on series CUT at the event (trunc rows): the crop budget only reaches passes <= k.
Strided fixed res (every 2nd / 4th pass) at a RANDOM stride phase: scores and costs averaged over lags 0..stride-1
(lag j: the last kept pass is j passes before the event; fewer kept passes).

    python -m exp.viz.geoid_trackN tradeoff       -> figures/geoidchange/01_tracking_mIoU-floodbIoU_vs_GMAC.pdf
        mean over event passes 12 / 16 / 20 / 24 of score and prefix cost (pass 8 only in 02); hue = pair
    python -m exp.viz.geoid_trackN tradeoff_col   -> figures/paper/05_GEOID.pdf (one column: panels stacked)
    python -m exp.viz.geoid_trackN per_position   -> figures/geoidchange/02_tracking_vs_GMAC_per_event_position.pdf
        one row per event position (stacked, shared axes): the later the event, the more a fixed model must spend
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from exp.viz.common import INK, MUTED  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 8})
OUT = Path("figures/geoidchange")
OUT_FILE = None
RES = "results/geoidchange"
COMBOS = [(128, 64), (128, 32), (64, 32), (64, 16)]
COLOR = dict(zip(COMBOS, plt.get_cmap("Set2").colors))           # Set2 per pair, as the main datasets figure
METRICS = [("miou", "mIoU (3-class)"), ("fbiou", "flood boundary IoU")]
KS = (8, 12, 16, 20, 24)
CP = False                       # cp=1 (Fig 12): every column on the SAME 1,266 long3 test series (31 passes of
                                 # history, cut at the event; trackN_rescore common_prefix=1), costs measured on them
KS_MAIN = (12, 16, 20, 24)      # 01: event at pass 8 left out (2 kept passes of history for a missed stride phase);
                                #     every position, pass 8 included, in 02
# fixed-res families: stride -> (line, marker, filled, legend label); every pass as the main figure's monthly
FAM = {1: ("-", "D", True, "every timestep")}      # sparse passes (every 2nd / 4th, random phase) not drawn: MiSTR
                                                 # could run on them too (not evaluated); rows stay in the results
SPARSE = {2: ("--", "s", False, "every 2nd timestep (random phase)"), 4: (":", "o", False, "every 4th timestep (random phase)")}
MISTR = "_qr"   # "_qr_or": the 14+14 oracle-warm-up MiSTR from OlmoEarth (pairs 128c32 + 64c16 only)
FIXED = ""                       # "cont": fixed points = the 28-epoch (+8) fixed models, training-matched to MiSTR
ROUND = "O"                      # O (train-only 4-week gap, no ignore, CutMix 10-40 %, q = share of the
                                 # whole series) or N; `round=N` on the command line
ROUNDS = {"N": dict(prefix="trkN_", q_max=0.32, cost=("tiletrack_N_prefix_flops", "tiletrack_N_prefix_flops_highq"),
                    q_text=r"$q \in$4-32 % of 4 timesteps"),
          "O": dict(prefix="trkO_", q_max=0.08, cost=("tiletrack_O_prefix_flops",), q_text=r"$q \in$1-8 % of the series")}
Q_TRAIN_MAX = 0.32               # budget-range training drew q from 4-32 %; only those budgets are drawn (set per round)
                                 # (q .5-2 rows exist in trackN_rescore.jsonl: not used)
GAINS = [(64, (128, 32)), (32, (64, 16))]   # fixed S_L -> MiSTR pair: compute saving at matched performance
X_LABEL = "GMACs to predict the event timestep (causal prefix, log)"


def _score(t, met):     # x 100: the paper reports IoUs in 0-100
    return 100 * (t["miou"] if met == "miou" else t["biou_cls"][2])


def load():
    """-> {(kind, L, C, q): {k: {miou, fbiou, gmac}}}; kind fixed / strided (C = stride) / range (MiSTR).
    Strided: scores and prefix costs averaged over the stride phases."""
    pref, pfix = {}, {}
    cfg = ROUNDS[ROUND]
    tables = ("tiletrack_O_prefix_flops_long3", "tiletrack_O_prefix_flops_long3_t5") if CP else \
        ("tiletrack_N_prefix_flops",) + cfg["cost"]
    for f in tables:   # fixed costs (crop 0) from round N's table: same models;
                                                           # MiSTR costs from the round's own table (its q unit)
        if Path(f"{RES}/{f}.csv").exists():
            for r in csv.DictReader(open(f"{RES}/{f}.csv")):
                if int(r["crop"]) and f == "tiletrack_N_prefix_flops" and ROUND != "N":
                    continue
                pref[(int(r["lr_ps"]), int(r["crop"]), round(float(r["q"]), 4), int(r["T"]))] = float(r["gmac"])
    if Path(f"{RES}/tiletrack_N_prefix_fixed_flops.csv").exists():
        for r in csv.DictReader(open(f"{RES}/tiletrack_N_prefix_fixed_flops.csv")):
            pfix[(int(r["lr_ps"]), int(r["T"]))] = float(r["gmac"])
    raw = defaultdict(list)
    for line in open(f"{RES}/" + ("trackN_rescore_cp4.jsonl" if CP else "trackN_rescore.jsonl")):
        r = json.loads(line)
        if not r["name"].startswith(cfg["prefix"]):
            continue
        n = r["name"][len(cfg["prefix"]):]
        if r.get("n"):                                # smoke-test subsets
            continue
        if r["q"] == 0 and r["stride"] == 1 and n.endswith("_cont") != (FIXED == "cont"):
            continue                                  # fixed=cont: the +8-epoch fixed models (training-matched to MiSTR,
                                                      # appendix); default: the 20-epoch ones (main Fig 05)
        if r.get("seed", 0) != 0 and not (CP and r["q"] == 0):   # S32 seeds 1/2: default every plotted point is
            continue                                  # seed 0; cp=1: fixed points = mean over their seeds (S32: 3)
        if r["stride"] > 1:
            key = ("strided", r["lr_ps"], r["stride"], 0.0)
        elif r["q"] == 0:
            key = ("fixed", r["lr_ps"], 0, 0.0)
        elif n.endswith(MISTR) and r.get("trunc") and r["q"] <= Q_TRAIN_MAX:   # training budgets only
            key = ("range", r["lr_ps"], r["crop"], round(r["q"], 4))
        else:
            continue                                 # dedicated-budget MiSTR / un-cut MiSTR rows: not drawn
        raw[key].append(r)
    out = {}
    for key, rs in raw.items():
        kind, L, C, q = key
        d = {}
        for k in KS:
            if not all(str(k) in r["test"]["per_k"] for r in rs):
                continue
            e = {m: float(np.mean([_score(r["test"]["per_k"][str(k)], m) for r in rs])) for m, _ in METRICS}
            if kind == "strided":
                g = [pfix.get((L, (k - r["lag"]) // C + 1)) for r in rs]
                e["gmac"] = float(np.mean(g)) if None not in g else None
            else:
                e["gmac"] = pref.get((L, C, q, k + 1))
            if e["gmac"] is not None:
                d[k] = e
        if d:
            out[key] = d
    return out


def series(acc, pick):
    """pick(d) -> {miou, fbiou, gmac} or None -> {"fixed": [(g, e, L)], ("strided", st): [...], (kind, L, C): [...]}"""
    s = defaultdict(list)
    for (kind, L, C, q), d in acc.items():
        e = pick(d)
        if e is None:
            continue
        key = "fixed" if kind == "fixed" else ("strided", C) if kind == "strided" else (kind, L, C)
        s[key].append((e["gmac"], e, L if kind in ("fixed", "strided") else q))
    return {k: sorted(v, key=lambda p: p[0]) for k, v in s.items()}


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, color="#e6e5e1", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def panel(ax, ser, met, labels=True):
    """One score-vs-prefix-GMAC panel: fixed-res families as lines, MiSTR as unjoined stars."""
    for st, (ls, mk, filled, _) in FAM.items():
        pts = ser.get("fixed" if st == 1 else ("strided", st), [])
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1][met] for p in pts]
        ax.plot(x, y, color=INK, ls=ls, lw=1.0, marker=mk, ms=4, mfc=INK if filled else "white", mew=0.8,
                alpha=0.75, zorder=3)
        if labels:
            for g, yy, (_, _, L) in zip(x, y, pts):
                ax.annotate(f"$S_{{{L}}}$" + (f"/{st}" if st > 1 else ""), (g, yy),   # every pass: above-left
                            xytext=(5, -3) if st > 1 or L == 128 else (-4, 4),
                            ha="right" if st == 1 and L != 128 else "left",
                            textcoords="offset points", fontsize=7, color=INK if st == 1 else MUTED)
    for L, C in COMBOS:
        pts = ser.get(("range", L, C), [])
        for inside in (True, False):                 # q within the training range (4-32 %): filled; above: hollow
            pp = [p for p in pts if (p[2] <= Q_TRAIN_MAX) == inside]
            if pp:
                ax.scatter([p[0] for p in pp], [p[1][met] for p in pp], marker="*", s=55,
                           facecolor=COLOR[(L, C)] if inside else "white", edgecolor=COLOR[(L, C)],
                           lw=0 if inside else 0.8, zorder=5)


def gains(ax, ser, met, fs=7):
    """Fixed S_L -> the CHEAPEST measured budget of the MiSTR pair that scores >= the fixed model (performance
    match): one horizontal arrow at the fixed model's score to that budget's cost, labelled x times less compute.
    No arrow when no training budget reaches the fixed score (printed)."""
    fx = {p[2]: p for p in ser.get("fixed", [])}
    for L, pair in GAINS:
        mx = sorted(ser.get(("range", *pair), []), key=lambda p: p[0])
        if L not in fx or not mx:
            continue
        g0, y0 = fx[L][0], fx[L][1][met]
        hit = next((p for p in mx if p[1][met] >= y0), None)
        if hit is None:
            print(f"  {met}: S{L} {y0:.3f} not matched by {pair} (best {max(p[1][met] for p in mx):.3f})")
            continue
        g1 = hit[0]
        col = COLOR[pair]
        ax.annotate("", (g1, y0), (g0, y0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.annotate(f"{g0 / g1:.1f}\u00d7 less", ((g0 * g1) ** 0.5, y0), xytext=(0, -2), textcoords="offset points",
                    color=col, fontsize=fs, ha="center", va="top", zorder=7,
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
        print(f"  {met}: S{L} {y0:.3f} @ {g0:.1f} = {pair} q {hit[2]:g} {hit[1][met]:.3f} @ {g1:.1f}: "
              f"{g0 / g1:.1f}x less")


def legend(fig, axes, ncol=2):
    """As the main datasets figure: "MiSTR (ours)" (one colour per resolution pair) and "fixed-res baselines"
    side by side, centred under the panels."""
    from matplotlib.transforms import Bbox
    star = dict(marker="*", ls="none", ms=8, mew=0)
    h = [Line2D([], [], color=COLOR[k], **star) for k in COMBOS]
    lab = [f"$S_{{{L}}}$ & $S_{{{C}}}$" for L, C in COMBOS]
    leg = fig.legend(h, lab, loc="lower left", borderaxespad=0, ncol=ncol, frameon=False, fontsize=8,
                     title=r"$\bigstar$ " + f"MiSTR (ours), {ROUNDS[ROUND]['q_text']}", title_fontproperties={"weight": "bold", "size": 8.5},
                     alignment="left", handlelength=1.2, columnspacing=1.2)
    fig.add_artist(leg)
    h = [Line2D([], [], color=INK, marker=mk, ms=4, ls=ls, lw=1.0, mfc=INK if f else "white", alpha=0.75)
         for ls, mk, f, _ in FAM.values()]
    leg2 = fig.legend(h, [v[3] for v in FAM.values()], loc="lower left", borderaxespad=0, frameon=False,
                      fontsize=8, title="fixed-res baseline", title_fontproperties={"weight": "bold", "size": 8.5},
                      alignment="left", handlelength=2.6)
    r = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    span = to_fig.transform_bbox(Bbox.union([ax.get_tightbbox(r) for ax in np.ravel(axes)]))
    w1, w2 = (to_fig.transform_bbox(lg.get_window_extent(r)).width for lg in (leg, leg2))
    left = (span.x0 + span.x1 - w1 - 0.05 - w2) / 2
    leg.set_bbox_to_anchor((left, 0.0), transform=fig.transFigure)
    leg2.set_bbox_to_anchor((left + w1 + 0.05, 0.0), transform=fig.transFigure)


def tradeoff():
    def pick(d):
        if not all(k in d for k in KS_MAIN):
            return None
        return {m: float(np.mean([d[k][m] for k in KS_MAIN])) for m in ("miou", "fbiou", "gmac")}
    ser = series(load(), pick)
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.3))
    for ax, (met, lab) in zip(axes, METRICS):
        style(ax)
        panel(ax, ser, met)
        gains(ax, ser, met)
        ax.set_title(f"{lab} " + r"$\uparrow$", fontsize=9, loc="left")
        ax.set_xlabel(X_LABEL, fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.36, top=0.92, wspace=0.22)
    legend(fig, axes)
    path = "01_tracking_mIoU-floodbIoU_vs_GMAC.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / path, bbox_inches="tight", pad_inches=0.02)
    print(OUT / path)
    for key, pts in sorted(ser.items(), key=str):
        print(key, [(round(g, 1), round(e["miou"], 3), round(e["fbiou"], 3)) for g, e, _ in pts])


def tradeoff_col():
    """One-column version of tradeoff (paper 05_GEOID: a single-column figure): the two panels stacked, one shared x
    axis, legend on the right (MiSTR pairs top, fixed baseline bottom). -> figures/paper/05_GEOID.pdf"""
    def pick(d):
        if not all(k in d for k in KS_MAIN):
            return None
        return {m: float(np.mean([d[k][m] for k in KS_MAIN])) for m in ("miou", "fbiou", "gmac")}
    ser = series(load(), pick)
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.0), sharex=True)
    for ax, (met, lab) in zip(axes, METRICS):
        style(ax)
        panel(ax, ser, met)
        gains(ax, ser, met, fs=6)
        ax.text(0.02, 0.98, f"{lab} " + r"$\uparrow$", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    axes[-1].set_xlabel("GMACs / flood series (log scale)", fontsize=7)
    star = dict(marker="*", ls="none", ms=7, mew=0)
    h = [Line2D([], [], color=COLOR[k], **star) for k in COMBOS]
    lab = [f"$S_{{{L}}}$ & $S_{{{C}}}$" for L, C in COMBOS]
    fig.legend(h, lab, loc="upper left", ncol=1, frameon=False, fontsize=7, handlelength=1.0,
                     handletextpad=0.4, title="\u2605 MiSTR",
                     title_fontproperties={"weight": "bold", "size": 7}, alignment="left",
                     bbox_to_anchor=(0.75, 0.97))
    fig.legend([Line2D([], [], color=INK, marker="D", ms=3.5, lw=1.0, alpha=0.75)], ["fixed res."],
               loc="upper left", frameon=False, fontsize=7, handlelength=1.6, handletextpad=0.4,
               title="baseline", title_fontproperties={"weight": "bold", "size": 7}, alignment="left",
               bbox_to_anchor=(0.75, 0.50))
    fig.subplots_adjust(left=0.10, right=0.74, top=0.99, bottom=0.14, hspace=0.10)
    path = Path("figures/paper/05_GEOID.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print(path)


def history_days():
    """Median real days of history before the event, per event position (test series; cp=1: the long3
    cache)."""
    from exp.geoidchange.tileseq import Series
    out = {}
    for k in KS:
        recs = json.load(open("data/GEOID-Flood-tileseries-long" + ("3" if CP else "2") + "/cache/test.json"))
        if CP:                                   # the common series: >= 31 passes of history
            out[k] = float(np.median([-Series.thin(r, 0)[2][Series.thin(r, 0)[1] - k] for r in recs
                                      if Series.thin(r, 0)[1] >= 31]))
            continue
        d = []
        for r in recs:
            idx, pp, rday = Series.thin(r, 0)
            if pp - k >= 0 and len(idx) - pp - 1 >= 31 - k:
                d.append(-rday[pp - k])
        out[k] = float(np.median(d))
    return out


def per_position4():
    """8 positions (cp=1 ks=4,...,31): 4 rows x 4 columns -- rows 1-2 mIoU (t = KS[:4], then KS[4:]), rows 3-4 flood
    bIoU, so every panel stays readable at text width."""
    acc = load()
    hist = history_days()
    fig, axes = plt.subplots(4, 4, figsize=(6.875, 6.6))
    for half in (0, 1):
        for i, k in enumerate(KS[4 * half:4 * half + 4]):
            ser = series(acc, lambda d, k=k: d.get(k))
            for j, (met, lab) in enumerate(METRICS):
                ax = axes[2 * j + half, i]
                style(ax)
                panel(ax, ser, met)
                gains(ax, ser, met, fs=5.5)
                ax.tick_params(labelsize=7)
                ax.xaxis.set_minor_formatter(plt.NullFormatter())
                ax.set_title(f"$t={k}$" + (" (last)" if k == 31 else "") + f", {hist[k]:.0f} d history",
                             fontsize=7.5, pad=3)
                if i == 0:
                    ax.set_ylabel(f"{lab} " + r"$\uparrow$", fontsize=8)
    for j in (0, 1):                       # shared y range per metric (both of its rows)
        rows = axes[2 * j:2 * j + 2].ravel()
        lo, hi = min(a.get_ylim()[0] for a in rows), max(a.get_ylim()[1] for a in rows)
        for a in rows:
            a.set_ylim(lo, hi)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.6, w_pad=0.3)
    fig.supxlabel(X_LABEL, fontsize=8.5, y=0.065, va="bottom")
    legend(fig, axes, ncol=4)
    path = "02_tracking_vs_GMAC_per_event_position_cp4" + ("_matched" if FIXED == "cont" else "") + ".pdf"
    target = Path(OUT_FILE) if OUT_FILE else OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
    print(target, hist)


def per_position():
    """Columns = event position (history length), rows = metric: wide and short, to sit at text width."""
    if len(KS) == 8:
        return per_position4()
    acc = load()
    hist = history_days()
    fig, axes = plt.subplots(2, len(KS), figsize=(6.875, 3.8), sharex="col", sharey="row")
    for i, k in enumerate(KS):
        ser = series(acc, lambda d, k=k: d.get(k))
        for j, (met, lab) in enumerate(METRICS):
            ax = axes[j, i]
            style(ax)
            panel(ax, ser, met)
            gains(ax, ser, met, fs=5.5)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            if j == 0:
                if len(KS) > 5:      # 7 columns: compact titles (0-based t; t = 31 is the last of 32)
                    ax.set_title(f"$t={k}$" + (" (last)" if k == 31 else "") + f"\n{hist[k]:.0f} d history",
                                 fontsize=7.5)
                else:
                    ax.set_title(f"event at timestep {k}\n({hist[k]:.0f} d history)", fontsize=8)
            if i == 0:
                ax.set_ylabel(f"{lab} " + r"$\uparrow$", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1), h_pad=0.4, w_pad=0.3)
    fig.supxlabel(X_LABEL, fontsize=8.5, y=0.12, va="bottom")
    legend(fig, axes, ncol=4)
    path = "02_tracking_vs_GMAC_per_event_position" + ("_cp" if CP else "") + ("_matched" if FIXED == "cont" else "") + ("_or" if MISTR == "_qr_or" else "") + ".pdf"
    target = Path(OUT_FILE) if OUT_FILE else OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
    print(target, hist)


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    CP = kw.get("cp") == "1"
    if CP:                                   # Fig 12: cp=1 ks=4,8,12,16,20,24,28,31
        KS = tuple(int(x) for x in kw.get("ks", "4,8,12,16,20,24,28,31").split(","))
        KS_MAIN = KS
    ROUND = kw.get("round", ROUND)
    FIXED = kw.get("fixed", FIXED)
    OUT_FILE = kw.get("out")                 # per_position: the paper's Fig 12 is out=figures/paper/12_GEOID-per-position.pdf
    MISTR = "_qr_or" if kw.get("mistr") == "or" else MISTR
    Q_TRAIN_MAX = ROUNDS[ROUND]["q_max"]
    {"tradeoff": tradeoff, "tradeoff_col": tradeoff_col, "per_position": per_position}[sys.argv[1]]()
