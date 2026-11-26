"""Track the flood over a same-orbit S1 series with sparse high-res crops (GEOID 512 px quadrants).

Differences from exp.geoidchange.tileseq (find-the-event):
- per-pass PSEUDO-LABELS (track_targets): the CEMS mask exists only at the post scene. A CEMS flood pixel that
  darkened at post (5x5-smoothed VV change vs the mean of passes 0..2, minus that pass's median change over
  labelled non-flood px, <= on dB) is "trackable": at another pass
  t it is 1 if its change <= on dB, 0 if >= off dB, ignored in between; non-trackable CEMS flood is ignored off
  post; labelled non-flood is 0 at every pass; unlabelled ignored. Post = the CEMS label.
- CAUSAL coarse pass: lr token at pass t attends to passes <= t only (the full-attention pass let post-1 copy
  the post flood). Crops at pass t read the cached lr tokens of passes <= t and the crops of passes <= t.
- JOINT space-time selector: slot (t, window) score = max over the two pass pairs touching t of the window-mean
  entropy of the pair's average prediction, H((p_{t-1} + p_t) / 2) (= mean entropy + JS divergence: unsure OR
  changed); pass 0 counts as "no flood". One top-k over all (pass, window) slots; budget q in units of one pass's
  windows (q .25 = as many crops as tileseq q .25 spends at its single pass). Every crop pass gets a refined map.

    python -m exp.geoidchange.tiletrack labels [split=test]    -> pseudo-label stats + overlays
    python -m exp.geoidchange.tiletrack train key=value ...    -> results/geoidchange/tiletrack.jsonl
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from torch import nn
from torch.utils.data import DataLoader

from exp.geoidchange.harness import OUT, IoU
from exp.geoidchange.tileseq import BASE, DB_FLOOR, NODATA, S, EventSeq, Series, collate as seq_collate
from mistr.encoders.olmoearth import _spatial_size, crop_patches

ON, OFF, REF_N = -3.0, -1.0, 3


def track_targets(x, y, k, valid, on=ON, off=OFF, smooth=5):
    """x (T, 2, S, S) dB, y (S, S) CEMS label {1 flood, 0, -1}, k post position -> (T - 1, S, S) int64
    targets for passes 1..T-1, plus the trackable mask."""
    T = len(x)
    ok = [s for s in range(min(REF_N, T)) if valid[s]] or [0]
    vv = np.stack([uniform_filter(x[t, 0], smooth) for t in range(T)])
    d = vv - vv[ok].mean(0)                                                  # (T, S, S) change vs early
    fl = y == 1
    bg = y == 0
    if bg.sum() > 1000:              # remove scene-wide shifts (wet soil, frost, ...): median change of non-flood
        d = d - np.median(d[:, bg][:, ::7], 1)[:, None, None]
    track = fl & (d[k] <= on)
    yt = np.where(y >= 0, 0, -1)[None].repeat(T - 1, 0)
    for t in range(1, T):
        if t == k:
            yt[t - 1] = y
            continue
        lab = np.full(y.shape, -1, np.int64)
        lab[track & (d[t] <= on)] = 1
        lab[track & (d[t] >= off)] = 0
        yt[t - 1][fl] = lab[fl]
        if not valid[t]:
            yt[t - 1] = -1
    return yt.astype(np.int64), track


class Track(Series):
    def __init__(self, *a, on=ON, off=OFF, three=False, **kw):
        self.y3 = self.three = three              # 3 classes: 0 dry, 1 permanent water, 2 flood
        super().__init__(*a, **kw)
        self.on, self.off = on, off

    cpu_labels = False          # True: pseudo-labels in the worker (numpy); False: on the GPU (track_targets_gpu)

    def __getitem__(self, i):
        x, days, valid, k, y, rday = super().__getitem__(i)
        if self.cpu_labels:
            yt, _ = track_targets(x, y, k, valid, self.on, self.off)
            return x, days, valid, k, y, rday, yt
        return x, days, valid, k, y, rday, None


MONTHLY = os.environ.get("GEOIDTSER_MONTHLY", "data/GEOID-Flood-tileseries-monthly/cache")


class MonthlyTrack(Track):
    """~Monthly same-orbit series (tileseries_fetch --monthly), 3 classes. Labels come only from the dataset:
    the event date = the GEOID-Flood label; every other date = its permanent water (class 1), everything else
    dry (0), unlabelled -1 (floods rarely last a month); on passes within near_days (40) of the event the event's
    flood px are ignored instead of dry (they may still be wet, e.g. +24 d). Training (augment): temporal CutMix -- with prob
    mix_p, n ~ U{1..3} other dates each get a random box (CutMix: side = S * sqrt(1 - lam), lam ~ U(0, 1),
    uniform centre, clipped) of the event date's pixels (VV + VH) AND its label, so floods appear on several
    dates with exact labels."""
    mix_p = 0.5
    near_days = 40                       # non-event passes within this many days: event flood px -> ignore
    mix_ref = True                       # False: never paste into the reference passes (t < REF_N)
    mix_flood = False                    # True: centre each box on a labelled-flood px of the event (none: no mix)
    mix_frac = None                      # (lo, hi): paste into round(U(lo, hi) x eligible passes) passes (>= 1);
                                         # None: 1-3 passes (the monthly T = 8 rule)
    keep_raw = False                     # viz: keep the unaugmented series in self.x_raw
    k_min = 1                            # labelled date at positions >= k_min (3: a full reference, = RefTrack)
    ref_n = REF_N                        # mix_ref=0: no pastes into passes t < max(ref_n, 1)

    def positions(self, r, T):
        return [p for p in Series.positions(r, T) if p >= self.k_min]

    def __init__(self, split, T=8, augment=False, n=0, **kw):
        kw.pop("three", None)
        super().__init__(split, T, augment=augment, n=n, three=True, root=MONTHLY, **kw)

    def __getitem__(self, i):
        x, days, valid, k, y, rday = Series.__getitem__(self, i)
        T = len(x)
        base = np.where(y == 1, 1, np.where(y >= 0, 0, -1))
        yt = np.repeat(base[None], T - 1, 0)
        yt[k - 1] = y
        for t in range(1, T):            # passes near the event: its flood px may still be wet -> ignore, not dry
            if t != k and abs(rday[t]) <= self.near_days:
                yt[t - 1][y == 2] = -1
        self.mixed = []                  # (t, y0, y1, x0, x1) of the CutMix boxes (viz)
        if self.keep_raw:
            self.x_raw = x.copy()
        fy, fx = np.nonzero(y == 2) if self.mix_flood else (None, None)
        if self.augment and np.random.rand() < self.mix_p and (not self.mix_flood or len(fy)):
            t0 = 0 if self.mix_ref else max(self.ref_n, 1)
            others = [t for t in range(t0, T) if t != k and valid[t]]
            n_mix = max(1, round(np.random.uniform(*self.mix_frac) * len(others))) if self.mix_frac else \
                np.random.randint(1, 4)
            for t in np.random.choice(others, min(len(others), n_mix), replace=False):
                side = int(S * np.sqrt(1 - np.random.rand()))
                if self.mix_flood:
                    j = np.random.randint(len(fy))
                    cy, cx = int(fy[j]), int(fx[j])
                else:
                    cy, cx = np.random.randint(S), np.random.randint(S)
                y0, y1 = max(cy - side // 2, 0), min(cy + side // 2, S)
                x0, x1 = max(cx - side // 2, 0), min(cx + side // 2, S)
                x[t, :, y0:y1, x0:x1] = x[k, :, y0:y1, x0:x1]
                self.mixed.append((int(t), y0, y1, x0, x1))
                if t >= 1:
                    yt[t - 1, y0:y1, x0:x1] = y[y0:y1, x0:x1]
        for t in range(1, T):
            if not valid[t]:
                yt[t - 1] = -1
        return x, days, valid, k, y, rday, yt.astype(np.int8 if FAST_PREP else np.int64)


LONG = os.environ.get("GEOIDTSER_LONG", "data/GEOID-Flood-tileseries-long/cache")


class LongTrack(MonthlyTrack):
    """Long DENSE same-orbit series (tileseries_fetch --before 24 --after 7: up to 24 passes before the event scene,
    7 after), 3 classes, for train, val AND test. Labels as MonthlyTrack: the event date = the GEOID-Flood label;
    other dates = its permanent water, else dry; the event's flood px ignored on dates within near_days (30) of the
    event. Window rule: the first pass is >= min_hist (30) days before the event; the event at any later position
    (train: random; eval: seeded per record). Temporal CutMix as MonthlyTrack (flags set by train)."""
    near_days = 30
    min_hist = 30.0
    root = LONG                          # data=long2: 24 passes before AND after (event anywhere in T = 32)
    stride = 1                           # >1: of the T-pass window keep every stride-th pass, anchored on the event
                                         # (same span, T / stride passes; the event at position k // stride)
    lag = 0                              # eval only (exp.geoidchange.stride_lag_eval): un-anchored stride phase
    gap_days = 0.0                       # TRAIN only: drop passes within +-gap_days of the event (flood may already /
                                         # still show; the model sees pass indices, not dates, so the cut is invisible)

    def __getitem__(self, i):
        x, days, valid, k, y, rday, yt = super().__getitem__(i)
        st = self.stride
        if st == 1:
            return x, days, valid, k, y, rday, yt
        kk = k - self.lag            # lag 0: the event pass is kept; lag j: the last kept pass is j passes before it
        sel = np.arange(kk % st, len(x), st)                                 # (scored against the event's label)
        return (np.ascontiguousarray(x[sel]), BASE + 12 * np.arange(len(sel), dtype=np.int64), valid[sel], kk // st,
                y, rday[sel], np.ascontiguousarray(yt[sel[1:] - 1]))

    def positions(self, r, T):
        return [p for p in Series.positions(r, T) if r["rday"][r["pp"] - p] <= -self.min_hist]

    def __init__(self, split, T=8, augment=False, n=0, ks=None, **kw):
        """ks: evaluate every series with its event at each of these positions (those feasible: >= min_hist days of
        history, a full window of real passes); the dataset becomes (series, k) pairs, r["k_fix"] = k."""
        kw.pop("three", None)
        Track.__init__(self, split, T, augment=augment, n=n, three=True, root=self.root, **kw)
        if self.gap_days and split == "train":
            n0 = len(self.recs)
            for r in self.recs:
                keep = [j for j, d in enumerate(r["rday"]) if j == r["pp"] or abs(d) >= self.gap_days]
                r["idx"], r["rday"] = [r["idx"][j] for j in keep], [r["rday"][j] for j in keep]
                r["pp"] = keep.index(r["pp"])
            self.recs = [r for r in self.recs if len(r["idx"]) >= T and self.positions(r, T)]
            print(f"LongTrack {split}: passes within {self.gap_days:g} d of the event dropped, "
                  f"{n0} -> {len(self.recs)} series", flush=True)
        if ks:
            n0 = len(self.recs)
            self.recs = [{**r, "k_fix": k} for r in self.recs for k in ks if k in self.positions(r, T)]
            self.ks = list(ks)
            print(f"LongTrack {split} T={T}: {n0} series x positions {list(ks)} -> {len(self.recs)} samples",
                  flush=True)


class RefTrack(Track):
    """Evaluation series with a MONTHLY reference: timesteps 0-2 = the last three monthly same-orbit passes at or
    before REF_DAY (~ -90 / -60 / -30 d; MonthlyTrack's series of the same quadrant), then 5 DENSE passes after the
    last of them (~6 d apart), the labelled date at position k in 3..7 (seeded per record, as Series' eval k; or
    k_post). The early dense passes (-42..-30 d), which can already show the flood, are never the reference.
    Quadrants without 3 such monthly passes or 5 dense passes around the event are dropped."""
    REF_DAY = -27.0                      # monthly passes at or before this many days count as reference

    def __init__(self, split="test", T=8, **kw):
        assert T == 8, "3 monthly reference passes + 5 dense"
        super().__init__(split, T, three=True, **kw)
        mon = Series(split, T, root=MONTHLY)
        twin = {(r["key"], r["quad"]): r for r in mon.recs}
        self.XM = mon.X
        keep = []
        for r in self.recs:
            m = twin.get((r["key"], r["quad"]))
            if m is None:
                continue
            ref = [j for j, d in enumerate(m["rday"]) if d <= self.REF_DAY][-REF_N:]
            if len(ref) < REF_N:
                continue
            last = m["rday"][ref[-1]]
            dense = [j for j, d in enumerate(r["rday"]) if d > last]
            e = dense.index(r["pp"])
            ks = [k for k in range(REF_N, T) if e >= k - REF_N and len(dense) - 1 - e >= T - 1 - k]
            if not ks:
                continue
            r["ref_m"], r["dense"], r["ks"] = (m["off"], [m["idx"][j] for j in ref], [m["rday"][j] for j in ref]), \
                [dense[i] for i in range(len(dense))], ks
            r["e_dense"] = e
            keep.append(r)
        self.dropped += len(self.recs) - len(keep)
        self.recs = keep

    def __getitem__(self, i):
        r = self.recs[i]
        T = self.T
        ks = r["ks"]
        k = min(ks, key=lambda p: abs(p - self.k_post)) if getattr(self, "k_post", None) else \
            ks[np.random.default_rng(r["rec"]).integers(len(ks))]
        moff, midx, mday = r["ref_m"]
        s0 = r["e_dense"] - (k - REF_N)
        dj = r["dense"][s0:s0 + T - REF_N]
        a = np.concatenate([np.asarray(self.XM[moff + np.asarray(midx)]),
                            np.asarray(self.X[r["off"] + np.asarray([r["idx"][j] for j in dj])])])
        rday = np.asarray(list(mday) + [r["rday"][j] for j in dj], dtype=np.float32)
        x = a.astype(np.float32) / 100                       # as Series.__getitem__
        ok = a != NODATA
        one = ok.any(1, keepdims=True) & ~ok
        x[one] = DB_FLOOR
        ok = ok | one
        valid = ok.mean((1, 2, 3)) > 0.05
        for t in range(T):
            for b in range(2):
                bad = ~ok[t, b]
                if bad.any():
                    x[t, b][bad] = np.median(x[t, b][~bad]) if (~bad).any() else DB_FLOOR
        yl = np.asarray(self.Y[r["rec"]])
        y = yl.astype(np.int64)
        y[yl == 255] = -1
        days = BASE + 12 * np.arange(T, dtype=np.int64)
        return x, days, valid, k, y, rday, None


FAST_PREP = False        # True: workers ship raw dB (FAST_DTYPE) + int8 labels; normalize + mask on the GPU (prep_gpu)
FAST_DTYPE = np.float32


def collate(items):
    """-> (sample, days, valid, post, y, mean VV, real days, LAST): LAST = pseudo-labels (B, T-1, S, S) if the
    dataset made them, else the raw passes (B, T, 2, S, S) for track_targets_gpu.
    FAST_PREP: sample = MaskedOlmoEarthSample(sentinel1 = RAW dB (B, T, 2, S, S), sentinel1_mask = None,
    timestamps); EventTrack.forward normalizes it on the GPU (identical to harness.to_sample)."""
    last = [it[6] for it in items] if items[0][6] is not None else [it[0] for it in items]
    if not FAST_PREP:
        return (*seq_collate([it[:6] for it in items]), torch.from_numpy(np.stack(last)))
    from exp.geoidchange.harness import _timestamps
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    x = np.stack([it[0] for it in items]).astype(FAST_DTYPE, copy=False)
    d = np.stack([it[1] for it in items])
    v = np.stack([it[2] for it in items])
    s = MaskedOlmoEarthSample(sentinel1=torch.from_numpy(x), sentinel1_mask=None,
                              timestamps=_timestamps(torch.from_numpy(d)))
    return (s, torch.from_numpy(d), torch.from_numpy(v), torch.tensor([it[3] for it in items]),
            torch.from_numpy(np.stack([it[4] for it in items])), torch.from_numpy(x[:, :, 0].mean((2, 3))),
            torch.from_numpy(np.stack([it[5] for it in items])), torch.from_numpy(np.stack(last)))


_AFF = None


def prep_gpu(s, valid):
    """FAST_PREP sample (sentinel1 = raw dB (B, T, 2, S, S) on the GPU) -> the normalized sample + mask, as
    harness.to_sample: (B, H, W, T, C) bands BAND, per-band affine of the OlmoEarth normalizer, mask MISSING on
    invalid passes."""
    global _AFF
    from exp.geoidchange.harness import BAND, INP
    from olmoearth_pretrain.datatypes import MaskValue
    if _AFF is None:                                 # the normalizer is affine per band: recover (scale, offset)
        C = len(BAND) if hasattr(BAND, "__len__") else 1
        z = INP.normalizer.normalize(INP.olmo["S1"], np.zeros((1, C)))
        o = INP.normalizer.normalize(INP.olmo["S1"], np.ones((1, C)))
        _AFF = (torch.tensor(o - z, dtype=torch.float64).reshape(-1), torch.tensor(z, dtype=torch.float64).reshape(-1))
    a, b = (t.to(s.sentinel1.device) for t in _AFF)
    arr = s.sentinel1[:, :, BAND].permute(0, 3, 4, 1, 2)                     # (B, H, W, T, C)
    s1 = (arr.double() * a + b).float()
    mask = torch.full_like(s1, MaskValue.ONLINE_ENCODER.value)
    bi, ti = torch.nonzero(~valid, as_tuple=True)
    mask[bi, :, :, ti] = MaskValue.MISSING.value
    return s._replace(sentinel1=s1, sentinel1_mask=mask)


def _box5(v):
    """scipy.ndimage.uniform_filter(v, 5) (mode reflect = symmetric) on (N, S, S)."""
    v = torch.cat([v[:, :2].flip(1), v, v[:, -2:].flip(1)], 1)
    v = torch.cat([v[:, :, :2].flip(2), v, v[:, :, -2:].flip(2)], 2)
    return F.avg_pool2d(v[:, None], 5, stride=1)[:, 0]


@torch.no_grad()
def track_targets_gpu(x, y, k, valid, on=ON, off=OFF, three=False):
    """track_targets for a batch on the GPU: x (B, T, 2, S, S) dB, y (B, S, S), k (B,), valid (B, T)
    -> (B, T-1, S, S) int64. three: y in {0 dry, 1 permanent water, 2 flood, -1}; permanent water is 1 at every
    pass, CEMS flood px follow the same rule with flood = 2 / dry = 0."""
    fc = 2 if three else 1
    B, T = x.shape[:2]
    vv = _box5(x[:, :, 0].flatten(0, 1).double()).view(B, T, *x.shape[-2:])
    out = []
    for b in range(B):
        ok = [s for s in range(min(REF_N, T)) if valid[b, s]] or [0]
        d = vv[b] - vv[b, ok].mean(0)
        fl, bg = y[b] == fc, y[b] == 0
        if bg.sum() > 1000:
            d = d - torch.quantile(d[:, bg][:, ::7], 0.5, dim=1)[:, None, None]
        kb = int(k[b])
        track = fl & (d[kb] <= on)
        base = torch.where(y[b] >= 0, 0, -1)
        if three:
            base = torch.where(y[b] == 1, 1, base)                           # permanent water: every pass
        yt = base[None].repeat(T - 1, 1, 1)
        lab = torch.full_like(yt, -1)
        dd = d[1:]
        lab[track[None] & (dd <= on)] = fc
        lab[track[None] & (dd >= off)] = 0
        yt = torch.where(fl[None], lab, yt)
        yt[kb - 1] = y[b]
        yt[~valid[b, 1:]] = -1
        out.append(yt)
    return torch.stack(out).long()


def batch_targets(last, y, post, v, ds):
    """Pseudo-labels on the GPU from collate's last element (or the precomputed ones)."""
    if last.dim() == 4:
        return last.cuda(non_blocking=True).long()
    return track_targets_gpu(last.cuda(non_blocking=True), y.cuda(), post, v.cuda(), ds.on, ds.off,
                             getattr(ds, "three", False))


# -- pseudo-label check -----------------------------------------------------------------------------------
def labels(kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = Track(kw.get("split", "test"), 8)
    ds.k_post = 4
    ds.cpu_labels = True
    n_fl = n_tr = 0
    per = {o: np.zeros(3) for o in range(-3, 4)}                             # [1, 0, ignore] in CEMS flood
    early = 0
    rows = []
    for i in range(len(ds)):
        x, _, v, k, y, _, yt = ds[i]
        fl = y == 1
        if fl.sum() < 0.01 * (y >= 0).sum():
            continue
        _, track = track_targets(x, y, k, v)
        n_fl += fl.sum()
        n_tr += track.sum()
        for t in range(1, len(x)):
            o = t - k
            if -3 <= o <= 3:
                lab = yt[t - 1][fl]
                per[o] += [(lab == 1).sum(), (lab == 0).sum(), (lab == -1).sum()]
        dd = np.stack([uniform_filter(x[t, 0], 5) for t in range(3)])
        dd = dd[2] - dd[:2].mean(0)
        early += (dd[fl].mean() - np.median(dd[y == 0])) <= ON              # flood already in pass 2
        rows.append(i)
    print(f"flooded crops {len(rows)}; CEMS flood px trackable (darkened <= {ON} dB at post): {n_tr / n_fl:.3f}; "
          f"crops whose flood had darkened by pass 2 (early ref contaminated): {early} ({early / len(rows):.3f})")
    for o, c in per.items():
        print(f"  offset {o:+d}: CEMS flood px -> 1 {c[0] / c.sum():.3f}  0 {c[1] / c.sum():.3f}  "
              f"ignore {c[2] / c.sum():.3f}")
    pick = [rows[int(j)] for j in np.linspace(0, len(rows) - 1, 6)]
    fig, ax = plt.subplots(2 * len(pick), 8, figsize=(8 * 1.4, 2 * len(pick) * 1.45))
    cm = matplotlib.colors.ListedColormap(["#d9d9d9", "#f4f4f2", "#2a78d6"])  # ignore, 0, 1
    for r, i in enumerate(pick):
        x, _, v, k, y, rd, yt = ds[i]
        ref = x[:3, 0].mean(0)
        for t in range(8):
            ax[2 * r, t].imshow(x[t, 0] - ref, cmap="RdBu", vmin=-8, vmax=8)
            ax[2 * r, t].set_title(f"{rd[t]:+.0f} d" + (" CEMS" if t == k else ""), fontsize=7,
                                   color="#c0392b" if t == k else "k")
            if t:
                ax[2 * r + 1, t].imshow(yt[t - 1] + 1, cmap=cm, vmin=0, vmax=2, interpolation="nearest")
        ax[2 * r, 0].set_ylabel(ds.recs[i]["key"], fontsize=6.5)
        ax[2 * r + 1, 0].set_ylabel("pseudo-label", fontsize=6.5)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Per-pass pseudo-labels: VV change vs passes 0-2 (red = darker) | blue = flood, white = 0, "
                 f"grey = ignore (trackable: <= {ON} dB at post; flood <= {ON}, dry >= {OFF})", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs("figures/geoidchange", exist_ok=True)
    fig.savefig("figures/geoidchange/tiletrack_labels.png", dpi=150)
    print("figures/geoidchange/tiletrack_labels.png")



# -- causal attention ------------------------------------------------------------------------------------
def token_times(lr):
    """(N_valid,) the pass index of every valid lr token (-1: a modality without time)."""
    out = []
    for m, shape in lr.dims.items():
        if len(shape) == 6:                                                  # (S, h, w, t, bandsets, D)
            h, w, T, b = shape[1:5]
            out.append((torch.arange(h * w * T * b, device=lr.x.device) // b) % T)
        else:
            out.append(torch.full((int(np.prod(shape[1:-1])),), -1, device=lr.x.device))
    return torch.cat(out).expand(len(lr.seqlens), -1)[lr.keep]


def _pad(x, seq, n_seq):
    """flat (N, ...) with sequence ids seq (N,) -> (S, M, ...), valid (S, M), slot of each (N,)."""
    order = torch.argsort(seq, stable=True)
    cnt = torch.bincount(seq, minlength=n_seq)
    start = torch.cumsum(cnt, 0) - cnt
    slot = torch.empty_like(seq)
    slot[order] = torch.arange(len(seq), device=seq.device) - start[seq[order]]
    M = int(cnt.max()) if len(seq) else 1
    out = x.new_zeros(n_seq, M, *x.shape[1:])
    out[seq, slot] = x
    ok = torch.zeros(n_seq, M, dtype=torch.bool, device=x.device)
    ok[seq, slot] = True
    return out, ok, slot


ATTN = "sdpa"           # chunk: the lr self-attention in query chunks of CHUNK tokens, each chunk's (S, C, N) mask
CHUNK = 2048            # slice built on the fly and recomputed in backward (checkpoint): never a dense (S, N, N)
                        # mask -> memory ~linear in tokens (fixed S16 at T = 32 = 32768 tokens); same math as sdpa


class LazyMask:
    """The causal_lr visibility rule, evaluated per query chunk: key k visible to query q iff ok[k] and (pass(k) <=
    pass(q) or either has no pass); padding query rows read key 0."""
    def __init__(self, ok, tp, causal):
        self.ok, self.tp, self.causal = ok, tp, causal

    def rows(self, i0, i1):
        ok, tp = self.ok, self.tp
        v = ok[:, None, :].expand(-1, i1 - i0, -1)
        if self.causal:
            tq = tp[:, i0:i1, None]
            v = v & ((tp[:, None, :] <= tq) | (tp[:, None, :] < 0) | (tq < 0))
        return v | (~ok[:, i0:i1, None] & (torch.arange(ok.shape[1], device=ok.device)[None, None] == 0))


class CrossMask:
    """causal_crops' visibility rule per query chunk (queries = crop tokens, keys = lr + crop tokens): key k visible to
    query q iff ok[k] and (pass(k) <= pass(q) or the key has no pass); padding query rows read key 0. Valid rows equal
    the dense mask exactly (padding rows differ, their outputs are discarded)."""
    def __init__(self, ok, tp, okq, tpq, causal):
        self.ok, self.tp, self.okq, self.tpq, self.causal = ok, tp, okq, tpq, causal

    def rows(self, i0, i1):
        return _cross_rows(self.ok, self.tp, self.okq, self.tpq, i0, i1, self.causal)


def _cross_rows(ok, tp, okq, tpq, i0, i1, causal):
    v = ok[:, None, :].expand(-1, i1 - i0, -1)
    if causal:
        v = v & ((tp[:, None, :] <= tpq[:, i0:i1, None]) | (tp[:, None, :] < 0))
    return v | (~okq[:, i0:i1, None] & (torch.arange(ok.shape[1], device=ok.device)[None, None] == 0))


def _chunk_cross(qc, k, v, ok, tp, okq, tpq, i0, causal, scale):
    m = _cross_rows(ok, tp, okq, tpq, i0, i0 + qc.shape[2], causal)
    return F.scaled_dot_product_attention(qc, k, v, attn_mask=m[:, None], scale=scale)


def _chunk_attn(qc, k, v, ok, tp, i0, causal, scale):
    m = LazyMask(ok, tp, causal).rows(i0, i0 + qc.shape[2])
    return F.scaled_dot_product_attention(qc, k, v, attn_mask=m[:, None], scale=scale)


def _block(blk, xq, xkv, mask):
    """One OlmoEarth block on queries xq (S, Q, D) reading keys / values xkv (S, K, D) under mask (S, Q, K)
    (or a FlexAttention BlockMask)."""
    a = blk.attn
    S, Q, _ = xq.shape
    heads = lambda v: v.view(v.shape[0], v.shape[1], a.num_heads, -1)        # noqa: E731
    nq = blk.norm1(xq)
    nk = nq if xkv is xq else blk.norm1(xkv)
    q, k, v = a.q_norm(heads(a.q(nq))), a.k_norm(heads(a.k(nk))), heads(a.v(nk))
    if isinstance(mask, torch.Tensor):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype),
                                           v.transpose(1, 2).to(q.dtype), attn_mask=mask[:, None], scale=a.scale)
    elif isinstance(mask, CrossMask):                                    # crops reading lr + crops: chunked
        from torch.utils.checkpoint import checkpoint
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype)
        o = torch.cat([checkpoint(_chunk_cross, qt[:, :, i:i + CHUNK], kt, vt, mask.ok, mask.tp, mask.okq, mask.tpq,
                                  i, mask.causal, a.scale, use_reentrant=False) for i in range(0, Q, CHUNK)], 2)
    else:                                                                # LazyMask: chunked + checkpointed
        from torch.utils.checkpoint import checkpoint
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype)
        o = torch.cat([checkpoint(_chunk_attn, qt[:, :, i:i + CHUNK], kt, vt, mask.ok, mask.tp, i, mask.causal,
                                  a.scale, use_reentrant=False) for i in range(0, Q, CHUNK)], 2)
    xq = xq + blk.drop_path(blk.ls1(a.proj_drop(a.proj(o.transpose(1, 2).reshape(S, Q, -1))))).to(xq.dtype)
    return xq + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(xq)))).to(xq.dtype)


def _seq_ids(seqlens):
    return torch.repeat_interleave(torch.arange(len(seqlens), device=seqlens.device), seqlens)


def causal_lr(mix, lr, causal=True):
    """Blocks [L, n) + norm on the lr sequences, pass t reading passes <= t (causal=False: all, = lr_trace).
    -> padded trace (list of (S, M, D)), normed flat output (N_valid, D), aux (seq, slot, ok, t_pad)."""
    seq = _seq_ids(lr.seqlens)
    t = token_times(lr)
    x, ok, slot = _pad(lr.x, seq, len(lr.seqlens))
    tp = torch.full(ok.shape, -1, dtype=t.dtype, device=t.device)
    tp[seq, slot] = t
    if ATTN == "chunk":                      # same visibility, never materialized as (S, N, N)
        lm = LazyMask(ok, tp, causal)
        trace = []
        for blk in mix.encoder.blocks[mix.L:]:
            trace.append(x)
            x = _block(blk, x, x, lm)
        return trace, mix.encoder.norm(x[seq, slot]), (ok, tp)
    vis = ok[:, None, :].expand(-1, ok.shape[1], -1)
    if causal:
        vis = vis & ((tp[:, None, :] <= tp[:, :, None]) | (tp[:, None, :] < 0) | (tp[:, :, None] < 0))
    vis = vis | (~ok[:, :, None] & (torch.arange(ok.shape[1], device=ok.device)[None, None] == 0))  # pad rows
    trace = []
    for blk in mix.encoder.blocks[mix.L:]:
        trace.append(x)
        x = _block(blk, x, x, vis)
    return trace, mix.encoder.norm(x[seq, slot]), (ok, tp)


def causal_crops(mix, trace, aux, hr, owner, t_crop, causal=True):
    """The crops' blocks [L, n) + norm: crop tokens of sample s at pass t read that sample's lr tokens and
    crop tokens of passes <= t (causal=False: all of them, = cross_hr). -> (N_crop_tokens, D)."""
    ok_lr, t_lr = aux
    S = ok_lr.shape[0]
    tok_s = torch.repeat_interleave(owner, hr.seqlens)
    tok_t = torch.repeat_interleave(t_crop, hr.seqlens)
    xc, ok_c, slot = _pad(mix._typed(hr, trace[0].dtype), tok_s, S)
    tc = torch.full(ok_c.shape, -1, dtype=tok_t.dtype, device=tok_t.device)
    tc[tok_s, slot] = tok_t
    if ATTN == "chunk":                       # never the dense (S, Q, K) mask: built per query chunk (CrossMask)
        vis = CrossMask(torch.cat([ok_lr, ok_c], 1), torch.cat([t_lr.to(tc.dtype), tc], 1), ok_c, tc, causal)
    else:
        see_lr = ok_lr[:, None, :].expand(-1, ok_c.shape[1], -1)
        see_c = ok_c[:, None, :].expand(-1, ok_c.shape[1], -1)
        if causal:
            see_lr = see_lr & ((t_lr[:, None, :] <= tc[:, :, None]) | (t_lr[:, None, :] < 0))
            see_c = see_c & (tc[:, None, :] <= tc[:, :, None])
        vis = torch.cat([see_lr, see_c], -1)
        vis[:, :, 0] |= ~ok_c                                                # padding rows read something
    for blk, x_lr in zip(mix.encoder.blocks[mix.L:], trace):
        xc = _block(blk, xc, torch.cat([x_lr, xc], 1), vis)
    return mix.encoder.norm(xc[tok_s, slot])


def bin_entropy(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def entropy(P, dim=2):
    """Shannon entropy over the class dim (2 classes: = bin_entropy of the flood prob)."""
    P = P.clamp_min(1e-6)
    return -(P * P.log()).sum(dim)


def budget_select(u, valid, target, T, WW, R=REF_N):
    """Slots (pass t >= 1, window) in descending score u (B, T-1, W, W), taken in order while the unique crops
    they need (the slot + its early reference crops at passes s < min(t, R)) stay <= target per sample.
    -> flat (sample, pass, window) of the taken slots."""
    B, dev = u.shape[0], u.device
    flat = u.flatten(1)
    N = flat.shape[1]
    order = flat.argsort(1, descending=True)
    ok = flat.gather(1, order) > float("-inf")
    t, w = order // WW + 1, order % WW
    rs = torch.arange(R, device=dev)
    rv = (rs[None, None] < t[..., None].clamp(max=R)) & valid[:, None, :R] & ok[..., None]
    keys = torch.cat([(t * WW + w)[..., None], rs[None, None] * WW + w[..., None]], -1)       # (B, N, 1 + R)
    kv = torch.cat([ok[..., None], rv], -1)
    item = torch.arange(N, device=dev)[None, :, None].expand_as(keys)
    bi = torch.arange(B, device=dev)[:, None, None].expand_as(keys)
    flat_k = (bi * T * WW + keys)[kv]
    first = torch.full((B * T * WW,), N, device=dev).scatter_reduce(0, flat_k, item[kv], "amin")
    seen = first < N
    kb = torch.div(torch.nonzero(seen)[:, 0], T * WW, rounding_mode="floor")
    new = torch.zeros(B, N, device=dev).index_put_((kb, first[seen]), torch.ones(len(kb), device=dev),
                                                   accumulate=True)
    n = ((new.cumsum(1) <= target) & ok).sum(1)                              # (B,) slots taken
    take = torch.arange(N, device=dev)[None] < n[:, None]
    sb = torch.arange(B, device=dev)[:, None].expand(B, N)[take]
    return sb, t[take], w[take]


# -- model -----------------------------------------------------------------------------------------------
ABS_T = 4                        # q_unit=abs: budget = q x ABS_T x W^2 crops per sequence, whatever T is


class EventTrack(EventSeq):
    causal = True
    ref_n = REF_N                # reference = mean of passes s < min(t, ref_n); 1 = the first pass; 0 = none

    def forward(self, s, days, valid, post, sel="joint", q=0.25, target=None):
        """sel oracle (training warm-up only): windows ranked by the coarse pass's cross-entropy against target
        (B, T-1, H, W; -1 ignored), as the label-oracle warm-up of the other datasets."""
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items() if v is not None})
        days, valid, post = days.cuda(), valid.cuda(), post.cuda()
        if s.sentinel1_mask is None:                                          # FAST_PREP batch: raw dB
            s = prep_gpu(s, valid)
        rel = (days - days[:, :1]).float() / 6
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.embed(s, self.lr_ps, times=rel)
            trace, enc_out, aux = causal_lr(self.mix, lr, self.causal)
        f = lr.pooled(enc_out).float()                                       # (B, T, h, w, D)
        B, T, h, w, D = f.shape
        P, dev = self.lr_ps, f.device
        v = valid.float()[:, :, None, None, None]
        R = self.ref_n
        if R:
            ref = torch.cumsum(f * v, 1)[:, :-1] / torch.cumsum(v, 1)[:, :-1].clamp(min=1)
            ref = ref[:, torch.arange(1, T, device=dev).clamp(max=R) - 1]     # mean of passes < min(t, R)
        else:                                                                 # head_in self: unused
            ref = f[:, 1:]
        m = self.tok2map(self.feat(f[:, 1:], ref).flatten(0, -2), P)
        C = self.n_cls
        m = m.view(B, T - 1, h, w, C, P, P).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T - 1, C, h * P, w * P)
        H = _spatial_size(s)
        lo = (h * P - H) // 2
        if h * P != H:
            m = m[..., lo:lo + H, lo:lo + H]
        info = dict(lr_tokens=int(valid.sum()) * h * w, crops=0, sel_b=None, sel_t=None,
                    slots=T * (H // self.crop) ** 2)
        if q <= 0:
            return m, m, info
        cp = self.crop
        W = H // cp
        WW = W * W
        with torch.no_grad():
            Pr = m.float().softmax(2)                                         # (B, T-1, C, H, W)
            p = Pr[:, :, C - 1]                                               # flood prob (last class)
            okp = valid[:, 1:]
            if sel in ("joint", "wet", "spatial", "post"):
                if sel in ("joint", "wet"):              # pair (t-1, t), pass 0 = pass 1 without its flood
                    p0 = Pr[:, :1].clone()
                    p0[:, :, 0] += p0[:, :, C - 1]
                    p0[:, :, C - 1] = 0
                    prev = torch.cat([p0, Pr[:, :-1]], 1)
                    pair = F.avg_pool2d(entropy((prev + Pr) / 2).flatten(0, 1)[:, None], cp)[:, 0]
                    pair = pair.view(B, T - 1, *pair.shape[-2:])              # (B, T-1, W, W): pair ending at t
                    pv = torch.cat([torch.ones_like(okp[:, :1]), okp[:, :-1]], 1) & okp
                    pair = pair.masked_fill(~pv[:, :, None, None], -1.0)
                    if sel == "joint":                   # both passes of the pair
                        nxt = torch.cat([pair[:, 1:], torch.full_like(pair[:, :1], -1.0)], 1)
                        u = torch.maximum(pair, nxt)                          # either pair touching t
                    else:                                # wet: only the pass of the pair predicted wetter
                        pw = F.avg_pool2d(p, cp)
                        pw_prev = torch.cat([torch.zeros_like(pw[:, :1]), pw[:, :-1]], 1)
                        own = pair.masked_fill(pw < pw_prev, -1.0)            # pair ending at t, t wetter
                        nx = pair[:, 1:].masked_fill(pw[:, :-1] <= pw[:, 1:], -1.0)   # pair ending t+1, t wetter
                        u = torch.maximum(own, torch.cat([nx, torch.full_like(pair[:, :1], -1.0)], 1))
                else:
                    u = F.avg_pool2d(entropy(Pr).flatten(0, 1)[:, None], cp)[:, 0].view(B, T - 1, W, W)
                if sel == "post":
                    u = u.masked_fill((torch.arange(1, T, device=dev)[None] != post[:, None])[:, :, None, None],
                                      float("-inf"))
            elif sel == "oracle":
                assert target is not None, "oracle selection needs the training target"
                ok = (target >= 0).float()
                ce = -torch.log(Pr.gather(2, target.clamp(min=0)[:, :, None]).squeeze(2).clamp(min=1e-6)) * ok
                u = (F.avg_pool2d(ce.flatten(0, 1)[:, None], cp)[:, 0]
                     / F.avg_pool2d(ok.flatten(0, 1)[:, None], cp)[:, 0].clamp(min=1e-6)).view(B, T - 1, W, W)
            else:
                u = torch.rand(B, T - 1, W, W, device=dev)
            u = u.masked_fill(~okp[:, :, None, None], float("-inf"))
            if getattr(self, "budget_scope", "sample") == "batch":   # ONE budget for the whole batch:
                assert R == 0, "batch budget: no reference crops (ref_n 0)"  # q x ABS_T x W^2 x B crops go to
                flat = u.flatten()                                           # the top slots of ANY sample, so
                Tq = (T - 1) if getattr(self, "q_unit", "abs") == "series" else ABS_T   # series: q of ALL slots
                K = min(round(q * Tq * WW * B), int((flat > float("-inf")).sum()))   # a flood-free series
                idx = flat.topk(K).indices                                   # can get (almost) none
                sb = torch.div(idx, (T - 1) * WW, rounding_mode="floor")
                st = torch.div(idx % ((T - 1) * WW), WW, rounding_mode="floor") + 1
                sw = idx % WW
            elif getattr(self, "q_unit", "pass") == "series":   # q of the (T - 1) x W^2 predicted-pass slots
                sb, st, sw = budget_select(u, valid, round(q * (T - 1) * WW), T, WW, R)
            elif getattr(self, "q_unit", "pass") in ("total", "abs"):   # total: q of all T x W^2 hr slots;
                Tq = ABS_T if self.q_unit == "abs" else T              # abs: q of ABS_T x W^2; refs included
                sb, st, sw = budget_select(u, valid, round(q * Tq * WW), T, WW, R)
            else:                                            # pass: q x one pass's windows, refs extra
                K = max(1, round(q * WW))
                if sel == "post":
                    K = min(K, WW)
                idx = u.flatten(1).topk(K, 1).indices                         # (B, K)
                sb = torch.arange(B, device=dev).repeat_interleave(K)
                st = idx.flatten() // WW + 1
                sw = idx.flatten() % WW
            # reference crops: passes s < min(t, R), valid
            rs = torch.arange(R, device=dev)
            rok = (rs[None] < st[:, None].clamp(max=R)) & valid[sb][:, :R]
            rb, rt, rw = sb[:, None].expand(-1, R)[rok], rs[None].expand(len(sb), -1)[rok], \
                sw[:, None].expand(-1, R)[rok]
            key = torch.cat([sb * T * WW + st * WW + sw, rb * T * WW + rt * WW + rw])
            uk, inv = torch.unique(key, return_inverse=True)
            cb, ct, cw = uk // (T * WW), (uk // WW) % T, uk % WW
        yy, xx = (cw // W) * cp, (cw % W) * cp
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crop_patches(s, cb, ct, yy, xx, cp, pixels=True), cp,
                                slot=rel[cb, ct], py=yy + lo, px=xx + lo)
            hx = causal_crops(self.mix, trace, aux, hr, cb, ct, self.causal)
        hf = hr.pooled(hx).float()[:, 0, 0, 0]                               # (N_unique, D)
        n_sel = len(sb)
        post_f = hf[inv[:n_sel]]
        rown = rok.nonzero(as_tuple=True)[0]                                 # ref -> its selected slot
        ref_sum = torch.zeros(n_sel, D, device=dev).index_add_(0, rown, hf[inv[n_sel:]])
        ref_n = torch.zeros(n_sel, device=dev).index_add_(0, rown, torch.ones(len(rown), device=dev))
        cm = self.tok2map(self.feat(post_f, ref_sum / ref_n.clamp(min=1)[:, None]), cp)  # (n_sel, 2, cp, cp)
        ar = torch.arange(cp, device=dev)
        sy, sx = (sw // W) * cp, (sw % W) * cp
        r, c = sy[:, None, None] + ar[None, :, None], sx[:, None, None] + ar[None, None, :]
        out = m.permute(0, 1, 3, 4, 2).clone()
        out[sb[:, None, None], (st - 1)[:, None, None], r, c] = cm.permute(0, 2, 3, 1).to(out.dtype)
        info.update(crops=len(uk), crop_b=cb, sel_b=sb, sel_t=st, win_y=sy, win_x=sx)
        return out.permute(0, 1, 4, 2, 3), m, info


BIOU_FRAC = 0.02                 # Boundary IoU band width: 2% of the image diagonal (Cheng et al., CVPR 2021)


@torch.no_grad()
def biou_counts(pred, lab, C, frac=BIOU_FRAC):
    """Boundary IoU counts per class (Cheng et al. 2021): band = pixels of a mask within d of its own contour,
    d = frac * diagonal; per class |G_d & P_d| and |G_d | P_d| over labelled pixels. Unlabelled pixels (lab < 0)
    and the tile border do not create contours. pred, lab (B, H, W) int tensors -> (inter (C,), union (C,))."""
    B, H, W = lab.shape
    d = max(1, int(round(frac * (H * H + W * W) ** 0.5)))
    ign = lab < 0

    def band(m):                                                     # m (B, H, W) bool
        f = (m | ign).float()[:, None]
        er = -F.max_pool2d(-F.pad(f, (d, d, d, d), mode="replicate"), 2 * d + 1, stride=1)
        return m & (er[:, 0] < 0.5)

    inter, union = np.zeros(C), np.zeros(C)
    ok = ~ign
    for c in range(C):
        g, p = band(lab == c) & ok, band(pred == c) & ok
        inter[c] += float((g & p).sum())
        union[c] += float((g | p).sum())
    return inter, union


# -- train / eval ----------------------------------------------------------------------------------------
OFFS = range(-3, 4)


@torch.no_grad()
def evaluate(net, ds, sels, q, bs=8):
    """Labelled date: per-class IoU from the confusion matrix (miou = mean over the C classes; iou = flood class;
    miou_fb = {non-flood, flood} mean, = miou when C = 2). Tracking: flood class vs the pseudo-labels.
    -> {sel: result} (q = 0: {"none": result})."""
    combos = [(sel, q) for sel in sels] if q > 0 else [("none", 0.0)]
    return {sel: r for (sel, _), r in evaluate_multi(net, ds, combos, bs).items()}


class _EvalAcc:
    """Running counts of one (selector, budget) over the evaluation batches (see evaluate)."""
    FB = (0.005, 0.05)                                                        # flood-share bins: none / low / high

    def __init__(self, C):
        self.C = C
        self.cm, self.cm0 = np.zeros((C, C)), np.zeros((C, C))                # rows = label, cols = prediction
        self.bi, self.bi0 = [np.zeros(C), np.zeros(C)], [np.zeros(C), np.zeros(C)]   # boundary IoU (inter, union)
        self.tr, self.tr0 = {o: IoU() for o in OFFS}, {o: IoU() for o in OFFS}
        self.tall, self.tall0 = IoU(), IoU()
        self.lt = self.ct = self.n = 0
        self.share = np.zeros(3)                                              # crops at post, +-1, other
        self.per_k = {}                                                       # labelled-date position -> counts
        self.fcrops, self.fn_ = np.zeros(3), np.zeros(3)
        self.slots = 1

    def update(self, out, m, info, post, ylab, ok, ytf):
        C, fc = self.C, self.C - 1
        B, T1 = out.shape[:2]
        ar = torch.arange(B, device=out.device)
        pidx = post.to(out.device) - 1
        off = torch.arange(1, T1 + 1, device=out.device)[None] - pidx[:, None] - 1   # (B, T-1): t - post
        for acc, mm, cmx, ta, tro in ((self.bi, out, self.cm, self.tall, self.tr),
                                      (self.bi0, m, self.cm0, self.tall0, self.tr0)):
            pr = mm.argmax(2)                                                 # (B, T-1, H, W)
            pl = pr[ar, pidx]
            i_, u_ = biou_counts(pl, ylab, C)                                 # boundary IoU on the labelled date
            acc[0] += i_
            acc[1] += u_
            cmx += torch.bincount(ylab[ok] * C + pl[ok], minlength=C * C).view(C, C).cpu().numpy()
            pf = pr == fc                                                     # tracking: per (b, t) counts
            tp = (pf & (ytf == 1)).sum((2, 3))
            fp = (pf & (ytf == 0)).sum((2, 3))
            fn = (~pf & (ytf == 1)).sum((2, 3))
            for o, it in [(None, ta)] + list(tro.items()):
                sel_bt = off != 0 if o is None else off == o
                it.tp += int(tp[sel_bt].sum())
                it.fp += int(fp[sel_bt].sum())
                it.fn += int(fn[sel_bt].sum())
        pl = out.argmax(2)[ar, pidx]                                          # per labelled-date position
        pk = post.to(out.device)
        for kk in pk.unique().tolist():
            mb = pk == kk
            c = self.per_k.setdefault(kk, [np.zeros((C, C)), np.zeros(C), np.zeros(C), 0])
            okb = ok[mb]
            c[0] += torch.bincount(ylab[mb][okb] * C + pl[mb][okb], minlength=C * C).view(C, C).cpu().numpy()
            i_, u_ = biou_counts(pl[mb], ylab[mb], C)
            c[1] += i_
            c[2] += u_
            c[3] += int(mb.sum())
        fs = ((ylab == fc) & ok).sum((1, 2)).float() / ok.sum((1, 2)).clamp(min=1)
        fbin = (fs >= self.FB[0]).long() + (fs >= self.FB[1]).long()          # 0 none, 1 low, 2 high
        ncrop = torch.bincount(info["crop_b"], minlength=B) if info.get("crop_b") is not None else \
            torch.zeros(B, device=out.device)
        for j in range(3):
            self.fcrops[j] += float(ncrop[fbin == j].sum())
            self.fn_[j] += int((fbin == j).sum())
        if info["sel_t"] is not None:
            o = (info["sel_t"] - post.cuda()[info["sel_b"]]).abs().cpu().numpy()
            self.share += [(o == 0).sum(), (o == 1).sum(), (o > 1).sum()]
        self.lt, self.ct, self.n = self.lt + info["lr_tokens"], self.ct + info["crops"], self.n + B
        self.slots = info.get("slots", 1)

    def result(self):
        C, fc = self.C, self.C - 1

        def ious(cm):
            tp = np.diag(cm)
            return tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)

        def fb(cm):                                                           # merge to {non-flood, flood}
            mm = np.zeros((2, 2))
            idx = [0] * (C - 1) + [1]
            for i in range(C):
                for j in range(C):
                    mm[idx[i], idx[j]] += cm[i, j]
            return ious(mm)

        I, I0 = ious(self.cm), ious(self.cm0)
        BI, BI0 = self.bi[0] / np.maximum(self.bi[1], 1), self.bi0[0] / np.maximum(self.bi0[1], 1)
        n = self.n
        return dict(miou=round(float(I.mean()), 4), miou_lr=round(float(I0.mean()), 4),
                    biou=round(float(BI.mean()), 4), biou_lr=round(float(BI0.mean()), 4),
                    biou_cls=[round(float(x), 4) for x in BI], biou_cls_lr=[round(float(x), 4) for x in BI0],
                    iou_cls=[round(float(x), 4) for x in I], iou_cls_lr=[round(float(x), 4) for x in I0],
                    miou_fb=round(float(fb(self.cm).mean()), 4), miou_fb_lr=round(float(fb(self.cm0).mean()), 4),
                    iou=round(float(I[fc]), 4), iou_lr=round(float(I0[fc]), 4),
                    track=round(self.tall.get(), 4), track_lr=round(self.tall0.get(), 4),
                    track_off={o: round(self.tr[o].get(), 4) for o in OFFS},
                    track_off_lr={o: round(self.tr0[o].get(), 4) for o in OFFS},
                    crop_share=[round(x, 3) for x in self.share / max(self.share.sum(), 1)],
                    hr_frac=round(self.ct / n / max(self.slots, 1), 4),
                    lr_tokens=round(self.lt / n, 1), crop_tokens=round(self.ct / n, 1), n=n,
                    per_k={kk: dict(miou=round(float(ious(c[0]).mean()), 4),
                                    biou=round(float((c[1] / np.maximum(c[2], 1)).mean()), 4), n=c[3],
                                    iou_cls=[round(float(x), 4) for x in ious(c[0])],
                                    biou_cls=[round(float(x), 4) for x in c[1] / np.maximum(c[2], 1)])
                           for kk, c in sorted(self.per_k.items())},
                    crops_by_flood={nm: round(float(self.fcrops[j] / max(self.fn_[j], 1)), 1)
                                    for j, nm in enumerate(("none<0.5%", "low<5%", "high>=5%"))},
                    n_by_flood=[int(x) for x in self.fn_])


@torch.no_grad()
def evaluate_multi(net, ds, combos, bs=8):
    """evaluate for several (selector, budget) pairs in ONE pass over the data: each batch is loaded (and moved to
    the GPU) once and run once per pair -> {(sel, q): result}. Same numbers as one evaluate per pair."""
    net.eval()
    C = net.n_cls
    fc = C - 1
    accs = {cq: _EvalAcc(C) for cq in combos}
    # shuffled with a fixed seed: with a batch-shared budget the batch mix matters (adjacent quadrants and the
    # same series at several positions would otherwise share batches)
    for s, d, v, post, y, _, _, last in DataLoader(ds, batch_size=bs, num_workers=7, collate_fn=collate,
                                                    pin_memory=True, prefetch_factor=4, shuffle=True,
                                                    generator=torch.Generator().manual_seed(0)):
        s = s._replace(**{k: t.cuda(non_blocking=True) for k, t in s._asdict().items() if t is not None})
        d, v, post = d.cuda(), v.cuda(), post.cuda()
        yt = batch_targets(last, y, post.cpu(), v, ds)                        # (B, T-1, H, W) on the GPU
        ylab = y.cuda()
        ok = ylab >= 0
        ytf = torch.where(yt >= 0, (yt == fc).long(), -1)                     # flood vs rest
        for (sel, q), acc in accs.items():
            out, m, info = net(s, d, v, post, sel, q)
            acc.update(out, m, info, post, ylab, ok, ytf)
    return {cq: a.result() for cq, a in accs.items()}


def train(kw):
    q, crop, lr_ps = float(kw.get("q", 0)), int(kw.get("crop", 32)), int(kw.get("lr_ps", 64))
    ep, seed, bs, T = int(kw.get("epochs", 20)), int(kw.get("seed", 0)), int(kw.get("bs", 8)), int(kw.get("T", 8))
    init, cw, tsel = kw.get("init", ""), float(kw.get("coarse_w", 1)), kw.get("train_sel", "joint")
    warmup = int(kw.get("warmup", 0))            # first `warmup` epochs select by the label oracle (sel oracle),
                                                 # then tsel; val picks the best epoch AFTER the warm-up only
    sels = kw.get("sels", "joint,wet,spatial,random,post").split(",")
    on, off = float(kw.get("on", ON)), float(kw.get("off", OFF))
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cls = int(kw.get("classes", 2))
    three = n_cls == 3
    ref_n = int(kw.get("ref_n", REF_N))          # reference passes: 3 (mean of passes < min(t, 3)), 1 (pass 0), 0
    # cat sees the pass itself (permanent water); ref_n 0: self = head(f_t), change only via the causal attention
    head_in = kw.get("head_in", "self" if ref_n == 0 else "cat" if three else "diff")
    net = EventTrack(lr_ps, crop, head_in, n_cls).cuda()
    net.ref_n = ref_n
    net.budget_scope = kw.get("budget", "sample")    # batch: one crop budget per batch (q x ABS_T x W^2 x B)
    net.causal = kw.get("causal", "1") == "1"
    net.q_unit = kw.get("q_unit", "total")       # total: q of all T x W^2 hr slots (refs incl.); pass: q of one pass's
    if init:
        net.load_state_dict(torch.load(init, map_location="cuda"))
    global FAST_PREP, ATTN
    FAST_PREP = kw.get("fast_prep", "0") == "1"
    ATTN = kw.get("attn", "sdpa")                 # chunk: chunked lr self-attention (fixed S16 at T = 32)   # raw dB + int8 labels from the workers, normalize on the GPU
    n_small = int(kw.get("n", 0))
    data = kw.get("data", "dense")
    long = data == "long"                              # long: LongTrack (dense, up to 32 passes) train/val/test
    monthly = data == "monthly" or long                # monthly: MonthlyTrack + temporal CutMix (3 classes)
    if monthly:
        assert three, "monthly / long data is 3-class"
        DS = LongTrack if long else MonthlyTrack
        DS.mix_p = float(kw.get("mix_p", 0.5))
        DS.near_days = float(kw.get("near_days", 30 if long else 40))
        DS.mix_ref = kw.get("mix_ref", "1") == "1"
        DS.mix_flood = kw.get("mix_flood", "0") == "1"
        DS.mix_frac = tuple(float(v) for v in kw["mix_frac"].split(",")) if kw.get("mix_frac") else None
        DS.k_min = int(kw.get("k_min", 1))
        DS.ref_n = ref_n
        if long:
            LongTrack.min_hist = float(kw.get("min_hist", 30))
            LongTrack.gap_days = float(kw.get("gap_days", 0))   # train only: no passes within +-gap_days of the event
            LongTrack.stride = int(kw.get("stride", 1))  # strided fixed-res baseline: T / stride passes, same span
            if kw.get("long_root"):                  # e.g. data/GEOID-Flood-tileseries-long2/cache
                LongTrack.root = kw["long_root"]
        tr = DS("train", T, augment=True, n=n_small)
        va = DS(kw.get("val_split", "val"), T, n=n_small)
    else:
        tr = Track("train", T, augment=True, n=n_small, on=on, off=off, three=three)
        va = Track(kw.get("val_split", "val"), T, n=n_small, on=on, off=off, three=three)
    ref_eval = kw.get("ref_eval", "0") == "1"     # test = RefTrack (monthly reference + dense), else the dense series
    ks_eval = [int(x) for x in kw["ks_eval"].split(",")] if kw.get("ks_eval") else None
    te = LongTrack("test", T, n=n_small, ks=ks_eval) if long else RefTrack("test", T, n=n_small) if ref_eval else \
        Track("test", T, n=n_small, on=on, off=off, three=three)
    te_m = MonthlyTrack("test", T, n=n_small) if monthly and not long else None
    print(f"[{kw.get('name')}] train {len(tr)} val {len(va)} test {len(te)} causal {net.causal}", flush=True)
    yy = np.stack([np.asarray(tr.Y[r["rec"]])[::8, ::8] for r in tr.recs[::3]])
    p1 = float((yy == 2).sum() / max((yy != 255).sum(), 1))
    w = torch.tensor([2 * p1, 2 * (1 - p1)]).float().cuda()
    if three:                                    # inverse class frequency, sum_c w_c f_c = 1
        f = np.array([(yy == c).sum() for c in range(3)], float) / max((yy != 255).sum(), 1)
        w = torch.tensor(1 / (3 * np.maximum(f, 1e-4))).float().cuda()
    dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=7, collate_fn=collate, drop_last=True,
                    persistent_workers=True, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4 if init else 1e-3)
    # q_set=.04,.08,.16,.32: train ONE model over a range of budgets (q drawn per batch); val / selection at q
    q_set = [float(x) for x in kw["q_set"].split(",")] if kw.get("q_set") else None
    best, curve, t0 = (-1.0, 0, None), [], time.time()
    max_min = float(kw.get("max_min", 140))
    ev_every = int(kw.get("eval_every", 2))
    # resume=1 (needs save=): state after every epoch in <save>.state; at max_min the job stops WITHOUT the final
    # eval (exit code 3) and the next job with the same command continues (chained sbatch)
    resume = kw.get("resume", "0") == "1" and bool(kw.get("save"))
    st_path, e0 = (kw.get("save", "") + ".state"), 0
    if resume and os.path.exists(st_path):
        st = torch.load(st_path, map_location="cuda", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        e0, curve, best = st["epoch"], st["curve"], st["best"]
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())
        print(f"[{kw.get('name')}] resumed after epoch {e0}", flush=True)
    # wall=<min>: stop before an epoch that would not finish by then (elapsed + 1.15 x the last epoch's duration)
    wall, t_ep = float(kw.get("wall", 0)), None
    for e in range(e0, ep):
        el = (time.time() - t0) / 60
        if el > max_min or (wall and t_ep is not None and el + 1.15 * t_ep > wall):
            print(f"[{kw.get('name')}] max_min reached after {e} epochs", flush=True)
            if resume:
                sys.exit(3)
            break
        net.mix.encoder.requires_grad_(e >= 1 or bool(init))
        if e == 1 and not init:
            for g in opt.param_groups:
                g["lr"] = 1e-4
        net.train()
        t_e0 = time.time()
        ls, nb = 0.0, 0
        for s, d, v, post, y, _, _, last in dl:
            qb = q_set[np.random.randint(len(q_set))] if q_set else q     # q_set: a budget drawn per batch
            yt = batch_targets(last, y, post, v, tr)
            out, m, _ = net(s, d, v, post, "oracle" if e < warmup else tsel, qb, target=yt)
            loss = cw * F.cross_entropy(m.flatten(0, 1).float(), yt.flatten(0, 1), weight=w, ignore_index=-1)
            if out is not m:
                loss = loss + F.cross_entropy(out.flatten(0, 1).float(), yt.flatten(0, 1), weight=w,
                                              ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ls, nb = ls + loss.item(), nb + 1
        msg = ""
        if ev_every and e + 1 > warmup and ((e + 1) % ev_every == 0 or e + 1 == ep):   # no val in the warm-up
            r = evaluate(net, va, [tsel], q)
            r = r[tsel if q > 0 else "none"]
            # 3 classes: val mIoU on the labelled date (the monthly val has no flood target off it, track = 0);
            # 2 classes (pseudo-label runs): flood IoU + tracking IoU
            score = r["miou"] if three else r["iou"] + r["track"]
            curve.append([e + 1, r["iou"], r["track"]])
            if score > best[0] and e + 1 > warmup:
                best = (score, e + 1, {k: v.detach().clone() for k, v in net.state_dict().items()})
            msg = f" val post {r['iou']:.4f} miou {r['miou']:.4f} track {r['track']:.4f}"
        print(f"[{kw.get('name')}] epoch {e + 1}/{ep} loss {ls / max(nb, 1):.4f}{msg} {time.time() - t0:.0f}s",
              flush=True)
        t_ep = (time.time() - t_e0) / 60
        if resume:
            os.makedirs(os.path.dirname(st_path) or ".", exist_ok=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": e + 1, "curve": curve,
                        "best": best, "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()},
                       st_path + ".tmp")
            os.replace(st_path + ".tmp", st_path)
    if best[2] is not None:
        net.load_state_dict(best[2])
    qs = [float(x) for x in kw["qs"].split(",")] if "qs" in kw else [q]    # eval-only budget sweep
    # the checkpoint is written only AFTER every result row (a link killed during the final eval is redone by the
    # next one: resume restores the finished state, rows already in the file are skipped)
    done = set()
    if resume and os.path.exists(f"{OUT}/tiletrack.jsonl"):
        for line in open(f"{OUT}/tiletrack.jsonl"):
            if f'"{kw.get("name")}"' in line:
                r = json.loads(line)
                if r.get("name") == kw.get("name") and r.get("seed") == seed:
                    done.add(r["q"])
    for qe in qs:
        if qe in done:
            print(f"[{kw.get('name')}] q {qe}: already in tiletrack.jsonl", flush=True)
    todo = [qe for qe in qs if qe not in done]
    combos = [(sel, qe) if qe > 0 else ("none", 0.0) for qe in todo for sel in (sels if qe > 0 else ["none"])]
    multi = {split: evaluate_multi(net, ds, combos) if ds is not None and combos else {}   # every budget in ONE
             for split, ds in (("val", va), ("test", te), ("test_monthly", te_m))}      # pass over each split
    for qe in todo:
        res = {"name": kw.get("name"), "q": qe, "q_train": q, "crop": crop, "lr_ps": lr_ps, "seed": seed,
               "epochs": ep, "T": T, "classes": n_cls, "head_in": head_in, "data": kw.get("data", "dense"), "mix_p": float(kw.get("mix_p", 0.5)) if monthly else 0,
               "near_days": DS.near_days if monthly else 0, "mix_ref": kw.get("mix_ref", "1") == "1" if monthly else None, "mix_flood": kw.get("mix_flood", "0") == "1" if monthly else None, "k_min": int(kw.get("k_min", 1)) if monthly else None, "test_set": "long" if long else "monthly_ref" if ref_eval else "dense", "ref_n": ref_n, "budget": net.budget_scope, "q_set": kw.get("q_set"),
               "stride": LongTrack.stride if long else 1, "ks_eval": ks_eval,
               "long_root": LongTrack.root if long else None,
               "min_hist": LongTrack.min_hist if long else None, "gap_days": LongTrack.gap_days if long else None,
               "mix_frac": kw.get("mix_frac"), "causal": net.causal, "q_unit": net.q_unit, "train_sel": tsel, "warmup": warmup, "on": on, "off": off, "init": init,
               "best_epoch": best[1], "curve": curve}
        for split, ds in (("val", va), ("test", te), ("test_monthly", te_m)):
            if ds is not None:
                res[split] = {sel: multi[split][(sel, qe if qe > 0 else 0.0)]
                              for sel in (sels if qe > 0 else ["none"])}
        print(json.dumps(res), flush=True)
        open(f"{OUT}/tiletrack.jsonl", "a").write(json.dumps(res) + "\n")
    if kw.get("save"):
        os.makedirs(os.path.dirname(kw["save"]) or ".", exist_ok=True)
        torch.save(net.state_dict(), kw["save"])

def evalref(kw):
    """Rescore saved trkM_* checkpoints on RefTrack (monthly reference) -> results/geoidchange/tiletrack_ref.jsonl.
    checkpoints/geoidchange/trkM_<fixPS[_cont] | LcC_q<q>_t<sel>>_s<seed>.pt; mixed: qs x sels as the warm runs."""
    import glob
    import re
    ds = RefTrack(kw.get("split", "test"), n=int(kw.get("n", 0)))
    print(f"RefTrack {kw.get('split', 'test')}: {len(ds)} quadrants ({ds.dropped} dropped)", flush=True)
    qs = [float(x) for x in kw.get("qs", "0.0625,0.125,0.25").split(",")]
    sels = kw.get("sels", "joint,spatial,random,post").split(",")
    done = set()
    out = kw.get("out", f"{OUT}/tiletrack_ref.jsonl")
    if os.path.exists(out):
        done = {(r["name"], r["seed"]) for r in map(json.loads, open(out))}
    paths = sorted({p for g in kw.get("ckpt", "checkpoints/geoidchange/trkM_*_s[0-9].pt").split(",")
                    for p in glob.glob(g)})
    for path in paths:
        stem = os.path.basename(path)[:-3]
        name, seed = stem.rsplit("_s", 1)
        if (name, int(seed)) in done:
            continue
        if m := re.fullmatch(r"trkM_fix(\d+)(_cont)?", name):
            lr_ps, crop, q_list, s_list = int(m[1]), 32, [0.0], ["none"]
        elif m := re.fullmatch(r"trkM_(\d+)c(\d+)_q([\d.]+)_t(\w+)", name):
            lr_ps, crop, q_list, s_list = int(m[1]), int(m[2]), qs, sels
        else:
            continue
        net = EventTrack(lr_ps, crop, "cat", 3).cuda()
        net.load_state_dict(torch.load(path, map_location="cuda"))
        net.q_unit = "total"
        t0 = time.time()
        for q in q_list:
            res = {"name": name, "seed": int(seed), "q": q, "lr_ps": lr_ps, "crop": crop if q else 0,
                   "protocol": "monthly_ref", "test": evaluate(net, ds, s_list if q else [], q)}
            open(out, "a").write(json.dumps(res) + "\n")
            k = "none" if q == 0 else "joint"
            print(f"{name} s{seed} q{q}: miou {res['test'][k]['miou']:.4f} biou {res['test'][k]['biou']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    {"labels": labels, "train": train, "evalref": evalref}[sys.argv[1]](kw)
