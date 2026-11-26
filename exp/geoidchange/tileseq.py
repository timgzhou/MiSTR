"""Temporal MiSTR on 512 px same-scene GEOID tiles: FIND THE EVENT in a same-orbit S1 RTC series.

Data (exp.geoidchange.tileseries_cache): per 512 px quadrant, consecutive passes of the relative
orbit of GEOID's post scene (the scene CEMS delineated on); label = the CEMS flood at that post
pass (tilecache .y.npy). A sample is a window of T consecutive passes holding the post pass at a
UNIFORM position k in 1..T-1 (train: random, eval: fixed per record), on a FAKE regular calendar
(pass i -> day BASE + 12 i): neither position nor timing tells where the event is.

Model (exp.geoidchange.mixseq.MixSeq parts): coarse pass at lr_ps on all T passes; per pass t >= 1
a flood map m_t = head(f_t - mean_{s < t} f_s) (causal reference). Temporal selection, no
attention: t* = argmax_t mean flood probability of m_t (tsel=score; random / oracle). Spatial:
the entropy (or random) windows of m_{t*}; a crop token at t* and at the nref passes before it in
each window, crop map = head(crop_t* - mean crop refs) replacing m_{t*} there. Output = the map at t*.
Train: per-pass coarse CE (post pass: the label; passes before it: no flood; after it: no NEW
flood when after=zero, ignored when after=ignore) + CE of the crop-refined map at the TRUE post
pass (teacher forcing: crops are learnt only at the right time).
Eval: hit (t* = post), hit within one pass, end-to-end IoU (map at t* vs the label), oracle IoU
(t* = post), random-date IoU, coarse-only IoU at t*, tokens per sample; + VV-drop heuristic hit.

    python -m exp.geoidchange.tileseq name=x lr_ps=64 q=0 epochs=20      -> results/geoidchange/tileseq.jsonl
"""

import json
import logging
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from exp.geoidchange.harness import to_sample
from exp.geoidchange.mixseq import MixSeq
from mistr.encoders.olmoearth import _spatial_size, crop_patches
logging.disable(logging.INFO)
ROOT = os.environ.get("GEOIDTSER_ROOT", "data/GEOID-Flood-tileseries/cache")
LAB = "data/GEOID-Flood-tiles512-samescene"
S = 512
NODATA = -32768
DB_FLOOR = -50.0
BASE = 18276


class Series(Dataset):
    def __init__(self, split, T=8, augment=False, n=0, gap=0, root=None):
        root = root or ROOT
        recs = json.load(open(f"{root}/{split}.json"))
        for r in recs:                                                         # gap > 0: thin to passes
            r["idx"], r["pp"], r["rday"] = self.thin(r, gap)                   # >= gap days apart
        self.recs = [r for r in recs if len(r["idx"]) >= T and self.positions(r, T)]
        self.dropped = len(recs) - len(self.recs)
        if n:
            self.recs = self.recs[:n]
        self.X = np.memmap(f"{root}/{split}.x", dtype=np.int16, mode="r").reshape(-1, 2, S, S)
        self.Y = np.load(f"{LAB}/{split}.y.npy", mmap_mode="r")
        self.T, self.augment = T, augment

    @staticmethod
    def thin(r, gap):
        """Stored pass indices kept, the post's position among them, real days from post. Greedy outward
        from the post pass: keep a pass when it is >= gap days from the last kept one (gap 0 = all)."""
        t = [datetime.fromisoformat(x).timestamp() / 86400 for x in r["time"]]
        p = r["post"]
        b, a = [p], [p]
        for i in range(p - 1, -1, -1):
            if t[b[-1]] - t[i] >= gap:
                b.append(i)
        for i in range(p + 1, len(t)):
            if t[i] - t[a[-1]] >= gap:
                a.append(i)
        idx = b[::-1] + a[1:]
        return idx, len(b) - 1, [round(t[i] - t[p], 2) for i in idx]

    @staticmethod
    def positions(r, T):
        A = len(r["idx"])
        lo, hi = max(1, r["pp"] - (A - T)), min(T - 1, r["pp"])
        return list(range(lo, hi + 1))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        pos = self.positions(r, self.T)
        if "k_fix" in r:                                                       # eval at a set position
            k = r["k_fix"]
        elif getattr(self, "k_post", None):                                     # viz: post at a set position
            k = min(pos, key=lambda p: abs(p - self.k_post))
        else:
            k = pos[np.random.randint(len(pos))] if self.augment else \
                pos[np.random.default_rng(r["rec"]).integers(len(pos))]
        s0 = r["pp"] - k
        a = np.asarray(self.X[r["off"] + np.asarray(r["idx"][s0:s0 + self.T])])   # (T, 2, S, S)
        rday = np.asarray(r["rday"][s0:s0 + self.T], dtype=np.float32)           # real days from post
        x = a.astype(np.float32) / 100
        ok = a != NODATA
        one = ok.any(1, keepdims=True) & ~ok                                   # one band missing
        x[one] = DB_FLOOR
        ok = ok | one
        valid = ok.mean((1, 2, 3)) > 0.05
        for t in range(self.T):
            for b in range(2):
                bad = ~ok[t, b]
                if bad.any():
                    x[t, b][bad] = np.median(x[t, b][~bad]) if (~bad).any() else DB_FLOOR
        yl = np.asarray(self.Y[r["rec"]])
        if getattr(self, "y3", False):            # 0 dry, 1 permanent water, 2 flood, -1 unlabelled
            y = yl.astype(np.int64)
        else:
            y = (yl == 2).astype(np.int64)
        y[yl == 255] = -1
        if self.augment:
            kk = np.random.randint(4)
            x, y = np.rot90(x, kk, (2, 3)), np.rot90(y, kk, (0, 1))
            if np.random.rand() < 0.5:
                x, y = x[..., ::-1], y[..., ::-1]
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        days = BASE + 12 * np.arange(self.T, dtype=np.int64)
        return x, days, valid, k, y, rday


def collate(items):
    x = np.stack([it[0] for it in items])
    d = np.stack([it[1] for it in items])
    v = np.stack([it[2] for it in items])
    return (to_sample(x, d, v), torch.from_numpy(d), torch.from_numpy(v),
            torch.tensor([it[3] for it in items]), torch.from_numpy(np.stack([it[4] for it in items])),
            torch.from_numpy(x[:, :, 0].mean((2, 3))),                        # mean VV per pass
            torch.from_numpy(np.stack([it[5] for it in items])))               # real days from post


class EventSeq(MixSeq):
    def forward(self, s, days, valid, post, tsel="score", q=0.25, sel="entropy", nref=2):
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items()
                          if v is not None})
        days, valid, post = days.cuda(), valid.cuda(), post.cuda()
        rel = (days - days[:, :1]).float() / 6
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.embed(s, self.lr_ps, times=rel)
            enc = self.mix.lr_trace(lr)
        f = lr.pooled(enc[1]).float()                                          # (B, T, h, w, D)
        B, T, h, w, D = f.shape
        P = self.lr_ps
        bi = torch.arange(B, device=f.device)
        v = valid.float()[:, :, None, None, None]
        ref = torch.cumsum(f * v, 1)[:, :-1] / torch.cumsum(v, 1)[:, :-1].clamp(min=1)   # mean_{s<t}
        E = getattr(self, "ref_n", 3)
        early = getattr(self, "ref_mode", "causal") == "early"
        if early:       # baseline reference: mean of passes s < min(t, E) (an early flood stays out)
            j = (torch.arange(1, T, device=f.device).clamp(max=E) - 1)
            ref = ref[:, j]
        d = self.feat(f[:, 1:], ref)
        m = self.tok2map(d.reshape(-1, d.shape[-1]), P)
        m = m.view(B, T - 1, h, w, 2, P, P).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T - 1, 2, h * P, w * P)
        H = _spatial_size(s)
        lo = (h * P - H) // 2
        if h * P != H:
            m = m[..., lo:lo + H, lo:lo + H]
        lr_tokens = int(valid.sum()) * h * w
        with torch.no_grad():
            score = m.float().softmax(2)[:, :, 1].mean((2, 3))                 # (B, T - 1)
            if getattr(self, "tscore", "max") == "rise":   # onset: largest jump over the previous pass
                prev = torch.cat([torch.zeros_like(score[:, :1]), score[:, :-1]], 1)
                pv = torch.cat([torch.ones_like(valid[:, :1]), valid[:, 1:-1]], 1)
                score = torch.where(pv, score - prev, score)
            score[~valid[:, 1:]] = -1.0
            if tsel == "oracle":
                ts = post
            elif tsel == "near":        # training: the pick when within one pass of post, else post
                pick = score.argmax(1) + 1
                ts = torch.where((pick - post).abs() <= 1, pick, post)
            elif tsel == "random":
                ts = torch.multinomial(valid[:, 1:].float() + 1e-6, 1)[:, 0] + 1
            else:
                ts = score.argmax(1) + 1
        m0 = m[bi, ts - 1]
        if q <= 0:
            return m0, m0, m, ts, lr_tokens, 0
        cp = self.crop
        with torch.no_grad():
            lp = m0.float().log_softmax(1)
            u = -(lp.exp() * lp).sum(1) if sel == "entropy" else torch.rand_like(m0[:, 0].float())
            win = F.avg_pool2d(u[:, None], cp)[:, 0]
            W = win.shape[-1]
            k = max(1, round(q * W * W))
            idx = win.flatten(1).topk(k, 1).indices
        wb = bi.repeat_interleave(k)
        wy, wx = (idx.flatten() // W) * cp, (idx.flatten() % W) * cp
        nwin = len(wb)
        if early:       # crop references = the same early passes (those before t*)
            refs = torch.arange(E, device=f.device)[None].expand(B, E)
            dts = torch.cat([ts[:, None], refs], 1)
            okd = torch.cat([torch.ones_like(ts[:, None], dtype=torch.bool), refs < ts[:, None]], 1)
        else:
            dts = ts[:, None] - torch.arange(nref + 1, device=f.device)[None]  # (B, 1 + nref): t*, refs
            okd = dts >= 0
        okd = okd & valid.gather(1, dts.clamp(min=0))
        okd[:, 0] = True
        ci, cj = okd[wb].nonzero(as_tuple=True)
        ct = dts[wb][ci, cj]
        bb, yy, xx = wb[ci], wy[ci], wx[ci]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crop_patches(s, bb, ct, yy, xx, cp, pixels=True), cp,
                                slot=rel[bb, ct], py=yy + lo, px=xx + lo)
            _, (hx,) = self.mix.tail_multi(lr, [(hr, bb)], enc)
        hf = hr.pooled(hx).float()[:, 0, 0, 0]
        is_post = cj == 0
        z = torch.zeros(nwin, D, device=hf.device)
        pre_sum = z.index_add(0, ci[~is_post], hf[~is_post])
        pre_n = torch.zeros(nwin, device=hf.device).index_add_(
            0, ci[~is_post], torch.ones_like(ci[~is_post], dtype=torch.float))
        post_f = z.index_add(0, ci[is_post], hf[is_post])
        cm = self.tok2map(self.feat(post_f, pre_sum / pre_n.clamp(min=1)[:, None]), cp)
        ar = torch.arange(cp, device=f.device)
        r = wy[:, None, None] + ar[None, :, None]
        c = wx[:, None, None] + ar[None, None, :]
        out = m0.permute(0, 2, 3, 1).clone()
        out[wb[:, None, None], r, c] = cm.permute(0, 2, 3, 1).to(out.dtype)
        return out.permute(0, 3, 1, 2), m0, m, ts, lr_tokens, len(ci)
