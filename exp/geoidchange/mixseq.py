"""Spatial mixed resolution on the GRD series (segmentation at GEOID's post date).

Pass 1: the sequence (K pre-event dates + post) at lr_ps, the change map = head(f_post -
mean f_pre) per token. Selected windows (entropy of that map; random as a control) get a crop
token at EVERY date of the sequence (attn cross_hr: crops read the season-level tokens and each
other); the window's map = head(crop_post - mean crop_pre), area-pooled to the crop, replacing
the coarse logits there. Sequences as exp.geoidchange.seqseg (pos=preev, ctx=K).

    GEOIDCHANGE_ROOT=data/GEOID-Flood-series-grd/cache GEOIDCHANGE_WINDOW=366 \\
        python -m exp.geoidchange.mixseq name=x sel=entropy q=0.25 crop=8 ctx=7
"""

import logging
import torch
import torch.nn.functional as F
from torch import nn
from exp.common.config import MODEL_SIZE_TO_ID
from exp.geoidchange.harness import INP
from mistr.encoders.olmoearth import OlmoEarthEncoder
from mistr.encoders.olmoearth import _spatial_size, crop_patches
from mistr.core.encoder import OneWayEncoder as MixedResOE
logging.disable(logging.INFO)


class MixSeq(nn.Module):
    def __init__(self, lr_ps=16, crop=8, head_in="diff", n_cls=2):
        super().__init__()
        enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], INP, 16, 64, name="tiny").encoder
        self.mix = MixedResOE(enc, 0, "center", lr_ps, attn="cross_hr")
        self.lr_ps, self.crop, self.head_in = lr_ps, crop, head_in
        # diff: head(post - mean pre); cat: head([post, post - mean pre]) (water is visible at post
        # alone, permanent water has no change); self: head(post) (no reference, change from the attention)
        self.n_cls = n_cls
        self.head = nn.Linear(enc.embedding_size * (2 if head_in == "cat" else 1), n_cls * 16 * 16)
        nn.init.zeros_(self.head.weight)

    def feat(self, post, ref):
        if self.head_in == "self":
            return post
        return torch.cat([post, post - ref], -1) if self.head_in == "cat" else post - ref

    def tok2map(self, d, ps):
        m = self.head(d).view(-1, self.n_cls, 16, 16)
        if ps > 16:                     # ps32: the 16 x 16 logits per token, upsampled
            return F.interpolate(m, size=(ps, ps), mode="bilinear", align_corners=False)
        return F.avg_pool2d(m, 16 // ps) if ps < 16 else m

    def forward(self, s, days, valid, post, sel="entropy", q=0.25):
        """s: (B, T) sequence, valid (B, T), post (B,) index of the post frame."""
        s = s._replace(**{k: v.cuda(non_blocking=True) for k, v in s._asdict().items()
                          if v is not None})
        days, valid, post = days.cuda(), valid.cuda(), post.cuda()
        rel = (days - days[:, :1]).float() / 6                              # time axis, as seqseg
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lr = self.mix.embed(s, self.lr_ps, times=rel)
            enc = self.mix.lr_trace(lr)
        f = lr.pooled(enc[1]).float()                                       # (B, T, h, w, D)
        B, T, h, w, D = f.shape
        bi = torch.arange(B, device=f.device)
        pre = (valid & (torch.arange(T, device=f.device)[None] != post[:, None])).float()
        ref = (f * pre[:, :, None, None, None]).sum(1) / pre.sum(1).clamp(min=1)[:, None, None, None]
        P = self.lr_ps
        m0 = self.tok2map(self.feat(f[bi, post], ref).reshape(B * h * w, -1), P)
        m0 = m0.view(B, h, w, 2, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, 2, h * P, w * P)
        H = _spatial_size(s)
        lo = (h * P - H) // 2           # lr_ps not dividing the tile: embed padded it, centred
        if lo or h * P != H:
            m0 = m0[:, :, lo:lo + H, lo:lo + H]
        lr_tokens = int(valid.sum()) * h * w
        if q <= 0:
            return m0, m0, lr_tokens, 0
        cp = self.crop
        with torch.no_grad():
            lp = m0.float().log_softmax(1)
            u = -(lp.exp() * lp).sum(1) if sel == "entropy" else torch.rand_like(m0[:, 0].float())
            win = F.avg_pool2d(u[:, None], cp)[:, 0]
            W = win.shape[-1]
            k = max(1, round(q * W * W))
            idx = win.flatten(1).topk(k, 1).indices                         # (B, k)
        # every (window, valid date): one crop token
        wb = bi.repeat_interleave(k)                                        # window -> sample
        wy, wx = (idx.flatten() // W) * cp, (idx.flatten() % W) * cp
        nwin = len(wb)
        vt = valid[wb]                                                      # (nwin, T)
        ci, ct = vt.nonzero(as_tuple=True)                                  # crop -> window, date
        bb, yy, xx = wb[ci], wy[ci], wx[ci]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hr = self.mix.embed(crop_patches(s, bb, ct, yy, xx, cp, pixels=True), cp,
                                slot=rel[bb, ct], py=yy + lo, px=xx + lo)
            _, (hx,) = self.mix.tail_multi(lr, [(hr, bb)], enc)
        hf = hr.pooled(hx).float()[:, 0, 0, 0]                              # (N, D)
        is_post = ct == post[bb]
        pre_sum = torch.zeros(nwin, D, device=hf.device).index_add_(0, ci[~is_post], hf[~is_post])
        pre_n = torch.zeros(nwin, device=hf.device).index_add_(0, ci[~is_post],
                                                               torch.ones_like(ci[~is_post], dtype=torch.float))
        post_f = torch.zeros(nwin, D, device=hf.device).index_add_(0, ci[is_post], hf[is_post])
        cm = self.tok2map(self.feat(post_f, pre_sum / pre_n.clamp(min=1)[:, None]), cp)   # (nwin, 2, cp, cp)
        ar = torch.arange(cp, device=f.device)
        r = wy[:, None, None] + ar[None, :, None]
        c = wx[:, None, None] + ar[None, None, :]
        out = m0.permute(0, 2, 3, 1).clone()
        out[wb[:, None, None], r, c] = cm.permute(0, 2, 3, 1).to(out.dtype)
        return out.permute(0, 3, 1, 2), m0, lr_tokens, len(ci)
