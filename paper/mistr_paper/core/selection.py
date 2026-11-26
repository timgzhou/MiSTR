"""Selecting the spatiotemporal windows MiSTR re-encodes at fine resolution (paper Sec. 3.2).

From the model's coarse pass (CoarsePass.out0), a WindowSelector picks
    where   which crop windows of the tile: those whose coarse-prediction uncertainty is above one
            global threshold (the (1 - q) quantile of the training batches, refit on val before
            testing; q = select.percentage); alternatives: random / grid / oracle windows,
            q x the windows of each sample
    when    each picked window's months: those of the coarse step the decoder attends to most
            there; alternatives: the step whose own map has the lowest entropy, a random step, or
            a fixed lattice over the used steps (grid)
    extra   (select.extra_steps) the next step of the most uncertain picked windows too,
            extra_steps more (window, step) pairs per picked window on average, by a second
            global threshold
and returns the Windows the model encodes. Windows tile the 64 px tile on their own grid (padded
and centred when the crop size does not divide 64, as the coarse tokens are).

The warm-up stage (train.warmup) uses a WarmupSelector: a lattice of windows (WarmupLattice),
scored by the label loss of the coarse prediction (oracle) or at random, the top n of each
sample, every present month.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

import mistr_paper.core as _pkg     # (FAST, read at call time)
from mistr_paper.encoders.olmoearth import _to_device
from mistr_paper.core.model import LR_PS, MiSTR
from mistr_paper.core.windows import LOOKUP_PS, TILE, Windows

H = TILE // LOOKUP_PS       # the 16 x 16 ps4 cell grid the warm-up selector's scores live on
NEG = float("-inf")
WHERE = ("uncertainty", "random", "grid", "oracle")
WHEN = ("attention", "entropy", "random", "grid")


# -- the coarse pass, as a selector reads it ----------------------------------------------
def topk_mask(scores, k):
    """Flat top-k over (T,H,W) per sample -> bool mask (B,T,H,W) with exactly k True each.

    Flat (not per-timestep) on purpose: k is a per-sample lookup BUDGET, and the model should
    be free to spend it unevenly across time -- a cloudy timestep deserves more lookups than a
    clear one.
    """
    B, T, H, W = scores.shape
    n = T * H * W
    k = min(k, n)
    flat = scores.reshape(B, n)
    idx = flat.topk(k, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(B, T, H, W)


class CoarsePass:
    """One batch as a selector sees it. out0 is the coarse pass (coarse tokens only) of the
    model being trained, computed once on first use; target only exists while training or for
    a label oracle."""

    def __init__(self, model: MiSTR, ms, ss, season, present, target=None, dates=None):
        dev = model.device
        self.dates = dates
        self.model = model
        self.ms, self.ss = _to_device(ms, dev, torch.bfloat16), _to_device(ss, dev, torch.bfloat16)
        self.season, self.present = season.to(dev), present.to(dev)
        self.target = None if target is None else target.to(dev)
        self.B, self.T = self.present.shape
        if model.cfg.coarse_time == "month":    # every month its own coarse step (MiSTR._inputs)
            self.ss = self.ms
            self.season = torch.arange(self.T, device=dev).expand(self.B, self.T).contiguous()
        self.NS = self.ss.timestamps.shape[1]
        self._out0 = None

    @classmethod
    def from_batch(cls, model: MiSTR, batch) -> "CoarsePass":
        """mistr_paper.core.data_cpu.Batch -> the context (the label included: oracles read it)."""
        return cls(model, batch.ms, batch.ss, batch.season, batch.present, batch.target,
                   batch.dates)

    @property
    def out0(self):
        if self._out0 is None:
            self._out0 = self.model.uniform(self.ms, self.ss, self.season, self.present)
        return self._out0

    @property
    def coarse_tokens(self):
        """The coarse pass's embedded coarse tokens if it ran, for reuse at inference."""
        return None if self._out0 is None else self._out0.coarse_tokens

    @property
    def coarse_cache(self):
        """The coarse pass's tokens through every block (one-way attention), if it ran."""
        return None if self._out0 is None else self._out0.coarse_cache

    @property
    def valid(self) -> torch.Tensor:
        """(B, T) month is present and in a coarse step."""
        return self.present & (self.season >= 0)

    @property
    def used(self) -> torch.Tensor:
        """(B, NS) coarse step holds a present month."""
        oh = F.one_hot(self.season.clamp(min=0), self.NS).bool() & self.valid[..., None]
        return oh.any(1)


def season_maps(model: MiSTR, lr_F: torch.Tensor) -> torch.Tensor:
    """(B, NS, h, w, D) coarse tokens -> (B NS, C, 64, 64): each coarse token's own head map
    (finer tokens area-pooled, as the decoder does). Precision follows the caller's autocast."""
    lin = model.head.linear
    NS, gh, gw = lr_F.shape[1:4]
    hp = getattr(model, "head_px", LR_PS)      # (model.coarse_head: the coarse patch)
    C = lin.out_features // hp ** 2
    m = lin(lr_F).view(-1, C, hp, hp)
    P = model.cfg.coarse_patch
    if P < hp:
        m = F.avg_pool2d(m, hp // P) if hp % P == 0 else F.adaptive_avg_pool2d(m, P)
    m = rearrange(m, "(b s gh gw) c i j -> (b s) c (gh i) (gw j)", s=NS, gh=gh, gw=gw)
    lo = model.lo           # ps6 / ps10 / ...: the tile padded to whole tokens, cropped back
    return m[..., lo:lo + TILE, lo:lo + TILE] if lo else m


def grid_steps(used: torch.Tensor, G: int) -> torch.Tensor:
    """when grid: (B, NS, G, G) True at the ((i + j) mod n)-th of a sample's n used coarse steps
    over window (i, j) (the steps cycle diagonally across the window lattice); none if n = 0."""
    B, NS = used.shape
    n = used.sum(1)                                                          # (B,)
    ij = torch.arange(G, device=used.device)
    k = (ij[:, None] + ij[None, :])[None] % n.clamp(min=1)[:, None, None]    # (B, G, G)
    order = used.long().cumsum(1) - 1                                        # rank among used
    hit = used[:, :, None, None] & (order[:, :, None, None] == k[:, None])
    return hit & (n > 0)[:, None, None, None]


def pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)
    return -(p * p.clamp(min=1e-8).log()).sum(1)                                # (N, 64, 64)


def spatial_uncertainty(ctx: CoarsePass) -> torch.Tensor:
    """(B, 64, 64) per-pixel uncertainty of the coarse pass: classification the entropy of the
    prediction; regression the variance over the used steps of each coarse token's own map
    (where the steps disagree)."""
    out0 = ctx.out0
    with torch.no_grad():
        if ctx.model.task.kind == "classification":
            return pixel_entropy(out0.prediction.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(ctx.model, out0.coarse_features)
        m = m.float().view(ctx.B, ctx.NS, -1, *m.shape[-2:]).mean(2)          # (B, NS, 64, 64)
        w = ctx.used.float()[..., None, None]
        n = w.sum(1).clamp(min=1)
        mu = (w * m).sum(1) / n
        return (w * (m - mu[:, None]) ** 2).sum(1) / n


NO_INPUT = -1.0     # window uncertainty of a window without input (uncertainties are >= 0)
_ZERO = {}          # task name -> S2 raw 0 after OlmoEarth normalization, (C,)


def _s2_zero(task) -> torch.Tensor:
    if task.name not in _ZERO:
        from mistr_paper.encoders.olmoearth import OlmoEarthInput
        if "S2" not in task.olmoearth.modalities:
            raise ValueError(f"select.input_mask: {task.name} has no S2 input")
        inp = OlmoEarthInput(task.olmoearth.modalities, task.olmoearth.impute)
        om = inp.olmo["S2"]
        z = inp.normalizer.normalize(om, np.zeros((1, 1, 1, 1, len(om.band_order)), np.float32))
        _ZERO[task.name] = torch.from_numpy(np.asarray(z, np.float32)).flatten()
    return _ZERO[task.name]


def input_coverage(ctx: CoarsePass) -> torch.Tensor:
    """(B, 64, 64) float: 1 where the pixel holds S2 input in at least one present month (the
    monthly sample, as the model reads it), 0 where it is raw 0 in every band of every present
    month (YieldSAT: outside the field polygon, tile padding). bfloat16 inputs: matched to
    normalize(0) within 0.03 per band."""
    x = ctx.ms.sentinel2_l2a                                       # (B, H, W, T, C), normalized
    z = _s2_zero(ctx.model.task).to(x.device)
    zero = ((x.float() - z).abs() < 0.03).all(-1)                  # (B, H, W, T)
    has = (~zero & ctx.present[:, None, None, :]).any(-1)
    return has.float()


def pixel_label_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, C, 64, 64) vs the label -> (N, 64, 64); 0 where unlabelled."""
    if logits.shape[1] > 1:
        return F.cross_entropy(logits, target.long(), ignore_index=-1, reduction="none")
    y = target[:, 0] if target.ndim == 4 else target           # YieldSAT: (value, field id)
    se = ((logits[:, 0] - y.float()) ** 2).nan_to_num(0.0)
    return se


@dataclass
class Selection:
    plan: Windows | list[Windows]
    mask: torch.Tensor | None = None    # (B, T, 16, 16) picks on the ps4 grid, where defined
    logw: torch.Tensor | None = None    # (N,) per-window decoder log-weights (none here)
    aux: torch.Tensor | None = None     # an auxiliary training loss (none here)


# -- the method's selector ----------------------------------------------------------------
class WindowThreshold(nn.Module):
    """The global thresholds: EMA tables (101 quantile levels) of the training batches'
    window uncertainty (tab_s) and of the extra-step scores (tab_x, ratio_x: candidates per
    picked window), saved with the selector, refit on val before testing."""

    LEVELS = 101

    def __init__(self, momentum: float = 0.95, extra: bool = False):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("tab_s", torch.zeros(self.LEVELS))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))
        if extra:
            self.register_buffer("tab_x", torch.zeros(self.LEVELS))
            self.register_buffer("ratio_x", torch.ones(()))
            self.register_buffer("seen_x", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, u_s):
        qs = torch.quantile(u_s.flatten().float(),
                            torch.linspace(0, 1, self.LEVELS, device=u_s.device))
        m = 0.0 if int(self.seen) == 0 else self.momentum
        self.tab_s.mul_(m).add_((1 - m) * qs)
        self.seen += 1

    @torch.no_grad()
    def update_x(self, vals, ratio):
        lv = torch.linspace(0, 1, self.LEVELS, device=vals.device)
        qx = torch.quantile(vals.float()[:16_000_000], lv)
        m = 0.0 if int(self.seen_x) == 0 else self.momentum
        self.tab_x.mul_(m).add_((1 - m) * qx)
        self.ratio_x.mul_(m).add_((1 - m) * ratio)
        self.seen_x += 1

    def threshold(self, tab, q):
        """The value above which a share q of the tracked distribution lies."""
        x = (1 - q) * (self.LEVELS - 1)
        i = min(int(x), self.LEVELS - 2)
        return tab[i] + (x - i) * (tab[i + 1] - tab[i])


class WindowSelector:
    """where x when (+ extra steps) on the windows' grid; see the module docstring.
    cfg: config.SelectConfig."""

    def __init__(self, window_px: int, cfg, coarse_patch: int, device=None):
        self.window_px, self.cfg, self.coarse_patch = window_px, cfg, coarse_patch
        if cfg.where not in WHERE or cfg.when not in WHEN:
            raise ValueError(f"select.where {cfg.where!r} / when {cfg.when!r}: {WHERE} / {WHEN}")
        # a patch that does not divide the tile (ps6 coarse tokens, 12 px windows) tiles it
        # padded, centred, to whole patches (66 / 72 px; OneWayEncoder.embed pads the coarse tokens
        # the same way)
        self.lo = self.frame_lo(coarse_patch)
        # nested: the windows tile each coarse cell (ps16 / ps8 with 8 / 4 px windows); else
        # each window reads the coarse cell under its centre
        self.nested = (coarse_patch % window_px == 0 and self.frame_lo(window_px) == self.lo
                       and window_px >= LOOKUP_PS)     # (2 px windows: smaller than a cell)
        self.extra = bool(cfg.extra_steps or cfg.extra_steps_choices)
        if self.extra and cfg.when != "attention":
            raise ValueError("select.extra_steps: with when attention only")
        self.t_set = [float(v) for v in cfg.extra_steps_choices.split(",") if v]
        self.tx = cfg.extra_steps    # this batch's extra_steps (training may draw it from the choices)
        self.state = WindowThreshold(extra=self.extra).to(device)
        self.last = {}

    def modules(self) -> list:
        return [self.state]

    @staticmethod
    def frame_lo(px: int) -> int:
        """Padding before the tile when px-px patches tile it padded to whole patches."""
        return (-(-TILE // px) * px - TILE) // 2

    def pool(self, u, px):
        """(N, 64, 64) -> the mean per px window (N, frame / px, frame / px), the map
        replicate-padded to px's padded frame first when px does not divide the tile."""
        lo = self.frame_lo(px)
        if TILE % px:
            i = (torch.arange(-(-TILE // px) * px, device=u.device) - lo).clamp(0, TILE - 1)
            u = u[:, i][:, :, i]
        return F.avg_pool2d(u[:, None], px)[:, 0]

    def pool_max(self, u, px):
        """As pool, the max per window."""
        lo = self.frame_lo(px)
        if TILE % px:
            i = (torch.arange(-(-TILE // px) * px, device=u.device) - lo).clamp(0, TILE - 1)
            u = u[:, i][:, :, i]
        return F.max_pool2d(u[:, None], px)[:, 0]

    def cell_of_window(self, n_windows: int, gh: int, device) -> torch.Tensor:
        """(n_windows,) the coarse cell (0..gh-1) under each window's centre."""
        cp = self.window_px
        c = (torch.arange(n_windows, device=device) * cp + cp / 2 - self.frame_lo(cp)
             + self.lo)                                     # centre in the coarse tokens' frame
        return (c // self.coarse_patch).long().clamp(0, gh - 1)

    def _up(self, gh: int, device):
        """coarse-cell maps (..., gh, gh) -> the windows' grid (..., G, G)."""
        f = self.coarse_patch // self.window_px
        if self.nested:
            return lambda m: m.repeat_interleave(f, -2).repeat_interleave(f, -1)
        ci = self.cell_of_window(-(-TILE // self.window_px), gh, device)
        return lambda m: m[..., ci, :][..., ci]

    def window_uncertainty(self, ctx) -> torch.Tensor:
        """(B, G, G) the mean coarse-prediction uncertainty over each window (>= 0; select.input_mask:
        NO_INPUT at windows less than half covered by S2 input)."""
        u = self.pool(spatial_uncertainty(ctx), self.window_px)
        if self.cfg.input_mask:
            u = u.masked_fill(self.pool(input_coverage(ctx), self.window_px) < 0.5, NO_INPUT)
        return u

    def threshold_values(self, ctx) -> torch.Tensor:
        """The window uncertainties the global threshold applies to, flat (val refit)."""
        return self.window_uncertainty(ctx).flatten()

    def attention_ranking(self, ctx):
        """(B, NS, gh, gw) the decoder's attention score of each coarse step at each coarse
        cell, -inf at unused steps; and its softmax over the steps."""
        with torch.no_grad():
            lr_F = ctx.out0.coarse_features.float()
            months = (F.one_hot(ctx.season.clamp(min=0), ctx.NS).float()
                      * ctx.valid.float()[..., None]).sum(1)                     # (B, NS)
            score = ctx.model.readout0._score(lr_F)
            unused = (months == 0)[:, :, None, None]
            w = score.masked_fill(unused, NEG).softmax(1)
        return score.masked_fill(unused, NEG), w

    def step_ranking(self, ctx, G: int) -> torch.Tensor:
        """when entropy / random / grid: (B, NS, G, G) a score per (coarse step, window),
        higher = re-encode that step, -inf at unused steps."""
        used = ctx.used[:, :, None, None]
        if self.cfg.when == "grid":     # a fixed lattice: the (i + j) mod n-th used step at (i, j)
            return grid_steps(ctx.used, G).float().masked_fill(~used, NEG)
        if self.cfg.when == "random":
            return torch.rand(ctx.B, ctx.NS, G, G, device=ctx.season.device).masked_fill(~used, NEG)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(ctx.model, ctx.out0.coarse_features)                    # (B NS, C, 64, 64)
        m = m.float()
        if ctx.model.task.kind != "classification":
            raise ValueError("when entropy: classification tasks only")
        px = pixel_entropy(m)
        return (-self.pool(px, self.window_px)).view(ctx.B, ctx.NS, G, G).masked_fill(~used, NEG)

    def step_disagreement(self, ctx, G: int) -> torch.Tensor:
        """(B, NS, NS, G, G) how much two coarse steps' own maps disagree over each window
        (extra_pick disagree): classification the mean Jensen-Shannon divergence of their class
        distributions (H(mean) - mean H), regression the mean absolute difference of their outputs;
        -inf on the diagonal and at unused steps."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m = season_maps(ctx.model, ctx.out0.coarse_features)                    # (B NS, C, 64, 64)
        B, NS = ctx.B, ctx.NS
        m = m.float().view(B, NS, *m.shape[1:])
        cls = ctx.model.task.kind == "classification"
        if cls:
            p = m.softmax(2)
            h = -(p * p.clamp(min=1e-8).log()).sum(2)                               # (B, NS, 64, 64)
        D = m.new_full((B, NS, NS, G, G), NEG)
        with torch.no_grad():
            for a in range(NS):
                for b in range(a + 1, NS):
                    if cls:
                        pm = (p[:, a] + p[:, b]) / 2
                        px = -(pm * pm.clamp(min=1e-8).log()).sum(1)
                        px = px - (h[:, a] + h[:, b]) / 2      # H(mean) - mean H: the JSD
                    else:
                        px = (m[:, a] - m[:, b]).abs().mean(1)
                    D[:, a, b] = D[:, b, a] = self.pool(px, self.window_px)
        ok = ctx.used[:, :, None] & ctx.used[:, None, :]
        return D.masked_fill(~ok[..., None, None], NEG)

    def picked_windows(self, ctx, u_s, q) -> torch.Tensor:
        """(B, G, G) bool: where."""
        w = self.cfg.where
        if w == "uncertainty":      # (u_s >= 0: never a NO_INPUT window, whatever the threshold)
            return (u_s >= self.state.threshold(self.state.tab_s, q)) & (u_s >= 0)
        B, G = u_s.shape[0], u_s.shape[-1]
        k = max(1, round(q * G * G))
        if w == "random":
            s = torch.rand(B, G, G, device=u_s.device)
        elif w == "grid":       # a lattice of k windows (random phase while training)
            m = max(1, round(G * G / k))
            c = max(1, round(m ** 0.5))
            i, j = torch.meshgrid(torch.arange(G, device=u_s.device),
                                  torch.arange(G, device=u_s.device), indexing="ij")
            phase = (torch.randint(0, m, (B, 1, 1), device=u_s.device) if ctx.model.training
                     else torch.zeros(B, 1, 1, dtype=torch.long, device=u_s.device))
            s = (((i + c * j)[None] + phase) % m == 0).float() + 1e-3 * torch.rand_like(u_s)
        else:                   # oracle: the coarse prediction's label loss
            if ctx.target is None:
                raise ValueError("where oracle: no label in the context")
            s = self.pool(pixel_label_loss(ctx.out0.prediction.float(), ctx.target), self.window_px)
        return topk_mask(s[:, None], k)[:, 0]

    def extra_candidates(self, u_s, hs, rank, ctx=None):
        """The extra steps: per picked window the second-ranked step (extra_pick attention) or
        the step whose own map disagrees most with the top step's (extra_pick disagree), scored by the window's uncertainty -> (candidates (B, NS, G, G), scores,
        the base pairs, picked windows)."""
        order = rank.argsort(1, descending=True).argsort(1)
        used = torch.isfinite(rank)
        base = hs[:, None] & (order == 0) & used
        if getattr(self.cfg, "extra_pick", "attention") == "disagree":
            NS, G = rank.shape[1], rank.shape[-1]
            D = self.step_disagreement(ctx, G)                                  # (B, NS, NS, G, G)
            top1 = rank.argmax(1)
            row = D.gather(1, top1[:, None, None].expand(-1, 1, NS, -1, -1))[:, 0]
            v, j = row.max(1)
            second = F.one_hot(j, NS).permute(0, 3, 1, 2).bool() & torch.isfinite(v)[:, None]
            cand = hs[:, None] & second & used
        else:
            cand = hs[:, None] & (order == 1) & used
        return cand, u_s[:, None].expand_as(cand), base, float(hs.sum())

    def extra_share(self):
        return min(1.0, self.tx / max(float(self.state.ratio_x), 1e-6))

    @torch.no_grad()
    def extra_values(self, ctx):
        """(candidate scores, candidates per picked window) of one batch under the current
        window threshold (exp.mistr.train val_threshold refits tab_x on val)."""
        u_s = self.window_uncertainty(ctx)
        hs = self.picked_windows(ctx, u_s, self.cfg.percentage)
        r, _ = self.attention_ranking(ctx)
        cand, score, _, unit = self.extra_candidates(u_s, hs, self._up(r.shape[-1], r.device)(r), ctx)
        return score[cand].float(), float(cand.sum()) / max(unit, 1.0)

    def __call__(self, ctx: CoarsePass) -> Selection:
        q = self.cfg
        training = ctx.model.training
        u_s = self.window_uncertainty(ctx)                                     # (B, G, G)
        G = u_s.shape[-1]
        if training and q.where == "uncertainty" and u_s.numel():
            self.state.update(u_s)
        qs = q.percentage
        self.tx = q.extra_steps
        if training and self.t_set:
            self.tx = self.t_set[int(torch.randint(len(self.t_set), ()))]
        hs = self.picked_windows(ctx, u_s, qs)
        valid = ctx.valid                                                      # (B, T)
        B, T = valid.shape
        if q.when == "attention":
            r, self.attention = self.attention_ranking(ctx)                    # coarse cells
            gh = r.shape[-1]
            up = self._up(gh, r.device)
            if self.extra:          # (window, step) pairs: the top step + extra by threshold
                cand, score, base, unit = self.extra_candidates(u_s, hs, up(r), ctx)
                if training and bool(cand.any()):
                    self.state.update_x(score[cand], float(cand.sum()) / max(unit, 1.0))
                share = self.extra_share()
                sel = (base | cand & (score >= self.state.threshold(self.state.tab_x, share))
                       if share > 0 else base)
                sg = ctx.season.clamp(min=0)[:, :, None, None].expand(B, T, G, G)
                hr = sel.gather(1, sg) & valid[:, :, None, None]
                self.last = {"crops/window": sel.sum() / sel[:, 0].numel() if _pkg.FAST
                             else float(sel.sum()) / sel[:, 0].numel()}
            else:                   # the most-attended step's months, per coarse cell
                top = (r.argsort(1, descending=True).argsort(1) < 1) & torch.isfinite(r)
                sidx = ctx.season.clamp(min=0)[:, :, None, None].expand(B, T, gh, gh)
                hr = hs[:, None] & up(top.gather(1, sidx) & valid[:, :, None, None])
        else:                       # one step per window, by the window's own ranking
            r = self.step_ranking(ctx, G)
            top = (r.argsort(1, descending=True).argsort(1) < 1) & torch.isfinite(r)
            sg = ctx.season.clamp(min=0)[:, :, None, None].expand(B, T, G, G)
            hr = hs[:, None] & top.gather(1, sg) & valid[:, :, None, None]
        plan = Windows.from_grid(hr, self.window_px)
        lo_c = self.frame_lo(self.window_px)
        mask = None
        if lo_c or not self.nested:     # the windows' padded frame -> the tile (edges clamped)
            plan = plan.shifted(-lo_c)
        else:
            g = TILE // self.window_px
            mask = hr.repeat_interleave(H // g, 2).repeat_interleave(H // g, 3)  # (B, T, 16, 16)
        self.last = {**self.last, "percentage": qs,      # (FAST: a 0-d tensor, read when logged)
                     "windows": hs.sum() / hs.numel() if _pkg.FAST else float(hs.sum()) / hs.numel()}
        return Selection(plan, mask)


# -- the warm-up stage's selector ---------------------------------------------------------
class WarmupLattice:
    """The warm-up's candidate windows: a lattice of crop_px windows spread evenly over the
    tile (stride loc_stride), each represented by the ps4 cell under its top-left corner."""

    def __init__(self, crop_px: int, loc_stride: int):
        if loc_stride < crop_px:
            raise ValueError(f"loc_stride {loc_stride} < crop_px {crop_px}")
        self.crop_px = crop_px
        n = (TILE - crop_px) // loc_stride + 1
        tl = sorted({round(i * (TILE - crop_px) / (n - 1)) for i in range(n)})
        cells = [v // LOOKUP_PS for v in tl]
        if len(set(cells)) != len(cells):
            raise ValueError(f"loc_stride {loc_stride}: two windows share an anchor cell")
        self.loc_px = torch.full((H,), -1, dtype=torch.long)
        self.loc_px[cells] = torch.tensor(tl)

    def window_mean(self, px: torch.Tensor) -> torch.Tensor:
        """(N, 64, 64) per-pixel values -> (N, 16, 16): each candidate's mean over its window
        (-inf: not a candidate)."""
        P_ = self.crop_px
        if P_ == LOOKUP_PS:
            return F.avg_pool2d(px[:, None], LOOKUP_PS)[:, 0]
        win = F.avg_pool2d(px[:, None], P_, stride=1)[:, 0]
        lut = self.loc_px.to(px.device)
        c = (lut >= 0).nonzero(as_tuple=True)[0]
        out = px.new_full((px.shape[0], H, H), NEG)
        out[:, c[:, None], c[None, :]] = win[:, lut[c]][:, :, lut[c]]
        return out

    def anchors(self, device) -> torch.Tensor:
        a = self.loc_px.to(device) >= 0
        return a[:, None] & a[None, :]


class WarmupSelector:
    """The warm-up: the top n candidate windows of each sample, every present month, scored by
    the coarse prediction's label loss (scorer "oracle", the location oracle) or at random."""

    def __init__(self, geometry: WarmupLattice, scorer: str, n: int):
        if scorer not in ("oracle", "random"):
            raise ValueError(f"warm-up scorer {scorer!r}: oracle | random")
        self.geo, self.kind, self.n = geometry, scorer, n

    def modules(self) -> list:
        return []

    def window_scores(self, ctx) -> torch.Tensor:
        if self.kind == "oracle":
            if ctx.target is None:
                raise ValueError("warm-up oracle: no target in the context")
            out0 = ctx.out0             # (the coarse pass runs outside autocast)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.geo.window_mean(pixel_label_loss(out0.prediction, ctx.target))
        s = torch.rand(ctx.B, ctx.T, H, H, device=ctx.present.device)
        s = s.masked_fill(~self.geo.anchors(s.device), NEG)
        where = ctx.present[:, :, None, None].expand_as(s)
        return s.masked_fill(~where, NEG).amax(1)

    def __call__(self, ctx: CoarsePass) -> Selection:
        where = ctx.present[:, :, None, None].expand(ctx.B, ctx.T, H, H)
        mask = topk_mask(self.window_scores(ctx)[:, None], max(1, self.n)) & where
        return Selection(Windows.from_mask(mask, self.geo.crop_px, self.geo.loc_px), mask)


def build_selector(cfg, window_px: int, model: MiSTR) -> WindowSelector:
    """config.SelectConfig -> the method's window selector."""
    return WindowSelector(window_px, cfg, model.cfg.coarse_patch, model.device)


def build_warmup(kind: str, window_px: int, n: int) -> WarmupSelector:
    """The warm-up selector: n windows of a window_px lattice (stride window_px) per sample,
    by the label oracle (kind "oracle") or at random."""
    # (2 px windows: one per 4 px cell, the lattice's anchor grid)
    return WarmupSelector(WarmupLattice(window_px, max(window_px, LOOKUP_PS)), kind, n)
