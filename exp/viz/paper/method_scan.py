"""Paper material, method figure: run the final method (s16m8: entropy space at the global
threshold + readout-attention time + cross_hr, schedule B) on every PASTIS test tile and rank the
tiles for the method figure: picks concentrated in one half of the tile (one half easy, one
hard), a moderate number of picks, picks spread over seasons, few cloudy months.

    source env_setup/env_olmo.sh && python -m exp.viz.paper.method_scan [--setup s16m8|s8m4]

-> figures/paper/method/scan_<setup>.pt (per-tile statistics) and scan_top_<setup>.png (contact sheet of the
best-ranked tiles without S2 nodata frames: monthly S2, picks per season, entropy, label, predictions)
"""
from exp.common import olmo_bootstrap  # noqa: F401
olmo_bootstrap.apply()

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp.common.tasks import TASKS  # noqa: E402
from mistr.core import MiSTR  # noqa: E402
from mistr.core.config import load  # noqa: E402
from mistr.core.selection import CoarsePass, build_selector, spatial_uncertainty  # noqa: E402

OUT = Path("figures/paper/method")
RUNS = {
    "s16m8": "mixres_pastis_ps16_quadg-skip-qs0.25-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c8_seed0_a11df7da",
    "s8m4": "mixres_pastis_ps8_quadg-skip-qs0.25-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c4_seed0_fbe9cd48",
    # q_s 1/2: half of the windows above the global threshold
    "s16m8_qs0.5": "mixres_pastis_ps16_quadg-skip-qs0.5-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c8_seed0_f0ff2d8e",
    "s8m4_qs0.5": "mixres_pastis_ps8_quadg-skip-qs0.5-qt0.5-hhone-crop-score-xhr-s1o50xhr_n0_c4_seed0_a736def1"}


def build(run: str, example):
    """A finished mixres run: model + selector with their best weights (val threshold table)."""
    ck = torch.load(f"checkpoints/{run}.pt", map_location="cpu", weights_only=False)
    cfg = load(ck["config"])
    model = MiSTR(TASKS[cfg.task], cfg.model, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    selector = build_selector(cfg.select, model, example)
    nn.ModuleList(selector.modules()).load_state_dict(ck["selector"])
    for m in selector.modules():
        m.eval()
    return model, selector, cfg


@torch.no_grad()
def run_batch(model, selector, batch):
    """-> the selection and both passes of one batch (inference, pass 1 reused)."""
    ctx = CoarsePass.from_batch(model, batch)
    sel = selector(ctx)
    plan = sel.plan[0]
    out = model(batch.ms, batch.ss, batch.season, batch.present, sel.plan, None, None,
                ctx.coarse_tokens, ctx.coarse_cache)
    u_s, u_t, top, r = selector.uncertainties(ctx, rank=True)
    ent = spatial_uncertainty(ctx)
    attn = ctx.model.readout0._score(ctx.out0.coarse_features.float())                # (B, NS, 4, 4)
    return ctx, plan, out, dict(ent=ent, u_t=u_t, top=top, attn=attn)
