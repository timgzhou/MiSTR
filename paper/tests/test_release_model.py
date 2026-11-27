"""The public package (../mistr) reproduces the paper code bit for bit: each released checkpoint,
exported (tools.export_pretrained), on the demo tiles at three budgets (and 1 / 1.5 / 2 seasons) -> the same windows, the
same coarse and final logits. Needs a GPU, the checkpoint and demo/pastis_test_samples.npz."""
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

CKS = [Path("checkpoints/mistr_pastis_S8Tsea+S4Tmon_q50_woracle50_ep100_seed0_9a7509fd.pt"),
       Path("checkpoints/mistr_pastis_S6Tsea+S3Tmon-w6_q50_xset0-0.5-1_xdis_woracle50_ep100_seed1_5dd30884.pt")]
SAMPLES = Path("demo/pastis_test_samples.npz")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and SAMPLES.exists()),
                                reason="needs a GPU and the demo samples")


@pytest.mark.parametrize("CK", CKS, ids=["S8+S4", "S6+S3"])
def test_release_model_matches_paper_code(tmp_path, CK):
    if not CK.exists():
        pytest.skip(f"{CK} not here")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repo root: mistr
    from mistr import Inputs, MiSTR
    from exp.demo import load_pretrained, load_samples, predict
    from mistr_paper.data.datasets.pastis import CLASS_NAMES
    from tools.export_pretrained import export

    _, _, old, sel = load_pretrained(CK)
    new = MiSTR.from_pretrained(str(export(str(CK), str(tmp_path / "m"), list(CLASS_NAMES[:19]))))
    batch = load_samples(SAMPLES)
    x = Inputs(batch.ms, batch.ss, batch.season, batch.present)
    for seasons, q in [(s, q) for s in ((1, 1.5, 2) if new.extra else (1,)) for q in (0.5, 0.25, 0.1)]:
        sel.cfg = dataclasses.replace(sel.cfg, percentage=q, extra_steps=seasons - 1.0)
        a, b = predict(old, sel, batch), new.predict_tiles(x, budget=q, seasons=seasons)
        p = a["plan"]
        assert torch.equal(torch.stack([p.b, p.t, p.py, p.px], 1).cpu(), b.windows)
        assert torch.equal(a["coarse_logits"], b.coarse_logits)
        assert torch.equal(a["logits"], b.logits)
