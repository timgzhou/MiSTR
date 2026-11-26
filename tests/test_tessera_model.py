"""exp.tessera.model: the masked forward reproduces Tessera v2's own encode, and masking a
token is the same as leaving it out."""
import pytest
import torch

pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from exp.tessera import model as tm

if not tm.CKPT.exists():
    pytest.skip(f"Tessera v2 weights not at {tm.CKPT}", allow_module_level=True)


@pytest.fixture(scope="module")
def m():
    return tm.load(torch.device("cuda"))


def _seq(P, T, C, dev):
    x = torch.randn(P, T, C, device=dev)
    doy = torch.randint(1, 366, (T,), device=dev).float()
    return x, doy


def _enc(m, x2, d2, v2, x1, d1, v1):
    z2 = tm.pool(*(lambda p: (p.h, p.logits))(tm.coarse_pass(m.s2_backbone, x2, d2, v2)))
    z1 = tm.pool(*(lambda p: (p.h, p.logits))(tm.coarse_pass(m.s1_backbone, x1, d1, v1)))
    return tm.fuse(m, z2, z1)


@torch.no_grad()
def test_matches_reference(m):
    dev = torch.device("cuda")
    x2, d2 = _seq(64, 13, 10, dev); x1, d1 = _seq(64, 9, 2, dev)
    ref = m.encode(torch.cat([x2, d2.expand(64, -1)[..., None]], -1),
                   torch.cat([x1, d1.expand(64, -1)[..., None]], -1))
    got = _enc(m, x2, d2, torch.ones(64, 13, dtype=torch.bool, device=dev),
               x1, d1, torch.ones(64, 9, dtype=torch.bool, device=dev))
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


@torch.no_grad()
def test_mask_equals_drop(m):
    dev = torch.device("cuda")
    x2, d2 = _seq(32, 12, 10, dev); x1, d1 = _seq(32, 8, 2, dev)
    keep2 = torch.ones(12, dtype=torch.bool, device=dev); keep2[[3, 7]] = False
    v2 = keep2.expand(32, -1).clone(); v1 = torch.ones(32, 8, dtype=torch.bool, device=dev)
    a = _enc(m, x2, d2, v2, x1, d1, v1)
    b = _enc(m, x2[:, keep2], d2[keep2], v2[:, keep2], x1, d1, v1)
    assert torch.allclose(a, b, atol=1e-4)


@torch.no_grad()
def test_one_way_no_fine_is_coarse(m):
    """a fine pass with every fine token masked leaves the coarse pool unchanged."""
    dev = torch.device("cuda")
    x, d = _seq(16, 6, 10, dev); xf, df = _seq(16, 4, 10, dev)
    v = torch.ones(16, 6, dtype=torch.bool, device=dev)
    cp = tm.coarse_pass(m.s2_backbone, x, d, v)
    fp = tm.fine_pass(m.s2_backbone, cp, xf, df, torch.zeros(16, 4, dtype=torch.bool, device=dev))
    a = tm.pool(cp.h, cp.logits)
    b = tm.pool(torch.cat([cp.h, fp.h], 1), torch.cat([cp.logits, fp.logits], 1))
    assert torch.allclose(a, b, atol=1e-5)
