"""mistr_paper.core.encoder: the mixed-resolution OlmoEarth pass reproduces OlmoEarth where it should."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _setup():
    from exp.common import olmo_bootstrap
    olmo_bootstrap.apply()
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    from exp.common.config import MODEL_SIZE_TO_ID
    from mistr_paper.encoders.olmoearth import OlmoEarthEncoder, OlmoEarthInput, _to_device
    from mistr_paper.core.encoder import OneWayEncoder

    enc = OlmoEarthEncoder(MODEL_SIZE_TO_ID["tiny"], OlmoEarthInput({"S1": "sentinel1"}),
                           16, 64, name="tiny").cuda().eval()
    g = torch.Generator().manual_seed(0)
    B, S, T = 2, 64, 4
    d = {"timestamps": torch.tensor([[[15, m, 2020] for m in (0, 3, 6, 9)]] * B)}
    for name in ("sentinel2_l2a", "sentinel1"):
        C = len(Modality.get(name).band_order)
        d[name] = torch.randn(B, S, S, T, C, generator=g)
        mask = torch.full_like(d[name], MaskValue.ONLINE_ENCODER.value)
        if name == "sentinel2_l2a":
            mask[1, :, :, 2] = MaskValue.MISSING.value            # one missing S2 frame
        d[name + "_mask"] = mask
    sample = _to_device(MaskedOlmoEarthSample(**d), "cuda", torch.bfloat16)
    # lr_ps = the crops' 4 px: a crop's centre offset is 0, as on its own grid
    return enc, OneWayEncoder(enc.encoder, 0, "center", 4).cuda().eval(), sample


@torch.no_grad()
def test_lr_only_pass_is_olmoearths_series_encoding():
    enc, mix, sample = _setup()
    ref = enc(sample, "series").float()                                        # (B,T,4,4,D)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        x, _ = mix.tail(lr, None, None)
    out = lr.pooled(x).float()
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@torch.no_grad()
def test_crop_is_encoded_at_its_true_position_and_slot():
    """The encodings a crop receives are those of its location on the full ps4 grid. (Its
    patch embedding differs near the crop border: the FlexiViT embedding resizes the image
    bicubically, so the full image's patches see their neighbours.)"""
    from mistr_paper.encoders.olmoearth import _slice_time, crop_patches
    _, mix, sample = _setup()
    t0 = 1
    one = _slice_time(sample, t0)                                              # T = 1
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1], [t0, t0], [3, 15], [7, 0]))
    crops = crop_patches(sample, b, t, y, x, 4)

    def encodings(s, **kw):                         # embed() minus the raw patch embedding
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = mix.embed(s, 4, **kw)
            raw = mix.encoder.patch_embeddings.forward(s, 4)
        raw = torch.cat([raw[m].flatten(1, -2) for m in p.dims], 1)[p.keep]
        return p, p.x.float() - raw.float()

    full, full_e = encodings(one)                                              # 16 x 16 grid
    _, crop_e = encodings(crops, slot=torch.zeros(2, dtype=torch.long, device="cuda"),
                          py=4 * y, px=4 * x)
    full_x = torch.zeros(*full.keep.shape, full_e.shape[-1], device="cuda")
    full_x[full.keep] = full_e
    per_mod, i = {}, 0
    for m, shape in full.dims.items():                                         # (B,16,16,1,S,D)
        n = shape[1] * shape[2] * shape[3] * shape[4]
        per_mod[m] = full_x[:, i:i + n].reshape(shape[0], 16, 16, shape[4], -1)
        i += n
    want = torch.cat([torch.cat([per_mod[m][bb, yy, xx] for m in full.dims])
                      for bb, yy, xx in zip(b.tolist(), y.tolist(), x.tolist())])
    torch.testing.assert_close(crop_e, want, atol=1e-2, rtol=1e-2)   # bf16 rounding
    # and a different position or slot is a different encoding
    _, moved = encodings(crops, slot=torch.ones(2, dtype=torch.long, device="cuda"),
                         py=4 * y, px=4 * x + 1)
    assert (moved - want).abs().amax() > 0.1


@torch.no_grad()
def test_injected_crops_only_touch_their_own_sample():
    from mistr_paper.encoders.olmoearth import crop_patches
    _, mix, sample = _setup()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        base, _ = mix.tail(lr, None, None)
        b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 0, 0], [0, 1, 3], [0, 5, 9],
                                                             [1, 5, 2]))
        hr = mix.head_blocks(mix.embed(crop_patches(sample, b, t, y, x, 4), 4,
                                       slot=t, py=4 * y, px=4 * x))
        with_hr, hr_out = mix.tail(lr, hr, b)
    n0 = int(lr.seqlens[0])
    assert hr_out.shape == hr.x.shape
    assert not torch.allclose(with_hr[:n0], base[:n0])                        # sample 0 moved
    torch.testing.assert_close(with_hr[n0:], base[n0:], atol=1e-2, rtol=1e-2)  # sample 1 not










def _splat_inputs(T=4, G=2, D=6):
    lr = torch.randn(1, G, 4, 4, D)
    season = torch.tensor([[0, 0, 1, 1]])[:, :T]
    present = torch.ones(1, T, dtype=torch.bool)
    hr = torch.full((1, D), 50.0)                                   # one crop, far from lr values
    idx = tuple(torch.tensor([v]) for v in (0, 1, 5, 9))            # (b, t, y, x)
    return lr, torch.ones(1, G, dtype=torch.bool), season, present, hr, idx








def test_seasons_used_ignores_padded_steps():
    from mistr_paper.core.seasons import seasons_used
    season = torch.tensor([[0, 0, 1, 0, -1]])               # steps 3, 4 are batch padding
    present = torch.tensor([[True, True, True, False, False]])
    assert seasons_used(season, present, 3)[0].tolist() == [True, True, False]



def _cellattn_inputs(B=2, NS=4, D=16):
    lr = torch.randn(B, NS, 4, 4, D)
    lr_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    b, t = torch.tensor([0, 0, 1]), torch.tensor([1, 5, 2])
    y, x = torch.tensor([3, 3, 15]), torch.tensor([7, 8, 0])
    return lr, lr_valid, torch.randn(3, D), (b, t, y, x)






@torch.no_grad()
def test_tail_multi_equals_tail_on_the_union_of_crop_sets():
    from mistr_paper.encoders.olmoearth import crop_patches
    from mistr_paper.core.encoder import Packed
    _, mix, sample = _setup()
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1, 0, 1], [0, 1, 3, 2],
                                                         [0, 5, 9, 1], [1, 5, 2, 7]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lr = mix.head_blocks(mix.embed(sample, 16))
        hr = mix.head_blocks(mix.embed(crop_patches(sample, b, t, y, x, 4), 4,
                                       slot=t, py=4 * y, px=4 * x))
        lr_one, hr_one = mix.tail(lr, hr, b)
        # the same crops as two sets: crops 0-1 and 2-3
        cut = int(hr.seqlens[:2].sum())
        sets = [(Packed(hr.x[:cut], hr.seqlens[:2], hr.keep[:2], hr.dims, hr.masks), b[:2]),
                (Packed(hr.x[cut:], hr.seqlens[2:], hr.keep[2:], hr.dims, hr.masks), b[2:])]
        lr_two, (h1, h2) = mix.tail_multi(lr, sets)
    # same tokens per sequence, in a different order inside it: equal up to bf16 attention noise
    torch.testing.assert_close(lr_two, lr_one, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(torch.cat([h1, h2]), hr_one, atol=3e-2, rtol=3e-2)


def test_within_places_a_step_inside_its_parent_step():
    from mistr_paper.core.data_cpu import within
    ts, te = torch.tensor([[0, 31, 61]]), torch.tensor([[30, 60, 91]])      # 3 months
    f = within(ts, te, torch.tensor([[0]]), torch.tensor([[91]]), torch.zeros(1, 3, dtype=torch.long))
    assert f[0, 0] < -0.3 and abs(f[0, 1]) < 0.02 and f[0, 2] > 0.3       # early / mid / late
    assert within(ts, te, torch.tensor([[0]]), torch.tensor([[91]]),
                  torch.tensor([[0, -1, 0]]))[0, 1] == 0                  # no parent -> 0












@pytest.mark.parametrize("ps", [8, 16])
@torch.no_grad()
def test_ps8_ps16_crops_are_encoded_at_their_true_position(ps):
    """As for ps4: a ps8 / ps16 crop's encodings are those of its place on the full ps8 / ps16
    grid of its frame."""
    from mistr_paper.encoders.olmoearth import _slice_time, crop_patches
    _, mix, sample = _setup()
    mix.lr_ps = ps          # (lr_ps = the crop patch: the centre offset is 0, as on its own grid)
    g = 64 // ps
    one = _slice_time(sample, 1)
    b, t, y, x = (torch.tensor(v, device="cuda") for v in ([0, 1], [1, 1], [g - 1, 0], [1, g - 1]))
    crops = crop_patches(sample, b, t, y, x, ps)

    def encodings(s, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = mix.embed(s, ps, **kw)
            raw = mix.encoder.patch_embeddings.forward(s, ps)
        raw = torch.cat([raw[m].flatten(1, -2) for m in p.dims], 1)[p.keep]
        return p, p.x.float() - raw.float()

    full, full_e = encodings(one)
    _, crop_e = encodings(crops, slot=torch.zeros(2, dtype=torch.long, device="cuda"),
                          py=ps * y, px=ps * x)
    full_x = torch.zeros(*full.keep.shape, full_e.shape[-1], device="cuda")
    full_x[full.keep] = full_e
    per_mod, i = {}, 0
    for m, shape in full.dims.items():
        n = shape[1] * shape[2] * shape[3] * shape[4]
        per_mod[m] = full_x[:, i:i + n].reshape(shape[0], g, g, shape[4], -1)
        i += n
    want = torch.cat([torch.cat([per_mod[m][bb, yy, xx] for m in full.dims])
                      for bb, yy, xx in zip(b.tolist(), y.tolist(), x.tolist())])
    torch.testing.assert_close(crop_e, want, atol=1e-2, rtol=1e-2)


def test_native_set_readout_footprints():
    """A ps8 crop (footprint 2x2 ps4 cells) and a ps16 crop (4x4) reach exactly the cells of
    their footprint, each cell getting its 4x4 px part of the crop's head map pooled to the
    footprint size."""
    import torch.nn.functional as F
    from mistr_paper.core.decoder import AttentionPoolDecoder
    g = torch.Generator().manual_seed(4)
    B, NS, h, T, D, C = 1, 2, 4, 4, 8, 2
    lin = torch.nn.Linear(D, C * 16 * 16)
    ro = AttentionPoolDecoder(D)
    ro.score = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(D, 1))
    with torch.no_grad():                     # score = feature channel 0: crops dominate
        ro.score[1].weight.zero_()
        ro.score[1].bias.zero_()
        ro.score[1].weight[0, 0] = 1.0
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    lr_F[..., 0] = 0.0
    season, present = torch.tensor([[0, 0, 1, 1]]), torch.ones(1, T, dtype=torch.bool)
    tok = torch.randn(2, D, generator=g)
    tok[:, 0] = 100.0
    idx = (torch.tensor([0, 0]), torch.tensor([0, 2]), torch.tensor([2, 8]), torch.tensor([6, 0]))
    size = torch.tensor([2, 4])                 # ps8 at cells (2..3, 6..7); ps16 at (8..11, 0..3)
    with torch.no_grad():
        base = ro(lin, lr_F, season, present, tok[:0], tuple(v[:0] for v in idx))
        got = ro(lin, lr_F, season, present, tok, idx, None, size)
        m8 = F.avg_pool2d(lin(tok[0]).view(1, C, 16, 16), 2)[0]              # (C, 8, 8)
        m16 = lin(tok[1]).view(C, 16, 16)
    # ps8 footprint: pixels 8..15 rows, 24..31 cols; ps16: rows 32..47, cols 0..15
    assert torch.allclose(got[0, :, 8:16, 24:32], m8, atol=1e-4)
    assert torch.allclose(got[0, :, 32:48, 0:16], m16, atol=1e-4)
    outside = torch.ones(64, 64, dtype=torch.bool)
    outside[8:16, 24:32] = False
    outside[32:48, 0:16] = False
    assert torch.allclose(got[0][:, outside], base[0][:, outside], atol=1e-6)


def test_native_set_leave_one_out_equals_rerunning_without_the_crop():
    """--joint's in-context value: removing crop j's weight from its cell's softmax pool,
    (out - w_j L_j) / (1 - w_j), equals the readout re-run without crop j."""
    from einops import rearrange
    from mistr_paper.core.decoder import AttentionPoolDecoder
    g = torch.Generator().manual_seed(5)
    B, NS, h, T, D, C = 2, 4, 4, 6, 8, 3
    lin = torch.nn.Linear(D, C * 16 * 16)
    ro = AttentionPoolDecoder(D)
    with torch.no_grad():
        ro.score[1].weight.normal_(generator=g)
    lr_F = torch.randn(B, NS, h, h, D, generator=g)
    season = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, -1, -1]])
    present = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool()
    b, t = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 5, 2])
    y, x = torch.tensor([6, 6, 2, 9]), torch.tensor([9, 9, 3, 1])       # two crops share a cell
    tok = torch.randn(4, D, generator=g)
    with torch.no_grad():
        _, (wgt, cand, out, idx) = ro(lin, lr_F, season, present, tok, (b, t, y, x), parts=True)
        for j in range(4):
            q = y[j] * 16 + x[j]
            m = (idx[b[j], q] == j).nonzero()[0, 0]
            w, L = wgt[b[j], q, NS + m], cand[b[j], q, NS + m]
            loo = (out[b[j], q] - w * L) / (1 - w)
            keep = torch.arange(4) != j
            ref = ro(lin, lr_F, season, present, tok[keep], tuple(v[keep] for v in (b, t, y, x)))
            ref = rearrange(ref, "b k (y i) (x j) -> b (y x) k (i j)", i=4, j=4)[b[j], q]
            assert torch.allclose(loo, ref, atol=1e-5)


def _time_setup(seed=6):
    g = torch.Generator().manual_seed(seed)
    B, NS, h, T, D = 1, 3, 4, 6, 8
    lr = torch.randn(B, NS, h, h, D, generator=g)
    lr_valid = torch.ones(B, NS, dtype=torch.bool)
    season = torch.tensor([[0, 0, 1, 1, 2, 2]])
    present = torch.ones(B, T, dtype=torch.bool)
    frac = torch.tensor([[-0.25, 0.25, -0.25, 0.25, -0.25, 0.25]])
    return g, lr, lr_valid, season, present, frac, D




















def test_native_set_offgrid_matches_grid():
    """AttentionPoolDecoder with pixel positions (--jitter): on-grid pixels reproduce the per-cell
    pool exactly; an off-grid crop only changes the pixels it covers."""
    import torch
    from torch import nn
    from mistr_paper.core.decoder import AttentionPoolDecoder
    torch.manual_seed(0)
    B, NS, D, C, N = 2, 4, 8, 3, 40
    ro = AttentionPoolDecoder(D)
    nn.init.normal_(ro.score[1].weight)
    lin = nn.Linear(D, C * 256)
    lr_F = torch.randn(B, NS, 4, 4, D)
    season = torch.arange(12)[None].expand(B, 12) // 3
    present = torch.ones(B, 12, dtype=torch.bool)
    hr = torch.randn(N, D)
    b, t = torch.randint(0, B, (N,)), torch.randint(0, 12, (N,))
    y, x = torch.randint(0, 16, (N,)), torch.randint(0, 16, (N,))
    ref, (w0, _, o0, _) = ro(lin, lr_F, season, present, hr, (b, t, y, x), parts=True)
    out, (w1, _, o1, cid) = ro(lin, lr_F, season, present, hr, (b, t, y, x), parts=True,
                               hr_px=(y * 4, x * 4))
    assert torch.allclose(ref, out, atol=1e-5) and torch.allclose(o0, o1, atol=1e-5)
    assert w1.ndim == 4 and torch.allclose(w1[..., 0], w0, atol=1e-6)
    # one crop, 2 px right and 1 px down of cell (3, 5) of sample 0
    one = (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]), torch.tensor([5]))
    base = ro(lin, lr_F, season, present, hr[:0], tuple(v[:0] for v in one))
    off = ro(lin, lr_F, season, present, hr[:1], one, torch.tensor([10.0]),      # a heavy crop
             hr_px=(torch.tensor([13]), torch.tensor([22])))
    changed = (off - base).abs().sum(1) > 1e-3                       # (B, 64, 64)
    want = torch.zeros_like(changed)
    want[0, 13:17, 22:26] = True
    assert (changed == want).all()


def test_native_set_ps16_window_matches_grid():
    """hr_fp = 16: a ps16 crop given by its top-left pixel == the same crop as a 4 x 4-cell
    footprint on the grid (hr_size 4)."""
    import torch
    from torch import nn
    from mistr_paper.core.decoder import AttentionPoolDecoder
    torch.manual_seed(1)
    B, NS, D, C, N = 2, 4, 8, 3, 30
    ro = AttentionPoolDecoder(D)
    nn.init.normal_(ro.score[1].weight)
    lin = nn.Linear(D, C * 256)
    lr_F = torch.randn(B, NS, 4, 4, D)
    season = torch.arange(12)[None].expand(B, 12) // 3
    present = torch.ones(B, 12, dtype=torch.bool)
    hr = torch.randn(N, D)
    b, t = torch.randint(0, B, (N,)), torch.randint(0, 12, (N,))
    y, x = torch.randint(0, 13, (N,)), torch.randint(0, 13, (N,))        # top-left ps4 cell
    ref = ro(lin, lr_F, season, present, hr, (b, t, y, x), hr_size=torch.full((N,), 4))
    out = ro(lin, lr_F, season, present, hr, (b, t, y, x), hr_px=(y * 4, x * 4), hr_fp=16)
    assert torch.allclose(ref, out, atol=1e-5)


def _apt_fixture(noisy_cell=(0, 0), B=2, T=6, H=16):
    """A MaskedOlmoEarthSample whose sentinel2_l2a is flat everywhere except one ps4 cell,
    which is spatially noisy AND (sample 1) also varies a lot across months."""
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

    g = torch.Generator().manual_seed(0)
    S, C = H * 4, len(Modality.get("sentinel2_l2a").band_order)
    s2 = torch.zeros(B, S, S, T, C)
    cy, cx = noisy_cell
    s2[:, cy * 4:cy * 4 + 4, cx * 4:cx * 4 + 4] = torch.randn(B, 4, 4, T, C, generator=g)
    ms = MaskedOlmoEarthSample(timestamps=torch.zeros(B, T, 3, dtype=torch.long),
                               sentinel2_l2a=s2)
    return ms








































