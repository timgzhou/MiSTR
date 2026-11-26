"""GeoidFloodDataset: conformance and chip geometry/time facts."""

import pytest

from mistr.data import All
from mistr.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset, DEFAULT_ROOT
from tests.mtmm_conformance import check_dataset

SEL = ChipSelection(128)
INDEX = DEFAULT_ROOT / f"chip_index_{SEL.key}.json"
pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="GEOID chip index not built")


def make(split, modalities, tiling):
    assert tiling is None
    return GeoidFloodDataset(split, modalities, SEL)


def test_conformance():
    ds = make("val", {"S1GRD": All()}, None)
    check_dataset(make, ds.SPEC, None, None)


def test_chips_are_windows_of_their_tile_with_real_geometry():
    s = make("val", {"S1GRD": All()}, None)[0]
    stem, rc = s.id.split("@r")
    row, col = map(int, rc.split("c"))
    geo = s.modalities["S1GRD"].geometry
    assert geo.shape == (128, 128) and geo.crs.startswith("EPSG:")
    assert geo.gsd == pytest.approx((10.0, 10.0))
    assert s.target_geometry == geo
    pre, post = s.modalities["S1GRD"].time.start.tolist()
    assert pre < post                                        # pre-event, then post-event


def test_selection_keeps_only_mapped_flooded_chips():
    ds = make("val", {"S1GRD": All()}, None)
    for i in range(0, len(ds), max(1, len(ds) // 20)):
        t = ds[i].target
        assert (t != 255).float().mean() >= 0.5 and (t == 2).any()


def test_unregistered_selection_is_refused():
    with pytest.raises(ValueError, match="declared counts"):
        GeoidFloodDataset("val", {"S1GRD": All()}, ChipSelection(96))
