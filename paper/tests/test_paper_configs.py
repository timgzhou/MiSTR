"""exp.mistr.paper_configs generates exactly the runs the shipped results come from (by name = config hash)."""
import csv
from pathlib import Path

from exp.mistr.paper_configs import runs


def test_generated_runs_are_the_results_rows():
    want = {r["run"] for f in Path("results").glob("*/mistr.csv") for r in csv.DictReader(open(f))}
    have = [c.name() for c in runs()]
    assert len(have) == len(set(have)), "duplicate runs"
    assert set(have) == want


def test_warm_starts_are_paper_runs():
    names = {c.name() for c in runs()}
    assert all(c.train.init_from in names for c in runs() if c.train.init_from)
