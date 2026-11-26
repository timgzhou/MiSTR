"""Concrete MTMM datasets, one module each."""
from mistr_paper.data.datasets.biomassters import BiomasstersDataset
from mistr_paper.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from mistr_paper.data.datasets.pastis import PastisDataset
from mistr_paper.data.datasets.planteur import PlanteurDataset
from mistr_paper.data.datasets.yieldsat import YieldSat24Dataset, YieldSatDataset

__all__ = ["BiomasstersDataset", "ChipSelection", "GeoidFloodDataset",
           "PastisDataset", "PlanteurDataset", "YieldSat24Dataset", "YieldSatDataset"]
