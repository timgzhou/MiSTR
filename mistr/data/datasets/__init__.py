"""Concrete MTMM datasets, one module each."""
from mistr.data.datasets.biomassters import BiomasstersDataset
from mistr.data.datasets.geoidflood import ChipSelection, GeoidFloodDataset
from mistr.data.datasets.pastis import PastisDataset
from mistr.data.datasets.planteur import PlanteurDataset
from mistr.data.datasets.yieldsat import YieldSat24Dataset, YieldSatDataset

__all__ = ["BiomasstersDataset", "ChipSelection", "GeoidFloodDataset",
           "PastisDataset", "PlanteurDataset", "YieldSat24Dataset", "YieldSatDataset"]
