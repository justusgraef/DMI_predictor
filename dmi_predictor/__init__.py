"""
DMI Predictor - Domain-Motif Interface prediction for protein-protein interactions.

This package provides tools to predict domain-motif interfaces (DMI) in protein
sequences and protein-protein interaction pairs using a trained random forest model.
"""

__version__ = "1.0.0"
__author__ = "Chop Yan Lee, Justus Graef"

from dmi_predictor.config import DMIPredictorConfig
from dmi_predictor.io.fasta_reader import FastaReader
from dmi_predictor.io.ppi_reader import PPIReader

__all__ = [
    "DMIPredictorConfig",
    "FastaReader",
    "PPIReader",
]
