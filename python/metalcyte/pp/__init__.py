"""Preprocessing: quality control, filtering, normalisation, gene selection, PCA and neighbours.

Each area of work has its own module. This file only re-exports their functions, so
adding an area never means editing the same file twice.
"""

from metalcyte._streaming import preprocess_backed
from metalcyte.pp._basics import (
    filter_cells,
    filter_genes,
    highly_variable_genes,
    log1p,
    neighbors,
    normalize_total,
    pca,
    scale,
)
from metalcyte.pp._batch import combat, regress_out
from metalcyte.pp._harmony import harmony_integrate
from metalcyte.pp._qc import calculate_qc_metrics, filter_genes_dispersion, normalize_per_cell, sqrt
from metalcyte.pp._sampling import downsample_counts, sample, subsample

__all__ = [
    "calculate_qc_metrics",
    "combat",
    "downsample_counts",
    "filter_cells",
    "filter_genes",
    "filter_genes_dispersion",
    "harmony_integrate",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_per_cell",
    "normalize_total",
    "pca",
    "preprocess_backed",
    "regress_out",
    "sample",
    "scale",
    "sqrt",
    "subsample",
]
