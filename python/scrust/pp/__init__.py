"""Preprocessing, mirroring `scanpy.pp`.

One module per area of responsibility; this file only re-exports, so adding an
area never means editing the same file twice.
"""

from scrust.pp._basics import (
    filter_cells,
    filter_genes,
    highly_variable_genes,
    log1p,
    neighbors,
    normalize_total,
    pca,
    scale,
)
from scrust.pp._batch import combat, regress_out
from scrust.pp._qc import calculate_qc_metrics, filter_genes_dispersion, normalize_per_cell, sqrt
from scrust.pp._sampling import downsample_counts, sample, subsample

__all__ = [
    "calculate_qc_metrics",
    "combat",
    "downsample_counts",
    "filter_cells",
    "filter_genes",
    "filter_genes_dispersion",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_per_cell",
    "normalize_total",
    "pca",
    "regress_out",
    "sample",
    "scale",
    "sqrt",
    "subsample",
]
