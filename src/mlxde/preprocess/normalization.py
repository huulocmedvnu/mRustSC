"""Per-sample size factor estimators.

Pure NumPy: normalisation is a single O(n_genes * n_samples) memory-bound pass,
so moving the matrix to a GPU would cost more than the arithmetic it saves.
"""

from __future__ import annotations

import numpy as np


def _validate_counts(counts: np.ndarray) -> np.ndarray:
    """Return ``counts`` as float, rejecting shapes and values we cannot scale."""
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-dimensional, got shape {counts.shape}")
    if counts.size == 0:
        raise ValueError("counts must contain at least one gene and one sample")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    return counts.astype(np.float64, copy=False)


def _normalise_to_unit_geometric_mean(factors: np.ndarray) -> np.ndarray:
    """Rescale strictly positive factors so their geometric mean is exactly 1.

    Shared by every estimator: the contract fixes the scale of size factors, but
    not how they are derived, so the constraint belongs in one place.
    """
    if np.any(factors <= 0) or not np.all(np.isfinite(factors)):
        raise ValueError("size factors must be finite and strictly positive")
    return factors / np.exp(np.mean(np.log(factors)))


class MedianOfRatiosSizeFactors:
    """DESeq2-style size factors: median ratio to a per-gene reference sample."""

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        counts = _validate_counts(counts)

        # Zero counts make the geometric-mean reference collapse to zero, so the
        # reference is built from genes expressed in every sample.
        usable = np.all(counts > 0, axis=1)
        if not np.any(usable):
            raise ValueError(
                "median-of-ratios needs at least one gene with a non-zero count in every "
                "sample; none qualified"
            )

        log_usable = np.log(counts[usable])
        log_reference = np.mean(log_usable, axis=1, keepdims=True)
        factors = np.exp(np.median(log_usable - log_reference, axis=0))
        return _normalise_to_unit_geometric_mean(factors)


class TotalCountSizeFactors:
    """Library-size size factors: each sample's total count, rescaled."""

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        counts = _validate_counts(counts)
        library_sizes = np.sum(counts, axis=0)
        if np.any(library_sizes <= 0):
            raise ValueError("every sample must have a strictly positive total count")
        return _normalise_to_unit_geometric_mean(library_sizes)
