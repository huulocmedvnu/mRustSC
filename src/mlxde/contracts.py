"""Abstractions every layer depends on.

This module is the only import shared across layers. High-level code (pipeline,
CLI) depends on the protocols declared here, never on a concrete implementation,
so a backend or an estimator can be swapped without touching its callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

Array = Any
"""An array owned by a ComputeBackend (``mlx.core.array`` or ``np.ndarray``)."""


@dataclass(frozen=True)
class CountMatrix:
    """Raw gene expression counts plus the metadata needed to model them."""

    counts: np.ndarray
    gene_ids: np.ndarray
    sample_ids: np.ndarray
    sample_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if self.counts.ndim != 2:
            raise ValueError(f"counts must be 2-dimensional, got shape {self.counts.shape}")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative")
        n_genes, n_samples = self.counts.shape
        if len(self.gene_ids) != n_genes:
            raise ValueError(f"expected {n_genes} gene ids, got {len(self.gene_ids)}")
        if len(self.sample_ids) != n_samples:
            raise ValueError(f"expected {n_samples} sample ids, got {len(self.sample_ids)}")
        if len(self.sample_metadata) != n_samples:
            raise ValueError(f"expected {n_samples} metadata rows, got {len(self.sample_metadata)}")

    @property
    def n_genes(self) -> int:
        return self.counts.shape[0]

    @property
    def n_samples(self) -> int:
        return self.counts.shape[1]

    def select_genes(self, mask: np.ndarray) -> CountMatrix:
        """Return a copy keeping only the genes where ``mask`` is True."""
        return CountMatrix(
            counts=self.counts[mask],
            gene_ids=self.gene_ids[mask],
            sample_ids=self.sample_ids,
            sample_metadata=self.sample_metadata,
        )


@dataclass(frozen=True)
class DesignMatrix:
    """Model matrix mapping samples to model coefficients."""

    matrix: np.ndarray
    coefficient_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"design matrix must be 2-dimensional, got shape {self.matrix.shape}")
        if self.matrix.shape[1] != len(self.coefficient_names):
            raise ValueError("coefficient_names must match the number of design columns")

    @property
    def n_samples(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_coefficients(self) -> int:
        return self.matrix.shape[1]

    def contrast(self, coefficient_name: str) -> np.ndarray:
        """Unit contrast vector isolating a single coefficient."""
        if coefficient_name not in self.coefficient_names:
            raise KeyError(
                f"unknown coefficient {coefficient_name!r}; available: {self.coefficient_names}"
            )
        contrast = np.zeros(self.n_coefficients)
        contrast[self.coefficient_names.index(coefficient_name)] = 1.0
        return contrast


@dataclass(frozen=True)
class GLMFit:
    """Per-gene negative binomial GLM fit on the natural-log scale."""

    coefficients: np.ndarray  # (n_genes, n_coefficients)
    covariance: np.ndarray  # (n_genes, n_coefficients, n_coefficients)
    dispersions: np.ndarray  # (n_genes,)
    fitted_means: np.ndarray  # (n_genes, n_samples)
    converged: np.ndarray  # (n_genes,) bool
    n_iterations: int


@dataclass(frozen=True)
class TestStatistics:
    """Outcome of a hypothesis test applied to one contrast of a GLMFit."""

    statistic: np.ndarray
    p_values: np.ndarray
    effect: np.ndarray  # contrast estimate, natural-log scale
    effect_standard_error: np.ndarray


@dataclass(frozen=True)
class DifferentialExpressionResult:
    """Per-gene differential expression table."""

    gene_ids: np.ndarray
    base_mean: np.ndarray
    log2_fold_change: np.ndarray
    log2_fold_change_standard_error: np.ndarray
    statistic: np.ndarray
    p_value: np.ndarray
    adjusted_p_value: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gene_id": self.gene_ids,
                "base_mean": self.base_mean,
                "log2_fold_change": self.log2_fold_change,
                "lfc_standard_error": self.log2_fold_change_standard_error,
                "statistic": self.statistic,
                "p_value": self.p_value,
                "adjusted_p_value": self.adjusted_p_value,
            }
        )

    def significant(
        self, alpha: float = 0.05, min_abs_log2_fold_change: float = 0.0
    ) -> pd.DataFrame:
        table = self.to_dataframe()
        is_significant = (table["adjusted_p_value"] <= alpha) & (
            table["log2_fold_change"].abs() >= min_abs_log2_fold_change
        )
        return table[is_significant].sort_values("adjusted_p_value")


@runtime_checkable
class ArrayOps(Protocol):
    """Element-wise and reduction operations a backend must provide."""

    def asarray(self, values: np.ndarray) -> Array: ...
    def asnumpy(self, values: Array) -> np.ndarray: ...
    def exp(self, values: Array) -> Array: ...
    def log(self, values: Array) -> Array: ...
    def sqrt(self, values: Array) -> Array: ...
    def abs(self, values: Array) -> Array: ...
    def clip(self, values: Array, low: float, high: float) -> Array: ...
    def maximum(self, values: Array, other: Array | float) -> Array: ...
    def where(self, condition: Array, if_true: Array, if_false: Array) -> Array: ...
    def sum(self, values: Array, axis: int | None = None) -> Array: ...
    def mean(self, values: Array, axis: int | None = None) -> Array: ...


@runtime_checkable
class LinearAlgebraOps(Protocol):
    """Matrix operations a backend must provide, batched over the gene axis."""

    def matmul(self, left: Array, right: Array) -> Array: ...
    def transpose(self, values: Array, axes: tuple[int, ...] | None = None) -> Array: ...
    def solve(self, matrices: Array, vectors: Array) -> Array: ...
    def inverse(self, matrices: Array) -> Array: ...


@runtime_checkable
class ComputeBackend(ArrayOps, LinearAlgebraOps, Protocol):
    """A device on which the numerical work runs."""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...


@runtime_checkable
class CountReader(Protocol):
    def read(self, counts_path: Path, metadata_path: Path) -> CountMatrix: ...


@runtime_checkable
class ResultWriter(Protocol):
    def write(self, result: DifferentialExpressionResult, path: Path) -> None: ...


@runtime_checkable
class GeneFilter(Protocol):
    def keep(self, counts: np.ndarray) -> np.ndarray:
        """Boolean mask (n_genes,) of genes worth testing."""
        ...


@runtime_checkable
class SizeFactorEstimator(Protocol):
    def estimate(self, counts: np.ndarray) -> np.ndarray:
        """Per-sample scaling factors (n_samples,), strictly positive."""
        ...


@runtime_checkable
class DispersionEstimator(Protocol):
    def estimate(
        self, counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix
    ) -> np.ndarray:
        """Per-gene negative binomial dispersions (n_genes,), strictly positive."""
        ...


@runtime_checkable
class GLMFitter(Protocol):
    def fit(
        self,
        counts: np.ndarray,
        design: DesignMatrix,
        size_factors: np.ndarray,
        dispersions: np.ndarray,
    ) -> GLMFit: ...


@runtime_checkable
class HypothesisTest(Protocol):
    def test(self, fit: GLMFit, contrast: np.ndarray) -> TestStatistics: ...


@runtime_checkable
class MultipleTestingCorrection(Protocol):
    def adjust(self, p_values: np.ndarray) -> np.ndarray: ...
