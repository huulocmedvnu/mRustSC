"""Negative binomial dispersion estimation.

The variance model is ``Var = mu + alpha * mu^2``: ``alpha`` is the per-gene
dispersion, the excess of biological variability over Poisson noise. Estimates
are produced by moments (fast, closed form) and may optionally be shrunk towards
a mean-dispersion trend, which is what makes small-sample designs usable.
"""

from __future__ import annotations

import numpy as np

from mlxde.contracts import Array, ComputeBackend, DesignMatrix, DispersionEstimator

MAXIMUM_DISPERSION = 1.0e3
"""Upper clamp.

A gene with alpha = 1000 already has a variance a thousand times its squared
mean; larger values are indistinguishable in likelihood terms but make the
downstream IRLS weights ``1 / (1/mu + alpha)`` underflow, so they are capped
rather than propagated.
"""

_MINIMUM_MEAN = 1.0e-8
"""Guard against dividing by the mean of an all-zero gene."""


def _normalised_counts(
    backend: ComputeBackend, counts: np.ndarray, size_factors: np.ndarray
) -> Array:
    """Counts on a common library-size scale, as a backend array."""
    return backend.asarray(counts) / backend.asarray(size_factors)


def _validate_shapes(counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix) -> None:
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-dimensional, got shape {counts.shape}")
    n_samples = counts.shape[1]
    if size_factors.shape != (n_samples,):
        raise ValueError(f"expected {n_samples} size factors, got shape {size_factors.shape}")
    if design.n_samples != n_samples:
        raise ValueError(f"design has {design.n_samples} rows, counts have {n_samples} columns")


class MethodOfMomentsDispersion:
    """Closed-form dispersion from the residual variance of normalised counts.

    The sample variance is taken about the fit of the design matrix rather than
    about the grand mean, so a gene that differs between conditions is not
    charged for that difference, and it is scaled by the residual degrees of
    freedom ``n_samples - n_coefficients`` to stay unbiased.
    """

    def __init__(self, backend: ComputeBackend, minimum: float = 1e-8) -> None:
        if minimum <= 0.0:
            raise ValueError(f"minimum must be strictly positive, got {minimum}")
        self._backend = backend
        self._minimum = minimum

    def estimate(
        self, counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix
    ) -> np.ndarray:
        _validate_shapes(counts, size_factors, design)
        backend = self._backend

        normalised = _normalised_counts(backend, counts, size_factors)
        residual_variance = self._residual_variance(normalised, design)

        mean = backend.mean(normalised, axis=1)
        safe_mean = backend.maximum(mean, _MINIMUM_MEAN)
        dispersion = (residual_variance - mean) / (safe_mean * safe_mean)

        clamped = backend.clip(dispersion, self._minimum, MAXIMUM_DISPERSION)
        return np.asarray(backend.asnumpy(clamped), dtype=np.float64)

    def _residual_variance(self, normalised: Array, design: DesignMatrix) -> Array:
        """Per-gene variance about the least-squares fit of the design."""
        backend = self._backend
        model = backend.asarray(design.matrix)
        transposed_model = backend.transpose(model)

        # Ordinary least squares per gene, batched as a single (p, n_genes) solve.
        coefficients = backend.solve(
            backend.matmul(transposed_model, model),
            backend.matmul(transposed_model, backend.transpose(normalised)),
        )
        residuals = normalised - backend.transpose(backend.matmul(model, coefficients))

        # A saturated design leaves no residual information; one degree of
        # freedom keeps the estimate finite (and, with zero residuals, minimal).
        degrees_of_freedom = max(design.n_samples - design.n_coefficients, 1)
        return backend.sum(residuals * residuals, axis=1) / degrees_of_freedom


class TrendedDispersion:
    """Shrink any dispersion estimate towards a fitted mean-dispersion trend.

    A decorator over an arbitrary :class:`~mlxde.contracts.DispersionEstimator`:
    it consumes only the numbers the base estimator returns, so it composes with
    every present and future estimator without knowing how they work.
    """

    def __init__(self, base: DispersionEstimator, shrinkage_weight: float = 0.5) -> None:
        if not 0.0 <= shrinkage_weight <= 1.0:
            raise ValueError(f"shrinkage_weight must lie in [0, 1], got {shrinkage_weight}")
        self._base = base
        self._shrinkage_weight = shrinkage_weight

    def estimate(
        self, counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix
    ) -> np.ndarray:
        base_dispersion = np.asarray(
            self._base.estimate(counts, size_factors, design), dtype=np.float64
        )
        if self._shrinkage_weight == 0.0:
            return base_dispersion

        mean = np.mean(np.asarray(counts, dtype=np.float64) / size_factors, axis=1)
        trend = self._fit_trend(mean, base_dispersion)

        # Geometric blend: dispersions are positive and roughly log-normal, so
        # interpolating in log space keeps the result positive and unbiased.
        weight = self._shrinkage_weight
        return np.exp((1.0 - weight) * np.log(base_dispersion) + weight * np.log(trend))

    @staticmethod
    def _fit_trend(mean: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
        """Least-squares line of log(dispersion) on log(mean).

        Preferred over the DESeq2 asymptotic form ``a / mu + b`` because it is a
        single unconstrained linear fit — no positivity constraints, no iteration
        — and it captures the same monotone decay of dispersion with expression.
        """
        usable = (mean > _MINIMUM_MEAN) & np.isfinite(dispersion) & (dispersion > 0.0)
        if usable.sum() < 2:
            return dispersion.copy()  # nothing to fit; fall back to the gene-wise values

        log_mean = np.log(mean[usable])
        if np.ptp(log_mean) == 0.0:  # a degenerate fit would be rank-deficient
            constant = float(np.exp(np.mean(np.log(dispersion[usable]))))
            return np.full_like(dispersion, constant)

        slope, intercept = np.polyfit(log_mean, np.log(dispersion[usable]), deg=1)
        trend = np.exp(intercept + slope * np.log(np.maximum(mean, _MINIMUM_MEAN)))
        return np.clip(trend, np.min(dispersion), MAXIMUM_DISPERSION)
