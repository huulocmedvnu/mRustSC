"""Family-wise and false-discovery-rate corrections for per-gene p-values.

Both corrections are pure sorting and cumulative reductions over a single vector,
so they stay on the CPU in NumPy: a `ComputeBackend` round trip would cost more
than the arithmetic it replaces.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _adjust_finite_only(
    p_values: np.ndarray, correction: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply ``correction`` to the non-NaN p-values, leaving NaN entries as NaN.

    Genes that could not be fitted carry a NaN p-value. They were never tested, so
    counting them would shrink every other gene's adjusted p-value towards
    significance; the effective number of tests is the number of finite entries.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1:
        raise ValueError(f"p_values must be 1-dimensional, got shape {p_values.shape}")

    is_tested = ~np.isnan(p_values)
    tested = p_values[is_tested]
    if np.any((tested < 0.0) | (tested > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")

    adjusted = np.full_like(p_values, np.nan)
    if tested.size:
        adjusted[is_tested] = correction(tested)
    return adjusted


class BenjaminiHochberg:
    """Benjamini-Hochberg step-up control of the false discovery rate."""

    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        return _adjust_finite_only(p_values, self._step_up)

    @staticmethod
    def _step_up(tested: np.ndarray) -> np.ndarray:
        n_tests = tested.size
        descending = np.argsort(tested)[::-1]
        ranks = np.arange(n_tests, 0, -1)

        # Enforce monotonicity from the largest p-value downwards, so a gene is never
        # more significant than a gene ranked above it (ties therefore share a value).
        scaled = n_tests / ranks * tested[descending]
        adjusted_descending = np.minimum.accumulate(scaled)

        adjusted = np.empty(n_tests)
        adjusted[descending] = adjusted_descending
        return np.clip(adjusted, 0.0, 1.0)


class Bonferroni:
    """Bonferroni control of the family-wise error rate."""

    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        return _adjust_finite_only(p_values, lambda tested: np.minimum(tested * tested.size, 1.0))
