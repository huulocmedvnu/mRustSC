"""Tests for the dispersion estimators."""

from __future__ import annotations

import numpy as np
import pytest

from mlxde.backend.numpy_backend import NumpyBackend
from mlxde.contracts import DesignMatrix, DispersionEstimator
from mlxde.stats.dispersion import MAXIMUM_DISPERSION, MethodOfMomentsDispersion, TrendedDispersion
from tests.conftest import make_synthetic_dataset

try:  # the GPU backend lives on another branch and may not exist yet
    from mlxde.backend.mlx_backend import MLXBackend
except ImportError:  # pragma: no cover - depends on the merge order
    MLXBackend = None


def two_group_design(n_samples_per_group: int) -> DesignMatrix:
    is_treated = np.repeat([0.0, 1.0], n_samples_per_group)
    return DesignMatrix(
        matrix=np.column_stack([np.ones(is_treated.size), is_treated]),
        coefficient_names=("intercept", "condition[treated]"),
    )


def negative_binomial_counts(
    n_genes: int, n_samples: int, mean: float, dispersion: float, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(
        n=1.0 / dispersion, p=1.0 / (1.0 + dispersion * mean), size=(n_genes, n_samples)
    ).astype(np.float64)


class ConstantDispersion:
    """Stub proving TrendedDispersion depends only on the protocol."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def estimate(
        self, counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix
    ) -> np.ndarray:
        return self._values


def test_is_a_dispersion_estimator(numpy_backend: NumpyBackend) -> None:
    estimator = MethodOfMomentsDispersion(numpy_backend)
    assert isinstance(estimator, DispersionEstimator)
    assert isinstance(TrendedDispersion(estimator), DispersionEstimator)


def test_recovers_known_dispersion(numpy_backend: NumpyBackend) -> None:
    truth = 0.3
    counts = negative_binomial_counts(n_genes=400, n_samples=12, mean=200.0, dispersion=truth)
    size_factors = np.ones(12)

    estimates = MethodOfMomentsDispersion(numpy_backend).estimate(
        counts, size_factors, two_group_design(6)
    )

    assert 1.0 / 1.5 <= np.median(estimates) / truth <= 1.5


def test_recovers_dispersion_of_synthetic_dataset(numpy_backend: NumpyBackend) -> None:
    dataset = make_synthetic_dataset(n_genes=500, n_samples_per_group=6, dispersion=0.2)

    estimates = MethodOfMomentsDispersion(numpy_backend).estimate(
        dataset.count_matrix.counts, np.ones(dataset.count_matrix.n_samples), dataset.design
    )

    assert 0.2 / 1.5 <= np.median(estimates) <= 0.2 * 1.5


def test_poisson_data_collapses_to_the_minimum(numpy_backend: NumpyBackend) -> None:
    rng = np.random.default_rng(1)
    counts = rng.poisson(lam=100.0, size=(300, 10)).astype(np.float64)
    minimum = 1e-6

    estimates = MethodOfMomentsDispersion(numpy_backend, minimum=minimum).estimate(
        counts, np.ones(10), two_group_design(5)
    )

    assert np.all(estimates > 0.0)
    assert np.median(estimates) < 1e-2
    assert np.min(estimates) == pytest.approx(minimum)


def test_estimates_are_positive_and_finite_with_zeros(numpy_backend: NumpyBackend) -> None:
    counts = np.zeros((4, 6))
    counts[1] = [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    counts[2] = [0.0, 5.0, 0.0, 300.0, 0.0, 12.0]
    counts[3] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    estimates = MethodOfMomentsDispersion(numpy_backend).estimate(
        counts, np.ones(6), two_group_design(3)
    )

    assert np.all(np.isfinite(estimates))
    assert np.all(estimates > 0.0)
    assert np.all(estimates <= MAXIMUM_DISPERSION)


def test_single_sample_per_group_has_no_residual_degrees_of_freedom(
    numpy_backend: NumpyBackend,
) -> None:
    counts = np.array([[10.0, 40.0], [0.0, 0.0], [5.0, 5.0]])

    estimates = MethodOfMomentsDispersion(numpy_backend).estimate(
        counts, np.array([1.0, 2.0]), two_group_design(1)
    )

    assert np.all(np.isfinite(estimates))
    assert np.all(estimates > 0.0)


def test_size_factors_rescale_the_counts(numpy_backend: NumpyBackend) -> None:
    counts = negative_binomial_counts(n_genes=50, n_samples=8, mean=150.0, dispersion=0.25)
    estimator = MethodOfMomentsDispersion(numpy_backend)
    design = two_group_design(4)

    unscaled = estimator.estimate(counts, np.ones(8), design)
    scaled = estimator.estimate(3.0 * counts, np.full(8, 3.0), design)

    np.testing.assert_allclose(scaled, unscaled, rtol=1e-10)


def test_minimum_must_be_positive(numpy_backend: NumpyBackend) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        MethodOfMomentsDispersion(numpy_backend, minimum=0.0)


def test_rejects_mismatched_shapes(numpy_backend: NumpyBackend) -> None:
    estimator = MethodOfMomentsDispersion(numpy_backend)
    counts = np.ones((3, 6))

    with pytest.raises(ValueError, match="size factors"):
        estimator.estimate(counts, np.ones(5), two_group_design(3))
    with pytest.raises(ValueError, match="design"):
        estimator.estimate(counts, np.ones(6), two_group_design(2))


def test_trend_with_zero_weight_reproduces_the_base_exactly() -> None:
    base_values = np.array([0.05, 0.4, 1.2, 0.02])
    trended = TrendedDispersion(ConstantDispersion(base_values), shrinkage_weight=0.0)

    estimates = trended.estimate(np.ones((4, 6)) * 10.0, np.ones(6), two_group_design(3))

    np.testing.assert_array_equal(estimates, base_values)


def test_trend_interpolates_between_gene_wise_and_trend(numpy_backend: NumpyBackend) -> None:
    dataset = make_synthetic_dataset(n_genes=300, n_samples_per_group=5, dispersion=0.2)
    counts = dataset.count_matrix.counts
    size_factors = np.ones(dataset.count_matrix.n_samples)
    base = MethodOfMomentsDispersion(numpy_backend)

    gene_wise = base.estimate(counts, size_factors, dataset.design)
    pure_trend = TrendedDispersion(base, shrinkage_weight=1.0).estimate(
        counts, size_factors, dataset.design
    )
    half = TrendedDispersion(base, shrinkage_weight=0.5).estimate(
        counts, size_factors, dataset.design
    )

    assert np.all(half > 0.0)
    np.testing.assert_allclose(half, np.sqrt(gene_wise * pure_trend), rtol=1e-10)
    lower = np.minimum(gene_wise, pure_trend)
    upper = np.maximum(gene_wise, pure_trend)
    assert np.all((half >= lower - 1e-12) & (half <= upper + 1e-12))


def test_pure_trend_is_smoother_than_the_gene_wise_estimate(numpy_backend: NumpyBackend) -> None:
    dataset = make_synthetic_dataset(n_genes=300, n_samples_per_group=5, dispersion=0.2)
    counts = dataset.count_matrix.counts
    size_factors = np.ones(dataset.count_matrix.n_samples)
    base = MethodOfMomentsDispersion(numpy_backend)

    gene_wise = base.estimate(counts, size_factors, dataset.design)
    pure_trend = TrendedDispersion(base, shrinkage_weight=1.0).estimate(
        counts, size_factors, dataset.design
    )

    assert np.std(np.log(pure_trend)) < np.std(np.log(gene_wise))


def test_trend_works_with_a_degenerate_base_estimate() -> None:
    constant = np.full(5, 0.25)
    trended = TrendedDispersion(ConstantDispersion(constant), shrinkage_weight=1.0)

    estimates = trended.estimate(np.full((5, 4), 20.0), np.ones(4), two_group_design(2))

    np.testing.assert_allclose(estimates, constant, rtol=1e-8)


def test_trend_rejects_weights_outside_the_unit_interval(numpy_backend: NumpyBackend) -> None:
    with pytest.raises(ValueError, match="shrinkage_weight"):
        TrendedDispersion(MethodOfMomentsDispersion(numpy_backend), shrinkage_weight=1.5)


@pytest.mark.skipif(MLXBackend is None, reason="mlx backend not available on this branch")
def test_backends_agree(numpy_backend: NumpyBackend) -> None:
    dataset = make_synthetic_dataset(n_genes=100, n_samples_per_group=4, dispersion=0.3)
    counts = dataset.count_matrix.counts
    size_factors = np.ones(dataset.count_matrix.n_samples)

    on_numpy = MethodOfMomentsDispersion(numpy_backend).estimate(
        counts, size_factors, dataset.design
    )
    on_mlx = MethodOfMomentsDispersion(MLXBackend()).estimate(counts, size_factors, dataset.design)

    np.testing.assert_allclose(on_mlx, on_numpy, rtol=1e-4, atol=1e-6)
