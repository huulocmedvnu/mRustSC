"""Tests for size factor estimation and gene filtering."""

from __future__ import annotations

import numpy as np
import pytest

from mlxde.contracts import GeneFilter, SizeFactorEstimator
from mlxde.preprocess.filtering import MinimumCountFilter
from mlxde.preprocess.normalization import (
    MedianOfRatiosSizeFactors,
    TotalCountSizeFactors,
    _normalise_to_unit_geometric_mean,
)

ESTIMATORS = [MedianOfRatiosSizeFactors, TotalCountSizeFactors]


def geometric_mean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


@pytest.fixture
def counts() -> np.ndarray:
    rng = np.random.default_rng(7)
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    baseline = rng.lognormal(mean=4.0, sigma=0.8, size=(300, 1))
    return rng.poisson(baseline * depths).astype(np.float64)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_implements_protocol(estimator_class: type) -> None:
    assert isinstance(estimator_class(), SizeFactorEstimator)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_factors_are_positive_with_unit_geometric_mean(
    estimator_class: type, counts: np.ndarray
) -> None:
    factors = estimator_class().estimate(counts)

    assert factors.shape == (counts.shape[1],)
    assert np.all(factors > 0)
    assert geometric_mean(factors) == pytest.approx(1.0)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_identical_libraries_give_unit_factors(estimator_class: type) -> None:
    library = np.array([[10.0], [20.0], [30.0], [40.0]])
    factors = estimator_class().estimate(np.tile(library, (1, 5)))

    assert factors == pytest.approx(np.ones(5))


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_scaling_one_sample_scales_only_its_factor(
    estimator_class: type, counts: np.ndarray
) -> None:
    estimator = estimator_class()
    scale = 3.0
    scaled_counts = counts.copy()
    scaled_counts[:, 2] *= scale

    baseline = estimator.estimate(counts)
    expected = baseline.copy()
    expected[2] *= scale

    # The unit-geometric-mean constraint spreads the change over every factor,
    # so the invariant is exact only after renormalising the expectation.
    assert estimator.estimate(scaled_counts) == pytest.approx(
        _normalise_to_unit_geometric_mean(expected)
    )


def test_median_of_ratios_resists_outlier_genes_that_move_total_counts(
    counts: np.ndarray,
) -> None:
    outliers = np.full((3, counts.shape[1]), 1.0)
    outliers[:, 0] = 1e7  # a handful of genes dominating one sample's library
    contaminated = np.vstack([counts, outliers])

    def shift(estimator: SizeFactorEstimator) -> float:
        before = estimator.estimate(counts)
        after = estimator.estimate(contaminated)
        return float(np.max(np.abs(after / before - 1.0)))

    median_shift = shift(MedianOfRatiosSizeFactors())
    total_count_shift = shift(TotalCountSizeFactors())

    assert median_shift < 0.01
    assert total_count_shift > 1.0
    assert median_shift < total_count_shift


def test_median_of_ratios_raises_when_no_gene_is_expressed_everywhere() -> None:
    counts = np.array([[5.0, 0.0], [0.0, 7.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="none qualified"):
        MedianOfRatiosSizeFactors().estimate(counts)


def test_total_count_raises_on_empty_sample() -> None:
    counts = np.array([[5.0, 0.0], [7.0, 0.0]])

    with pytest.raises(ValueError, match="strictly positive total count"):
        TotalCountSizeFactors().estimate(counts)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_negative_counts_are_rejected(estimator_class: type) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimator_class().estimate(np.array([[1.0, -1.0], [2.0, 3.0]]))


def test_filter_implements_protocol() -> None:
    assert isinstance(MinimumCountFilter(), GeneFilter)


def test_filter_mask_is_boolean_and_gene_shaped(counts: np.ndarray) -> None:
    mask = MinimumCountFilter().keep(counts)

    assert mask.dtype == np.bool_
    assert mask.shape == (counts.shape[0],)


def test_filter_boundaries_are_inclusive() -> None:
    counts = np.array(
        [
            [10.0, 10.0, 10.0, 0.0],  # exactly min_count in exactly min_samples
            [9.0, 10.0, 10.0, 10.0],  # one sample just below min_count
            [10.0, 10.0, 0.0, 0.0],  # one sample short of min_samples
            [0.0, 0.0, 0.0, 0.0],  # never expressed
        ]
    )

    mask = MinimumCountFilter(min_count=10, min_samples=3).keep(counts)

    assert mask.tolist() == [True, True, False, False]


def test_filter_selects_the_expected_genes(synthetic_dataset) -> None:
    count_matrix = synthetic_dataset.count_matrix
    gene_filter = MinimumCountFilter(min_count=20, min_samples=4)

    mask = gene_filter.keep(count_matrix.counts)
    filtered = count_matrix.select_genes(mask)

    assert filtered.n_genes == int(np.sum(mask))
    assert filtered.n_samples == count_matrix.n_samples
    assert np.array_equal(filtered.gene_ids, count_matrix.gene_ids[mask])
    assert np.array_equal(filtered.counts, count_matrix.counts[mask])
    assert np.all(np.sum(filtered.counts >= 20, axis=1) >= 4)


@pytest.mark.parametrize(("min_count", "min_samples"), [(-1, 3), (10, 0)])
def test_filter_rejects_invalid_thresholds(min_count: int, min_samples: int) -> None:
    with pytest.raises(ValueError):
        MinimumCountFilter(min_count=min_count, min_samples=min_samples)
