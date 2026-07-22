"""Tests for the multiple-testing corrections.

`scipy.stats.false_discovery_control` is used as an independent reference only;
the implementation under test deliberately depends on NumPy alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from mlxde.contracts import MultipleTestingCorrection
from mlxde.stats.multiple_testing import BenjaminiHochberg, Bonferroni


@pytest.fixture
def benjamini_hochberg() -> BenjaminiHochberg:
    return BenjaminiHochberg()


@pytest.fixture
def bonferroni() -> Bonferroni:
    return Bonferroni()


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_satisfies_protocol(correction: MultipleTestingCorrection) -> None:
    assert isinstance(correction, MultipleTestingCorrection)


@pytest.mark.parametrize("seed", range(5))
def test_matches_scipy_on_random_p_values(benjamini_hochberg: BenjaminiHochberg, seed: int) -> None:
    p_values = np.random.default_rng(seed).uniform(size=500)
    np.testing.assert_allclose(
        benjamini_hochberg.adjust(p_values), false_discovery_control(p_values, method="bh")
    )


def test_matches_scipy_with_ties_and_duplicates(benjamini_hochberg: BenjaminiHochberg) -> None:
    rng = np.random.default_rng(7)
    p_values = np.repeat(rng.choice([0.001, 0.01, 0.04, 0.5, 0.5, 1.0], size=30), 3)
    np.testing.assert_allclose(
        benjamini_hochberg.adjust(p_values), false_discovery_control(p_values, method="bh")
    )


def test_matches_hand_computed_table(benjamini_hochberg: BenjaminiHochberg) -> None:
    p_values = np.array([0.005, 0.01, 0.04, 0.2, 0.5])
    # n/rank * p = [0.025, 0.025, 0.0667, 0.25, 0.5], already monotone.
    expected = np.array([0.025, 0.025, 0.0666666667, 0.25, 0.5])
    np.testing.assert_allclose(benjamini_hochberg.adjust(p_values), expected)


def test_ties_receive_identical_adjusted_values(benjamini_hochberg: BenjaminiHochberg) -> None:
    adjusted = benjamini_hochberg.adjust(np.array([0.02, 0.02, 0.02, 0.9]))
    assert len(np.unique(adjusted[:3])) == 1


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_adjusted_values_are_bounded_monotone_and_conservative(
    correction: MultipleTestingCorrection,
) -> None:
    p_values = np.random.default_rng(1).uniform(size=200)
    adjusted = correction.adjust(p_values)

    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    assert np.all(adjusted >= p_values - 1e-12)

    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_nan_positions_are_preserved(correction: MultipleTestingCorrection) -> None:
    p_values = np.array([0.01, np.nan, 0.2, np.nan, 0.5])
    adjusted = correction.adjust(p_values)
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(p_values))


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_nan_values_do_not_inflate_the_number_of_tests(
    correction: MultipleTestingCorrection,
) -> None:
    tested = np.array([0.001, 0.02, 0.3, 0.7])
    with_nan = np.array([0.001, np.nan, 0.02, 0.3, np.nan, 0.7])

    adjusted = correction.adjust(with_nan)
    np.testing.assert_allclose(adjusted[~np.isnan(with_nan)], correction.adjust(tested))


def test_all_nan_input_returns_all_nan(benjamini_hochberg: BenjaminiHochberg) -> None:
    adjusted = benjamini_hochberg.adjust(np.full(4, np.nan))
    assert np.all(np.isnan(adjusted))


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_empty_and_single_element_inputs(correction: MultipleTestingCorrection) -> None:
    assert correction.adjust(np.array([])).shape == (0,)
    np.testing.assert_allclose(correction.adjust(np.array([0.03])), [0.03])


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_rejects_non_vector_input(correction: MultipleTestingCorrection) -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        correction.adjust(np.zeros((2, 3)))


@pytest.mark.parametrize("correction", [BenjaminiHochberg(), Bonferroni()])
def test_rejects_p_values_outside_the_unit_interval(
    correction: MultipleTestingCorrection,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        correction.adjust(np.array([0.5, 1.5]))


def test_bonferroni_is_p_times_number_of_valid_tests(bonferroni: Bonferroni) -> None:
    p_values = np.array([0.001, 0.01, np.nan, 0.4])
    n_valid = 3
    expected = np.minimum(p_values * n_valid, 1.0)
    np.testing.assert_allclose(bonferroni.adjust(p_values), expected)


def test_bonferroni_is_never_more_significant_than_benjamini_hochberg(
    benjamini_hochberg: BenjaminiHochberg, bonferroni: Bonferroni
) -> None:
    p_values = np.random.default_rng(3).uniform(size=300)
    assert np.all(bonferroni.adjust(p_values) >= benjamini_hochberg.adjust(p_values) - 1e-12)


def test_global_null_yields_almost_no_discoveries(benjamini_hochberg: BenjaminiHochberg) -> None:
    p_values = np.random.default_rng(11).uniform(size=5000)
    assert np.mean(benjamini_hochberg.adjust(p_values) <= 0.05) < 0.01


def test_planted_signal_is_recovered(benjamini_hochberg: BenjaminiHochberg) -> None:
    rng = np.random.default_rng(13)
    n_genes, n_differential = 1000, 50
    p_values = rng.uniform(size=n_genes)
    differential = rng.choice(n_genes, size=n_differential, replace=False)
    p_values[differential] = rng.uniform(0.0, 1e-6, size=n_differential)

    discovered = benjamini_hochberg.adjust(p_values) <= 0.05
    assert discovered.sum() >= n_differential
    assert set(np.flatnonzero(discovered)) >= set(differential)
