"""Tests for the Wald and likelihood ratio hypothesis tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mlxde.contracts import DesignMatrix, GLMFit, GLMFitter, HypothesisTest
from mlxde.stats.hypothesis import (
    LikelihoodRatioTest,
    WaldTest,
    negative_binomial_log_likelihood,
)


def make_fit(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    *,
    fitted_means: np.ndarray | None = None,
    dispersions: np.ndarray | None = None,
) -> GLMFit:
    """A GLMFit built by hand, so the tests never depend on the real GLM module."""
    n_genes = coefficients.shape[0]
    if fitted_means is None:
        fitted_means = np.full((n_genes, 4), 10.0)
    if dispersions is None:
        dispersions = np.full(n_genes, 0.1)
    return GLMFit(
        coefficients=coefficients,
        covariance=covariance,
        dispersions=dispersions,
        fitted_means=fitted_means,
        converged=np.ones(n_genes, dtype=bool),
        n_iterations=5,
    )


class StubFitter:
    """Minimal GLMFitter returning fitted means supplied at construction time."""

    def __init__(self, fitted_means: np.ndarray, dispersions: np.ndarray) -> None:
        self._fitted_means = fitted_means
        self._dispersions = dispersions
        self.designs: list[DesignMatrix] = []

    def fit(
        self,
        counts: np.ndarray,
        design: DesignMatrix,
        size_factors: np.ndarray,
        dispersions: np.ndarray,
    ) -> GLMFit:
        self.designs.append(design)
        n_genes = counts.shape[0]
        return GLMFit(
            coefficients=np.zeros((n_genes, design.n_coefficients)),
            covariance=np.zeros((n_genes, design.n_coefficients, design.n_coefficients)),
            dispersions=self._dispersions,
            fitted_means=self._fitted_means,
            converged=np.ones(n_genes, dtype=bool),
            n_iterations=1,
        )


def test_both_tests_implement_the_protocol() -> None:
    lrt = LikelihoodRatioTest(
        fitter=StubFitter(np.full((2, 4), 10.0), np.full(2, 0.1)),
        design=DesignMatrix(np.ones((4, 2)), ("intercept", "condition[treated]")),
        counts=np.full((2, 4), 10.0),
        size_factors=np.ones(4),
    )
    assert isinstance(WaldTest(), HypothesisTest)
    assert isinstance(lrt, HypothesisTest)
    assert isinstance(StubFitter(np.zeros((1, 1)), np.ones(1)), GLMFitter)


def test_wald_reproduces_the_textbook_statistic_and_p_value() -> None:
    coefficients = np.array([[0.5, 1.2]])
    covariance = np.array([[[0.04, 0.0], [0.0, 0.09]]])
    contrast = np.array([0.0, 1.0])

    result = WaldTest().test(make_fit(coefficients, covariance), contrast)

    expected_z = 1.2 / 0.3
    assert result.effect == pytest.approx(1.2)
    assert result.effect_standard_error == pytest.approx(0.3)
    assert result.statistic[0] == pytest.approx(expected_z)
    assert result.p_values[0] == pytest.approx(2.0 * stats.norm.sf(expected_z))


def test_wald_handles_null_and_large_effects() -> None:
    coefficients = np.array([[0.0, 0.0], [0.0, 5.0]])
    covariance = np.tile(np.eye(2) * 0.01, (2, 1, 1))

    result = WaldTest().test(make_fit(coefficients, covariance), np.array([0.0, 1.0]))

    assert result.p_values[0] == pytest.approx(1.0)
    assert result.p_values[1] < 1e-10


def test_wald_p_values_are_bounded_and_sign_symmetric() -> None:
    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=(50, 3))
    covariance = np.tile(np.eye(3) * 0.25, (50, 1, 1))
    fit = make_fit(coefficients, covariance)
    contrast = np.array([0.0, 1.0, -0.5])

    positive = WaldTest().test(fit, contrast)
    negative = WaldTest().test(fit, -contrast)

    assert np.all((positive.p_values >= 0.0) & (positive.p_values <= 1.0))
    assert positive.p_values == pytest.approx(negative.p_values)
    assert positive.statistic == pytest.approx(-negative.statistic)


def test_wald_p_values_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(0)
    standard_error = 0.4
    n_genes = 5000
    coefficients = np.column_stack(
        [np.zeros(n_genes), rng.normal(scale=standard_error, size=n_genes)]
    )
    covariance = np.tile(np.diag([1.0, standard_error**2]), (n_genes, 1, 1))

    result = WaldTest().test(make_fit(coefficients, covariance), np.array([0.0, 1.0]))

    assert stats.kstest(result.p_values, "uniform").pvalue > 0.05


def test_wald_returns_one_for_degenerate_standard_errors() -> None:
    coefficients = np.array([[1.0, 2.0], [1.0, 2.0]])
    covariance = np.stack([np.zeros((2, 2)), np.full((2, 2), np.nan)])

    result = WaldTest().test(make_fit(coefficients, covariance), np.array([0.0, 1.0]))

    assert result.p_values == pytest.approx([1.0, 1.0])
    assert not np.any(np.isnan(result.p_values))


def test_contrast_shape_is_validated() -> None:
    fit = make_fit(np.zeros((2, 2)), np.tile(np.eye(2), (2, 1, 1)))
    with pytest.raises(ValueError, match="coefficients"):
        WaldTest().test(fit, np.array([1.0, 0.0, 0.0]))


def make_lrt_case(
    full_means: np.ndarray, reduced_means: np.ndarray
) -> tuple[LikelihoodRatioTest, GLMFit, StubFitter]:
    n_genes, n_samples = full_means.shape
    counts = np.tile(np.arange(1.0, n_samples + 1.0), (n_genes, 1))
    dispersions = np.full(n_genes, 0.1)
    design = DesignMatrix(
        matrix=np.column_stack([np.ones(n_samples), np.repeat([0.0, 1.0], n_samples // 2)]),
        coefficient_names=("intercept", "condition[treated]"),
    )
    fitter = StubFitter(reduced_means, dispersions)
    test = LikelihoodRatioTest(
        fitter=fitter, design=design, counts=counts, size_factors=np.ones(n_samples)
    )
    fit = make_fit(
        coefficients=np.tile([1.0, 0.5], (n_genes, 1)),
        covariance=np.tile(np.eye(2) * 0.04, (n_genes, 1, 1)),
        fitted_means=full_means,
        dispersions=dispersions,
    )
    return test, fit, fitter


def test_lrt_statistic_is_non_negative_and_drops_the_tested_column() -> None:
    full_means = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 6.0, 6.0]])
    reduced_means = np.full((2, 4), 2.5)

    test, fit, fitter = make_lrt_case(full_means, reduced_means)
    result = test.test(fit, np.array([0.0, 1.0]))

    assert np.all(result.statistic >= 0.0)
    assert np.all((result.p_values >= 0.0) & (result.p_values <= 1.0))
    assert fitter.designs[-1].coefficient_names == ("intercept",)
    assert fitter.designs[-1].n_coefficients == 1


def test_lrt_gives_p_one_when_the_models_fit_identically() -> None:
    means = np.array([[1.0, 2.0, 3.0, 4.0]])

    test, fit, _ = make_lrt_case(means, means)
    result = test.test(fit, np.array([0.0, 1.0]))

    assert result.statistic == pytest.approx(0.0)
    assert result.p_values == pytest.approx(1.0)


def test_lrt_degrees_of_freedom_follow_the_contrast() -> None:
    n_samples, n_genes = 4, 3
    counts = np.tile(np.arange(1.0, n_samples + 1.0), (n_genes, 1))
    dispersions = np.full(n_genes, 0.1)
    design = DesignMatrix(
        matrix=np.column_stack(
            [np.ones(n_samples), np.repeat([0.0, 1.0], 2), np.arange(float(n_samples))]
        ),
        coefficient_names=("intercept", "condition[treated]", "batch"),
    )
    full_means = np.full((n_genes, n_samples), 3.0)
    reduced_means = np.full((n_genes, n_samples), 2.0)
    fit = make_fit(
        coefficients=np.tile([1.0, 0.5, 0.2], (n_genes, 1)),
        covariance=np.tile(np.eye(3) * 0.04, (n_genes, 1, 1)),
        fitted_means=full_means,
        dispersions=dispersions,
    )
    fitter = StubFitter(reduced_means, dispersions)
    test = LikelihoodRatioTest(fitter, design, counts, np.ones(n_samples))

    result = test.test(fit, np.array([0.0, 1.0, 1.0]))

    assert fitter.designs[-1].coefficient_names == ("intercept",)
    expected_statistic = 2.0 * (
        negative_binomial_log_likelihood(counts, full_means, dispersions)
        - negative_binomial_log_likelihood(counts, reduced_means, dispersions)
    )
    assert result.statistic == pytest.approx(expected_statistic)
    assert result.p_values == pytest.approx(stats.chi2.sf(expected_statistic, 2))


def test_lrt_reports_the_full_model_effect() -> None:
    means = np.full((2, 4), 3.0)
    test, fit, _ = make_lrt_case(means, means)

    result = test.test(fit, np.array([0.0, 1.0]))

    assert result.effect == pytest.approx(0.5)
    assert result.effect_standard_error == pytest.approx(0.2)


def test_lrt_rejects_a_contrast_that_empties_the_design() -> None:
    means = np.full((1, 4), 3.0)
    test, fit, _ = make_lrt_case(means, means)

    with pytest.raises(ValueError, match="at least one coefficient"):
        test.test(fit, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="at least one coefficient"):
        test.test(fit, np.array([0.0, 0.0]))


def test_negative_binomial_log_likelihood_matches_scipy() -> None:
    counts = np.array([[0.0, 3.0, 7.0]])
    means = np.array([[1.0, 2.0, 5.0]])
    dispersion = 0.25
    size = 1.0 / dispersion
    expected = stats.nbinom.logpmf(counts, size, size / (size + means)).sum()

    actual = negative_binomial_log_likelihood(counts, means, np.array([dispersion]))

    assert actual[0] == pytest.approx(expected)
