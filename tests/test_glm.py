"""Tests for the batched negative binomial GLM."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import gammaln

from mlxde.backend.numpy_backend import NumpyBackend
from mlxde.contracts import DesignMatrix, GLMFitter
from mlxde.stats.glm import NegativeBinomialGLM
from tests.conftest import make_synthetic_dataset

POISSON_LIMIT = 1e-8


def reference_coefficients(
    counts_row: np.ndarray,
    design: np.ndarray,
    size_factors: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """Maximum likelihood fit of one gene by direct numerical optimisation."""

    def negative_log_likelihood(coefficients: np.ndarray) -> float:
        means = size_factors * np.exp(design @ coefficients)
        if dispersion <= POISSON_LIMIT:
            # The negative binomial gammaln terms cancel as alpha -> 0, and
            # evaluating them at 1/alpha = 1e8 would lose all precision.
            return float(np.sum(means - counts_row * np.log(means)))
        shape = 1.0 / dispersion
        terms = (
            gammaln(counts_row + shape)
            - gammaln(shape)
            + counts_row * np.log(dispersion * means)
            - (counts_row + shape) * np.log1p(dispersion * means)
        )
        return float(-np.sum(terms))

    start = np.zeros(design.shape[1])
    start[0] = np.log(np.mean(counts_row / size_factors) + 0.1)
    result = minimize(negative_log_likelihood, start, method="BFGS", options={"gtol": 1e-10})
    return result.x


def two_group_design(n_per_group: int) -> DesignMatrix:
    is_treated = np.repeat([0.0, 1.0], n_per_group)
    return DesignMatrix(
        matrix=np.column_stack([np.ones(2 * n_per_group), is_treated]),
        coefficient_names=("intercept", "condition[treated]"),
    )


@pytest.fixture
def fitter(numpy_backend: NumpyBackend) -> NegativeBinomialGLM:
    return NegativeBinomialGLM(numpy_backend)


def test_implements_glm_fitter_protocol(fitter: NegativeBinomialGLM) -> None:
    assert isinstance(fitter, GLMFitter)


@pytest.mark.parametrize("dispersion", [POISSON_LIMIT, 0.25])
def test_matches_maximum_likelihood_reference(
    fitter: NegativeBinomialGLM, dispersion: float
) -> None:
    rng = np.random.default_rng(11)
    design = two_group_design(5)
    size_factors = np.linspace(0.7, 1.4, design.n_samples)
    means = size_factors * np.array([40.0, 40.0, 40.0, 40.0, 40.0, 90.0, 90.0, 90.0, 90.0, 90.0])
    counts = rng.poisson(means, size=(6, design.n_samples)).astype(np.float64)

    fit = fitter.fit(counts, design, size_factors, np.full(counts.shape[0], dispersion))

    assert fit.converged.all()
    for gene in range(counts.shape[0]):
        expected = reference_coefficients(counts[gene], design.matrix, size_factors, dispersion)
        assert fit.coefficients[gene] == pytest.approx(expected, abs=1e-4)


def test_fitted_means_are_consistent_with_coefficients(fitter: NegativeBinomialGLM) -> None:
    dataset = make_synthetic_dataset(n_genes=30, n_differential=5, seed=3)
    size_factors = np.linspace(0.8, 1.2, dataset.count_matrix.n_samples)

    fit = fitter.fit(
        dataset.count_matrix.counts,
        dataset.design,
        size_factors,
        np.full(dataset.count_matrix.n_genes, 0.2),
    )

    expected = size_factors * np.exp(fit.coefficients @ dataset.design.matrix.T)
    assert fit.fitted_means == pytest.approx(expected, rel=1e-8)


def test_recovers_synthetic_fold_changes(fitter: NegativeBinomialGLM) -> None:
    dispersion = 0.1
    dataset = make_synthetic_dataset(
        n_samples_per_group=8, n_differential=60, dispersion=dispersion
    )
    size_factors = np.ones(dataset.count_matrix.n_samples)

    fit = fitter.fit(
        dataset.count_matrix.counts,
        dataset.design,
        size_factors,
        np.full(dataset.count_matrix.n_genes, dispersion),
    )

    estimated_log2 = fit.coefficients[:, 1] / np.log(2.0)
    correlation = np.corrcoef(estimated_log2, dataset.true_log2_fold_change)[0, 1]
    differential = dataset.differential_genes
    mean_absolute_error = np.abs(
        estimated_log2[differential] - dataset.true_log2_fold_change[differential]
    ).mean()

    assert fit.converged.all()
    assert correlation > 0.95
    assert mean_absolute_error < 0.3


def test_size_factors_enter_the_model_as_an_offset(fitter: NegativeBinomialGLM) -> None:
    """Deeper sequencing of one sample must not change the coefficients.

    Counts are set to their exact means, so the maximum likelihood solution is
    the generating coefficient vector regardless of how the samples are scaled;
    any deviation is the offset being handled incorrectly rather than noise.
    """
    design = two_group_design(4)
    true_coefficients = np.array([[4.0, 1.5], [2.0, -0.75], [6.0, 0.0]])
    size_factors = np.linspace(0.6, 1.5, design.n_samples)
    counts = size_factors * np.exp(true_coefficients @ design.matrix.T)
    dispersions = np.full(counts.shape[0], 0.2)

    scaled_counts = counts.copy()
    scaled_counts[:, 0] *= 4.0
    scaled_size_factors = size_factors.copy()
    scaled_size_factors[0] *= 4.0

    baseline = fitter.fit(counts, design, size_factors, dispersions)
    rescaled = fitter.fit(scaled_counts, design, scaled_size_factors, dispersions)

    assert baseline.coefficients == pytest.approx(true_coefficients, abs=1e-8)
    assert rescaled.coefficients == pytest.approx(true_coefficients, abs=1e-8)


def test_standard_errors_shrink_with_duplicated_samples(fitter: NegativeBinomialGLM) -> None:
    dataset = make_synthetic_dataset(n_genes=40, n_differential=8, seed=7)
    counts = dataset.count_matrix.counts
    design = dataset.design
    size_factors = np.ones(design.n_samples)
    dispersions = np.full(counts.shape[0], 0.2)

    doubled_design = DesignMatrix(
        matrix=np.vstack([design.matrix, design.matrix]),
        coefficient_names=design.coefficient_names,
    )
    doubled = fitter.fit(
        np.hstack([counts, counts]),
        doubled_design,
        np.hstack([size_factors, size_factors]),
        dispersions,
    )
    single = fitter.fit(counts, design, size_factors, dispersions)

    ratio = np.sqrt(np.diagonal(doubled.covariance, axis1=1, axis2=2)) / np.sqrt(
        np.diagonal(single.covariance, axis1=1, axis2=2)
    )
    assert doubled.coefficients == pytest.approx(single.coefficients, abs=1e-5)
    assert ratio == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-4)


def test_covariance_matches_the_information_matrix(fitter: NegativeBinomialGLM) -> None:
    dataset = make_synthetic_dataset(n_genes=15, n_differential=4, seed=9)
    dispersions = np.full(dataset.count_matrix.n_genes, 0.15)
    size_factors = np.linspace(0.9, 1.1, dataset.count_matrix.n_samples)

    fit = fitter.fit(dataset.count_matrix.counts, dataset.design, size_factors, dispersions)

    design = dataset.design.matrix
    for gene in range(dataset.count_matrix.n_genes):
        means = fit.fitted_means[gene]
        weights = means / (1.0 + dispersions[gene] * means)
        information = design.T @ (weights[:, None] * design)
        assert fit.covariance[gene] == pytest.approx(np.linalg.inv(information), rel=1e-6)


def test_all_zero_gene_does_not_contaminate_other_genes(fitter: NegativeBinomialGLM) -> None:
    dataset = make_synthetic_dataset(n_genes=20, n_differential=5, seed=1)
    counts = dataset.count_matrix.counts
    size_factors = np.ones(dataset.design.n_samples)
    dispersions = np.full(counts.shape[0], 0.2)

    healthy = fitter.fit(counts, dataset.design, size_factors, dispersions)

    with_zero_gene = counts.copy()
    with_zero_gene[0] = 0.0
    degenerate = fitter.fit(with_zero_gene, dataset.design, size_factors, dispersions)

    assert np.isfinite(degenerate.coefficients).all()
    assert np.isfinite(degenerate.covariance).all()
    # An all-zero gene has no information about its mean; the clamps keep it at
    # a finite, very negative intercept instead of producing NaN.
    assert degenerate.coefficients[0, 0] < -10.0
    assert degenerate.converged[1:].all()
    assert degenerate.coefficients[1:] == pytest.approx(healthy.coefficients[1:], abs=1e-8)


def test_reports_non_convergence_instead_of_raising(numpy_backend: NumpyBackend) -> None:
    dataset = make_synthetic_dataset(n_genes=10, n_differential=3, seed=2)
    impatient = NegativeBinomialGLM(numpy_backend, max_iterations=1)

    fit = impatient.fit(
        dataset.count_matrix.counts,
        dataset.design,
        np.ones(dataset.design.n_samples),
        np.full(dataset.count_matrix.n_genes, 0.2),
    )

    assert fit.n_iterations == 1
    assert not fit.converged.any()
    assert np.isfinite(fit.coefficients).all()


def test_rejects_inconsistent_inputs(fitter: NegativeBinomialGLM) -> None:
    design = two_group_design(3)
    counts = np.ones((4, design.n_samples))
    with pytest.raises(ValueError, match="size factors"):
        fitter.fit(counts, design, np.ones(design.n_samples - 1), np.ones(4))
    with pytest.raises(ValueError, match="dispersions"):
        fitter.fit(counts, design, np.ones(design.n_samples), np.ones(3))
    with pytest.raises(ValueError, match="strictly positive"):
        fitter.fit(counts, design, np.zeros(design.n_samples), np.ones(4))


def test_backends_agree() -> None:
    try:
        from mlxde.backend.mlx_backend import MLXBackend
    except ImportError:  # the GPU backend lives on another branch
        pytest.skip("MLXBackend is not available")

    backend = MLXBackend()
    if not backend.is_available():
        pytest.skip("MLX backend reports no device")

    dataset = make_synthetic_dataset(n_genes=50, n_differential=10, seed=4)
    arguments = (
        dataset.count_matrix.counts,
        dataset.design,
        np.linspace(0.8, 1.2, dataset.count_matrix.n_samples),
        np.full(dataset.count_matrix.n_genes, 0.2),
    )
    on_cpu = NegativeBinomialGLM(NumpyBackend()).fit(*arguments)
    on_gpu = NegativeBinomialGLM(backend).fit(*arguments)

    assert on_gpu.coefficients == pytest.approx(on_cpu.coefficients, abs=2e-3)
    assert on_gpu.fitted_means == pytest.approx(on_cpu.fitted_means, rel=2e-3)
    assert on_gpu.covariance == pytest.approx(on_cpu.covariance, rel=5e-3, abs=1e-8)
