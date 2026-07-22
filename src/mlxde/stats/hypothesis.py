"""Hypothesis tests turning a fitted GLM and a contrast into per-gene p-values.

Two interchangeable strategies are offered: a Wald test, which reads the answer
off the fitted coefficient covariance, and a likelihood ratio test, which refits
a reduced model without the contrasted coefficients. Both implement
``HypothesisTest`` and populate the same ``TestStatistics`` fields, so the
pipeline can swap one for the other without knowing which it holds.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import gammaln

from mlxde.contracts import DesignMatrix, GLMFit, GLMFitter, TestStatistics

# A gene whose standard error is zero, negative or non-finite carries no usable
# evidence: the fit degenerated (all-zero counts, separation, a singular
# information matrix). Reporting p = 1 keeps such genes in the table and lets
# multiple-testing correction treat them as uninformative, whereas NaN would
# silently propagate into every downstream summary.
_UNINFORMATIVE_P_VALUE = 1.0


def _contrast_effect(fit: GLMFit, contrast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene contrast estimate and its standard error, vectorised over genes.

    Shared by both tests so the projection maths exists in exactly one place.
    """
    contrast = np.asarray(contrast, dtype=float)
    if contrast.ndim != 1:
        raise ValueError(f"contrast must be 1-dimensional, got shape {contrast.shape}")
    n_coefficients = fit.coefficients.shape[1]
    if contrast.shape[0] != n_coefficients:
        raise ValueError(
            f"contrast has {contrast.shape[0]} entries but the fit has "
            f"{n_coefficients} coefficients"
        )

    effect = np.einsum("gc,c->g", fit.coefficients, contrast)
    variance = np.einsum("c,gcd,d->g", contrast, fit.covariance, contrast)
    # Rounding can push a variance a hair below zero; the sqrt of a negative
    # would become NaN and be indistinguishable from a genuinely failed fit.
    standard_error = np.sqrt(np.where(variance > 0.0, variance, np.nan))
    return effect, standard_error


def _is_usable(standard_error: np.ndarray) -> np.ndarray:
    """Genes whose standard error supports a meaningful test statistic."""
    return np.isfinite(standard_error) & (standard_error > 0.0)


class WaldTest:
    """Normal-approximation test: the contrast estimate divided by its standard error."""

    def test(self, fit: GLMFit, contrast: np.ndarray) -> TestStatistics:
        effect, standard_error = _contrast_effect(fit, contrast)
        usable = _is_usable(standard_error)

        statistic = np.divide(effect, standard_error, out=np.zeros_like(effect), where=usable)
        p_values = np.where(usable, 2.0 * stats.norm.sf(np.abs(statistic)), _UNINFORMATIVE_P_VALUE)
        return TestStatistics(
            statistic=statistic,
            p_values=p_values,
            effect=effect,
            effect_standard_error=standard_error,
        )


def negative_binomial_log_likelihood(
    counts: np.ndarray, means: np.ndarray, dispersions: np.ndarray
) -> np.ndarray:
    """Per-gene negative binomial log-likelihood summed over samples.

    Parameterised by the mean and the dispersion ``alpha`` (variance is
    ``mu + alpha * mu**2``), matching the GLM fitter's convention.
    """
    size = 1.0 / dispersions[:, None]
    means = np.maximum(means, np.finfo(float).tiny)
    per_sample = (
        gammaln(counts + size)
        - gammaln(size)
        - gammaln(counts + 1.0)
        + size * np.log(size / (size + means))
        + counts * np.log(means / (size + means))
    )
    return per_sample.sum(axis=1)


class LikelihoodRatioTest:
    """Chi-squared test comparing the full model with the contrast's coefficients dropped.

    Depends on the ``GLMFitter`` protocol rather than a concrete GLM, so the
    reduced model is fitted by whatever implementation the caller injected.
    """

    def __init__(
        self,
        fitter: GLMFitter,
        design: DesignMatrix,
        counts: np.ndarray,
        size_factors: np.ndarray,
    ) -> None:
        self._fitter = fitter
        self._design = design
        self._counts = counts
        self._size_factors = size_factors

    def test(self, fit: GLMFit, contrast: np.ndarray) -> TestStatistics:
        # DESeq2 reports the Wald-style effect alongside the LRT statistic, so
        # the two tests remain substitutable in the result table.
        effect, standard_error = _contrast_effect(fit, contrast)

        reduced_design, degrees_of_freedom = self._reduce(np.asarray(contrast, dtype=float))
        reduced_fit = self._fitter.fit(
            counts=self._counts,
            design=reduced_design,
            size_factors=self._size_factors,
            dispersions=fit.dispersions,
        )

        log_likelihood_full = negative_binomial_log_likelihood(
            self._counts, fit.fitted_means, fit.dispersions
        )
        log_likelihood_reduced = negative_binomial_log_likelihood(
            self._counts, reduced_fit.fitted_means, reduced_fit.dispersions
        )
        # A reduced model can never fit better; anything below zero is rounding.
        statistic = np.maximum(2.0 * (log_likelihood_full - log_likelihood_reduced), 0.0)

        usable = np.isfinite(statistic)
        statistic = np.where(usable, statistic, 0.0)
        p_values = np.where(
            usable, stats.chi2.sf(statistic, degrees_of_freedom), _UNINFORMATIVE_P_VALUE
        )
        return TestStatistics(
            statistic=statistic,
            p_values=p_values,
            effect=effect,
            effect_standard_error=standard_error,
        )

    def _reduce(self, contrast: np.ndarray) -> tuple[DesignMatrix, int]:
        """Design with the contrasted coefficients removed, and how many were removed."""
        tested = contrast != 0.0
        if not tested.any():
            raise ValueError("contrast must select at least one coefficient")
        if tested.all():
            raise ValueError("contrast must leave at least one coefficient in the reduced model")
        kept = ~tested
        reduced = DesignMatrix(
            matrix=self._design.matrix[:, kept],
            coefficient_names=tuple(
                name
                for name, keep in zip(self._design.coefficient_names, kept, strict=True)
                if keep
            ),
        )
        return reduced, int(tested.sum())
