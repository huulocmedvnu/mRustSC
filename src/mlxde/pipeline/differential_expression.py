"""Orchestration of a differential expression analysis.

The pipeline owns the *order* of the analysis and nothing else: every numerical
step is delegated to a collaborator injected through the constructor, so this
module imports no concrete estimator, backend or file format.
"""

from __future__ import annotations

import numpy as np

from mlxde.contracts import (
    CountMatrix,
    DesignMatrix,
    DifferentialExpressionResult,
    DispersionEstimator,
    GeneFilter,
    GLMFit,
    GLMFitter,
    HypothesisTest,
    MultipleTestingCorrection,
    SizeFactorEstimator,
    TestStatistics,
)

_LOG_2 = np.log(2.0)


class DifferentialExpressionPipeline:
    """Runs the DESeq2-style sequence of estimators over one count matrix.

    The steps are: estimate size factors, filter genes, estimate dispersions on
    the retained genes, fit the GLM, apply the hypothesis test, adjust the
    p-values, assemble the result table.

    Genes rejected by the filter are excluded from testing *and* from the
    multiple-testing correction — that is the point of DESeq2's independent
    filtering, since testing hopeless genes only costs power for the rest. They
    are still reported, with NaN statistics, so the returned table stays aligned
    row-for-row with the input gene ids, and their ``base_mean`` is computed like
    everyone else's.

    Instances are stateless: ``run`` may be called repeatedly and mutates
    neither the pipeline nor its arguments.
    """

    def __init__(
        self,
        size_factors: SizeFactorEstimator,
        gene_filter: GeneFilter,
        dispersions: DispersionEstimator,
        fitter: GLMFitter,
        hypothesis_test: HypothesisTest,
        correction: MultipleTestingCorrection,
    ) -> None:
        self._size_factors = size_factors
        self._gene_filter = gene_filter
        self._dispersions = dispersions
        self._fitter = fitter
        self._hypothesis_test = hypothesis_test
        self._correction = correction

    def run(
        self, count_matrix: CountMatrix, design: DesignMatrix, contrast: np.ndarray
    ) -> DifferentialExpressionResult:
        """Test ``contrast`` for every gene of ``count_matrix`` under ``design``."""
        self._validate(count_matrix, design, contrast)

        size_factors = self._size_factors.estimate(count_matrix.counts)
        base_mean = self._base_mean(count_matrix.counts, size_factors)

        retained = self._gene_filter.keep(count_matrix.counts)
        tested_counts = count_matrix.counts[retained]

        fit = self._fit(tested_counts, design, size_factors)
        statistics = self._hypothesis_test.test(fit, contrast)
        adjusted_p_values = self._correction.adjust(statistics.p_values)

        return self._assemble(
            count_matrix.gene_ids, base_mean, retained, statistics, adjusted_p_values
        )

    @staticmethod
    def _validate(count_matrix: CountMatrix, design: DesignMatrix, contrast: np.ndarray) -> None:
        if design.n_samples != count_matrix.n_samples:
            raise ValueError(
                f"design has {design.n_samples} samples but the count matrix has "
                f"{count_matrix.n_samples}"
            )
        contrast = np.asarray(contrast)
        if contrast.shape != (design.n_coefficients,):
            raise ValueError(
                f"contrast must have shape ({design.n_coefficients},) to match the design "
                f"coefficients {design.coefficient_names}, got shape {contrast.shape}"
            )

    @staticmethod
    def _base_mean(counts: np.ndarray, size_factors: np.ndarray) -> np.ndarray:
        """Mean size-factor-normalised count per gene, for filtered genes too."""
        return np.mean(counts / size_factors, axis=1)

    def _fit(self, counts: np.ndarray, design: DesignMatrix, size_factors: np.ndarray) -> GLMFit:
        dispersions = self._dispersions.estimate(counts, size_factors, design)
        return self._fitter.fit(counts, design, size_factors, dispersions)

    @staticmethod
    def _assemble(
        gene_ids: np.ndarray,
        base_mean: np.ndarray,
        retained: np.ndarray,
        statistics: TestStatistics,
        adjusted_p_values: np.ndarray,
    ) -> DifferentialExpressionResult:
        """Scatter the tested genes back into full-length, NaN-padded columns."""

        def full(tested_values: np.ndarray) -> np.ndarray:
            column = np.full(len(gene_ids), np.nan)
            column[retained] = tested_values
            return column

        # Coefficients are fitted on the natural-log scale; this is the single
        # place where the reported table switches to log2.
        return DifferentialExpressionResult(
            gene_ids=gene_ids,
            base_mean=base_mean,
            log2_fold_change=full(statistics.effect / _LOG_2),
            log2_fold_change_standard_error=full(statistics.effect_standard_error / _LOG_2),
            statistic=full(statistics.statistic),
            p_value=full(statistics.p_values),
            adjusted_p_value=full(adjusted_p_values),
        )
