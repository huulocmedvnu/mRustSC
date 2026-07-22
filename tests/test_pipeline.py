"""Pipeline orchestration tests.

The collaborators are stubs implementing the protocols from ``mlxde.contracts``
with prescribed outputs, so the expected result table is known exactly and the
test says nothing about how any estimator computes its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from mlxde.contracts import (
    CountMatrix,
    DesignMatrix,
    GLMFit,
    TestStatistics,
)
from mlxde.pipeline.differential_expression import DifferentialExpressionPipeline

LN_4 = float(np.log(4.0))


@dataclass
class ConstantSizeFactors:
    """SizeFactorEstimator returning the same factor for every sample."""

    value: float = 1.0

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        return np.full(counts.shape[1], self.value)


@dataclass
class DropGenesFilter:
    """GeneFilter rejecting a fixed set of gene positions."""

    dropped: tuple[int, ...] = ()

    def keep(self, counts: np.ndarray) -> np.ndarray:
        mask = np.ones(counts.shape[0], dtype=bool)
        mask[list(self.dropped)] = False
        return mask


@dataclass
class ConstantDispersion:
    """DispersionEstimator recording the shape of the genes it was given."""

    value: float = 0.1
    seen_gene_counts: list[int] = field(default_factory=list)

    def estimate(
        self, counts: np.ndarray, size_factors: np.ndarray, design: DesignMatrix
    ) -> np.ndarray:
        self.seen_gene_counts.append(counts.shape[0])
        return np.full(counts.shape[0], self.value)


@dataclass
class PrescribedFitter:
    """GLMFitter returning ``coefficient_row`` for every gene it is handed."""

    coefficient_row: tuple[float, ...] = (1.0, LN_4)
    variance: float = 0.25
    seen_dispersions: list[np.ndarray] = field(default_factory=list)

    def fit(
        self,
        counts: np.ndarray,
        design: DesignMatrix,
        size_factors: np.ndarray,
        dispersions: np.ndarray,
    ) -> GLMFit:
        self.seen_dispersions.append(dispersions)
        n_genes = counts.shape[0]
        n_coefficients = design.n_coefficients
        coefficients = np.tile(np.asarray(self.coefficient_row), (n_genes, 1))
        covariance = np.tile(self.variance * np.eye(n_coefficients), (n_genes, 1, 1))
        return GLMFit(
            coefficients=coefficients,
            covariance=covariance,
            dispersions=dispersions,
            fitted_means=np.ones_like(counts),
            converged=np.ones(n_genes, dtype=bool),
            n_iterations=3,
        )


@dataclass
class ContrastTest:
    """HypothesisTest applying the contrast exactly, with a fixed p-value ramp."""

    p_values: np.ndarray | None = None

    def test(self, fit: GLMFit, contrast: np.ndarray) -> TestStatistics:
        effect = fit.coefficients @ contrast
        variance = np.einsum("i,gij,j->g", contrast, fit.covariance, contrast)
        standard_error = np.sqrt(variance)
        p_values = np.linspace(0.01, 0.5, len(effect)) if self.p_values is None else self.p_values
        return TestStatistics(
            statistic=effect / standard_error,
            p_values=np.asarray(p_values, dtype=float),
            effect=effect,
            effect_standard_error=standard_error,
        )


@dataclass
class HalvingCorrection:
    """MultipleTestingCorrection recording how many p-values it received."""

    seen_sizes: list[int] = field(default_factory=list)

    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        self.seen_sizes.append(len(p_values))
        return np.minimum(p_values * 2.0, 1.0)


def make_count_matrix(counts: np.ndarray) -> CountMatrix:
    n_genes, n_samples = counts.shape
    sample_ids = np.array([f"sample_{index}" for index in range(n_samples)])
    return CountMatrix(
        counts=counts,
        gene_ids=np.array([f"gene_{index}" for index in range(n_genes)]),
        sample_ids=sample_ids,
        sample_metadata=pd.DataFrame({"condition": ["control", "treated"] * (n_samples // 2)}),
    )


def make_design(n_samples: int) -> DesignMatrix:
    is_treated = np.tile([0.0, 1.0], n_samples // 2)
    return DesignMatrix(
        matrix=np.column_stack([np.ones(n_samples), is_treated]),
        coefficient_names=("intercept", "condition[treated]"),
    )


@pytest.fixture
def counts() -> np.ndarray:
    return np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [0.0, 0.0, 1.0, 1.0],
            [100.0, 100.0, 100.0, 100.0],
            [5.0, 15.0, 25.0, 35.0],
        ]
    )


def build_pipeline(
    dropped: tuple[int, ...] = (),
    size_factor: float = 1.0,
    coefficient_row: tuple[float, ...] = (1.0, LN_4),
    p_values: np.ndarray | None = None,
) -> tuple[DifferentialExpressionPipeline, dict[str, object]]:
    collaborators: dict[str, object] = {
        "size_factors": ConstantSizeFactors(size_factor),
        "gene_filter": DropGenesFilter(dropped),
        "dispersions": ConstantDispersion(),
        "fitter": PrescribedFitter(coefficient_row=coefficient_row),
        "hypothesis_test": ContrastTest(p_values),
        "correction": HalvingCorrection(),
    }
    return DifferentialExpressionPipeline(**collaborators), collaborators  # type: ignore[arg-type]


def test_result_columns_are_fully_determined(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline(p_values=np.array([0.01, 0.02, 0.03, 0.04]))
    count_matrix = make_count_matrix(counts)

    result = pipeline.run(count_matrix, make_design(4), np.array([0.0, 1.0]))

    assert list(result.gene_ids) == ["gene_0", "gene_1", "gene_2", "gene_3"]
    np.testing.assert_allclose(result.base_mean, [25.0, 0.5, 100.0, 20.0])
    np.testing.assert_allclose(result.log2_fold_change, np.full(4, 2.0))
    np.testing.assert_allclose(result.log2_fold_change_standard_error, np.full(4, 0.5 / np.log(2)))
    np.testing.assert_allclose(result.statistic, np.full(4, LN_4 / 0.5))
    np.testing.assert_allclose(result.p_value, [0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(result.adjusted_p_value, [0.02, 0.04, 0.06, 0.08])


def test_base_mean_uses_size_factors(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline(size_factor=2.0)

    result = pipeline.run(make_count_matrix(counts), make_design(4), np.array([0.0, 1.0]))

    np.testing.assert_allclose(result.base_mean, [12.5, 0.25, 50.0, 10.0])


def test_natural_log_coefficient_is_reported_as_log2(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline(coefficient_row=(1.0, LN_4))

    result = pipeline.run(make_count_matrix(counts), make_design(4), np.array([0.0, 1.0]))

    np.testing.assert_allclose(result.log2_fold_change, 2.0)


def test_filtered_genes_are_reported_as_nan_and_never_tested(counts: np.ndarray) -> None:
    pipeline, collaborators = build_pipeline(dropped=(1, 2))

    result = pipeline.run(make_count_matrix(counts), make_design(4), np.array([0.0, 1.0]))

    filtered = [1, 2]
    for column in (
        result.log2_fold_change,
        result.log2_fold_change_standard_error,
        result.statistic,
        result.p_value,
        result.adjusted_p_value,
    ):
        assert np.all(np.isnan(column[filtered]))
        assert np.all(np.isfinite(np.delete(column, filtered)))

    # base_mean is reported for every gene, filtered or not.
    assert np.all(np.isfinite(result.base_mean))
    assert collaborators["dispersions"].seen_gene_counts == [2]
    assert collaborators["correction"].seen_sizes == [2]


def test_design_sample_mismatch_is_rejected(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline()

    with pytest.raises(ValueError, match="design has 6 samples"):
        pipeline.run(make_count_matrix(counts), make_design(6), np.array([0.0, 1.0]))


def test_wrong_length_contrast_is_rejected(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline()

    with pytest.raises(ValueError, match=r"contrast must have shape \(2,\)"):
        pipeline.run(make_count_matrix(counts), make_design(4), np.array([0.0, 1.0, 0.0]))


def test_running_twice_is_reproducible_and_leaves_inputs_untouched(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline(dropped=(1,))
    count_matrix = make_count_matrix(counts)
    design = make_design(4)
    contrast = np.array([0.0, 1.0])
    original_counts = counts.copy()
    original_design = design.matrix.copy()
    original_contrast = contrast.copy()

    first = pipeline.run(count_matrix, design, contrast).to_dataframe()
    second = pipeline.run(count_matrix, design, contrast).to_dataframe()

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(count_matrix.counts, original_counts)
    np.testing.assert_array_equal(design.matrix, original_design)
    np.testing.assert_array_equal(contrast, original_contrast)


def test_result_round_trips_through_dataframe_and_significant(counts: np.ndarray) -> None:
    pipeline, _ = build_pipeline(dropped=(3,), p_values=np.array([0.001, 0.4, 0.02]))

    result = pipeline.run(make_count_matrix(counts), make_design(4), np.array([0.0, 1.0]))

    table = result.to_dataframe()
    assert list(table.columns) == [
        "gene_id",
        "base_mean",
        "log2_fold_change",
        "lfc_standard_error",
        "statistic",
        "p_value",
        "adjusted_p_value",
    ]
    assert len(table) == 4

    significant = result.significant(alpha=0.05, min_abs_log2_fold_change=1.0)
    assert list(significant["gene_id"]) == ["gene_0", "gene_2"]
    assert significant["adjusted_p_value"].is_monotonic_increasing
