"""Tests for the reporting layer: plots and the textual summary."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as pyplot
import numpy as np
import pytest
from matplotlib.figure import Figure

from mlxde.contracts import DifferentialExpressionResult
from mlxde.report.plots import ma_plot, volcano_plot
from mlxde.report.summary import summarize

PlotFunction = Callable[[DifferentialExpressionResult], Figure]


def make_result(
    log2_fold_change: list[float],
    adjusted_p_value: list[float],
    base_mean: list[float] | None = None,
) -> DifferentialExpressionResult:
    """Build a result directly, so the reporting layer is testable without a pipeline."""
    log2_fold_change_array = np.array(log2_fold_change, dtype=float)
    adjusted = np.array(adjusted_p_value, dtype=float)
    n_genes = len(log2_fold_change_array)
    return DifferentialExpressionResult(
        gene_ids=np.array([f"gene_{index:03d}" for index in range(n_genes)]),
        base_mean=np.full(n_genes, 100.0) if base_mean is None else np.array(base_mean, float),
        log2_fold_change=log2_fold_change_array,
        log2_fold_change_standard_error=np.full(n_genes, 0.2),
        statistic=log2_fold_change_array / 0.2,
        p_value=adjusted / 2.0,
        adjusted_p_value=adjusted,
    )


@pytest.fixture
def mixed_result() -> DifferentialExpressionResult:
    """Two significant up, one significant down, one weak effect, one filtered gene."""
    return make_result(
        log2_fold_change=[3.0, 1.5, -2.0, 0.2, np.nan],
        adjusted_p_value=[1e-6, 1e-4, 1e-5, 1e-3, np.nan],
        base_mean=[500.0, 200.0, 300.0, 50.0, np.nan],
    )


def significant_points(figure: Figure) -> int:
    return len(figure.axes[0].collections[1].get_offsets())


def background_points(figure: Figure) -> int:
    return len(figure.axes[0].collections[0].get_offsets())


def test_volcano_plot_returns_labelled_figure(mixed_result: DifferentialExpressionResult) -> None:
    figure = volcano_plot(mixed_result)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert axes.get_xlabel() == "log2 fold change"
    assert axes.get_ylabel() == "-log10(p-value)"
    assert significant_points(figure) + background_points(figure) > 0


def test_ma_plot_returns_labelled_figure(mixed_result: DifferentialExpressionResult) -> None:
    figure = ma_plot(mixed_result)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert axes.get_xlabel() == "log10(base mean)"
    assert axes.get_ylabel() == "log2 fold change"
    assert significant_points(figure) + background_points(figure) > 0


def test_volcano_highlights_exactly_the_genes_passing_both_thresholds(
    mixed_result: DifferentialExpressionResult,
) -> None:
    expected = len(mixed_result.significant(alpha=0.05, min_abs_log2_fold_change=1.0))

    assert expected == 3
    assert significant_points(volcano_plot(mixed_result)) == expected


def test_ma_plot_highlights_every_significant_gene(
    mixed_result: DifferentialExpressionResult,
) -> None:
    expected = len(mixed_result.significant(alpha=0.05))

    assert expected == 4
    assert significant_points(ma_plot(mixed_result)) == expected


def test_filtered_genes_are_dropped_not_plotted_as_zeros(
    mixed_result: DifferentialExpressionResult,
) -> None:
    figure = volcano_plot(mixed_result)

    n_plotted = significant_points(figure) + background_points(figure)
    assert n_plotted == len(mixed_result.gene_ids) - 1


@pytest.mark.parametrize("plot", [volcano_plot, ma_plot])
def test_all_filtered_result_produces_empty_but_valid_figure(plot: PlotFunction) -> None:
    empty = make_result(
        log2_fold_change=[np.nan, np.nan],
        adjusted_p_value=[np.nan, np.nan],
        base_mean=[np.nan, np.nan],
    )

    figure = plot(empty)

    assert len(figure.axes) == 1
    assert significant_points(figure) + background_points(figure) == 0


def test_ma_plot_drops_genes_without_expression() -> None:
    figure = ma_plot(make_result([1.0, 2.0], [1e-3, 1e-3], base_mean=[0.0, 10.0]))

    assert significant_points(figure) + background_points(figure) == 1


@pytest.mark.parametrize("plot", [volcano_plot, ma_plot])
def test_plots_do_not_touch_the_pyplot_figure_registry(
    plot: PlotFunction, mixed_result: DifferentialExpressionResult
) -> None:
    plot(mixed_result)
    plot(mixed_result)

    assert len(pyplot.get_fignums()) == 0


def test_summarize_reports_counts_for_a_hand_built_result(
    mixed_result: DifferentialExpressionResult,
) -> None:
    report = summarize(mixed_result)

    assert "Genes in result:      5" in report
    assert "Genes tested:         4" in report
    assert "Genes filtered out:   1" in report
    assert "Significant at FDR <= 0.05: 4" in report
    assert "up-regulated:       3" in report
    assert "down-regulated:     1" in report
    assert "gene_000" in report


def test_summarize_counts_agree_with_significant(
    mixed_result: DifferentialExpressionResult,
) -> None:
    n_significant = len(mixed_result.significant(alpha=0.001))

    assert f"Significant at FDR <= 0.001: {n_significant}" in summarize(mixed_result, alpha=0.001)


def test_summarize_handles_no_significant_genes() -> None:
    report = summarize(make_result([0.1, -0.1], [0.9, 0.8]))

    assert "Significant at FDR <= 0.05: 0" in report
    assert "No significant genes at FDR <= 0.05." in report
    assert "Strongest hits" not in report


def test_summarize_returns_a_string_without_printing(
    mixed_result: DifferentialExpressionResult, capsys: pytest.CaptureFixture[str]
) -> None:
    report = summarize(mixed_result)

    assert isinstance(report, str)
    assert capsys.readouterr().out == ""
