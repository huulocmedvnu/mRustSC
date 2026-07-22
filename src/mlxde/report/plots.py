"""Diagnostic plots for a differential expression result.

The figures are built through ``matplotlib.figure.Figure`` + ``FigureCanvasAgg``
instead of ``pyplot`` because pyplot keeps every figure it creates in a global
registry and switches the process-wide backend on import: that leaks memory when
plotting in a loop and is unsafe in a server or notebook the caller also uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mlxde.contracts import DifferentialExpressionResult

_SIGNIFICANT_COLOUR = "#E69F00"  # orange/blue instead of red/green: colour-blind safe
_BACKGROUND_COLOUR = "#0072B2"
_GUIDE_COLOUR = "#767676"  # readable on both light and dark backgrounds
_POINT_SIZE = 12.0
_REQUIRED_COLUMNS = ("base_mean", "log2_fold_change", "p_value", "adjusted_p_value")


def volcano_plot(
    result: DifferentialExpressionResult,
    alpha: float = 0.05,
    min_abs_log2_fold_change: float = 1.0,
) -> Figure:
    """Effect size against evidence: x = log2 fold change, y = -log10(p-value)."""
    table, is_significant = _classify(result, alpha, min_abs_log2_fold_change)

    figure, axes = _new_figure()
    log_p_values = -np.log10(table["p_value"].to_numpy())
    _scatter_by_significance(
        axes, table["log2_fold_change"].to_numpy(), log_p_values, is_significant
    )

    for threshold in (-min_abs_log2_fold_change, min_abs_log2_fold_change):
        axes.axvline(threshold, color=_GUIDE_COLOUR, linestyle="--", linewidth=0.8)
    axes.axhline(-np.log10(alpha), color=_GUIDE_COLOUR, linestyle="--", linewidth=0.8)

    axes.set_xlabel("log2 fold change")
    axes.set_ylabel("-log10(p-value)")
    axes.set_title(f"Volcano plot (FDR <= {alpha:g}, |log2FC| >= {min_abs_log2_fold_change:g})")
    axes.legend(loc="best", frameon=False)
    return figure


def ma_plot(result: DifferentialExpressionResult, alpha: float = 0.05) -> Figure:
    """Effect size against expression level: x = log10(base mean), y = log2 fold change."""
    table, is_significant = _classify(result, alpha, min_abs_log2_fold_change=0.0)

    # A gene with zero base mean has no defined position on a log expression axis.
    is_expressed = table["base_mean"].to_numpy() > 0.0
    table, is_significant = table[is_expressed], is_significant[is_expressed]

    figure, axes = _new_figure()
    log_base_mean = np.log10(table["base_mean"].to_numpy())
    _scatter_by_significance(
        axes, log_base_mean, table["log2_fold_change"].to_numpy(), is_significant
    )
    axes.axhline(0.0, color=_GUIDE_COLOUR, linestyle="-", linewidth=0.8)

    axes.set_xlabel("log10(base mean)")
    axes.set_ylabel("log2 fold change")
    axes.set_title(f"MA plot (FDR <= {alpha:g})")
    axes.legend(loc="best", frameon=False)
    return figure


def _classify(
    result: DifferentialExpressionResult, alpha: float, min_abs_log2_fold_change: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop untestable genes and flag the ones passing both thresholds.

    Genes removed by the gene filter carry NaN statistics; they are absent from
    the plots rather than drawn at the origin, which would invent an effect of 0.
    """
    table = result.to_dataframe()
    is_testable = np.isfinite(table.loc[:, list(_REQUIRED_COLUMNS)].to_numpy()).all(axis=1)
    table = table[is_testable]
    is_significant = (table["adjusted_p_value"] <= alpha) & (
        table["log2_fold_change"].abs() >= min_abs_log2_fold_change
    )
    return table, is_significant.to_numpy()


def _new_figure() -> tuple[Figure, Axes]:
    figure = Figure(figsize=(6.0, 5.0), layout="constrained")
    FigureCanvasAgg(figure)  # attaching a canvas makes the figure renderable without pyplot
    return figure, figure.add_subplot(1, 1, 1)


def _scatter_by_significance(
    axes: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    is_significant: np.ndarray,
) -> None:
    axes.scatter(
        x_values[~is_significant],
        y_values[~is_significant],
        s=_POINT_SIZE,
        color=_BACKGROUND_COLOUR,
        alpha=0.4,
        linewidths=0.0,
        label="not significant",
    )
    axes.scatter(
        x_values[is_significant],
        y_values[is_significant],
        s=_POINT_SIZE,
        color=_SIGNIFICANT_COLOUR,
        alpha=0.9,
        linewidths=0.0,
        label="significant",
    )
