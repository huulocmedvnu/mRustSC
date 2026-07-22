"""Quality-control metrics and the legacy normalisation helpers. Owned by feat/qc-metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = ["calculate_qc_metrics", "filter_genes_dispersion", "normalize_per_cell", "sqrt"]


def calculate_qc_metrics(
    adata: AnnData,
    *,
    qc_vars: Sequence[str] = (),
    percent_top: Sequence[int] = (50, 100, 200, 500),
    log1p: bool = True,
    inplace: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Per-cell and per-gene QC metrics, as `scanpy.pp.calculate_qc_metrics`."""
    raise NotImplementedError("feat/qc-metrics")


def normalize_per_cell(
    adata: AnnData, *, counts_per_cell_after: float | None = None, inplace: bool = True
) -> None:
    """scanpy's legacy per-cell normalisation, kept because pipelines still call it."""
    raise NotImplementedError("feat/qc-metrics")


def sqrt(adata: AnnData, *, inplace: bool = True) -> None:
    """Square-root transform, as `scanpy.pp.sqrt`."""
    raise NotImplementedError("feat/qc-metrics")


def filter_genes_dispersion(
    adata: AnnData,
    *,
    flavor: str = "seurat",
    n_top_genes: int | None = None,
    inplace: bool = True,
) -> None:
    """The pre-`highly_variable_genes` dispersion filter scanpy still ships."""
    raise NotImplementedError("feat/qc-metrics")
