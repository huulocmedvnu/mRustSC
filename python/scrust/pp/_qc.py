"""Quality-control metrics and the legacy normalisation helpers. Owned by feat/qc-metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust._shared import _VALUE_DTYPE, _csr_args, _csr_from_parts, _extension
from scrust.pp._basics import filter_cells, highly_variable_genes, normalize_total

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = ["calculate_qc_metrics", "filter_genes_dispersion", "normalize_per_cell", "sqrt"]

# scanpy names its columns after what is being counted; `calculate_qc_metrics`
# exposes both words as arguments, and every caller leaves them at these values.
_EXPR_TYPE = "counts"
_VAR_TYPE = "genes"

# Counts are integers, so scanpy's per-cell and per-gene occupancy counts are too.
_COUNT_DTYPE = np.int32

# `normalize_per_cell` drops cells below this many counts before normalising;
# scanpy exposes it as `min_counts` and defaults it to 1.
_MIN_COUNTS = 1

# The dispersion cut-offs `filter_genes_dispersion` falls back on when no
# `n_top_genes` is given, straight from scanpy's defaults.
_MIN_DISPERSION = 0.5
_MIN_MEAN = 0.0125
_MAX_MEAN = 3.0


def calculate_qc_metrics(
    adata: AnnData,
    *,
    qc_vars: Sequence[str] = (),
    percent_top: Sequence[int] = (50, 100, 200, 500),
    log1p: bool = True,
    inplace: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Per-cell and per-gene QC metrics, as `scanpy.pp.calculate_qc_metrics`.

    `qc_vars` names boolean columns of `adata.var` — `"mt"` for the
    mitochondrial genes, say — and each one adds the totals and the percentage
    of a cell's counts that fall in it.
    """
    import pandas as pd

    # scanpy sorts the requested depths, and the columns come out in that order.
    percent_top = sorted(percent_top) if percent_top else []
    subsets = [np.asarray(adata.var[name], dtype=bool) for name in qc_vars]
    cells, genes = _extension().qc_metrics(*_csr_args(adata.X), percent_top, subsets)

    obs_metrics = pd.DataFrame(index=adata.obs_names)
    obs_metrics[f"n_{_VAR_TYPE}_by_{_EXPR_TYPE}"] = np.asarray(
        cells["n_genes_by_counts"], dtype=_COUNT_DTYPE
    )
    obs_metrics[f"total_{_EXPR_TYPE}"] = np.asarray(cells["total_counts"], dtype=_VALUE_DTYPE)
    for depth, fractions in zip(percent_top, cells["pct_counts_in_top"], strict=True):
        obs_metrics[f"pct_{_EXPR_TYPE}_in_top_{depth}_{_VAR_TYPE}"] = np.asarray(fractions) * 100
    for name, totals in zip(qc_vars, cells["subset_totals"], strict=True):
        obs_metrics[f"total_{_EXPR_TYPE}_{name}"] = totals
        obs_metrics[f"pct_{_EXPR_TYPE}_{name}"] = totals / obs_metrics[f"total_{_EXPR_TYPE}"] * 100

    var_metrics = pd.DataFrame(index=adata.var_names)
    var_metrics[f"n_cells_by_{_EXPR_TYPE}"] = np.asarray(
        genes["n_cells_by_counts"], dtype=_COUNT_DTYPE
    )
    var_metrics[f"mean_{_EXPR_TYPE}"] = np.asarray(genes["mean_counts"], dtype=_VALUE_DTYPE)
    var_metrics[f"pct_dropout_by_{_EXPR_TYPE}"] = np.asarray(
        genes["pct_dropout_by_counts"], dtype=_VALUE_DTYPE
    )
    var_metrics[f"total_{_EXPR_TYPE}"] = np.asarray(genes["total_counts"], dtype=_VALUE_DTYPE)

    if log1p:
        _insert_log1p_columns(
            obs_metrics, [f"n_{_VAR_TYPE}_by_{_EXPR_TYPE}", f"total_{_EXPR_TYPE}"]
        )
        _insert_log1p_columns(obs_metrics, [f"total_{_EXPR_TYPE}_{name}" for name in qc_vars])
        _insert_log1p_columns(var_metrics, [f"mean_{_EXPR_TYPE}", f"total_{_EXPR_TYPE}"])

    if not inplace:
        return obs_metrics, var_metrics
    adata.obs[obs_metrics.columns] = obs_metrics
    adata.var[var_metrics.columns] = var_metrics
    return None


def _insert_log1p_columns(metrics: pd.DataFrame, columns: Sequence[str]) -> None:
    """Add `log1p_<column>` directly after each of `columns`, where scanpy puts it.

    The transform is a display convenience on a column already computed, which is
    why it lives here and not in the core — scanpy derives these columns in its
    Python layer too, from exactly these values.
    """
    for column in columns:
        metrics.insert(
            metrics.columns.get_loc(column) + 1, f"log1p_{column}", np.log1p(metrics[column])
        )


def normalize_per_cell(
    adata: AnnData, *, counts_per_cell_after: float | None = None, inplace: bool = True
) -> None:
    """scanpy's legacy per-cell normalisation, kept because pipelines still call it.

    It is `normalize_total` with two habits of its own, both preserved here: the
    pre-normalisation totals are recorded in `adata.obs["n_counts"]`, and cells
    with no counts at all are dropped instead of being left unscaled. Dropping
    them first is also what makes the default target — the median count — the
    median over the *remaining* cells, as scanpy takes it.

    Those two are part of the legacy semantics rather than of writing back a
    matrix, so they happen whatever `inplace` says; only the normalised matrix is
    returned instead of stored when it is `False`.
    """
    cells, _ = _extension().qc_metrics(*_csr_args(adata.X), [], [])
    adata.obs["n_counts"] = np.asarray(cells["total_counts"], dtype=_VALUE_DTYPE)
    filter_cells(adata, min_counts=_MIN_COUNTS)
    return normalize_total(adata, target_sum=counts_per_cell_after, inplace=inplace)


def sqrt(adata: AnnData, *, inplace: bool = True) -> None:
    """Square-root transform, as `scanpy.pp.sqrt`."""
    rooted = _csr_from_parts(_extension().sqrt(*_csr_args(adata.X)), adata.shape)
    if not inplace:
        return rooted
    adata.X = rooted
    return None


def filter_genes_dispersion(
    adata: AnnData,
    *,
    flavor: str = "seurat",
    n_top_genes: int | None = None,
    inplace: bool = True,
) -> None:
    """The pre-`highly_variable_genes` dispersion filter scanpy still ships.

    The dispersions are `highly_variable_genes`' own, so the two agree by
    construction. What survives from the legacy function is its selection rule:
    without `n_top_genes` it keeps every gene inside a fixed window of mean
    expression and above a dispersion cut-off, rather than a fixed count.

    Unlike scanpy's version this never subsets `adata`; the flag is written to
    `adata.var["highly_variable"]`, which is scanpy's `subset=False` behaviour.
    """
    # The cut-off rule ranks nothing, so it needs the statistics rather than the
    # flag; asking for every gene keeps the one core call that produces both.
    wanted = adata.n_vars if n_top_genes is None else n_top_genes
    table = highly_variable_genes(adata, n_top_genes=wanted, flavor=flavor, inplace=False)
    if n_top_genes is None:
        table["highly_variable"] = _within_dispersion_cutoffs(table)

    if not inplace:
        return table
    for column in table.columns:
        adata.var[column] = table[column]
    return None


def _within_dispersion_cutoffs(table: pd.DataFrame) -> np.ndarray:
    """scanpy's cut-off selection: a window on the mean, a floor on the dispersion.

    A gene whose dispersion is undefined is treated as having none rather than
    being dropped, which is how scanpy keeps it out of the selection.
    """
    means = table["means"].to_numpy()
    dispersions = np.nan_to_num(table["dispersions_norm"].to_numpy())
    return (means > _MIN_MEAN) & (means < _MAX_MEAN) & (dispersions > _MIN_DISPERSION)
