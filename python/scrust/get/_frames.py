"""Pulling tidy frames out of an AnnData. Owned by feat/accessors."""

from __future__ import annotations

import warnings
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["aggregate", "obs_df", "rank_genes_groups_df", "var_df"]

# The fields `tl.rank_genes_groups` writes, in the column order scanpy's frame has.
_DE_COLUMNS = ("names", "scores", "logfoldchanges", "pvals", "pvals_adj")
# logreg reports a coefficient and nothing else, so its frame is narrower.
_LOGREG_COLUMNS = ("names", "scores")

_AGGREGATIONS = ("count_nonzero", "mean", "median", "sum", "var")
# scanpy's default degrees of freedom for `var`: the sample, not population, variance.
_DOF = 1


def obs_df(
    adata: AnnData,
    keys: Sequence[str] = (),
    *,
    obsm_keys: Sequence[tuple[str, int]] = (),
    layer: str | None = None,
) -> pd.DataFrame:
    """Per-cell frame of genes, `obs` columns and `obsm` slices, as `scanpy.get.obs_df`."""
    keys = _as_key_list(keys)
    obs_columns, gene_keys = _split_keys(adata.obs, adata.var_names, dim="obs", keys=keys)

    frame = pd.DataFrame(index=adata.obs_names)
    if gene_keys:
        expression = _slice_along(
            _matrix(adata, layer), adata.var_names, gene_keys, axis=1, backed=adata.isbacked
        )
        frame = pd.concat(
            [frame, pd.DataFrame(expression, columns=gene_keys, index=adata.obs_names)], axis=1
        )
    if obs_columns:
        frame = pd.concat([frame, adata.obs[obs_columns]], axis=1)
    if keys:
        # Back to the caller's order, repeating any key they asked for twice.
        frame = frame[keys]

    _append_slices(frame, adata.obsm, obsm_keys)
    return frame


def var_df(
    adata: AnnData, keys: Sequence[str] = (), *, varm_keys: Sequence[tuple[str, int]] = ()
) -> pd.DataFrame:
    """Per-gene frame, as `scanpy.get.var_df`."""
    keys = _as_key_list(keys)
    var_columns, cell_keys = _split_keys(adata.var, adata.obs_names, dim="var", keys=keys)

    frame = pd.DataFrame(index=adata.var_names)
    if cell_keys:
        expression = _slice_along(
            adata.X, adata.obs_names, cell_keys, axis=0, backed=adata.isbacked
        ).T
        frame = pd.concat(
            [frame, pd.DataFrame(expression, columns=cell_keys, index=adata.var_names)], axis=1
        )
    if var_columns:
        frame = pd.concat([frame, adata.var[var_columns]], axis=1)
    if keys:
        frame = frame[keys]

    _append_slices(frame, adata.varm, varm_keys)
    return frame


def rank_genes_groups_df(
    adata: AnnData,
    group: str | Sequence[str] | None,
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
) -> pd.DataFrame:
    """The differential expression result as a tidy frame, as `scanpy.get.rank_genes_groups_df`."""
    result = adata.uns[key]  # a missing key raises KeyError naming it, as scanpy does
    if group is None:
        groups = list(result["names"].dtype.names)
    else:
        groups = [group] if isinstance(group, str) else [str(name) for name in group]

    is_logreg = result["params"]["method"] == "logreg"
    columns = list(_LOGREG_COLUMNS if is_logreg else _DE_COLUMNS)

    # One structured array per field, each with one column per group; stacking the
    # group level turns the (gene rank x group) grid into one row per gene per group.
    frame = pd.concat(
        [pd.DataFrame(result[column])[groups] for column in columns],
        axis=1,
        names=[None, "group"],
        keys=columns,
    )
    frame = frame.stack(level=1, future_stack=True).reset_index()
    frame["group"] = pd.Categorical(frame["group"], categories=groups)
    frame = frame.sort_values(["group", "level_0"]).drop(columns="level_0")

    if not is_logreg:
        if pval_cutoff is not None:
            frame = frame[frame["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None:
            frame = frame[frame["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None:
            frame = frame[frame["logfoldchanges"] < log2fc_max]

    if len(groups) == 1:
        # scanpy drops the constant column for a single group and callers rely on it.
        frame = frame.drop(columns="group")
    return frame.reset_index(drop=True)


def aggregate(
    adata: AnnData,
    by: str | Sequence[str],
    func: str | Sequence[str],
    *,
    axis: int = 0,
    layer: str | None = None,
    device: str = "auto",
) -> AnnData:
    """Group cells and reduce, as `scanpy.get.aggregate`.

    `device` is accepted for signature parity only: the reductions here are
    scipy sparse products and per-group medians, not core algorithms, so there is
    no Rust or Metal path behind them yet.
    """
    functions = _as_key_list(func)
    if unknown := sorted(set(functions) - set(_AGGREGATIONS)):
        raise ValueError(f"func {unknown} is not one of {list(_AGGREGATIONS)}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

    # Grouping is always over the rows of `data`, so aggregating genes transposes first.
    data = _matrix(adata, layer)
    grouping_frame, feature_frame = (adata.obs, adata.var) if axis == 0 else (adata.var, adata.obs)
    if axis == 1:
        data = data.T

    grouping, labels = _combine_categories(grouping_frame, by)
    counts = np.bincount(grouping.codes[grouping.codes >= 0], minlength=len(grouping.categories))
    labels["n_obs_aggregated"] = counts

    result = AnnData(
        layers=_reduce_groups(data, grouping, counts, functions), obs=labels, var=feature_frame
    )
    return result.T if axis == 1 else result


def _as_key_list(keys: str | Sequence[str]) -> list[str]:
    """Accept a lone key as well as a sequence of them, as every scanpy accessor does."""
    return [keys] if isinstance(keys, str) else list(keys)


def _matrix(adata: AnnData, layer: str | None) -> Any:
    """The expression matrix an accessor reads: a named layer, or `X`."""
    return adata.X if layer is None else adata.layers[layer]


def _split_keys(
    dim_frame: pd.DataFrame,
    alt_index: pd.Index,
    *,
    dim: Literal["obs", "var"],
    keys: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split `keys` into annotation columns and names of the opposite axis.

    A key found in both is an error rather than a preference: silently choosing
    between a gene and an identically named annotation column corrupts data.
    """
    alt_dim = "var" if dim == "obs" else "obs"
    if not dim_frame.columns.is_unique:
        duplicated = dim_frame.columns[dim_frame.columns.duplicated()].tolist()
        raise ValueError(
            f"adata.{dim} contains duplicated columns. Please rename or remove them "
            f"first.\nDuplicated columns {duplicated}"
        )
    if not alt_index.is_unique:
        raise ValueError(
            f"adata.{alt_dim}_names contains duplicated items\nPlease rename these "
            f"{alt_dim} names first for example using `adata.{alt_dim}_names_make_unique()`"
        )

    column_keys: list[str] = []
    index_keys: list[str] = []
    not_found: list[str] = []
    for key in dict.fromkeys(keys):  # unique, in the order given
        in_columns, in_index = key in dim_frame.columns, key in alt_index
        if in_columns and in_index:
            raise KeyError(
                f"The key {key!r} is found in both adata.{dim} and adata.{alt_dim}_names."
            )
        if in_columns:
            column_keys.append(key)
        elif in_index:
            index_keys.append(key)
        else:
            not_found.append(key)
    if not_found:
        raise KeyError(
            f"Could not find keys {not_found!r} in columns of `adata.{dim}` or in "
            f"adata.{alt_dim}_names."
        )
    return column_keys, index_keys


def _slice_along(
    matrix: Any, dim_names: pd.Index, keys: Sequence[str], *, axis: int, backed: bool
) -> np.ndarray:
    """Take the rows or columns named by `keys`, densified and in the order asked for."""
    positions = dim_names.get_indexer(keys)
    indexer: list[Any] = [slice(None), slice(None)]
    if backed:
        # A backed sparse dataset only accepts increasing indices, so read in sorted
        # order and undo the sort afterwards rather than returning shuffled columns.
        sorted_order = np.argsort(positions)
        restore: list[Any] = [slice(None), slice(None)]
        indexer[axis] = positions[sorted_order]
        restore[axis] = np.argsort(sorted_order)
        block = matrix[tuple(indexer)][tuple(restore)]
    else:
        indexer[axis] = positions
        block = matrix[tuple(indexer)]
    return block.toarray() if sp.issparse(block) else np.asarray(block)


def _append_slices(
    frame: pd.DataFrame, mapping: Mapping[str, Any], entries: Sequence[tuple[str, int]]
) -> None:
    """Add one `key-index` column per `(key, index)` pair, as scanpy names them."""
    for key, index in entries:
        value = mapping[key]
        if isinstance(value, pd.DataFrame):
            column = value.loc[:, index]
        elif sp.issparse(value):
            column = np.ravel(value[:, index].toarray())
        else:
            column = np.ravel(np.asarray(value)[:, index])
        frame[f"{key}-{index}"] = column


def _combine_categories(
    annotation: pd.DataFrame, by: str | Sequence[str]
) -> tuple[pd.Categorical, pd.DataFrame]:
    """Return the grouping as one categorical, plus a frame labelling each group.

    Several `by` columns combine into the product of their categories, joined by
    `_` and ordered as `itertools.product` yields them, which is the group order
    scanpy produces.
    """
    columns = _as_key_list(by)
    if missing := [column for column in columns if column not in annotation]:
        raise KeyError(f"grouping columns {missing} are not in the annotation frame")
    grouped = {
        column: pd.Categorical(annotation[column]).remove_unused_categories() for column in columns
    }

    combinations = list(product(*(values.categories for values in grouped.values())))
    names = pd.Index(["_".join(map(str, combination)) for combination in combinations])
    labels = pd.DataFrame(
        {
            column: pd.Categorical(
                [combination[position] for combination in combinations],
                categories=grouped[column].categories,
            )
            for position, column in enumerate(columns)
        },
        index=names,
    )

    # Mixed-radix encoding of the per-column codes, which is exactly `product` order.
    codes = np.zeros(len(annotation), dtype=np.int64)
    unassigned = np.zeros(len(annotation), dtype=bool)
    for values in grouped.values():
        codes = codes * len(values.categories) + values.codes
        unassigned |= values.codes < 0
    codes[unassigned] = -1

    grouping = pd.Categorical.from_codes(codes, categories=names).remove_unused_categories()
    return grouping, labels.loc[grouping.categories]


def _reduce_groups(
    data: Any, grouping: pd.Categorical, counts: np.ndarray, functions: Sequence[str]
) -> dict[str, np.ndarray]:
    """Reduce the rows of `data` belonging to each group, one array per function."""
    requested = set(functions)
    indicator = _indicator(grouping)
    per_group = counts[:, None]
    reduced: dict[str, np.ndarray] = {}

    if requested & {"sum", "mean", "var"}:
        totals = _densify(indicator @ data)
    if "sum" in requested:
        reduced["sum"] = totals
    if requested & {"mean", "var"}:
        means = totals / per_group
    if "mean" in requested:
        reduced["mean"] = means
    if "var" in requested:
        reduced["var"] = _variance(indicator, data, means, counts, grouping.categories)
    if "count_nonzero" in requested:
        reduced["count_nonzero"] = _densify(indicator @ _nonzero(data)).astype(np.int64)
    if "median" in requested:
        reduced["median"] = _medians(data, grouping)
    return reduced


def _indicator(grouping: pd.Categorical) -> sp.csr_matrix:
    """Group-by-cell 0/1 matrix; `indicator @ data` sums each group in one sparse pass."""
    assigned = np.flatnonzero(grouping.codes >= 0)
    return sp.csr_matrix(
        (np.ones(assigned.size), (grouping.codes[assigned], assigned)),
        shape=(len(grouping.categories), len(grouping)),
    )


def _variance(
    indicator: sp.csr_matrix,
    data: Any,
    means: np.ndarray,
    counts: np.ndarray,
    categories: pd.Index,
) -> np.ndarray:
    """Sample variance per group, `nan` where a group is too small to have one."""
    squares = _densify(indicator @ _squared(data))
    # Expanding sum((x - m)^2) to sum(x^2) - n * m^2 keeps the pass over the data
    # sparse; the alternative centres every entry and so densifies the matrix.
    deviations = squares - counts[:, None] * means**2
    if (too_small := counts <= _DOF).any():
        warnings.warn(
            f"groups {categories[too_small].tolist()} have {_DOF} or fewer observations, "
            "so their var is nan",
            RuntimeWarning,
            stacklevel=2,
        )
    return deviations / np.where(counts > _DOF, counts - _DOF, np.nan)[:, None]


def _medians(data: Any, grouping: pd.Categorical) -> np.ndarray:
    """Median per group, densifying one group's rows at a time.

    A median needs every value of a group at once, so this is the one reduction
    that cannot stream; a group is a row block, never the whole matrix.
    """
    return np.stack(
        [
            np.median(_densify(data[grouping.codes == code]), axis=0)
            for code in range(len(grouping.categories))
        ]
    )


def _densify(matrix: Any) -> np.ndarray:
    """A dense `float64` array, the accumulator width every reduction reports in."""
    array = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return array.astype(np.float64, copy=False)


def _squared(data: Any) -> Any:
    """Elementwise square, keeping a sparse matrix sparse."""
    return data.power(2) if sp.issparse(data) else np.asarray(data) ** 2


def _nonzero(data: Any) -> Any:
    """Elementwise 0/1 mask, keeping a sparse matrix sparse."""
    return (data != 0).astype(np.uint8)
