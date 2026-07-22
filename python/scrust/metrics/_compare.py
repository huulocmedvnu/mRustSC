"""Comparing labellings. Owned by feat/metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from natsort import natsorted
from pandas.api.types import CategoricalDtype

from scrust._shared import _csr_args, _extension

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["confusion_matrix", "modularity"]

# scanpy's default when it hands a graph to igraph: plain Newman modularity.
_RESOLUTION = 1.0


def confusion_matrix(
    orig: Any, new: Any, data: Any = None, *, normalize: bool = True
) -> pd.DataFrame:
    """Contingency table of two labellings, as `scanpy.metrics.confusion_matrix`.

    `orig` and `new` are either label arrays or, when `data` is given, column
    names in it. Rows are the original labels and columns the new ones, in
    category order where the labels are categorical and natural sort order
    otherwise. `normalize` divides each row by its own total.
    """
    if data is not None:
        orig = data[orig] if isinstance(orig, str) else orig
        new = data[new] if isinstance(new, str) else new
    orig, new = pd.Series(orig), pd.Series(new)
    if len(orig) != len(new):
        raise ValueError(f"orig and new must be the same length, got {len(orig)} and {len(new)}")

    # One shared label set for both axes, in first-seen order, so a label that
    # occurs in only one of the two labellings still gets a row and a column.
    labels = pd.unique(np.concatenate((orig.to_numpy(), new.to_numpy())))
    counts = _contingency(orig, new, labels)
    if normalize:
        totals = counts.sum(axis=1)[:, np.newaxis]
        counts = np.divide(counts, totals, where=totals != 0, out=counts.astype(np.float64))

    table = pd.DataFrame(
        counts,
        index=pd.Index(labels, name="Original labels" if orig.name is None else orig.name),
        columns=pd.Index(labels, name="New Labels" if new.name is None else new.name),
    )
    return table.loc[np.array(_ordered(orig)), np.array(_ordered(new))]


def _contingency(orig: pd.Series, new: pd.Series, labels: np.ndarray) -> np.ndarray:
    """Counts of every `(orig, new)` pair, over the shared `labels` on both axes."""
    size = len(labels)
    rows = pd.Categorical(orig, categories=labels).codes.astype(np.intp)
    columns = pd.Categorical(new, categories=labels).codes.astype(np.intp)
    flat = np.bincount(rows * size + columns, minlength=size * size)
    return flat.reshape(size, size)


def _ordered(labels: pd.Series) -> Any:
    """The order an axis is presented in: category order, else natural sort."""
    if isinstance(labels.dtype, CategoricalDtype):
        return labels.cat.categories
    return natsorted(pd.unique(labels))


def modularity(adata: AnnData, keys: str, *, neighbors_key: str = "neighbors") -> float:
    """Newman modularity of a labelling on the neighbour graph.

    `keys` names a column of `adata.obs`; its categories become the partition.
    The graph is the connectivities `neighbors_key` points at, so a labelling and
    the graph it was found on are always scored together.
    """
    graph = adata.obsp[_connectivities_key(adata, neighbors_key)]
    if keys not in adata.obs:
        raise KeyError(f"adata.obs has no {keys!r}")
    labels = pd.Categorical(adata.obs[keys]).codes.astype(np.uint32)
    return _extension().modularity(*_csr_args(graph), labels, _RESOLUTION)


def _connectivities_key(adata: AnnData, neighbors_key: str) -> str:
    """Where `pp.neighbors` recorded its connectivities, as scanpy resolves it."""
    if neighbors_key not in adata.uns:
        raise KeyError(f"adata.uns has no {neighbors_key!r}; run scrust.pp.neighbors first")
    return adata.uns[neighbors_key].get("connectivities_key", "connectivities")
