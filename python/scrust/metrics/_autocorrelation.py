"""Spatial autocorrelation over the neighbour graph. Owned by feat/metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from scrust._shared import _csr_args, _dense, _extension

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["gearys_c", "morans_i"]


def gearys_c(
    adata: AnnData, *, vals: Any = None, use_graph: str = "connectivities", device: str = "auto"
) -> np.ndarray:
    """Geary's C for each gene over the neighbour graph, as `scanpy.metrics.gearys_c`.

    See `morans_i` for how `vals` is resolved. Low values mean strong spatial
    correlation; a constant feature has no statistic and comes back as `nan`.
    """
    return _autocorrelation("gearys_c", adata, vals, use_graph, device)


def morans_i(
    adata: AnnData, *, vals: Any = None, use_graph: str = "connectivities", device: str = "auto"
) -> np.ndarray:
    """Moran's I for each gene over the neighbour graph, as `scanpy.metrics.morans_i`.

    `vals` may be omitted, in which case every gene of `adata.X` is scored, or it
    may name one gene or one `obs` column, name several of them, or be an
    explicit array. As in scanpy, an explicit 2-D array is `(n_features,
    n_cells)` and a single feature returns a scalar rather than a length-1 array.
    """
    return _autocorrelation("morans_i", adata, vals, use_graph, device)


def _autocorrelation(
    name: str, adata: AnnData, vals: Any, use_graph: str, device: str
) -> np.ndarray:
    """Both statistics take the same arguments and differ only in the core call."""
    if use_graph not in adata.obsp:
        raise KeyError(f"adata.obsp has no {use_graph!r}; run scrust.pp.neighbors first")
    graph = adata.obsp[use_graph]

    features, is_scalar = _features(adata, vals)
    if features.shape[0] != adata.n_obs:
        raise ValueError(f"vals must cover {adata.n_obs} cells, got {features.shape[0]}")

    statistic = getattr(_extension(), name)(*_csr_args(graph), *_csr_args(features), device)
    return statistic[0] if is_scalar else statistic


def _features(adata: AnnData, vals: Any) -> tuple[Any, bool]:
    """Resolve `vals` to a cells-by-features matrix, and whether it is a single one.

    The core scores columns of a cells-by-genes matrix, which is `adata.X`'s own
    layout, so the only case that needs transposing is scanpy's explicit
    `(n_features, n_cells)` array.
    """
    if vals is None:
        return adata.X, False
    if isinstance(vals, str):
        return _vector(adata, vals).reshape(-1, 1), True
    if _is_name_sequence(vals):
        return np.column_stack([_vector(adata, name) for name in vals]), False

    # A sparse operand stays sparse: densifying it here is exactly the
    # `(n_cells, n_genes)` intermediate the core goes to lengths to avoid.
    array = vals if sp.issparse(vals) else np.asarray(vals)
    if array.ndim == 1:
        return array.reshape(-1, 1), True
    if array.ndim == 2:
        return array.T, False
    raise ValueError(f"vals must be 1- or 2-dimensional, got {array.ndim} dimensions")


def _vector(adata: AnnData, name: str) -> np.ndarray:
    """One `obs` column or one gene, resolved the way `adata.obs_vector` resolves it."""
    if name in adata.obs:
        return np.asarray(adata.obs[name])
    if name in adata.var_names:
        return _dense(adata[:, name].X).ravel()
    raise KeyError(f"{name!r} is neither a column of adata.obs nor a gene in adata.var_names")


def _is_name_sequence(vals: Any) -> bool:
    """A list of gene or `obs` names, as opposed to a numeric array."""
    return (
        not hasattr(vals, "ndim")  # excludes numpy and scipy.sparse alike
        and hasattr(vals, "__iter__")
        and all(isinstance(name, str) for name in vals)
    )
