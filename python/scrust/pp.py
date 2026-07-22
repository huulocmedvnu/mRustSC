"""Preprocessing, mirroring `scanpy.pp`. Owned by feat/python-pp.

This module is AnnData plumbing and defaults only: it pulls a matrix out of an
`AnnData`, hands it to the Rust core as flat typed arrays, and writes the result
back into the slot scanpy uses. It also holds the private helpers that
`scrust.tl` reuses, so the conventions live in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    import pandas as pd
    from anndata import AnnData

__all__ = [
    "filter_cells",
    "filter_genes",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_total",
    "pca",
    "scale",
]

# The three CSR arrays cross the boundary with these dtypes: `f32` values match
# the core's "f32 throughout" rule, 32-bit offsets match its index type.
_INDEX_DTYPE = np.uint32
_VALUE_DTYPE = np.float32

# The contract gives no `device` argument to these functions, so they always ask
# the core to choose.
_DEFAULT_DEVICE = "auto"


def _extension() -> ModuleType:
    """Import the compiled core on use, not on import.

    `scrust.pp` must stay importable while the extension is still being built,
    and tests replace it wholesale in `sys.modules`.
    """
    from scrust import _scrust

    return _scrust


def _as_csr(matrix: Any) -> sp.csr_matrix:
    """Return `matrix` as CSR, the one format the Rust core accepts."""
    if isinstance(matrix, sp.csr_matrix):
        return matrix
    if sp.issparse(matrix):
        return matrix.tocsr()
    if isinstance(matrix, np.ndarray):
        return sp.csr_matrix(matrix)
    raise TypeError(
        f"adata.X must be a scipy.sparse matrix or a numpy.ndarray, got {type(matrix).__name__}"
    )


def _csr_args(matrix: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Flatten a matrix into the `(indptr, indices, values, n_cols)` call convention."""
    csr = _as_csr(matrix)
    return (
        csr.indptr.astype(_INDEX_DTYPE, copy=False),
        csr.indices.astype(_INDEX_DTYPE, copy=False),
        csr.data.astype(_VALUE_DTYPE, copy=False),
        csr.shape[1],
    )


def _csr_from_parts(parts: Sequence[np.ndarray], shape: tuple[int, int]) -> sp.csr_matrix:
    """Rebuild a CSR matrix from a sparse return value.

    Only the first three entries are read, so a core that echoes `n_cols` back
    as a fourth element is accepted unchanged.
    """
    indptr, indices, values = parts[0], parts[1], parts[2]
    return sp.csr_matrix((values, indices, indptr), shape=shape)


def _dense(matrix: Any) -> np.ndarray:
    """Return a C-contiguous `f32` dense view, for the cell-by-k embeddings."""
    array = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return np.ascontiguousarray(array, dtype=_VALUE_DTYPE)


def _representation(adata: AnnData, use_rep: str) -> np.ndarray:
    """Resolve scanpy's `use_rep` to a dense embedding."""
    if use_rep == "X":
        return _dense(adata.X)
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm has no {use_rep!r}; run scrust.pp.pca first or pass use_rep")
    return _dense(adata.obsm[use_rep])


def filter_cells(
    adata: AnnData,
    *,
    min_genes: int | None = None,
    min_counts: int | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Filter out cells below `min_genes` expressed genes or `min_counts` total counts."""
    if min_genes is None and min_counts is None:
        raise ValueError("provide at least one of min_genes or min_counts")
    mask = np.asarray(
        _extension().filter_cells(*_csr_args(adata.X), min_genes, min_counts), dtype=bool
    )
    if not inplace:
        return mask
    adata._inplace_subset_obs(mask)
    return None


def filter_genes(
    adata: AnnData,
    *,
    min_cells: int | None = None,
    min_counts: int | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Filter out genes seen in fewer than `min_cells` cells or below `min_counts` counts."""
    if min_cells is None and min_counts is None:
        raise ValueError("provide at least one of min_cells or min_counts")
    mask = np.asarray(
        _extension().filter_genes(*_csr_args(adata.X), min_cells, min_counts), dtype=bool
    )
    if not inplace:
        return mask
    adata._inplace_subset_var(mask)
    return None


def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    inplace: bool = True,
) -> sp.csr_matrix | None:
    """Normalise every cell to `target_sum` counts, or to the median count if `None`."""
    parts = _extension().normalize_total(*_csr_args(adata.X), target_sum, _DEFAULT_DEVICE)
    normalized = _csr_from_parts(parts, adata.shape)
    if not inplace:
        return normalized
    adata.X = normalized
    return None


def log1p(adata: AnnData, *, inplace: bool = True) -> sp.csr_matrix | None:
    """Apply `log(1 + x)` to the count matrix."""
    logged = _csr_from_parts(_extension().log1p(*_csr_args(adata.X)), adata.shape)
    if not inplace:
        return logged
    adata.X = logged
    # scanpy records the base so downstream tools know the data is logarithmised.
    adata.uns["log1p"] = {"base": None}
    return None


def highly_variable_genes(
    adata: AnnData,
    *,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """Select the `n_top_genes` most variable genes."""
    import pandas as pd

    result = _extension().highly_variable_genes(
        *_csr_args(adata.X), n_top_genes, flavor, _DEFAULT_DEVICE
    )
    table = pd.DataFrame(
        {
            "highly_variable": np.asarray(result["highly_variable"], dtype=bool),
            "means": np.asarray(result["means"], dtype=_VALUE_DTYPE),
            "dispersions_norm": np.asarray(result["normalised_dispersions"], dtype=_VALUE_DTYPE),
        },
        index=adata.var_names,
    )
    if not inplace:
        return table
    for column in table.columns:
        adata.var[column] = table[column]
    return None


def scale(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Scale genes to unit variance, optionally centring and clipping at `max_value`."""
    scaled = np.asarray(
        _extension().scale(*_csr_args(adata.X), zero_center, max_value, _DEFAULT_DEVICE),
        dtype=_VALUE_DTYPE,
    )
    if not inplace:
        return scaled
    adata.X = scaled
    return None


def pca(
    adata: AnnData,
    *,
    n_comps: int = 50,
    zero_center: bool = True,
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Principal component analysis by randomised SVD."""
    result = _extension().pca(*_csr_args(adata.X), n_comps, zero_center, random_state, device)
    adata.obsm["X_pca"] = np.asarray(result["embedding"], dtype=_VALUE_DTYPE)
    # The core returns components as (n_components, n_genes); scanpy stores the transpose.
    adata.varm["PCs"] = np.asarray(result["components"], dtype=_VALUE_DTYPE).T.copy()
    adata.uns["pca"] = {
        "variance_ratio": np.asarray(result["explained_variance_ratio"], dtype=_VALUE_DTYPE),
        "variance": np.asarray(result["explained_variance"], dtype=_VALUE_DTYPE),
        "params": {"zero_center": zero_center, "n_comps": n_comps, "random_state": random_state},
    }


def neighbors(
    adata: AnnData,
    *,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    device: str = "auto",
) -> None:
    """Build the k-nearest-neighbour graph and its UMAP connectivities."""
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}")
    extension = _extension()
    # scanpy counts the cell itself among its n_neighbors; the core does not.
    indices, distances = extension.knn(_representation(adata, use_rep), n_neighbors - 1, device)
    indices = np.asarray(indices)
    distances = np.asarray(distances, dtype=_VALUE_DTYPE)

    # knn returns one fixed-width row per cell, so the CSR offsets are the row starts.
    n_obs, k = indices.shape
    indptr = np.arange(0, n_obs * k + 1, k, dtype=_INDEX_DTYPE)
    adata.obsp["distances"] = _csr_from_parts(
        (indptr, indices.ravel(), distances.ravel()), (n_obs, n_obs)
    )
    adata.obsp["connectivities"] = _csr_from_parts(
        extension.connectivities(indices, distances), (n_obs, n_obs)
    )
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": n_neighbors, "method": "umap", "use_rep": use_rep},
    }
