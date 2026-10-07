"""Filtering, normalisation, scaling, HVG, PCA and neighbours.

This module is AnnData plumbing and defaults only: it pulls a matrix out of an
`AnnData`, hands it to the Rust core as flat typed arrays, and writes the result
back into the slot scanpy uses. It also holds the private helpers that
`metalcyte.tl` reuses, so the conventions live in exactly one place.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from metalcyte._shared import (
    _INDEX_DTYPE,
    _VALUE_DTYPE,
    _csr_args,
    _csr_from_parts,
    _default_device,
    _extension,
    _representation,
    _resolve_device,
)

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

#: Cells above which `pp.neighbors(method="auto")` switches from the exact search to
#: NN-descent. Measured on an M3 Pro (benches/knn_methods.py): the exact search is
#: faster on the GPU up to about 100 000 cells and on the CPU up to about 50 000; at
#: 250 000 cells NN-descent is 2.4 times faster than the GPU and 6 times faster than the
#: CPU search. The threshold sits where the exact graph, which matches scanpy's search
#: cell for cell, stops being cheap.
APPROXIMATE_FROM = 200_000

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


def _fast_path(ext, name: str) -> bool:
    """Whether the zero-copy multi-core kernel `name` is available and not switched off.

    `METALCYTE_FORCE_COPY=1` disables every fast path so the ablation in
    `benches/ablation.py` can measure what the in-place kernels buy; it is not a
    user-facing setting.
    """
    return hasattr(ext, name) and os.environ.get("METALCYTE_FORCE_COPY") != "1"


def _fast_csr_copy(matrix) -> sp.csr_matrix | None:
    """A float32 CSR copy of `matrix` for the zero-copy kernels, or `None` to fall back.

    One memcpy per array (or one cast when the values are not float32); the index
    arrays keep scipy's own dtype, which the kernels read directly. The input is never
    modified, so a layer that shares `adata.X` keeps its counts.
    """
    if not (sp.issparse(matrix) and matrix.format == "csr"):
        return None  # csr_matrix or csr_array; anything else takes the general path
    if matrix.indices.dtype not in (np.int32, np.int64, np.uint32):
        return None
    out = matrix.astype(_VALUE_DTYPE, copy=True)
    if not (out.data.flags.c_contiguous and out.indptr.flags.c_contiguous):
        return None
    return out


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
    ext = _extension()
    if _fast_path(ext, "filter_cells_mask") and sp.isspmatrix_csr(adata.X):
        values = np.ascontiguousarray(adata.X.data, dtype=np.float32)
        mask = np.asarray(ext.filter_cells_mask(adata.X.indptr, values, min_genes, min_counts))
    else:
        mask = np.asarray(ext.filter_cells(*_csr_args(adata.X), min_genes, min_counts), dtype=bool)
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
    ext = _extension()
    if _fast_path(ext, "filter_genes_mask") and sp.isspmatrix_csr(adata.X):
        x = adata.X
        values = np.ascontiguousarray(x.data, dtype=np.float32)
        mask = np.asarray(
            ext.filter_genes_mask(x.indptr, x.indices, values, x.shape[1], min_cells, min_counts)
        )
    else:
        mask = np.asarray(ext.filter_genes(*_csr_args(adata.X), min_cells, min_counts), dtype=bool)
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
    if getattr(adata, "isbacked", False):
        from metalcyte._backed import normalize_total_backed

        normalize_total_backed(adata, target_sum)  # streams X on disk, one block in RAM
        return None
    ext = _extension()
    normalized = _fast_csr_copy(adata.X) if _fast_path(ext, "normalize_total_inplace") else None
    if normalized is not None:
        # Zero-copy, multi-core kernel straight on the copy's numpy buffers.
        ext.normalize_total_inplace(normalized.indptr, normalized.data, target_sum)
    else:
        parts = _extension().normalize_total(*_csr_args(adata.X), target_sum, _default_device())
        normalized = _csr_from_parts(parts, adata.shape)
    if not inplace:
        return normalized
    adata.X = normalized
    return None


def log1p(adata: AnnData, *, inplace: bool = True) -> sp.csr_matrix | None:
    """Apply `log(1 + x)` to the count matrix."""
    if getattr(adata, "isbacked", False):
        from metalcyte._backed import log1p_backed

        log1p_backed(adata)  # streams X on disk, one block in RAM
        adata.uns["log1p"] = {"base": None}
        return None
    ext = _extension()
    logged = _fast_csr_copy(adata.X) if _fast_path(ext, "log1p_inplace") else None
    if logged is not None:
        ext.log1p_inplace(logged.data)
    else:
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
        *_csr_args(adata.X), n_top_genes, flavor, _default_device()
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
    x, ext = adata.X, _extension()
    if (
        _fast_path(ext, "scale_dense")
        and sp.issparse(x)
        and x.format == "csr"
        and x.indices.dtype in (np.int32, np.int64, np.uint32)
    ):
        data = x.data if x.data.dtype == _VALUE_DTYPE else x.data.astype(_VALUE_DTYPE)
        # One fused multi-core pass from CSR into the dense result; no device round trip.
        scaled = ext.scale_dense(
            np.ascontiguousarray(x.indptr),
            np.ascontiguousarray(x.indices),
            np.ascontiguousarray(data),
            x.shape[1],
            zero_center,
            max_value,
        )
    else:
        scaled = np.asarray(
            _extension().scale(*_csr_args(x), zero_center, max_value, _default_device()),
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
    device: str | None = None,
) -> None:
    """Principal component analysis by randomised SVD."""
    device = _resolve_device(device)
    ext, x = _extension(), adata.X
    if isinstance(x, np.ndarray) and x.ndim == 2 and _fast_path(ext, "pca_dense"):
        # Dense X (e.g. after pp.scale): straight to the device, no CSR round trip.
        dense = np.ascontiguousarray(x, dtype=_VALUE_DTYPE)
        result = ext.pca_dense(dense, n_comps, zero_center, random_state, device)
    else:
        result = ext.pca(*_csr_args(x), n_comps, zero_center, random_state, device)
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
    method: str = "auto",
    random_state: int = 0,
    device: str | None = None,
) -> None:
    """Build the k-nearest-neighbour graph and its UMAP connectivities.

    `method` is `"exact"`, `"approximate"` or `"auto"`. The exact search compares every
    pair of cells and costs quadratic time; the approximate search is NN-descent seeded
    with a random-projection forest, close to linear in the number of cells, with a
    recall above 0.95 on PCA embeddings. `"auto"` runs the exact search up to
    `APPROXIMATE_FROM` cells and the approximate one above. `random_state` seeds the
    approximate search only.
    """
    device = _resolve_device(device)
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}")
    if method not in ("auto", "exact", "approximate"):
        raise ValueError(f"method must be 'auto', 'exact' or 'approximate', got {method!r}")
    extension = _extension()
    representation = _representation(adata, use_rep)
    if method == "auto":
        method = "approximate" if adata.n_obs > APPROXIMATE_FROM else "exact"
    # scanpy counts the cell itself among its n_neighbors; the core does not.
    if method == "exact":
        indices, distances = extension.knn(representation, n_neighbors - 1, device)
    else:
        indices, distances = extension.knn_approximate(
            representation, n_neighbors - 1, int(random_state)
        )
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
        "params": {
            "n_neighbors": n_neighbors,
            "method": "umap",
            "use_rep": use_rep,
            "knn_method": method,
            "random_state": random_state,
        },
    }
