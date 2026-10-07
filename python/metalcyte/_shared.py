"""Helpers for reading and writing `AnnData` objects, shared by every module.

They live in their own file so that adding one feature does not require editing a
file that another feature also uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from anndata import AnnData

# The three CSR arrays cross the boundary with these dtypes: `f32` values match
# the core's "f32 throughout" rule, and the core indexes with unsigned 32-bit.
_INDEX_DTYPE = np.uint32
_VALUE_DTYPE = np.float32
_LABEL_DTYPE = np.uint32


def _default_device() -> str:
    """The device a function uses when its caller did not name one.

    Read through `settings` rather than frozen at import, so setting
    `metalcyte.settings.device` takes effect on the next call.
    """
    from metalcyte.settings import settings

    return settings.resolve_device()


def _resolve_device(device: str | None) -> str:
    """The device an API call runs on: the caller's explicit choice, else `settings.device`.

    Every public function takes `device=None` and goes through here, so setting
    `metalcyte.settings.device = "cpu"` really does keep the whole pipeline off the GPU.
    """
    from metalcyte.settings import settings

    return settings.resolve_device(device)


def _extension() -> ModuleType:
    """Import the compiled core on use, not on import.

    `metalcyte.pp` must stay importable while the extension is still being built,
    and tests replace it wholesale in `sys.modules`.
    """
    from metalcyte import _metalcyte

    return _metalcyte


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
        raise KeyError(f"adata.obsm has no {use_rep!r}; run metalcyte.pp.pca first or pass use_rep")
    return _dense(adata.obsm[use_rep])


def _neighbor_graph(adata: AnnData):
    """Return the connectivities graph UMAP lays out."""
    if "connectivities" not in adata.obsp:
        raise KeyError("adata.obsp has no 'connectivities'; run metalcyte.pp.neighbors first")
    return adata.obsp["connectivities"]
