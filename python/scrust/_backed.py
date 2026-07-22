"""Row blocks read straight from an `.h5ad`, for matrices that do not fit in memory.

Nothing here is wired into `scrust.pp` yet — those functions still take an
in-memory `AnnData`. This is the piece they adopt when they stop doing that, so
the surface is deliberately the smallest one a chunked preprocessing step needs:
open a file, iterate row blocks, close it.

    with open_backed("atlas.h5ad") as backed:
        for start, block in backed.blocks():
            ...  # `block` is a scipy CSR matrix of at most `block_size` cells

The matrix is never read whole: `anndata`'s backed mode leaves `X` on disk and
each block is one HDF5 read of the slice it covers. Peak memory is therefore the
block, not the dataset, and the block is sized against
`scrust.settings.max_memory_gb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata
import numpy as np
import scipy.sparse as sp

from scrust.settings import settings

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    import pandas as pd

# The block cost model, mirroring `scrust_core::chunked::rows_per_block`. A block
# is charged for its CSR arrays (a `uint32` column index and a `float32` value
# per stored entry, a `uint32` row pointer per row) *and* for the dense
# `(rows, n_vars)` buffer a caller densifies it into, because the two are live at
# the same time and the dense one dominates at single-cell densities.
_BYTES_PER_STORED_ENTRY = 8
_BYTES_PER_ROW_POINTER = 4
_BYTES_PER_DENSE_ENTRY = 4
_BYTES_PER_GB = 1024**3

_SUFFIX = ".h5ad"
_VALUE_DTYPE = np.float32


def block_size_for(
    n_vars: int,
    density: float = 1.0,
    max_memory_gb: float | None = None,
) -> int:
    """How many cells fit in one block of a matrix `n_vars` genes wide.

    `density` is the mean fraction of stored entries per row; the default of 1.0
    is the worst case, a fully dense matrix. `max_memory_gb` defaults to
    `scrust.settings.max_memory_gb`, the budget the chunked paths size
    themselves against.

    Never returns zero: a caller has to make progress even when a single row
    exceeds the budget.
    """
    if n_vars < 1:
        raise ValueError(f"n_vars must be at least 1, got {n_vars}")
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must lie in [0, 1], got {density}")
    budget_gb = settings.max_memory_gb if max_memory_gb is None else max_memory_gb
    if budget_gb <= 0:
        raise ValueError(f"max_memory_gb must be positive, got {budget_gb}")

    bytes_per_row = (
        n_vars * _BYTES_PER_DENSE_ENTRY
        + int(np.ceil(density * n_vars)) * _BYTES_PER_STORED_ENTRY
        + _BYTES_PER_ROW_POINTER
    )
    return max(1, int(budget_gb * _BYTES_PER_GB) // bytes_per_row)


class BackedMatrix:
    """An open `.h5ad` handing out row blocks of `X`.

    Build one with `open_backed`. It owns the open HDF5 file, so use it as a
    context manager or call `close` when done.
    """

    def __init__(self, adata: anndata.AnnData, path: Path) -> None:
        self._adata = adata
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_obs, n_vars)` — cells by genes, as everywhere else."""
        return (self.n_obs, self.n_vars)

    @property
    def n_obs(self) -> int:
        return int(self._adata.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self._adata.shape[1])

    @property
    def obs(self) -> pd.DataFrame:
        """Cell annotations. Small enough to be read eagerly, unlike `X`."""
        return self._adata.obs

    @property
    def var(self) -> pd.DataFrame:
        """Gene annotations."""
        return self._adata.var

    @property
    def density(self) -> float:
        """Fraction of `X` that is stored, read from the file's own bookkeeping.

        The count of stored entries is the length of the `indices` array, which
        HDF5 reports from the header without reading any of it.
        """
        matrix = self._x()
        if not _is_sparse_dataset(matrix):
            return 1.0
        stored = int(matrix.group["indices"].shape[0])
        return stored / max(1, self.n_obs * self.n_vars)

    def block_size(self, max_memory_gb: float | None = None) -> int:
        """The block size this file's shape and density imply under the budget."""
        return block_size_for(self.n_vars, self.density, max_memory_gb)

    def blocks(self, block_size: int | None = None) -> Iterator[tuple[int, sp.csr_matrix]]:
        """Yield `(first cell index, block)` in row order, one HDF5 read each.

        `block_size` defaults to `self.block_size()`. Blocks are CSR with
        `float32` values, the form the Rust core takes; the last one is short
        whenever the cell count is not a multiple of the block size.
        """
        if block_size is None:
            block_size = self.block_size()
        if block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {block_size}")
        matrix = self._x()
        for start in range(0, self.n_obs, block_size):
            stop = min(start + block_size, self.n_obs)
            yield start, _as_csr(matrix[start:stop])

    def close(self) -> None:
        """Close the backing file. Idempotent, so `with` blocks nest safely."""
        self._adata.file.close()

    def __enter__(self) -> BackedMatrix:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"BackedMatrix({self._path.name!r}, {self.n_obs} cells x {self.n_vars} genes)"

    def _x(self) -> Any:
        """The on-disk `X`, still on disk."""
        matrix = self._adata.X
        if matrix is None:
            raise ValueError(f"{self._path.name} has no X to stream")
        return matrix


def open_backed(path: str | os.PathLike[str]) -> BackedMatrix:
    """Open an `.h5ad` in backed mode, leaving `X` on disk.

    Raises `FileNotFoundError` if the file does not exist, `ValueError` if it is
    not an `.h5ad`, and `TypeError` if its `X` cannot be sliced by rows — a
    column-major (CSC) `X` would need the whole file read to assemble one block.
    """
    path = Path(path)
    if path.suffix != _SUFFIX:
        raise ValueError(f"expected an {_SUFFIX} file, got {path.name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    adata = anndata.read_h5ad(path, backed="r")
    backed = BackedMatrix(adata, path)
    matrix = adata.X
    if matrix is not None and getattr(matrix, "format", "csr") != "csr":
        backed.close()
        raise TypeError(
            f"{path.name} stores X as {matrix.format}; row blocks need a CSR or dense X. "
            "Rewrite it with `adata.X = adata.X.tocsr()`."
        )
    return backed


def _is_sparse_dataset(matrix: Any) -> bool:
    """True for anndata's on-disk sparse `X`, false for an h5py dense dataset."""
    return hasattr(matrix, "group")


def _as_csr(block: Any) -> sp.csr_matrix:
    """A read of one row block as CSR, whether `X` is stored sparse or dense."""
    matrix = block if sp.issparse(block) else sp.csr_matrix(np.asarray(block))
    return matrix.astype(_VALUE_DTYPE, copy=False).tocsr()
