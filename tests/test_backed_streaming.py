"""Backed streaming parity: `pp.normalize_total` and `pp.log1p` on an on-disk AnnData.

When `adata.isbacked`, these stream `X` in row blocks and rewrite it on disk, so peak
memory is one block rather than the whole matrix. The result must equal the in-memory
result: normalisation is per-row and log1p is element-wise, so a block sees exactly the
rows it would in memory. A given `target_sum` is therefore bit-for-bit; `target_sum=None`
depends on the global median, which is computed from streamed totals and agrees to `f32`.

The benchmark that shows the memory reduction is `benches/backed_transform.py`.
"""

from __future__ import annotations

import anndata
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

import scrust as sr

_BLOCK = 64  # small, so a 400-cell matrix streams in several blocks rather than one


def _counts(
    n_cells: int = 400, n_genes: int = 1500, per_row: int = 90, seed: int = 0
) -> anndata.AnnData:
    """A small integer-count CSR AnnData, built directly so the test needs no download."""
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, n_cells * per_row + 1, per_row, dtype=np.int32)
    indices = rng.integers(0, n_genes, n_cells * per_row).astype(np.int32)
    data = rng.integers(1, 50, n_cells * per_row).astype(np.float32)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
    matrix.sum_duplicates()  # collapse repeated columns so the pattern is well formed
    return anndata.AnnData(matrix)


@pytest.fixture
def backed_and_memory(tmp_path):
    """The same counts twice: one backed on disk, one held in memory."""
    base = _counts()
    path = tmp_path / "counts.h5ad"
    base.write_h5ad(path)
    backed = anndata.read_h5ad(path, backed="r")
    assert backed.isbacked
    memory = base.copy()
    yield backed, memory
    backed.file.close()


def _streamed(adata: anndata.AnnData, call) -> None:
    sr.settings.chunk_size = _BLOCK
    try:
        call(adata)
    finally:
        sr.settings.chunk_size = 0


def _data(adata: anndata.AnnData) -> np.ndarray:
    resolved = adata.to_memory() if adata.isbacked else adata
    return resolved.X.tocsr().data


def test_normalize_total_backed_matches_memory_bit_for_bit(backed_and_memory) -> None:
    backed, memory = backed_and_memory
    _streamed(backed, lambda a: sr.pp.normalize_total(a, target_sum=1e4))
    sr.pp.normalize_total(memory, target_sum=1e4)
    assert_array_equal(_data(backed), _data(memory))


def test_log1p_backed_matches_memory_bit_for_bit(backed_and_memory) -> None:
    backed, memory = backed_and_memory
    _streamed(backed, sr.pp.log1p)
    sr.pp.log1p(memory)
    assert_array_equal(_data(backed), _data(memory))
    assert backed.uns["log1p"] == {"base": None}


def test_normalize_then_log1p_backed_matches_memory(backed_and_memory) -> None:
    """The whole pre-processing head, streamed on disk, equals the in-memory head."""
    backed, memory = backed_and_memory

    def head(adata: anndata.AnnData) -> None:
        sr.pp.normalize_total(adata, target_sum=1e4)
        sr.pp.log1p(adata)

    _streamed(backed, head)
    head(memory)
    assert_array_equal(_data(backed), _data(memory))


def test_normalize_total_median_backed_matches_memory_to_f32(backed_and_memory) -> None:
    """`target_sum=None` uses the global median of per-cell totals, streamed to `f32`."""
    backed, memory = backed_and_memory
    _streamed(backed, sr.pp.normalize_total)  # None -> median
    sr.pp.normalize_total(memory)
    assert_allclose(_data(backed), _data(memory), rtol=1e-5, atol=1e-6)


def test_in_memory_path_is_unchanged_by_the_backed_branch() -> None:
    """A plain in-memory AnnData must not touch the streaming path at all."""
    adata = _counts(n_cells=120, n_genes=300, per_row=40)
    assert not adata.isbacked
    sr.pp.normalize_total(adata, target_sum=1e4)
    sr.pp.log1p(adata)
    assert adata.uns["log1p"] == {"base": None}
    assert np.isfinite(adata.X.data).all()
