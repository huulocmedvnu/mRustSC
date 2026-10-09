"""Backed `.h5ad` row blocks: `metalcyte._backed`.

The point of the module is that peak memory follows the block, not the dataset,
so the last test streams a matrix tens of times larger than its block budget and
measures what it actually costs.
"""

from __future__ import annotations

import subprocess
import sys
import tracemalloc

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from metalcyte._backed import block_size_for, open_backed
from metalcyte.settings import settings

N_OBS, N_VARS, DENSITY = 240, 60, 0.2

# The streaming test's matrix: 100k cells by 1k genes at 10% density is 380 MB
# once densified and 115 MB as CSR, against a 4 MB block budget.
STREAMED_OBS, STREAMED_VARS, STREAMED_DENSITY = 100_000, 1_000, 0.1
STREAMED_BUDGET_GB = 4 / 1024
# Blocks are a few MB; most of the allowance is the obs frame, which backed mode
# does read eagerly — 100k cell names cost more than the matrix blocks do. Still
# an order of magnitude below the matrix this pass covers.
PEAK_ALLOWANCE_BYTES = 32 * 1024**2


# Peak resident memory is a property of a whole process, so it is measured in
# one: a fresh interpreter that does nothing but the pass under measurement.
# Both snippets import the same modules, so the difference between them is the
# matrix and nothing else.
_STREAMING_PASS = """
import numpy as np
import pandas as pd
from metalcyte._backed import open_backed

with open_backed({path!r}) as backed:
    for _, block in backed.blocks(backed.block_size(max_memory_gb={budget!r})):
        np.asarray(block.sum(axis=1)).ravel()
"""

_IN_MEMORY_PASS = """
import anndata, numpy as np
import metalcyte._backed  # imported for a like-for-like baseline

adata = anndata.read_h5ad({path!r})
np.asarray(adata.X.sum(axis=1)).ravel()
"""

_REPORT_PEAK = """
import resource, sys

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak / (1024**2 if sys.platform == "darwin" else 1024))
"""


def peak_rss_mb(body: str) -> float:
    """Run `body` in a fresh interpreter and return its peak resident set, in MB."""
    completed = subprocess.run(
        [sys.executable, "-c", body + _REPORT_PEAK],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(completed.stdout.split()[-1])


def write_h5ad(path, matrix) -> anndata.AnnData:
    """Write `matrix` as the `X` of a minimal `.h5ad` and return what was written."""
    adata = anndata.AnnData(matrix)
    adata.write_h5ad(path)
    return adata


def random_csr(n_obs: int, n_vars: int, density: float, seed: int = 0) -> sp.csr_matrix:
    return sp.random_array(
        (n_obs, n_vars),
        density=density,
        format="csr",
        dtype=np.float32,
        rng=np.random.default_rng(seed),
    )


def constant_density_csr(n_obs: int, n_vars: int, density: float) -> sp.csr_matrix:
    """A matrix with exactly `density * n_vars` stored entries in every row."""
    per_row = int(np.ceil(density * n_vars))
    columns = np.tile(np.arange(per_row, dtype=np.int32), n_obs)
    indptr = np.arange(n_obs + 1, dtype=np.int32) * per_row
    values = np.arange(1, n_obs * per_row + 1, dtype=np.float32)
    return sp.csr_matrix((values, columns, indptr), shape=(n_obs, n_vars))


@pytest.fixture
def h5ad_path(tmp_path):
    """A small sparse `.h5ad`, with the matrix that was written to it."""
    matrix = random_csr(N_OBS, N_VARS, DENSITY)
    path = tmp_path / "small.h5ad"
    write_h5ad(path, matrix)
    return path, matrix


@pytest.mark.parametrize("block_size", [1, 7, N_OBS - 1, N_OBS, N_OBS + 100])
def test_blocks_reassemble_the_matrix(h5ad_path, block_size):
    path, matrix = h5ad_path
    with open_backed(path) as backed:
        assert backed.shape == matrix.shape
        starts, blocks = zip(*backed.blocks(block_size), strict=True)

    assert list(starts) == list(range(0, N_OBS, block_size))
    assert sum(block.shape[0] for block in blocks) == N_OBS
    assert all(block.shape[0] <= block_size for block in blocks)
    assert_allclose(sp.vstack(blocks).toarray(), matrix.toarray())


def test_blocks_reassemble_a_dense_matrix(tmp_path):
    dense = np.random.default_rng(1).random((50, 8), dtype=np.float32)
    path = tmp_path / "dense.h5ad"
    write_h5ad(path, dense)

    with open_backed(path) as backed:
        blocks = [block for _, block in backed.blocks(9)]

    assert all(sp.issparse(block) for block in blocks)
    assert_allclose(sp.vstack(blocks).toarray(), dense)


def test_density_and_block_size_come_from_the_file(h5ad_path):
    path, matrix = h5ad_path
    with open_backed(path) as backed:
        assert backed.density == pytest.approx(matrix.nnz / (N_OBS * N_VARS))
        assert backed.block_size() == block_size_for(N_VARS, backed.density)


def test_no_block_exceeds_the_requested_budget(tmp_path):
    """Every block fits the budget under the model `block_size_for` uses: the
    dense buffer a caller would build from it, plus the CSR arrays it arrives in.

    The model is stated in terms of the *mean* density, so the bound is exact
    only when the rows share it; this matrix therefore stores the same number of
    entries in every row. On a lumpier matrix a block of unusually full rows
    overshoots by the row-to-row spread, which is what the docstring says.
    """
    path = tmp_path / "even.h5ad"
    write_h5ad(path, constant_density_csr(N_OBS, N_VARS, DENSITY))
    budget_gb = 8 / 1024**2  # 8 KB, so the file needs many blocks
    budget_bytes = budget_gb * 1024**3

    with open_backed(path) as backed:
        block_size = backed.block_size(max_memory_gb=budget_gb)
        assert 1 <= block_size < N_OBS
        peak = max(
            block.shape[0] * N_VARS * 4 + block.nnz * 8 + block.shape[0] * 4
            for _, block in backed.blocks(block_size)
        )

    assert peak <= budget_bytes, f"peak block {peak} B over a {budget_bytes:.0f} B budget"


def test_block_size_follows_the_settings_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_memory_gb", 1 / 1024)
    small = block_size_for(N_VARS, DENSITY)
    monkeypatch.setattr(settings, "max_memory_gb", 16 / 1024)
    assert block_size_for(N_VARS, DENSITY) == 16 * small


def test_block_size_is_never_zero():
    assert block_size_for(200_000, 1.0, max_memory_gb=1e-9) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_vars": 0}, "n_vars"),
        ({"n_vars": 10, "density": 1.5}, "density"),
        ({"n_vars": 10, "max_memory_gb": 0.0}, "max_memory_gb"),
    ],
)
def test_block_size_for_rejects_impossible_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        block_size_for(**kwargs)


def test_rejects_a_file_that_is_not_h5ad(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("gene,count\n")
    with pytest.raises(ValueError, match=r"expected an \.h5ad file, got 'counts\.csv'"):
        open_backed(path)


def test_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"absent\.h5ad"):
        open_backed(tmp_path / "absent.h5ad")


def test_rejects_a_column_major_matrix(tmp_path):
    path = tmp_path / "csc.h5ad"
    write_h5ad(path, random_csr(20, 10, 0.3).tocsc())
    with pytest.raises(TypeError, match="csc"):
        open_backed(path)


def test_rejects_a_block_size_below_one(h5ad_path):
    path, _ = h5ad_path
    with open_backed(path) as backed, pytest.raises(ValueError, match="block_size"):
        next(backed.blocks(0))


def test_streams_a_matrix_far_larger_than_the_block_budget(tmp_path):
    """The argument for this branch: a full pass over a matrix that would need
    380 MB densified, in blocks budgeted at 4 MB, without ever holding it."""
    path = tmp_path / "large.h5ad"
    matrix = random_csr(STREAMED_OBS, STREAMED_VARS, STREAMED_DENSITY)
    write_h5ad(path, matrix)
    expected_totals = np.asarray(matrix.sum(axis=1)).ravel()
    dense_bytes = STREAMED_OBS * STREAMED_VARS * 4
    del matrix

    tracemalloc.start()
    try:
        totals = np.empty(STREAMED_OBS, dtype=np.float64)
        with open_backed(path) as backed:
            block_size = backed.block_size(max_memory_gb=STREAMED_BUDGET_GB)
            n_blocks = 0
            for start, block in backed.blocks(block_size):
                totals[start : start + block.shape[0]] = np.asarray(block.sum(axis=1)).ravel()
                n_blocks += 1
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    streamed_rss = peak_rss_mb(_STREAMING_PASS.format(path=str(path), budget=STREAMED_BUDGET_GB))
    in_memory_rss = peak_rss_mb(_IN_MEMORY_PASS.format(path=str(path)))
    print(
        f"\nstreamed {STREAMED_OBS} x {STREAMED_VARS} ({dense_bytes / 1024**2:.0f} MB dense, "
        f"{path.stat().st_size / 1024**2:.0f} MB on disk) in {n_blocks} blocks of "
        f"{block_size} cells: {peak / 1024**2:.1f} MB allocated at peak. "
        f"Whole interpreter peak RSS: {streamed_rss:.0f} MB streaming, "
        f"{in_memory_rss:.0f} MB reading the same file into memory."
    )
    assert n_blocks > 10, "the budget must force many blocks for this to prove anything"
    assert peak < PEAK_ALLOWANCE_BYTES, f"peak {peak} B"
    assert peak < dense_bytes / 4, "streaming must cost far less than the dense matrix"
    assert in_memory_rss - streamed_rss > 40, (
        f"streaming must save more than the noise floor: {streamed_rss} vs {in_memory_rss} MB"
    )
    assert_allclose(totals, expected_totals, rtol=1e-5)


@pytest.mark.parametrize(
    ("index_dtype", "chunks", "compression"),
    [
        (np.int32, None, None),
        (np.int64, None, None),
        (np.int32, 97, None),
        (np.int64, 97, None),
        (np.int64, 97, "gzip"),
    ],
)
def test_direct_reader_returns_the_blocks_anndata_returns(
    tmp_path, monkeypatch, index_dtype, chunks, compression
):
    """The Rust chunk reader gives anndata's blocks for either gene index width and for
    contiguous or chunked storage, and steps aside for a compressed file."""
    import h5py

    rng = np.random.default_rng(7)
    matrix = sp.random(533, 61, density=0.2, format="csr", random_state=rng, dtype=np.float32)
    matrix.data = np.round(matrix.data * 20).astype(np.float32) + 1
    keep = np.ones(533, dtype=np.float32)
    keep[17] = 0  # an empty row
    matrix = sp.csr_matrix(sp.diags(keep) @ matrix)
    matrix.eliminate_zeros()
    path = tmp_path / "counts.h5ad"
    anndata.AnnData(
        obs=pd.DataFrame(index=[f"c{i}" for i in range(533)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(61)]),
    ).write_h5ad(path)
    with h5py.File(path, "r+") as f:
        if "X" in f:
            del f["X"]
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array(matrix.shape, dtype=np.int64)
        g.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        opts = dict(chunks=(chunks,) if chunks else None, compression=compression)
        g.create_dataset("data", data=matrix.data, **opts)
        g.create_dataset("indices", data=matrix.indices.astype(index_dtype), **opts)

    def read(direct):
        monkeypatch.setenv("METALCYTE_DIRECT_READ", "1" if direct else "0")
        with open_backed(path) as backed:
            used = backed._direct() is not None
            return used, list(backed.blocks(50))

    used, direct_blocks = read(True)
    assert used == (compression is None)
    _, reference = read(False)
    assert len(direct_blocks) == len(reference) == 11
    for (s1, a), (s2, b) in zip(direct_blocks, reference, strict=True):
        assert s1 == s2 and a.shape == b.shape and a.data.dtype == np.float32
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.data, b.data)
    assert (sp.vstack([b for _, b in direct_blocks]) != matrix).nnz == 0
