"""Audit of `crates/scrust-core/src/chunked.rs` — the block-streaming layer.

There is no scanpy equivalent to cross-check against, so this file pins the one
property streaming has to have: **the answer does not depend on the block size**.

A note on reach, established by reading `crates/scrust-py/src/*.rs`: *nothing*
in `chunked.rs` is bound to Python. `crates/scrust-py/src/lib.rs` registers
thirteen submodules and none of them is `chunked`; `RowBlocks`, `CsrRowBlocks`,
`rows_per_block`, `streaming_gene_statistics` and `streaming_cell_totals` are
reachable only from Rust. What *is* reachable is `python/scrust/_backed.py`,
which re-implements the same block iteration and the same cost model in Python
(its own comment says it mirrors `scrust_core::chunked::rows_per_block`), plus
the in-memory kernels in `scrust._scrust` that a chunked pass would call per
block. So the tests below split three ways:

* the cost model — `block_size_for` against an independent transcription of
  `rows_per_block`, which is the only way left to check the two agree;
* the block iterator — `_backed.BackedMatrix.blocks`, the reachable twin of
  `CsrRowBlocks::next_block`;
* block-size invariance of real reductions, computed by driving the *crate's*
  kernels (`qc_metrics`, `log1p`, `normalize_total`, `scale`) over those blocks
  and comparing against the same kernels run on the whole matrix.

The Rust `GeneMoments` accumulator is mirrored here in `stream_gene_statistics`
so that its Chan-Golub-LeVeque merge can be checked against the crate's
in-memory gene statistics, which is the agreement the module promises.
"""

from __future__ import annotations

import math

import anndata
import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from numpy.testing import assert_allclose

from scrust import _scrust
from scrust._backed import block_size_for, open_backed
from scrust.settings import settings
from scrust_call import scrust_call

N_OBS, N_VARS = 240, 60

# The public names of `chunked.rs`, none of which reaches Python.
CHUNKED_NAMES = (
    "rows_per_block",
    "streaming_gene_statistics",
    "streaming_cell_totals",
    "RowBlocks",
    "CsrRowBlocks",
)

# Constants transcribed from `chunked.rs`.
BYTES_PER_STORED_ENTRY = 8  # u32 column index + f32 value
BYTES_PER_ROW_POINTER = 4  # one indptr entry
BYTES_PER_DENSE_ENTRY = 4  # one f32 of the dense buffer the caller builds
BYTES_PER_GB = 1024**3


def rust_rows_per_block(n_cols: int, density: float, budget_bytes: int) -> int:
    """`scrust_core::chunked::rows_per_block`, transcribed line for line.

    The Rust is not callable from Python, so agreement with the Python mirror
    can only be checked against a hand transcription. Kept deliberately literal.
    """
    density = 1.0 if math.isnan(density) else min(max(density, 0.0), 1.0)
    stored_per_row = math.ceil(density * n_cols)
    bytes_per_row = (
        n_cols * BYTES_PER_DENSE_ENTRY
        + stored_per_row * BYTES_PER_STORED_ENTRY
        + BYTES_PER_ROW_POINTER
    )
    return max(budget_bytes // bytes_per_row, 1)


def write_h5ad(path, matrix) -> None:
    anndata.AnnData(matrix).write_h5ad(path)


def counts_matrix(n_obs: int = N_OBS, n_vars: int = N_VARS, seed: int = 0) -> sp.csr_matrix:
    """Sparse counts with no empty row and no empty gene.

    Empty rows would make `normalize_total` divide by zero and empty genes would
    make `scale` substitute a deviation of 1, both of which would mask rather
    than expose a block-size dependence.
    """
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.4, size=(n_obs, n_vars)).astype(np.float32)
    dense[np.arange(n_obs), rng.integers(0, n_vars, n_obs)] += 3.0
    dense[rng.integers(0, n_obs, n_vars), np.arange(n_vars)] += 5.0
    return sp.csr_matrix(dense)


def large_mean_matrix(n_obs: int = 400, n_vars: int = 8) -> sp.csr_matrix:
    """Every value near 1e6, the spread around it of order one.

    The regime `chunked.rs` says a raw sum of squares cannot survive: the
    variance is twelve orders of magnitude below the second moment.
    """
    rng = np.random.default_rng(7)
    offsets = np.arange(n_vars, dtype=np.float64) * 1000.0
    dense = 1.0e6 + offsets + rng.uniform(-1.0, 1.0, size=(n_obs, n_vars))
    return sp.csr_matrix(dense.astype(np.float32))


def stream_gene_statistics(blocks) -> tuple[np.ndarray, np.ndarray]:
    """`GeneMoments` from `chunked.rs`, mirrored: a Chan-Golub-LeVeque merge.

    Per-block moments are taken about the *block's own* mean and merged with the
    shift correction, so nothing large is ever squared. Returns the mean and the
    sample variance (ddof = 1), which is what the Rust `finish` returns.
    """
    mean = squared_deviations = None
    n_rows = 0
    for block in blocks:
        dense = np.asarray(block.todense(), dtype=np.float64)
        block_rows = dense.shape[0]
        if block_rows == 0:
            continue
        if mean is None:
            mean = np.zeros(dense.shape[1])
            squared_deviations = np.zeros(dense.shape[1])
        block_mean = dense.mean(axis=0)
        block_deviations = ((dense - block_mean) ** 2).sum(axis=0)
        total = n_rows + block_rows
        shift = block_mean - mean
        squared_deviations += block_deviations + shift**2 * (n_rows * block_rows / total)
        mean = mean + shift * block_rows / total
        n_rows = total
    if n_rows == 0:
        raise ValueError("an empty matrix has no gene statistics")
    return mean, squared_deviations / max(n_rows - 1, 1)


def naive_gene_variance(blocks) -> np.ndarray:
    """The formula `chunked.rs` exists to avoid: `sum(x^2) - n * mean^2` in f32."""
    total = squares = None
    n_rows = 0
    for block in blocks:
        dense = np.asarray(block.todense(), dtype=np.float32)
        if total is None:
            total = np.zeros(dense.shape[1], dtype=np.float32)
            squares = np.zeros(dense.shape[1], dtype=np.float32)
        total += dense.sum(axis=0, dtype=np.float32)
        squares += (dense * dense).sum(axis=0, dtype=np.float32)
        n_rows += dense.shape[0]
    return (squares - total * total / np.float32(n_rows)) / np.float32(n_rows - 1)


def recover_mean_and_deviation(
    matrix: sp.csr_matrix, scaled: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The per-gene mean and deviation the crate used, read back out of `scale`.

    `scale(zero_center=True, max_value=None)` returns `(x - mean) / deviation`,
    an exact affine function of the column, so a least-squares fit of the column
    against its scaled form recovers both constants. This is the only way to see
    the crate's in-memory gene statistics: they are not exposed directly.
    """
    dense = np.asarray(matrix.todense(), dtype=np.float64)
    scaled = np.asarray(scaled, dtype=np.float64)
    means, deviations = [], []
    for gene in range(dense.shape[1]):
        design = np.column_stack([np.ones(dense.shape[0]), scaled[:, gene]])
        intercept, slope = np.linalg.lstsq(design, dense[:, gene], rcond=None)[0]
        means.append(intercept)
        deviations.append(slope)
    return np.asarray(means), np.asarray(deviations)


@pytest.fixture
def h5ad(tmp_path):
    """A small sparse `.h5ad` and the matrix that was written to it."""
    matrix = counts_matrix()
    path = tmp_path / "counts.h5ad"
    write_h5ad(path, matrix)
    return path, matrix


def read_blocks(path, block_size: int) -> list[sp.csr_matrix]:
    with open_backed(path) as backed:
        return [block for _, block in backed.blocks(block_size)]


# --------------------------------------------------------------------------
# What is reachable at all
# --------------------------------------------------------------------------


def test_no_part_of_chunked_rs_is_bound_to_python():
    """Pins the premise of this audit: the Rust streaming layer has no binding.

    `crates/scrust-py/src/lib.rs` registers no `chunked` module, so every name in
    `chunked.rs` is dead from Python's point of view and the Python mirror in
    `scrust._backed` is the only streaming code a user can reach. If a binding is
    ever added this fails, which is the signal to test the Rust directly instead
    of its transcription.
    """
    bound = [name for name in CHUNKED_NAMES if hasattr(_scrust, name)]
    assert bound == [], f"chunked.rs is now bound as {bound}; audit the binding, not the mirror"
    # The mirror exists and is pure Python, not a re-export of the extension.
    assert block_size_for.__module__ == "scrust._backed"


# --------------------------------------------------------------------------
# The cost model: `rows_per_block` / `block_size_for`
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_vars", [1, 100, 2_000, 30_000])
@pytest.mark.parametrize("density", [0.0, 0.05, 0.5, 1.0])
@pytest.mark.parametrize("budget_gb", [1e-6, 1 / 1024, 0.5, 4.0])
def test_block_size_for_reproduces_the_rust_cost_model(n_vars, density, budget_gb):
    """The Python block size equals `rows_per_block` exactly, not approximately.

    Two implementations of one cost model is the defect this module invites: the
    Rust sizes the blocks a Rust caller streams and the Python sizes the blocks
    an `.h5ad` reader streams, and if they drift the same file is processed in
    different block sizes depending on which side opened it. Every term is
    checked — the dense buffer, the CSR arrays, the row pointer, the ceiling on
    the stored count and the `max(1, ...)` floor.
    """
    expected = rust_rows_per_block(n_vars, density, int(budget_gb * BYTES_PER_GB))
    assert block_size_for(n_vars, density, max_memory_gb=budget_gb) == expected


@pytest.mark.parametrize("n_vars", [1, 60, 200_000])
@pytest.mark.parametrize("budget_gb", [1e-9, 8 / 1024**2, 1.0])
def test_a_block_is_never_empty_and_never_over_budget(n_vars, budget_gb):
    """One row always fits, and any larger block stays inside the budget.

    Both halves matter: returning zero would stall a stream forever, and
    returning a row too many would break the promise the whole module rests on,
    that peak memory follows the block. The bound is asserted against the
    module's own model — dense buffer plus CSR arrays plus row pointers — and is
    waived only for the single-row block, which the floor may force over budget.
    """
    rows = block_size_for(n_vars, 1.0, max_memory_gb=budget_gb)
    assert rows >= 1
    used = rows * (n_vars * BYTES_PER_DENSE_ENTRY + n_vars * BYTES_PER_STORED_ENTRY)
    used += rows * BYTES_PER_ROW_POINTER
    assert rows == 1 or used <= budget_gb * BYTES_PER_GB, f"{rows} rows need {used} B"


def test_the_two_cost_models_disagree_on_a_density_outside_zero_to_one():
    """Documented divergence: Rust clamps a bad density, Python rejects it.

    `rows_per_block` clamps to [0, 1] and reads NaN as fully dense "because this
    function has no error channel"; `block_size_for` raises. Callers therefore
    cannot pass the same arguments to both. Python's is the safer contract, so
    this pins the divergence rather than calling it a defect — but a NaN density
    (which `BackedMatrix.density` would produce from a zero-sized matrix) is a
    hard error on one side and a full-density block on the other.
    """
    assert rust_rows_per_block(1_000, float("nan"), 1 << 20) == rust_rows_per_block(
        1_000, 1.0, 1 << 20
    )
    assert rust_rows_per_block(1_000, 5.0, 1 << 20) == rust_rows_per_block(1_000, 1.0, 1 << 20)
    assert rust_rows_per_block(1_000, -5.0, 1 << 20) == rust_rows_per_block(1_000, 0.0, 1 << 20)
    for bad in (float("nan"), 5.0, -5.0):
        with pytest.raises(ValueError, match="density"):
            block_size_for(1_000, bad, max_memory_gb=1.0)


def test_the_default_block_size_tracks_settings_max_memory_gb(h5ad):
    """`settings.max_memory_gb` is the budget, end to end through an open file.

    Not just that `block_size_for` reads the setting, but that the blocks a
    reader actually gets shrink when the budget does: doubling the budget must
    double the block and halve the number of reads.
    """
    path, _ = h5ad
    sizes = {}
    for budget_gb in (8 / 1024**2, 16 / 1024**2):
        settings.max_memory_gb = budget_gb
        try:
            with open_backed(path) as backed:
                blocks = [block.shape[0] for _, block in backed.blocks()]
            sizes[budget_gb] = (backed.block_size(), blocks)
        finally:
            settings.max_memory_gb = 4.0

    small, large = sizes[8 / 1024**2], sizes[16 / 1024**2]
    assert 1 < small[0] < N_OBS, f"the small budget must force many blocks, got {small[0]}"
    # Proportional to the budget up to the integer floor, which can leave one row.
    assert 2 * small[0] <= large[0] <= 2 * small[0] + 1
    assert max(large[1]) == large[0] and max(small[1]) == small[0]
    assert len(large[1]) < len(small[1])


def test_settings_chunk_size_is_never_read(h5ad):
    """DEFECT (documentation): `settings.chunk_size` has no effect anywhere.

    It is documented as "Rows per streamed block; `0` derives one from
    `max_memory_gb`", but `_backed.blocks` defaults to `self.block_size()` and
    never consults it, and no Rust binding reads it either. Setting it to 3 must
    therefore change nothing; this pins that it does not, so the day someone
    wires it up the test says where the contract changed.
    """
    path, _ = h5ad
    settings.max_memory_gb = 8 / 1024**2
    settings.chunk_size = 3
    try:
        with open_backed(path) as backed:
            derived = backed.block_size()
            first = next(iter(backed.blocks()))[1].shape[0]
    finally:
        settings.chunk_size = 0
        settings.max_memory_gb = 4.0
    assert derived != 3, "pick a budget whose block size is not the chunk_size under test"
    assert first == derived, "chunk_size now has an effect; the setting was dead when audited"


# --------------------------------------------------------------------------
# The block iterator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 7, N_OBS - 1, N_OBS, N_OBS + 100])
def test_blocks_partition_the_rows_exactly_once_and_in_order(h5ad, block_size):
    """Every row appears once, in its original position, whatever the block size.

    `CsrRowBlocks::next_block` rebases each block's `indptr` off the first
    offset; a mistake there duplicates or drops rows at a boundary, and 7 and
    `N_OBS - 1` both leave a short final block. Reassembling and comparing to the
    source catches a dropped row, a repeated row and a reordered one, none of
    which a row-count check would.
    """
    path, matrix = h5ad
    with open_backed(path) as backed:
        starts, blocks = zip(*backed.blocks(block_size), strict=True)

    assert list(starts) == list(range(0, N_OBS, block_size))
    assert [block.shape[0] for block in blocks] == [
        min(block_size, N_OBS - start) for start in starts
    ]
    assert_allclose(sp.vstack(blocks).toarray(), matrix.toarray())


def test_an_empty_matrix_streams_to_no_blocks(tmp_path):
    """Zero cells is an empty stream, not an error and not one empty block.

    The Rust says the same: `next_block` returns `None` immediately,
    `streaming_cell_totals` returns an empty vector, and only
    `streaming_gene_statistics` errors ("at least one cell"). The mirrored
    accumulator is held to that last part too.
    """
    path = tmp_path / "empty.h5ad"
    write_h5ad(path, sp.csr_matrix((0, N_VARS), dtype=np.float32))
    with open_backed(path) as backed:
        assert backed.shape == (0, N_VARS)
        blocks = list(backed.blocks(5))
    assert blocks == []
    with pytest.raises(ValueError, match="empty"):
        stream_gene_statistics([])


# --------------------------------------------------------------------------
# Block-size invariance of the crate's own kernels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 7, N_OBS])
def test_per_cell_qc_metrics_do_not_depend_on_the_block_size(h5ad, block_size):
    """The crate's per-cell QC metrics computed per block match the whole matrix.

    This is `streaming_cell_totals`' invariant driven through code that is
    actually reachable: total counts, genes detected and the `percent_top`
    fractions are all per-row, so a block boundary must be invisible. A kernel
    that took a matrix-wide quantity into a per-cell number — a proportion of the
    grand total, say — would fail here at every block size but the last.
    """
    path, matrix = h5ad
    whole, _ = scrust_call(
        "pp.calculate_qc_metrics", AnnData(matrix), percent_top=(5, 20), inplace=False
    )
    frames = [
        scrust_call("pp.calculate_qc_metrics", AnnData(block), percent_top=(5, 20), inplace=False)[
            0
        ]
        for block in read_blocks(path, block_size)
    ]
    streamed = np.concatenate([frame.to_numpy() for frame in frames])
    assert streamed.shape == whole.to_numpy().shape
    assert_allclose(streamed, whole.to_numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("block_size", [1, 7, N_OBS])
def test_per_gene_qc_totals_reassemble_from_the_blocks(h5ad, block_size):
    """A per-gene reduction accumulated over blocks equals the one-shot answer.

    Per-gene quantities are the hard direction: every block touches every gene,
    so the accumulation has to be correct rather than merely concatenated.
    Totals add, detected-cell counts add, and the mean is the weighted mean of
    the block means — the same merge `GeneMoments` performs for the first moment.
    """
    path, matrix = h5ad
    _, whole = scrust_call(
        "pp.calculate_qc_metrics", AnnData(matrix), percent_top=(5, 20), inplace=False
    )
    totals = np.zeros(N_VARS)
    detected = np.zeros(N_VARS)
    weighted_mean = np.zeros(N_VARS)
    n_rows = 0
    for block in read_blocks(path, block_size):
        _, genes = scrust_call(
            "pp.calculate_qc_metrics", AnnData(block), percent_top=(5, 20), inplace=False
        )
        totals += genes["total_counts"].to_numpy(dtype=np.float64)
        detected += genes["n_cells_by_counts"].to_numpy(dtype=np.float64)
        weighted_mean += genes["mean_counts"].to_numpy(dtype=np.float64) * block.shape[0]
        n_rows += block.shape[0]

    assert n_rows == N_OBS
    assert_allclose(totals, whole["total_counts"].to_numpy(dtype=np.float64), rtol=1e-6)
    assert_allclose(detected, whole["n_cells_by_counts"].to_numpy(dtype=np.float64))
    assert_allclose(
        weighted_mean / n_rows, whole["mean_counts"].to_numpy(dtype=np.float64), rtol=1e-5
    )


@pytest.mark.parametrize("block_size", [1, 7, N_OBS])
def test_log1p_is_identical_block_by_block(h5ad, block_size):
    """An elementwise kernel must be bit-for-bit block independent.

    `log1p` has no cross-row term at all, so this is the strictest invariance
    available: not `allclose` but exact equality of the f32 outputs. Any
    dependence on the number of rows in the argument — a vectorisation that
    changes summation order, a tail loop that differs — shows up here and nowhere
    else in the suite.
    """
    path, matrix = h5ad
    whole = scrust_call("pp.log1p", AnnData(matrix), inplace=False)
    parts = [
        scrust_call("pp.log1p", AnnData(block), inplace=False)
        for block in read_blocks(path, block_size)
    ]
    streamed = sp.vstack(parts).toarray()
    assert np.array_equal(streamed, whole.toarray()), "log1p differs across block boundaries"


@pytest.mark.parametrize("block_size", [1, 7])
def test_normalize_total_streams_only_with_an_explicit_target(h5ad, block_size):
    """DEFECT (streaming hazard): the default `target_sum=None` is not stream-safe.

    With an explicit target the kernel is per-row and blockwise application is
    exact. With `target_sum=None` scanpy's rule is to normalise to the *median*
    cell count, a matrix-wide statistic, so the same cell gets a different answer
    depending on which cells shared its block. Nothing in `chunked.rs` or in
    `_backed` warns of this, and `streaming_cell_totals` gives a chunked caller
    exactly the totals it would need to compute the median first. The test
    asserts both halves, including that the divergence is real and not a
    tolerance artefact.
    """
    path, matrix = h5ad
    blocks = read_blocks(path, block_size)

    exact = scrust_call("pp.normalize_total", AnnData(matrix), target_sum=1e4, inplace=False)
    streamed = sp.vstack(
        [
            scrust_call("pp.normalize_total", AnnData(block), target_sum=1e4, inplace=False)
            for block in blocks
        ]
    )
    assert_allclose(streamed.toarray(), exact.toarray(), rtol=1e-6)

    median = scrust_call("pp.normalize_total", AnnData(matrix), inplace=False)
    median_streamed = sp.vstack(
        [scrust_call("pp.normalize_total", AnnData(block), inplace=False) for block in blocks]
    )
    gap = np.abs(median_streamed.toarray() - median.toarray()).max()
    assert gap > 1e-3, (
        "target_sum=None was expected to be block dependent (it normalises to the "
        f"median cell count); the largest disagreement was only {gap}"
    )


# --------------------------------------------------------------------------
# The gene statistics the module exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 7, N_OBS, N_OBS + 100])
def test_streamed_gene_statistics_match_the_crates_in_memory_scale(h5ad, block_size):
    """The merged per-gene mean and variance equal the ones `scale` used in memory.

    This is the module's headline claim, checked across the whole span of block
    sizes: one row per block, a size that divides nothing, exactly the matrix,
    and larger than the matrix. The in-memory side is the crate's own
    `gene_mean_and_deviation`, read back out of `pp.scale` (it is not exposed
    directly), and it carries Bessel's correction — so the streamed variance must
    use ddof = 1 as `GeneMoments::finish` says, not ddof = 0. The last assertion
    demonstrates the test's power: with ddof = 0 the same comparison fails.
    """
    path, matrix = h5ad
    scaled = scrust_call("pp.scale", AnnData(matrix), max_value=None, inplace=False)
    crate_mean, crate_deviation = recover_mean_and_deviation(matrix, scaled)

    mean, variance = stream_gene_statistics(read_blocks(path, block_size))
    assert_allclose(mean, crate_mean, rtol=1e-5, atol=1e-6)
    assert_allclose(np.sqrt(variance), crate_deviation, rtol=1e-4)

    population = variance * (N_OBS - 1) / N_OBS
    assert not np.allclose(np.sqrt(population), crate_deviation, rtol=1e-4), (
        "ddof = 0 must not pass this comparison, or the tolerance is measuring nothing"
    )


def test_the_streamed_merge_survives_a_mean_a_sum_of_squares_cannot(tmp_path):
    """Large mean, tiny variance: the merge holds, `sum(x^2) - n*mean^2` collapses.

    The reason `GeneMoments` uses the Chan-Golub-LeVeque update rather than a
    running sum of squares. The streamed variance is checked against a two-pass
    f64 reference *and* against the crate's own in-memory deviation, and the
    naive accumulation is run over the identical blocks to show the failure is
    the formula and not the data: it is out by orders of magnitude.
    """
    matrix = large_mean_matrix()
    path = tmp_path / "large_mean.h5ad"
    write_h5ad(path, matrix)
    blocks = read_blocks(path, 32)
    assert len(blocks) > 10, "the block size must actually split the matrix"

    dense = np.asarray(matrix.todense(), dtype=np.float64)
    reference = dense.var(axis=0, ddof=1)

    _, streamed = stream_gene_statistics(blocks)
    assert_allclose(streamed, reference, rtol=1e-9)

    scaled = scrust_call("pp.scale", AnnData(matrix), max_value=None, inplace=False)
    _, crate_deviation = recover_mean_and_deviation(matrix, scaled)
    assert_allclose(np.sqrt(streamed), crate_deviation, rtol=5e-2)

    naive = naive_gene_variance(blocks)
    naive_error = np.abs(naive - reference).max() / reference.max()
    streamed_error = np.abs(streamed - reference).max() / reference.max()
    assert naive_error > 1000 * max(streamed_error, 1e-12), (
        f"the naive formula was expected to collapse here; it was out by {naive_error} "
        f"against the merge's {streamed_error}"
    )
