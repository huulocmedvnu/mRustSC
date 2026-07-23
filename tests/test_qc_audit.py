"""Cross-check `calculate_qc_metrics`, `sqrt` and the two filters against scanpy.

These run before anything else in a pipeline, so an error here is an error in every
result downstream, and it is the kind that does not announce itself: a QC metric that
is quietly 3% off still looks like a QC metric.

The reference is scanpy driven on the same matrix. What each test pins is in its
docstring; the ones carrying the most risk are:

* `test_explicit_zeros_are_not_expressed_genes`, because scanpy calls
  `eliminate_zeros()` before it counts anything and the core has to skip them by hand;
* `test_percent_top_of_a_cell_with_no_counts_is_not_a_number`, because the ratio is
  0/0 and a silent 0 there would look like a real measurement;
* `test_filter_counts_positive_entries_where_qc_counts_non_zero_ones`, because the two
  modules genuinely disagree on what "expressed" means, and so does scanpy.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from scrust_call import scrust_call


def csr_args(matrix: sparse.csr_matrix):
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def scrust_qc(matrix, percent_top=(50,), subsets=()):
    return scrust_call(
        "_scrust.qc_metrics",
        *csr_args(matrix),
        list(percent_top),
        [np.asarray(s, dtype=bool) for s in subsets],
    )


def scrust_filter_cells(matrix, *, min_genes=None, min_counts=None):
    return np.asarray(scrust_call("_scrust.filter_cells", *csr_args(matrix), min_genes, min_counts))


def scrust_filter_genes(matrix, *, min_cells=None, min_counts=None):
    return np.asarray(scrust_call("_scrust.filter_genes", *csr_args(matrix), min_cells, min_counts))


def counts(n_cells=120, n_genes=80, seed=0, sparsity=0.7):
    rng = np.random.default_rng(seed)
    scale = rng.lognormal(0.5, 1.2, size=n_genes)
    dense = rng.poisson(scale, size=(n_cells, n_genes)).astype(np.float32)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(dense)


def scanpy_qc(matrix, percent_top=(50,), subsets=None):
    """scanpy's obs and var frames for the same matrix.

    `calculate_qc_metrics` mutates nothing when `inplace=False`; the subsets arrive as
    boolean columns in `var`, which is how scanpy spells `qc_vars`.
    """
    adata = AnnData(matrix.copy())
    qc_vars = []
    for i, subset in enumerate(subsets or []):
        name = f"s{i}"
        adata.var[name] = np.asarray(subset, dtype=bool)
        qc_vars.append(name)
    return sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, percent_top=list(percent_top) or None, inplace=False, log1p=False
    )


# --------------------------------------------------------------------------------
# 1. The metrics themselves.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("percent_top", [(50,), (10, 50), (1, 5, 20, 50)])
def test_cell_metrics_match_scanpy(percent_top):
    matrix = counts()
    cells, _ = scrust_qc(matrix, percent_top)
    obs, _ = scanpy_qc(matrix, percent_top)

    np.testing.assert_array_equal(
        np.asarray(cells["n_genes_by_counts"]), obs["n_genes_by_counts"].to_numpy()
    )
    np.testing.assert_allclose(
        np.asarray(cells["total_counts"]), obs["total_counts"].to_numpy(), rtol=1e-6
    )
    for i, n in enumerate(sorted(percent_top)):
        np.testing.assert_allclose(
            np.asarray(cells["pct_counts_in_top"])[i] * 100.0,
            obs[f"pct_counts_in_top_{n}_genes"].to_numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"top {n}",
        )


def test_gene_metrics_match_scanpy():
    matrix = counts(seed=2)
    _, genes = scrust_qc(matrix)
    _, var = scanpy_qc(matrix)

    np.testing.assert_array_equal(
        np.asarray(genes["n_cells_by_counts"]), var["n_cells_by_counts"].to_numpy()
    )
    np.testing.assert_allclose(
        np.asarray(genes["mean_counts"]), var["mean_counts"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(genes["total_counts"]), var["total_counts"].to_numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(genes["pct_dropout_by_counts"]),
        var["pct_dropout_by_counts"].to_numpy(),
        rtol=1e-5,
    )


def test_gene_subset_totals_match_scanpys_qc_vars():
    """`pct_counts_mt` and friends. The core returns the totals; the percentage scanpy
    reports is that total over `total_counts`, so both halves are checked here."""
    matrix = counts(seed=4)
    rng = np.random.default_rng(0)
    mito = rng.random(matrix.shape[1]) < 0.1
    ribo = rng.random(matrix.shape[1]) < 0.2
    assert mito.any() and ribo.any()

    cells, _ = scrust_qc(matrix, (50,), (mito, ribo))
    obs, _ = scanpy_qc(matrix, (50,), (mito, ribo))

    totals = np.asarray(cells["subset_totals"])
    for i, name in enumerate(("s0", "s1")):
        np.testing.assert_allclose(totals[i], obs[f"total_counts_{name}"].to_numpy(), rtol=1e-6)
        percentage = totals[i] / np.asarray(cells["total_counts"]) * 100.0
        np.testing.assert_allclose(
            percentage, obs[f"pct_counts_{name}"].to_numpy(), rtol=1e-5, atol=1e-5
        )


# --------------------------------------------------------------------------------
# 2. The edges, where a metric can be wrong without looking wrong.
# --------------------------------------------------------------------------------


def test_explicit_zeros_are_not_expressed_genes():
    """scanpy calls `eliminate_zeros()` before it counts (`_qc.py:88-91`), so a stored
    zero is not an expressed gene and not an occupied cell. The core walks the stored
    entries and has to skip them by hand.

    The two matrices here are equal as dense arrays and differ only in what is stored,
    so any difference between them is this bug and nothing else.
    """
    matrix = counts(seed=6)
    dense = matrix.toarray()

    explicit = sparse.csr_matrix(np.ones_like(dense))
    explicit.data = dense.ravel().copy()  # every entry stored, most of them zero
    assert explicit.nnz == dense.size
    assert explicit.count_nonzero() < explicit.nnz

    compact_cells, compact_genes = scrust_qc(matrix)
    stored_cells, stored_genes = scrust_qc(explicit)

    for key in ("n_genes_by_counts", "total_counts"):
        np.testing.assert_allclose(
            np.asarray(stored_cells[key]), np.asarray(compact_cells[key]), rtol=1e-6
        )
    for key in ("n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"):
        np.testing.assert_allclose(
            np.asarray(stored_genes[key]), np.asarray(compact_genes[key]), rtol=1e-6
        )

    obs, var = scanpy_qc(matrix)
    np.testing.assert_array_equal(
        np.asarray(stored_cells["n_genes_by_counts"]), obs["n_genes_by_counts"].to_numpy()
    )
    np.testing.assert_array_equal(
        np.asarray(stored_genes["n_cells_by_counts"]), var["n_cells_by_counts"].to_numpy()
    )


def test_percent_top_of_a_cell_with_no_counts_is_not_a_number():
    """An empty cell has no total to divide by. The ratio is 0/0, and NaN is the honest
    answer -- a 0 would read as "this cell holds none of its counts in its top genes",
    which is a statement about a cell that has no counts to hold.

    scanpy produces NaN here too, from the same division.
    """
    matrix = counts(n_cells=40, seed=8).tolil()
    matrix[0, :] = 0
    matrix = sparse.csr_matrix(matrix)
    assert matrix[0].nnz == 0

    cells, _ = scrust_qc(matrix, (10,))
    obs, _ = scanpy_qc(matrix, (10,))

    ours = np.asarray(cells["pct_counts_in_top"])[0]
    theirs = obs["pct_counts_in_top_10_genes"].to_numpy()
    assert np.isnan(ours[0]), "an empty cell must not report a top-gene fraction"
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
    np.testing.assert_allclose(
        ours[~np.isnan(ours)] * 100.0, theirs[~np.isnan(theirs)], rtol=1e-5, atol=1e-5
    )


def test_percent_top_deeper_than_the_cell_expresses_is_all_of_it():
    """Asking for the top 500 genes of a cell that expresses 12 is asking for all of
    them, so the fraction is 1. The core falls back to the last cumulative total rather
    than indexing past the end."""
    matrix = counts(n_cells=30, n_genes=40, seed=10, sparsity=0.85)
    expressed = np.diff(matrix.indptr)
    assert expressed.min() < 40, "some cell has to express fewer genes than we ask for"

    cells, _ = scrust_qc(matrix, (40,))
    fractions = np.asarray(cells["pct_counts_in_top"])[0]
    totals = np.asarray(cells["total_counts"])
    np.testing.assert_allclose(fractions[totals > 0], 1.0, rtol=1e-6)

    obs, _ = scanpy_qc(matrix, (40,))
    theirs = obs["pct_counts_in_top_40_genes"].to_numpy()
    np.testing.assert_allclose(
        fractions[totals > 0] * 100.0, theirs[totals > 0], rtol=1e-5, atol=1e-5
    )


def test_a_gene_in_no_cell_has_dropped_out_of_all_of_them():
    matrix = counts(n_cells=50, seed=12).tolil()
    matrix[:, 3] = 0
    matrix = sparse.csr_matrix(matrix)

    _, genes = scrust_qc(matrix)
    assert np.asarray(genes["n_cells_by_counts"])[3] == 0
    assert np.asarray(genes["pct_dropout_by_counts"])[3] == 100.0
    assert np.asarray(genes["mean_counts"])[3] == 0.0

    _, var = scanpy_qc(matrix)
    np.testing.assert_allclose(
        np.asarray(genes["pct_dropout_by_counts"]),
        var["pct_dropout_by_counts"].to_numpy(),
        rtol=1e-5,
    )


# --------------------------------------------------------------------------------
# 3. sqrt
# --------------------------------------------------------------------------------


def test_sqrt_matches_scipy_and_stays_sparse():
    matrix = counts(seed=14)
    indptr, indices, values, _ = scrust_call("_scrust.sqrt", *csr_args(matrix))
    ours = sparse.csr_matrix(
        (np.asarray(values), np.asarray(indices), np.asarray(indptr)), shape=matrix.shape
    )
    np.testing.assert_allclose(ours.toarray(), np.sqrt(matrix.toarray()), rtol=1e-6)
    np.testing.assert_array_equal(ours.indptr, matrix.indptr)


def test_sqrt_of_a_negative_is_a_nan_rather_than_an_error():
    """`scipy.sparse.csr_matrix.sqrt` yields NaN on a negative stored value rather than
    refusing the matrix, and the core follows it: rejecting input scanpy accepts is a
    compatibility break, as the t-SNE perplexity guard was."""
    dense = np.array([[4.0, -1.0], [0.0, 9.0]], dtype=np.float32)
    matrix = sparse.csr_matrix(dense)
    _, _, values, _ = scrust_call("_scrust.sqrt", *csr_args(matrix))
    values = np.asarray(values)
    assert values[0] == 2.0
    assert np.isnan(values[1])
    assert values[2] == 3.0


# --------------------------------------------------------------------------------
# 4. The filters.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("min_genes", [1, 5, 20])
def test_filter_cells_by_gene_count_matches_scanpy(min_genes):
    matrix = counts(seed=16)
    adata = AnnData(matrix.copy())
    keep, _ = sc.pp.filter_cells(adata, min_genes=min_genes, inplace=False)
    np.testing.assert_array_equal(scrust_filter_cells(matrix, min_genes=min_genes), keep)


@pytest.mark.parametrize("min_counts", [1.0, 50.0, 200.0])
def test_filter_cells_by_total_counts_matches_scanpy(min_counts):
    matrix = counts(seed=18)
    adata = AnnData(matrix.copy())
    keep, _ = sc.pp.filter_cells(adata, min_counts=min_counts, inplace=False)
    np.testing.assert_array_equal(scrust_filter_cells(matrix, min_counts=min_counts), keep)


@pytest.mark.parametrize("min_cells", [1, 10, 40])
def test_filter_genes_by_cell_count_matches_scanpy(min_cells):
    matrix = counts(seed=20)
    adata = AnnData(matrix.copy())
    keep, _ = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    np.testing.assert_array_equal(scrust_filter_genes(matrix, min_cells=min_cells), keep)


@pytest.mark.parametrize("min_counts", [1.0, 30.0, 150.0])
def test_filter_genes_by_total_counts_matches_scanpy(min_counts):
    matrix = counts(seed=22)
    adata = AnnData(matrix.copy())
    keep, _ = sc.pp.filter_genes(adata, min_counts=min_counts, inplace=False)
    np.testing.assert_array_equal(scrust_filter_genes(matrix, min_counts=min_counts), keep)


def test_filter_counts_positive_entries_where_qc_counts_non_zero_ones():
    """The two modules mean different things by "expressed", and so does scanpy.

    `filter_cells(min_genes=)` counts entries *greater than zero*; `calculate_qc_metrics`
    counts entries that are *not zero*. On counts they agree, because a count cannot be
    negative. On a matrix that has been centred or regressed they do not, and the gap is
    every negative entry.

    Pinned rather than reconciled: both follow scanpy, which draws the line in these two
    places itself. The test exists so that a caller who filters a centred matrix knows
    what they are getting.
    """
    dense = np.array(
        [
            [1.0, -2.0, 0.0, 3.0],
            [-1.0, -1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    matrix = sparse.csr_matrix(dense)

    cells, _ = scrust_qc(matrix, ())
    non_zero = np.asarray(cells["n_genes_by_counts"])
    np.testing.assert_array_equal(non_zero, np.array([3, 3, 0], dtype=non_zero.dtype))

    positive_only = scrust_filter_cells(matrix, min_genes=1)
    np.testing.assert_array_equal(positive_only, np.array([True, False, False]))

    adata = AnnData(matrix.copy())
    keep, _ = sc.pp.filter_cells(adata, min_genes=1, inplace=False)
    np.testing.assert_array_equal(positive_only, keep)
