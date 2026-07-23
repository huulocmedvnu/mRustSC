"""Cross-check `normalize_total`, `log1p` and `highly_variable_genes` against scanpy.

Every reference here is scanpy driven on the same matrix, not a transcription, except
where the note says otherwise.

The three places this looks hardest are the three that carry the most risk:

* the median that `normalize_total` picks when `target_sum` is None, because scanpy
  computes it differently depending on how the matrix is *stored*;
* the one-gene bin in `highly_variable_genes`, where scanpy swaps its centre and its
  spread rather than dividing by a missing standard deviation;
* the `n_top_genes` cut-off, which scanpy applies as a threshold rather than a count,
  so ties on the boundary make the result longer than asked for.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from scrust_call import DEVICE, scrust_call

FLAVORS = ("seurat", "cell_ranger")


def csr_args(matrix: sparse.csr_matrix):
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def scrust_normalize(matrix, target_sum=None):
    result = scrust_call("_scrust.normalize_total", *csr_args(matrix), target_sum, DEVICE)
    return _to_csr(result, matrix.shape)


def scrust_log1p(matrix):
    return _to_csr(scrust_call("_scrust.log1p", *csr_args(matrix)), matrix.shape)


def _to_csr(result, shape):
    indptr, indices, values, _n_cols = result
    return sparse.csr_matrix(
        (np.asarray(values), np.asarray(indices), np.asarray(indptr)), shape=shape
    )


def scrust_hvg(matrix, n_top_genes, flavor):
    return scrust_call(
        "_scrust.highly_variable_genes", *csr_args(matrix), n_top_genes, flavor, DEVICE
    )


def counts(n_cells=200, n_genes=60, seed=0, sparsity=0.6):
    """A count matrix with a wide spread of means, which is what makes the mean bins
    non-trivial."""
    rng = np.random.default_rng(seed)
    scale = rng.lognormal(0.0, 1.5, size=n_genes).astype(np.float32)
    dense = rng.poisson(scale, size=(n_cells, n_genes)).astype(np.float32)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(dense)


def logged(matrix, target_sum=1e4):
    """scanpy's own normalize + log1p, so the HVG comparison starts from data scanpy
    produced rather than data the core produced."""
    adata = AnnData(matrix.copy())
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return sparse.csr_matrix(adata.X)


# --------------------------------------------------------------------------------
# 1. normalize_total
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("target_sum", [1e4, 1.0, 1e6])
def test_normalize_total_matches_scanpy_for_an_explicit_target(target_sum):
    matrix = counts()
    adata = AnnData(matrix.copy())
    sc.pp.normalize_total(adata, target_sum=target_sum)
    np.testing.assert_allclose(
        scrust_normalize(matrix, target_sum).toarray(), adata.X.toarray(), rtol=1e-5, atol=1e-6
    )


def test_normalize_total_matches_scanpy_when_the_median_is_implied():
    """`target_sum=None` on a matrix where every cell has counts, so scanpy's two
    median rules agree and only the arithmetic is under test."""
    matrix = counts(seed=2)
    assert np.all(np.asarray(matrix.sum(axis=1)).ravel() > 0)

    adata = AnnData(matrix.copy())
    sc.pp.normalize_total(adata, target_sum=None)
    np.testing.assert_allclose(
        scrust_normalize(matrix).toarray(), adata.X.toarray(), rtol=1e-5, atol=1e-6
    )


def test_the_implied_median_follows_scanpys_sparse_rule_not_its_dense_one():
    """DOCUMENTED DIVERGENCE, and it is scanpy that is of two minds, not the core.

    With `target_sum=None` scanpy picks the target differently depending on how the
    matrix is stored (`_normalization.py:93-117`):

    * CSR: `np.median(counts_per_cell)`, over *every* cell;
    * anything else: `_compute_nnz_median`, over cells with a non-zero count.

    They differ as soon as an empty cell exists -- which is exactly what a
    `min_genes` filter has not yet removed. The core always takes a CSR and so always
    follows the first rule. This test states which scanpy it agrees with, and measures
    the gap to the other one so nobody has to guess whether it matters.
    """
    matrix = counts(n_cells=60, seed=3).tolil()
    matrix[:12, :] = 0  # a fifth of the cells hold no counts at all
    matrix = sparse.csr_matrix(matrix)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    assert (totals == 0).sum() == 12

    sparse_rule = float(np.median(totals))
    dense_rule = float(np.median(totals[totals > 0]))
    assert sparse_rule != dense_rule, "this matrix has to separate the two rules"

    ours = scrust_normalize(matrix).toarray()

    from_csr = AnnData(matrix.copy())
    sc.pp.normalize_total(from_csr, target_sum=None)
    np.testing.assert_allclose(ours, from_csr.X.toarray(), rtol=1e-5, atol=1e-6)

    from_dense = AnnData(np.asarray(matrix.todense()))
    sc.pp.normalize_total(from_dense, target_sum=None)
    assert not np.allclose(ours, np.asarray(from_dense.X), rtol=1e-3, atol=1e-3), (
        "scanpy's two paths have converged; the core can stop choosing between them"
    )
    # The gap is exactly the ratio of the two targets.
    np.testing.assert_allclose(
        ours, np.asarray(from_dense.X) * (sparse_rule / dense_rule), rtol=1e-5, atol=1e-6
    )


def test_a_cell_with_no_counts_is_left_alone_rather_than_divided_by_zero():
    matrix = counts(n_cells=40, seed=5).tolil()
    matrix[0, :] = 0
    matrix = sparse.csr_matrix(matrix)

    ours = scrust_normalize(matrix, 1e4).toarray()
    assert np.all(ours[0] == 0.0)
    assert np.isfinite(ours).all()

    adata = AnnData(matrix.copy())
    sc.pp.normalize_total(adata, target_sum=1e4)
    np.testing.assert_allclose(ours, adata.X.toarray(), rtol=1e-5, atol=1e-6)


def test_log1p_matches_scanpy_and_keeps_the_sparsity_pattern():
    matrix = counts(seed=7)
    ours = scrust_log1p(matrix)
    adata = AnnData(matrix.copy())
    sc.pp.log1p(adata)
    np.testing.assert_allclose(ours.toarray(), adata.X.toarray(), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(ours.indptr, matrix.indptr)
    np.testing.assert_array_equal(ours.indices, matrix.indices)


# --------------------------------------------------------------------------------
# 2. highly_variable_genes, against scanpy on the same log data
# --------------------------------------------------------------------------------


def scanpy_hvg(matrix, n_top_genes, flavor):
    adata = AnnData(matrix.copy())
    return sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, inplace=False)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("n_top_genes", [10, 25])
def test_hvg_matches_scanpy(flavor, n_top_genes):
    """Means, dispersions, normalised dispersions and the selection itself."""
    matrix = logged(counts(seed=11))
    ours = scrust_hvg(matrix, n_top_genes, flavor)
    theirs = scanpy_hvg(matrix, n_top_genes, flavor)

    np.testing.assert_allclose(
        np.asarray(ours["means"]), theirs["means"].to_numpy(), rtol=1e-4, atol=1e-6
    )
    for ours_key, theirs_key in (
        ("dispersions", "dispersions"),
        ("normalised_dispersions", "dispersions_norm"),
    ):
        got = np.asarray(ours[ours_key], dtype=np.float64)
        want = theirs[theirs_key].to_numpy().astype(np.float64)
        both = np.isfinite(got) & np.isfinite(want)
        np.testing.assert_array_equal(
            np.isfinite(got), np.isfinite(want), err_msg=f"{ours_key}: NaN pattern"
        )
        np.testing.assert_allclose(got[both], want[both], rtol=1e-4, atol=1e-6, err_msg=ours_key)

    np.testing.assert_array_equal(
        np.asarray(ours["highly_variable"]), theirs["highly_variable"].to_numpy()
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_hvg_matches_scanpy_on_a_dense_matrix(flavor):
    """No zeros at all, so every gene has a real mean and no bin is empty.

    200 genes over 20 bins, not 40: at two genes per bin the `cell_ranger` statistic
    is degenerate, which is its own test below rather than a confound here.
    """
    rng = np.random.default_rng(13)
    dense = rng.lognormal(0.0, 1.0, size=(150, 200)).astype(np.float32)
    matrix = logged(sparse.csr_matrix(dense))
    ours = scrust_hvg(matrix, 12, flavor)
    theirs = scanpy_hvg(matrix, 12, flavor)
    np.testing.assert_allclose(
        np.asarray(ours["normalised_dispersions"], dtype=np.float64),
        theirs["dispersions_norm"].to_numpy().astype(np.float64),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(ours["highly_variable"]), theirs["highly_variable"].to_numpy()
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_gene_that_never_varies_carries_no_dispersion(flavor):
    """A constant gene has zero variance, so its dispersion is zero and its log is not
    a number. scanpy carries that through as NaN rather than -inf, and a NaN must not
    shift or widen the bin it lands in -- pandas' groupby skips it."""
    matrix = logged(counts(seed=17))
    dense = matrix.toarray()
    dense[:, 0] = 0.0  # never expressed
    dense[:, 1] = 1.5  # expressed, but identically
    matrix = sparse.csr_matrix(dense)

    ours = scrust_hvg(matrix, 10, flavor)
    theirs = scanpy_hvg(matrix, 10, flavor)
    got = np.asarray(ours["normalised_dispersions"], dtype=np.float64)
    want = theirs["dispersions_norm"].to_numpy().astype(np.float64)
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
    both = np.isfinite(got) & np.isfinite(want)
    np.testing.assert_allclose(got[both], want[both], rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(ours["highly_variable"]), theirs["highly_variable"].to_numpy()
    )


def test_a_bin_holding_one_gene_normalises_it_to_exactly_one():
    """scanpy's odd corner, `_highly_variable_genes.py:490-502`.

    A bin with a single gene has no standard deviation, so pandas gives NaN. Rather
    than propagate it scanpy sets the bin's spread to its mean and its centre to zero,
    which sends that gene's normalised dispersion to exactly 1 -- not to 0, and not to
    NaN. It only shows up when the mean bins are wide enough to isolate a gene, so the
    matrix here has one gene far above the rest.
    """
    rng = np.random.default_rng(19)
    dense = rng.poisson(3.0, size=(120, 30)).astype(np.float32)
    dense[:, 0] = rng.poisson(4000.0, size=120).astype(np.float32)  # alone in its bin
    matrix = logged(sparse.csr_matrix(dense))

    theirs = scanpy_hvg(matrix, 5, "seurat")
    ours = scrust_hvg(matrix, 5, "seurat")
    got = np.asarray(ours["normalised_dispersions"], dtype=np.float64)
    want = theirs["dispersions_norm"].to_numpy().astype(np.float64)

    assert np.isclose(want[0], 1.0), "the fixture stopped isolating gene 0 in its own bin"
    np.testing.assert_allclose(got[0], 1.0, rtol=1e-9)
    both = np.isfinite(got) & np.isfinite(want)
    np.testing.assert_allclose(got[both], want[both], rtol=1e-4, atol=1e-6)


def test_cell_ranger_bins_holding_two_genes_are_degenerate():
    """DOCUMENTED DIVERGENCE, and the core is the one that is arithmetically right.

    `cell_ranger` centres each bin on its median and scales by its MAD. For a bin
    holding exactly two dispersions `a` and `b`, the median is `(a + b) / 2` and both
    deviations are `|a - b| / 2`, so the MAD is `|a - b| / 2` and *every* gene in such a
    bin normalises to exactly `+-c`, where `c = 0.674489750196` is the constant
    `statsmodels.robust.mad` divides by. That is exact arithmetic, not a coincidence:
    with two points a robust scale carries no information.

    The core reproduces it exactly, so 40 genes over 20 bins give it 6 distinct
    normalised dispersions. scanpy computes the same quantity through pandas and
    statsmodels and its values land a few ulps apart instead, so it sees 31 distinct
    values and its `>= cutoff` selection stops earlier. Asking for the top 12 genes
    then gives the core 17 and scanpy 13, and neither list is more correct -- the
    ranking is simply not defined once the values are genuinely tied.

    This is confined to the degenerate regime. Measured over the same fixture:

        genes/bin    distinct    core    scanpy    identical
              2.0           6      17        13    no
              3.0          27      12        12    yes
              5.0          67      12        12    yes
             10.0         200      12        12    yes

    Real data has thousands of genes over 20 bins, so it never gets near this. The
    test exists so that a difference in gene lists here is recognised rather than
    chased.
    """
    rng = np.random.default_rng(13)
    dense = rng.lognormal(0.0, 1.0, size=(150, 40)).astype(np.float32)
    matrix = logged(sparse.csr_matrix(dense))

    ours = scrust_hvg(matrix, 12, "cell_ranger")
    theirs = scanpy_hvg(matrix, 12, "cell_ranger")
    got = np.asarray(ours["normalised_dispersions"], dtype=np.float64)
    want = theirs["dispersions_norm"].to_numpy().astype(np.float64)

    mad_scale = 0.6744897501960817
    # The core's values are the exact ones, and there are only a handful of them.
    finite = got[np.isfinite(got)]
    assert np.unique(finite).size < 10, "the bins are no longer degenerate"
    tied = finite[np.isclose(finite, mad_scale, rtol=1e-6)]
    assert tied.size > 10, "the +-c tie should dominate a two-gene-per-bin matrix"
    assert np.unique(tied).size == 1, "the core's tied values are bit-identical"

    # scanpy agrees to the precision the quantity is computable at, and no further:
    # the dispersions come back as f32, so 1e-6 relative is the floor here.
    both = np.isfinite(got) & np.isfinite(want)
    np.testing.assert_allclose(got[both], want[both], rtol=1e-5, atol=1e-8)
    assert np.unique(want[np.isfinite(want)]).size > np.unique(finite).size, (
        "scanpy's values are no longer perturbed apart; the selections may now agree"
    )

    # And so the selections differ, by exactly the width of the tie.
    assert np.asarray(ours["highly_variable"]).sum() > (theirs["highly_variable"].to_numpy().sum())


def test_the_top_gene_cut_off_is_a_threshold_so_ties_widen_the_selection():
    """scanpy takes the n-th largest normalised dispersion and keeps everything at or
    above it, so a tie on the boundary returns more than `n_top_genes` genes. Building
    the tie deliberately: duplicated genes get identical dispersions and land in the
    same bin, so they normalise identically too.
    """
    rng = np.random.default_rng(23)
    dense = rng.poisson(2.0, size=(100, 20)).astype(np.float32)
    dense[:, 10:] = dense[:, :10]  # every gene has an exact twin
    matrix = logged(sparse.csr_matrix(dense))

    for n_top_genes in (3, 5):
        ours = scrust_hvg(matrix, n_top_genes, "seurat")
        theirs = scanpy_hvg(matrix, n_top_genes, "seurat")
        chosen = np.asarray(ours["highly_variable"])
        np.testing.assert_array_equal(chosen, theirs["highly_variable"].to_numpy())
        assert chosen.sum() >= n_top_genes


@pytest.mark.parametrize("flavor", FLAVORS)
def test_hvg_asked_for_more_genes_than_exist_keeps_them_all(flavor):
    matrix = logged(counts(n_genes=15, seed=29))
    ours = scrust_hvg(matrix, 100, flavor)
    chosen = np.asarray(ours["highly_variable"])
    finite = np.isfinite(np.asarray(ours["normalised_dispersions"], dtype=np.float64))
    assert chosen[finite].all(), "every gene with a dispersion should survive"
