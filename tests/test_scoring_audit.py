"""Cross-check `score_genes` and the two sampling routines against scanpy.

`score_genes` makes an unusually strong claim: that it reproduces scanpy's random draw
*exactly*, by reimplementing numpy's legacy Mersenne Twister and Fisher-Yates shuffle in
Rust. Either it does or it does not, and the difference is not a tolerance -- a
different control set is a different score for every cell. That claim is what most of
this file is about.

The chain it has to match, from `scanpy/tools/_score_genes.py`:

    np.random.seed(random_state)                          # the *global* legacy state
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items        # equal-count bins
    r_genes = r_genes.to_series().sample(ctrl_size).index  # pandas, not numpy directly

`Series.sample(k)` turns out to be `np.random.permutation(n)[:k]` exactly -- checked in
`test_pandas_sample_is_a_legacy_permutation` rather than assumed, because the whole
audit rests on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from scrust_call import DEVICE, scrust_call


def csr_args(matrix: sparse.csr_matrix):
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def scrust_score(matrix, gene_set, ctrl_size=50, n_bins=25, seed=0):
    return np.asarray(
        scrust_call(
            "_scrust.score_genes",
            *csr_args(matrix),
            np.asarray(gene_set, dtype=np.uint32),
            ctrl_size,
            n_bins,
            seed,
            DEVICE,
        )
    )


def scrust_subsample(n_cells, n_keep, replace, seed):
    return np.asarray(scrust_call("_scrust.subsample", n_cells, n_keep, replace, seed))


def expression(n_cells=200, n_genes=300, seed=0, sparsity=0.5):
    """Log-scale data with a wide spread of gene means, so the expression bins are not
    all the same and the control draw actually has choices to make."""
    rng = np.random.default_rng(seed)
    scale = rng.lognormal(0.0, 1.5, size=n_genes)
    dense = rng.poisson(scale, size=(n_cells, n_genes)).astype(np.float32)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(np.log1p(dense).astype(np.float32))


def scanpy_score(matrix, gene_set, ctrl_size=50, n_bins=25, seed=0):
    names = [f"g{i}" for i in range(matrix.shape[1])]
    adata = AnnData(matrix.copy())
    adata.var_names = names
    sc.tl.score_genes(
        adata,
        [names[i] for i in gene_set],
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        random_state=seed,
        score_name="s",
        use_raw=False,
    )
    return adata.obs["s"].to_numpy()


# --------------------------------------------------------------------------------
# 1. The premise the rest of the file rests on.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "k", "seed"), [(100, 7, 0), (50, 50, 3), (1000, 25, 11)])
def test_pandas_sample_is_a_legacy_permutation(n, k, seed):
    """scanpy draws its control genes with `Series.sample`, which is pandas, not numpy.
    The core reimplements `np.random.permutation`. Those are only the same draw because
    pandas routes `sample` through the global legacy `RandomState`, and because
    `choice(replace=False)` is itself a truncated permutation.

    Asserted rather than assumed: if a pandas release changes this, the core's scores
    stop matching scanpy's and this test says why before the others do.
    """
    np.random.seed(seed)
    from_pandas = pd.Series(range(n)).sample(k).index.to_numpy()
    np.random.seed(seed)
    from_numpy = np.random.permutation(n)[:k]
    np.testing.assert_array_equal(from_pandas, from_numpy)


# --------------------------------------------------------------------------------
# 2. score_genes
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_score_genes_matches_scanpy(seed):
    """The whole thing: same bins, same control draw, same score, to f32.

    If the Mersenne Twister port were wrong by a single draw the control set would
    differ, and the scores would not be close -- this is not a tolerance test dressed up
    as an exactness one.
    """
    matrix = expression()
    gene_set = list(range(10, 35))
    ours = scrust_score(matrix, gene_set, seed=seed)
    theirs = scanpy_score(matrix, gene_set, seed=seed)
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ctrl_size", [10, 50, 100])
def test_score_genes_matches_scanpy_across_control_sizes(ctrl_size):
    matrix = expression(seed=2)
    gene_set = list(range(40, 60))
    np.testing.assert_allclose(
        scrust_score(matrix, gene_set, ctrl_size=ctrl_size),
        scanpy_score(matrix, gene_set, ctrl_size=ctrl_size),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("n_bins", [5, 25, 50])
def test_score_genes_matches_scanpy_across_bin_counts(n_bins):
    """`n_items = round(len(obs_avg) / (n_bins - 1))` and `rank(method="min") // n_items`
    is an equal-*count* binning whose last bin is short whenever the gene count is not a
    multiple of the bin size. The off-by-one lives here."""
    matrix = expression(seed=4)
    gene_set = list(range(5, 25))
    np.testing.assert_allclose(
        scrust_score(matrix, gene_set, n_bins=n_bins),
        scanpy_score(matrix, gene_set, n_bins=n_bins),
        rtol=1e-4,
        atol=1e-5,
    )


def test_a_different_seed_gives_a_different_control_set():
    """The seed has to reach the draw. A score that ignored it would match scanpy at
    seed 0 and quietly disagree everywhere else.

    The matrix has to be big enough for a draw to happen at all -- see
    `test_the_seed_does_nothing_until_a_bin_is_larger_than_the_control_size`, which is
    why this uses 2000 genes rather than the 300 the rest of the file uses.
    """
    matrix = expression(n_genes=2000, seed=6)
    gene_set = list(range(10, 30))
    first = scrust_score(matrix, gene_set, seed=0)
    second = scrust_score(matrix, gene_set, seed=7)
    assert not np.allclose(first, second), "the seed does not reach the control draw"
    np.testing.assert_allclose(first, scanpy_score(matrix, gene_set, seed=0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(second, scanpy_score(matrix, gene_set, seed=7), rtol=1e-4, atol=1e-5)


def test_the_seed_does_nothing_until_a_bin_is_larger_than_the_control_size():
    """A property of `score_genes` that surprises, and that both sides share.

    Genes are cut into `n_bins` equal-count bins, so a bin holds about
    `n_genes / (n_bins - 1)` genes. `ctrl_size` genes are then drawn from each bin --
    but only `if ctrl_size < len(bin)`. Below that the whole bin is taken and there is
    no draw, so `random_state` changes nothing.

    At scanpy's defaults, `ctrl_size=50` and `n_bins=25`, that threshold is about 1200
    genes. Under it -- a subsetted panel, a marker-gene matrix, most toy examples --
    `score_genes` is deterministic no matter what seed is passed, which is easy to
    mistake for a seed that is not wired up. It is wired up; there is simply nothing
    for it to do.

    Pinned on both sides, so the claim is about `score_genes` and not about this crate.
    """
    gene_set = list(range(10, 30))

    small = expression(n_genes=300, seed=6)  # about 12 genes per bin, under ctrl_size
    assert np.allclose(scrust_score(small, gene_set, seed=0), scrust_score(small, gene_set, seed=7))
    assert np.allclose(scanpy_score(small, gene_set, seed=0), scanpy_score(small, gene_set, seed=7))

    large = expression(n_genes=2000, seed=6)  # about 83 genes per bin, over it
    assert not np.allclose(
        scrust_score(large, gene_set, seed=0), scrust_score(large, gene_set, seed=7)
    )
    assert not np.allclose(
        scanpy_score(large, gene_set, seed=0), scanpy_score(large, gene_set, seed=7)
    )


def test_score_genes_is_deterministic_for_a_fixed_seed():
    matrix = expression(seed=8)
    gene_set = list(range(3, 18))
    first = scrust_score(matrix, gene_set, seed=5)
    second = scrust_score(matrix, gene_set, seed=5)
    np.testing.assert_array_equal(first, second)


def test_a_control_set_larger_than_its_bin_takes_the_whole_bin():
    """When `ctrl_size` is not smaller than the bin, scanpy skips the sampling entirely
    and takes every gene in it. The core has to skip the draw too -- consuming random
    numbers there would desynchronise every later bin."""
    matrix = expression(n_genes=60, seed=10)
    gene_set = list(range(0, 12))
    ours = scrust_score(matrix, gene_set, ctrl_size=1000, n_bins=5)
    theirs = scanpy_score(matrix, gene_set, ctrl_size=1000, n_bins=5)
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-5)


def test_scoring_a_gene_against_itself_is_not_zero_but_is_centred():
    """A sanity bound independent of scanpy: the score is the gene set's mean minus a
    control mean drawn from the same expression bins, so scoring a random set should
    straddle zero rather than sit anywhere in particular."""
    matrix = expression(seed=12)
    scores = scrust_score(matrix, list(range(100, 140)))
    assert np.isfinite(scores).all()
    assert scores.min() < 0.0 < scores.max(), "a random gene set should straddle zero"


# --------------------------------------------------------------------------------
# 3. subsample
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_keep", [1, 25, 199])
def test_subsample_without_replacement_keeps_distinct_cells(n_keep):
    kept = scrust_subsample(200, n_keep, False, 0)
    assert len(kept) == n_keep
    assert len(set(kept.tolist())) == n_keep, "a cell was drawn twice without replacement"
    assert kept.min() >= 0 and kept.max() < 200


def test_subsample_with_replacement_may_repeat_and_may_exceed_the_population():
    drawn = scrust_subsample(50, 500, True, 3)
    assert len(drawn) == 500
    assert len(set(drawn.tolist())) < 500, "500 draws from 50 cells produced no repeat"
    assert drawn.min() >= 0 and drawn.max() < 50


def test_subsample_is_deterministic_for_a_fixed_seed_and_varies_otherwise():
    first = scrust_subsample(300, 40, False, 11)
    np.testing.assert_array_equal(first, scrust_subsample(300, 40, False, 11))
    assert not np.array_equal(first, scrust_subsample(300, 40, False, 12))


def test_subsampling_everything_is_a_permutation():
    """`n_keep == n_cells` without replacement has to return every cell exactly once,
    which is the case an off-by-one in the Fisher-Yates loop would break."""
    kept = np.sort(scrust_subsample(64, 64, False, 5))
    np.testing.assert_array_equal(kept, np.arange(64))


# --------------------------------------------------------------------------------
# 4. downsample_counts
# --------------------------------------------------------------------------------


def test_downsample_counts_per_cell_matches_scanpys_totals():
    """Every cell above the target comes down to it exactly; every cell below is left
    alone. The draw itself is random, so what is compared against scanpy is the
    invariant, not the values."""
    rng = np.random.default_rng(15)
    dense = rng.poisson(6.0, size=(60, 30)).astype(np.float32)
    matrix = sparse.csr_matrix(dense)
    target = 40

    indptr, indices, values, _ = scrust_call(
        "_scrust.downsample_counts", *csr_args(matrix), target, None, False, 0
    )
    ours = sparse.csr_matrix(
        (np.asarray(values), np.asarray(indices), np.asarray(indptr)), shape=matrix.shape
    )
    before = np.asarray(matrix.sum(axis=1)).ravel()
    after = np.asarray(ours.sum(axis=1)).ravel()

    np.testing.assert_array_equal(after[before > target], target)
    np.testing.assert_array_equal(after[before <= target], before[before <= target])
    assert (ours.toarray() <= dense + 1e-6).all(), "downsampling created counts"

    adata = AnnData(matrix.copy())
    sc.pp.downsample_counts(adata, counts_per_cell=target, random_state=0)
    theirs = np.asarray(adata.X.sum(axis=1)).ravel()
    np.testing.assert_array_equal(after, theirs)


def test_downsample_counts_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(17)
    matrix = sparse.csr_matrix(rng.poisson(5.0, size=(40, 20)).astype(np.float32))
    first = scrust_call("_scrust.downsample_counts", *csr_args(matrix), 25, None, False, 9)
    second = scrust_call("_scrust.downsample_counts", *csr_args(matrix), 25, None, False, 9)
    np.testing.assert_array_equal(np.asarray(first[2]), np.asarray(second[2]))
