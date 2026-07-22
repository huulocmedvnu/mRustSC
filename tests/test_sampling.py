"""Cell subsampling and count thinning: `pp.subsample`, `pp.sample`, `pp.downsample_counts`.

Two layers are checked here.

The AnnData plumbing — `copy` versus in place, obs/var alignment, the argument
errors — goes through the public `pp` functions.

The statistics go through the compiled core directly, and are the real bar:
exact totals, no entry growing without replacement, and per-gene expectations
proportional to the original counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from scrust import _scrust, pp

N_OBS = 8
N_VARS = 5
GROUPS = ["a", "b", "a", "c", "b", "a", "c", "b"]

# One cell deep enough for a distributional check, with a lopsided gene profile.
DEEP_CELL = np.array([[1000.0, 2000.0, 3000.0, 4000.0]], dtype=np.float32)
DEEP_TARGET = 1000
# Repetitions of the draw behind every expectation assertion. 400 puts the
# standard error of the mean at ~0.45 counts on the smallest gene, so a
# four-sigma band is under 2 counts wide on an expectation of 100.
REPEATS = 400


def _adata(counts: np.ndarray | None = None) -> AnnData:
    if counts is None:
        counts = np.arange(1, N_OBS * N_VARS + 1, dtype=np.float32).reshape(N_OBS, N_VARS)
    n_obs, n_vars = counts.shape
    return AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"group": pd.Categorical(GROUPS[:n_obs])},
            index=[f"cell{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame({"marker": np.arange(n_vars)}, index=[f"gene{j}" for j in range(n_vars)]),
    )


# --- the Python layer --------------------------------------------------------


def test_subsample_in_place_keeps_obs_and_var_aligned() -> None:
    adata = _adata()
    assert pp.subsample(adata, n_obs=3) is None
    assert adata.n_obs == 3
    assert adata.n_vars == N_VARS
    assert list(adata.obs.index) == list(adata.obs_names)
    assert len(adata.obs["group"]) == 3
    assert list(adata.var_names) == [f"gene{j}" for j in range(N_VARS)]
    # Each kept row still carries its own counts: row `cellN` is N*N_VARS+1 upwards.
    for name, row in zip(adata.obs_names, adata.X.toarray(), strict=True):
        expected = int(name.removeprefix("cell")) * N_VARS + 1
        assert row[0] == expected


def test_subsample_copy_leaves_the_input_untouched() -> None:
    adata = _adata()
    before = adata.X.toarray()
    subset = pp.subsample(adata, n_obs=3, copy=True)
    assert adata.n_obs == N_OBS
    np.testing.assert_array_equal(adata.X.toarray(), before)
    assert subset is not None
    assert subset.n_obs == 3
    assert set(subset.obs_names) <= set(adata.obs_names)


def test_subsample_by_fraction_matches_the_equivalent_count() -> None:
    by_fraction = pp.subsample(_adata(), 0.5, copy=True)
    by_count = pp.subsample(_adata(), n_obs=N_OBS // 2, copy=True)
    assert by_fraction is not None and by_count is not None
    assert list(by_fraction.obs_names) == list(by_count.obs_names)


def test_subsample_is_reproducible_and_seed_dependent() -> None:
    first = pp.subsample(_adata(), n_obs=4, random_state=3, copy=True)
    again = pp.subsample(_adata(), n_obs=4, random_state=3, copy=True)
    other = pp.subsample(_adata(), n_obs=4, random_state=4, copy=True)
    assert list(first.obs_names) == list(again.obs_names)
    assert list(first.obs_names) != list(other.obs_names)


def test_sample_with_replacement_may_draw_more_cells_than_exist() -> None:
    drawn = pp.sample(_adata(), n=N_OBS * 2, replace=True, copy=True)
    assert drawn is not None
    assert drawn.n_obs == N_OBS * 2
    assert len(set(drawn.obs_names)) < drawn.n_obs


def test_sample_rejects_both_or_neither_criterion() -> None:
    with pytest.raises(TypeError, match="exactly one"):
        pp.sample(_adata(), 0.5, n=2)
    with pytest.raises(TypeError, match="exactly one"):
        pp.sample(_adata())
    with pytest.raises(TypeError, match="exactly one"):
        pp.subsample(_adata(), 0.5, n_obs=2)


def test_subsample_rejects_more_cells_than_exist_without_replacement() -> None:
    with pytest.raises(ValueError, match="at most"):
        pp.subsample(_adata(), n_obs=N_OBS + 1)
    # With replacement the same request is legitimate.
    assert pp.sample(_adata(), n=N_OBS + 1, replace=True, copy=True).n_obs == N_OBS + 1


def test_subsample_rejects_a_negative_or_oversized_fraction() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        pp.subsample(_adata(), -0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pp.sample(_adata(), 1.5)
    # `replace=True` is exactly the case where a fraction above 1 is allowed.
    assert pp.sample(_adata(), 1.5, replace=True, copy=True).n_obs == int(1.5 * N_OBS)


def test_downsample_copy_leaves_the_input_untouched() -> None:
    adata = _adata()
    before = adata.X.toarray()
    thinned = pp.downsample_counts(adata, counts_per_cell=5, copy=True)
    np.testing.assert_array_equal(adata.X.toarray(), before)
    assert thinned is not None
    assert thinned.shape == adata.shape
    assert list(thinned.obs_names) == list(adata.obs_names)
    assert list(thinned.var_names) == list(adata.var_names)


def test_downsample_in_place_replaces_x_and_keeps_the_shape() -> None:
    adata = _adata()
    assert pp.downsample_counts(adata, counts_per_cell=5) is None
    assert adata.shape == (N_OBS, N_VARS)
    assert sp.issparse(adata.X)
    np.testing.assert_array_equal(adata.X.toarray().sum(axis=1), np.full(N_OBS, 5.0))


def test_downsample_by_total_counts_hits_the_matrix_total() -> None:
    """The other criterion, end to end: the wrapper passes the two positionally."""
    adata = _adata()
    pp.downsample_counts(adata, total_counts=100)
    assert adata.X.sum() == 100.0
    assert adata.shape == (N_OBS, N_VARS)


def test_downsample_rejects_both_or_neither_criterion() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        pp.downsample_counts(_adata(), counts_per_cell=5, total_counts=50)
    with pytest.raises(ValueError, match="exactly one"):
        pp.downsample_counts(_adata())


# --- the core itself ---------------------------------------------------------


def _csr_args(matrix: sp.csr_matrix):
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def _thin(counts: np.ndarray, target: int, *, replace: bool = False, seed: int = 0) -> np.ndarray:
    """Thin a dense count block through the compiled core, back to dense."""
    original = sp.csr_matrix(counts.astype(np.float32))
    indptr, indices, values, _ = _scrust.downsample_counts(
        *_csr_args(original), float(target), None, replace, seed
    )
    thinned = sp.csr_matrix((values, indices, indptr), shape=original.shape)
    assert np.all(thinned.data != 0.0), "a thinned matrix must carry no stored zeros"
    return thinned.toarray()


def _rows() -> np.ndarray:
    """Cells with very different depths, so both branches of the target test."""
    return np.array(
        [
            [10.0, 0.0, 20.0, 30.0],  # 60 counts
            [1.0, 2.0, 0.0, 0.0],  # 3 counts, below any target used here
            [40.0, 40.0, 40.0, 40.0],  # 160 counts
            [0.0, 0.0, 0.0, 0.0],  # empty
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("replace", [False, True])
def test_every_cell_hits_the_target_or_keeps_its_own_total(replace: bool) -> None:
    counts = _rows()
    thinned = _thin(counts, 20, replace=replace, seed=1)
    expected = np.minimum(counts.sum(axis=1), 20)
    np.testing.assert_array_equal(thinned.sum(axis=1), expected)


def test_no_count_exceeds_its_original_without_replacement() -> None:
    counts = _rows()
    for seed in range(20):
        thinned = _thin(counts, 20, replace=False, seed=seed)
        assert np.all(thinned <= counts), "a hypergeometric draw cannot invent counts"


def test_with_replacement_a_gene_may_take_more_than_it_had() -> None:
    # 20 draws with replacement out of a cell whose smallest gene holds 10 will
    # eventually hand that gene more than 10, which is the visible difference
    # between the multinomial and the hypergeometric draw.
    counts = np.array([[10.0, 10.0]], dtype=np.float32)
    grew = any(np.any(_thin(counts, 15, replace=True, seed=s) > 10) for s in range(30))
    assert grew


def test_total_counts_thins_the_whole_matrix() -> None:
    original = sp.csr_matrix(_rows())
    indptr, _indices, values, _ = _scrust.downsample_counts(
        *_csr_args(original), None, 50.0, False, 0
    )
    assert values.sum() == 50.0
    assert np.all(values != 0.0)
    assert len(indptr) == original.shape[0] + 1


def _mean_counts(counts: np.ndarray, target: int, replace: bool, repeats: int) -> np.ndarray:
    """Mean count per gene over `repeats` seeds."""
    return np.mean([_thin(counts, target, replace=replace, seed=s) for s in range(repeats)], axis=0)


def _standard_error(counts: np.ndarray, target: int, replace: bool, repeats: int) -> np.ndarray:
    """Standard error of `_mean_counts`, from the exact variance of the draw.

    A gene's count is binomial for the multinomial draw, and the same variance
    times the finite population correction for the hypergeometric one.
    """
    total = counts.sum()
    share = counts / total
    variance = target * share * (1.0 - share)
    if not replace:
        variance = variance * (total - target) / (total - 1.0)
    return np.sqrt(variance / repeats)


@pytest.mark.parametrize("replace", [False, True])
def test_expected_counts_are_proportional_to_the_original(replace: bool) -> None:
    counts = DEEP_CELL
    means = _mean_counts(counts, DEEP_TARGET, replace, REPEATS)
    expected = DEEP_TARGET * counts / counts.sum()
    # Four standard errors: a correct draw fails this about once in 16000 runs,
    # and the seeds are fixed, so it does not flake. Anything looser would not
    # notice a systematically skewed draw.
    tolerance = 4.0 * _standard_error(counts, DEEP_TARGET, replace, REPEATS)
    np.testing.assert_array_less(np.abs(means - expected), tolerance)


def test_the_same_seed_gives_the_same_matrix() -> None:
    counts = _rows()
    np.testing.assert_array_equal(_thin(counts, 20, seed=5), _thin(counts, 20, seed=5))
    assert not np.array_equal(_thin(counts, 20, seed=5), _thin(counts, 20, seed=6))


def test_a_cell_draws_the_same_counts_whatever_it_is_processed_with() -> None:
    """Each cell's stream is keyed on its own index, not on the loop order."""
    counts = _rows()
    whole = _thin(counts, 20, seed=9)
    head = _thin(counts[:2], 20, seed=9)
    np.testing.assert_array_equal(whole[:2], head)


def test_subsample_returns_the_right_cells() -> None:
    kept = np.asarray(_scrust.subsample(100, 30, False, 0))
    assert kept.shape == (30,)
    assert len(set(kept.tolist())) == 30
    assert kept.max() < 100
    np.testing.assert_array_equal(kept, np.asarray(_scrust.subsample(100, 30, False, 0)))
    assert not np.array_equal(kept, np.asarray(_scrust.subsample(100, 30, False, 1)))

    with_replacement = np.asarray(_scrust.subsample(5, 50, True, 0))
    assert with_replacement.shape == (50,)
    assert len(set(with_replacement.tolist())) < 50

    with pytest.raises(ValueError, match="n_keep"):
        _scrust.subsample(10, 11, False, 0)


# --- against scanpy ----------------------------------------------------------


def test_downsample_agrees_with_scanpy_where_the_draw_is_determined() -> None:
    """The draw itself cannot be matched, so the determined quantities are.

    scanpy thins with `numpy.random.choice` seeded per cell from its own legacy
    generator; we thin with a seeded ChaCha stream. No seed maps one to the
    other, so an element-wise comparison is not available at any tolerance.
    What *is* determined regardless of the generator is asserted here: the new
    total of every cell, the absence of stored zeros, and the fact that no
    entry grows. The expectation is compared against scanpy's own in the test
    below.
    """
    scanpy = pytest.importorskip("scanpy")
    counts = _rows()
    reference = AnnData(sp.csr_matrix(counts.copy()))
    scanpy.pp.downsample_counts(reference, counts_per_cell=20, random_state=0)
    ours = _thin(counts, 20, seed=0)

    np.testing.assert_array_equal(ours.sum(axis=1), np.asarray(reference.X.sum(axis=1)).ravel())
    assert reference.X.nnz == np.count_nonzero(ours)
    assert np.all(ours <= counts)


def test_downsample_expectation_matches_scanpys_over_many_draws() -> None:
    """Both draws must land on the same per-gene expectation."""
    scanpy = pytest.importorskip("scanpy")
    counts = DEEP_CELL
    ours = _mean_counts(counts, DEEP_TARGET, False, REPEATS)

    theirs = np.zeros_like(counts)
    for seed in range(REPEATS):
        adata = AnnData(sp.csr_matrix(counts.copy()))
        scanpy.pp.downsample_counts(adata, counts_per_cell=DEEP_TARGET, random_state=seed)
        theirs = theirs + adata.X.toarray()
    theirs = theirs / REPEATS

    # Two independent estimators of the same mean, so the gap they may show is
    # the standard error of their difference: sqrt(2) times one of them.
    tolerance = 4.0 * np.sqrt(2.0) * _standard_error(counts, DEEP_TARGET, False, REPEATS)
    np.testing.assert_array_less(np.abs(ours - theirs), tolerance)
