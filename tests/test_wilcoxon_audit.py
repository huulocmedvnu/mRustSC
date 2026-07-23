"""Cross-check the Wilcoxon rank-sum path against scanpy's `rank_genes_groups`.

The reference is driven, not transcribed, wherever scanpy can be driven: the tests
below call `sc.tl.rank_genes_groups(method="wilcoxon")` on the same matrix and compare
scores, p-values, adjusted p-values and log fold changes gene by gene. scanpy returns
its results sorted per group and as a structured recarray, so `_scanpy_table` puts them
back into (group, gene) order first.

Where scanpy cannot be driven -- the far tail of the normal survival function, where
every score it produces has already underflowed -- the reference is scipy.

What each test pins is in its own docstring. The three that matter most:

* `test_explicit_zeros_rank_with_the_structural_ones`, because the core never
  materialises the zero block and instead ranks it in closed form;
* `test_negative_values_rank_around_the_zero_block`, because that closed form assumes
  the zero block sits between the negatives and the positives;
* `test_far_tail_p_values_do_not_underflow`, because scanpy's own p-values reach zero
  there and the core's do not.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse, stats

from scrust_call import scrust_call

TIE_CORRECT = (False, True)


def csr_args(matrix: sparse.csr_matrix):
    """The four positional arguments the binding takes a CSR as."""
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def scrust_wilcoxon(matrix, labels, n_groups, *, reference=None, tie_correct=False):
    return scrust_call(
        "_scrust.rank_genes_groups_wilcoxon",
        *csr_args(matrix),
        np.asarray(labels, dtype=np.uint32),
        n_groups,
        reference,
        tie_correct,
        "cpu",
    )


def _scanpy_table(matrix, labels, n_groups, *, reference=None, tie_correct=False):
    """scanpy's answer, unsorted, as (n_groups, n_genes) arrays.

    scanpy sorts each group's genes by score and reports names, so the ordering has to
    be undone before anything can be compared. The reference group, which scanpy omits
    entirely, comes back as the neutral row the core produces for it.
    """
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    n_genes = dense.shape[1]
    names = [f"g{i}" for i in range(n_genes)]
    adata = AnnData(
        sparse.csr_matrix(dense),
        obs={"group": [str(label) for label in labels]},
        var={"name": names},
    )
    adata.var_names = names
    adata.obs["group"] = adata.obs["group"].astype("category")

    sc.tl.rank_genes_groups(
        adata,
        "group",
        method="wilcoxon",
        reference="rest" if reference is None else str(reference),
        tie_correct=tie_correct,
        use_raw=False,
        n_genes=n_genes,
    )
    result = adata.uns["rank_genes_groups"]

    out = {
        key: np.zeros((n_groups, n_genes))
        for key in ("scores", "pvals", "pvals_adj", "logfoldchanges")
    }
    # The reference group is not tested at all, so it keeps the neutral row: a p-value
    # of 1, and a Benjamini-Hochberg adjustment of a vector of ones, which is ones.
    if reference is not None:
        out["pvals"][reference, :] = 1.0
        out["pvals_adj"][reference, :] = 1.0

    order = {name: index for index, name in enumerate(names)}
    for group in result["names"].dtype.names:
        row = int(group)
        columns = [order[name] for name in result["names"][group]]
        for key in out:
            out[key][row, columns] = result[key][group]
    return out


def blobs(n_cells=90, n_genes=25, n_groups=3, seed=0, sparsity=0.5, scale=1.0):
    """A count-like matrix with a real group effect and a realistic number of zeros."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups).astype(np.uint32)
    dense = rng.gamma(2.0, scale, size=(n_cells, n_genes)).astype(np.float32)
    for group in range(n_groups):
        rows = labels == group
        genes = slice(group * 5, group * 5 + 5)
        dense[rows, genes] *= np.float32(3.0)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(dense), labels


def assert_matches(ours, theirs, *, rtol=1e-5, atol=1e-6, keys=None):
    keys = keys or {
        "scores": "scores",
        "p_values": "pvals",
        "adjusted_p_values": "pvals_adj",
        "log2_fold_changes": "logfoldchanges",
    }
    for ours_key, theirs_key in keys.items():
        got = np.asarray(ours[ours_key], dtype=np.float64)
        want = np.asarray(theirs[theirs_key], dtype=np.float64)
        finite = np.isfinite(want)
        np.testing.assert_allclose(
            got[finite], want[finite], rtol=rtol, atol=atol, err_msg=ours_key
        )


# --------------------------------------------------------------------------------
# 1. The comparison scanpy makes by default.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_matches_scanpy_against_rest(tie_correct):
    """Each group against every other cell, which is scanpy's default `reference="rest"`."""
    matrix, labels = blobs()
    ours = scrust_wilcoxon(matrix, labels, 3, tie_correct=tie_correct)
    theirs = _scanpy_table(matrix, labels, 3, tie_correct=tie_correct)
    assert_matches(ours, theirs)


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
@pytest.mark.parametrize("reference", [0, 2])
def test_matches_scanpy_against_a_reference_group(reference, tie_correct):
    """One named group as the reference, where the ranking covers two groups rather
    than the whole matrix, and scanpy drops the reference group from its output."""
    matrix, labels = blobs()
    ours = scrust_wilcoxon(matrix, labels, 3, reference=reference, tie_correct=tie_correct)
    theirs = _scanpy_table(matrix, labels, 3, reference=reference, tie_correct=tie_correct)
    assert_matches(ours, theirs)

    # The row scanpy does not produce at all.
    assert np.all(np.asarray(ours["scores"])[reference] == 0.0)
    assert np.all(np.asarray(ours["p_values"])[reference] == 1.0)
    assert np.all(np.asarray(ours["log2_fold_changes"])[reference] == 0.0)


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_matches_scanpy_on_a_dense_matrix_with_no_zeros_at_all(tie_correct):
    """The zero block is empty here, so the closed form contributes nothing and the
    ranking is the ordinary one. It separates a bug in the closed form from a bug in
    the ranking itself."""
    matrix, labels = blobs(sparsity=0.0)
    ours = scrust_wilcoxon(matrix, labels, 3, tie_correct=tie_correct)
    theirs = _scanpy_table(matrix, labels, 3, tie_correct=tie_correct)
    assert_matches(ours, theirs)


# --------------------------------------------------------------------------------
# 2. The zero block, which the core ranks without materialising.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_explicit_zeros_rank_with_the_structural_ones(tie_correct):
    """A stored zero and a structural zero must rank identically.

    scanpy densifies each chunk, so it cannot tell them apart. The core walks the
    stored entries and counts the rest, so it can -- it drops stored zeros back into
    the counted block on purpose (`rank_gene`). If that ever stops happening, the zero
    block splits in two and every rank past it shifts.

    The two matrices here are equal as dense arrays and differ only in what is stored.
    """
    matrix, labels = blobs(seed=4)
    dense = np.asarray(matrix.todense(), dtype=np.float32)

    # Same values, but every entry stored, half of them explicit zeros.
    explicit = sparse.csr_matrix(np.ones_like(dense))
    explicit.data = dense.ravel().copy()
    assert explicit.nnz == dense.size
    assert (explicit.count_nonzero()) < explicit.nnz, "there have to be stored zeros"

    compact = scrust_wilcoxon(matrix, labels, 3, tie_correct=tie_correct)
    stored = scrust_wilcoxon(explicit, labels, 3, tie_correct=tie_correct)
    for key in ("scores", "p_values", "adjusted_p_values", "log2_fold_changes"):
        np.testing.assert_allclose(
            np.asarray(stored[key], dtype=np.float64),
            np.asarray(compact[key], dtype=np.float64),
            rtol=1e-6,
            atol=1e-9,
            err_msg=key,
        )

    assert_matches(stored, _scanpy_table(matrix, labels, 3, tie_correct=tie_correct))


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_negative_values_rank_around_the_zero_block(tie_correct):
    """`rank_gene` places the whole zero block between the negative and the positive
    stored values, in one step, and gives it the average rank
    `n_negative + (n_zero + 1) / 2`. That is only right if the split point is exactly
    zero, so a matrix that has been centred -- which is what `pp.scale` produces, and
    a perfectly ordinary thing to rank -- is the case that tests it.
    """
    matrix, labels = blobs(seed=6)
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    dense[dense > 0] -= np.float32(1.5)  # sends about half the stored values negative
    shifted = sparse.csr_matrix(dense)
    assert (shifted.data < 0).any() and (shifted.data > 0).any()
    assert (np.asarray(shifted.todense()) == 0).any(), "the zero block has to survive"

    ours = scrust_wilcoxon(shifted, labels, 3, tie_correct=tie_correct)
    theirs = _scanpy_table(shifted, labels, 3, tie_correct=tie_correct)
    assert_matches(ours, theirs)


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_an_all_zero_gene_is_one_tied_block(tie_correct):
    """Every cell tied. The tie coefficient goes to zero, the variance with it, and
    scanpy maps the resulting NaN score to 0 -- so the p-value is 1, not NaN."""
    matrix, labels = blobs(seed=8)
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    dense[:, 0] = 0.0
    dense[:, 1] = 2.5  # constant but non-zero: tied without being the zero block
    matrix = sparse.csr_matrix(dense)

    ours = scrust_wilcoxon(matrix, labels, 3, tie_correct=tie_correct)
    theirs = _scanpy_table(matrix, labels, 3, tie_correct=tie_correct)
    assert_matches(ours, theirs)

    if tie_correct:
        assert np.all(np.asarray(ours["scores"])[:, :2] == 0.0)
        assert np.all(np.asarray(ours["p_values"])[:, :2] == 1.0)


# --------------------------------------------------------------------------------
# 3. Group shapes.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("tie_correct", TIE_CORRECT)
def test_a_single_cell_group_scores_where_scanpy_refuses(tie_correct):
    """DOCUMENTED DIVERGENCE, in the direction that needs stating: the core answers a
    question scanpy declines to answer.

    scanpy raises `ValueError: Could not calculate statistics for groups ... since they
    only contain one sample` for any selected group with fewer than two cells
    (`_rank_genes_groups.py:145-155`). The core computes the comparison instead, and
    `wilcoxon::tests::group_of_one_cell_still_scores` pins that on purpose.

    Neither is wrong about the *rank test*: with `n_active = 1` the variance
    `n(N - n)(N + 1) / 12` is positive and the statistic is well defined. scanpy's guard
    protects its own `_basic_stats`, which takes a Bessel-corrected variance per group
    and so divides by `n - 1`; the Wilcoxon path never needs that number.

    The consequence for a caller is real, though, and is why this is pinned rather than
    left implicit: a pipeline that leans on scanpy raising here -- a cluster that came
    out as one cell, say -- gets a plausible-looking p-value from the core instead of an
    error. Whether to adopt scanpy's guard is a contract decision, not an audit one.
    """
    matrix, labels = blobs(n_cells=60, n_groups=2, seed=11)
    labels = labels.copy()
    labels[0] = 2  # one cell, alone in its own group
    assert int((labels == 2).sum()) == 1

    with pytest.raises(ValueError, match="only contain one sample"):
        _scanpy_table(matrix, labels, 3, tie_correct=tie_correct)

    ours = scrust_wilcoxon(matrix, labels, 3, tie_correct=tie_correct)
    scores = np.asarray(ours["scores"], dtype=np.float64)
    p_values = np.asarray(ours["p_values"], dtype=np.float64)
    assert np.isfinite(scores).all() and np.isfinite(p_values).all()
    assert ((p_values >= 0.0) & (p_values <= 1.0)).all()
    # Not a degenerate row: the one cell does separate some gene from the other 59.
    assert np.abs(scores[2]).max() > 1.0

    # The two groups that are not degenerate still agree with a scanpy run that has
    # the singleton removed, so the divergence is confined to the group scanpy rejects.
    keep = labels != 2
    kept_matrix = matrix[keep]
    kept_labels = labels[keep]
    kept_ours = scrust_wilcoxon(kept_matrix, kept_labels, 2, tie_correct=tie_correct)
    kept_theirs = _scanpy_table(kept_matrix, kept_labels, 2, tie_correct=tie_correct)
    assert_matches(kept_ours, kept_theirs)


def test_scores_are_antisymmetric_for_two_groups():
    """With two groups and no reference, each is the other's rest, so the two rows of
    scores are negatives of one another and the p-values are equal. Nothing in the
    implementation enforces this; it follows from the rank sums being complementary,
    which makes it a good check on the closed-form zero block."""
    matrix, labels = blobs(n_groups=2, n_cells=80, seed=13)
    ours = scrust_wilcoxon(matrix, labels, 2, tie_correct=True)
    scores = np.asarray(ours["scores"], dtype=np.float64)
    p_values = np.asarray(ours["p_values"], dtype=np.float64)
    np.testing.assert_allclose(scores[0], -scores[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p_values[0], p_values[1], rtol=1e-9, atol=0.0)


# --------------------------------------------------------------------------------
# 4. The tails, where scanpy's own numbers stop being usable.
# --------------------------------------------------------------------------------


def test_far_tail_p_values_do_not_underflow():
    """The p-value has to survive a score that a `1 - cdf` formulation could not reach.

    The core writes the tail as `erfc(|z| / sqrt 2)` rather than `1 - cdf`, so it keeps
    full relative precision where the complement would have rounded to 1 and the
    p-value to 0. At the separating gene below the score is past 24 and the p-value is
    of order 1e-132, which is the thing being checked.

    The tolerance is not tight, and cannot be: see
    `test_the_reported_score_is_f32_so_the_p_value_cannot_be_recomputed_from_it`.
    """
    # The core's tail, reached through a real comparison: two groups that separate
    # perfectly give the largest score the data allows.
    n_per_group = 400
    labels = np.repeat([0, 1], n_per_group).astype(np.uint32)
    dense = np.zeros((2 * n_per_group, 2), dtype=np.float32)
    dense[:n_per_group, 0] = np.linspace(1.0, 2.0, n_per_group)
    dense[n_per_group:, 0] = np.linspace(11.0, 12.0, n_per_group)
    dense[:, 1] = np.linspace(1.0, 2.0, 2 * n_per_group)
    matrix = sparse.csr_matrix(dense)

    ours = scrust_wilcoxon(matrix, labels, 2, tie_correct=True)
    scores = np.asarray(ours["scores"], dtype=np.float64)
    p_values = np.asarray(ours["p_values"], dtype=np.float64)

    separating = abs(scores[0, 0])
    assert separating > 20.0, f"the separating gene should saturate the score, got {separating}"
    assert p_values[0, 0] > 0.0, "the tail underflowed to zero"
    assert p_values[0, 0] < 1e-100, "the tail is not deep enough to test anything"
    np.testing.assert_allclose(p_values[0, 0], 2.0 * stats.norm.sf(separating), rtol=1e-4)


def test_the_reported_score_is_f32_so_the_p_value_cannot_be_recomputed_from_it():
    """A property worth knowing before trusting a recomputed p-value.

    `scores` is reported as f32 and the p-value is computed from the f64 score before
    that rounding, so `2 * norm.sf(|reported score|)` is not the reported p-value. The
    gap is the f32 rounding of the score carried through the exponential tail: about
    1e-7 relative near z = 3, growing to 1e-5 by z = 25.

    scanpy does exactly the same -- its `scores` field is declared float32
    (`_rank_genes_groups.py:742`) while `pvals` comes from the float64 array -- so this
    is agreement with the reference, not a divergence from it. It is pinned because the
    tolerance of every other tail test here depends on it.
    """
    matrix, labels = blobs(n_cells=120, n_genes=40, seed=21)
    ours = scrust_wilcoxon(matrix, labels, 3, tie_correct=True)
    scores = np.asarray(ours["scores"], dtype=np.float64)
    p_values = np.asarray(ours["p_values"], dtype=np.float64)
    recomputed = 2.0 * stats.norm.sf(np.abs(scores))

    # Close, because the two agree as functions...
    np.testing.assert_allclose(p_values, recomputed, rtol=1e-4, atol=0.0)
    # ...but not to f64, because the score went through f32 on the way out.
    worst = np.max(np.abs(p_values - recomputed) / np.maximum(recomputed, 1e-300))
    assert worst > 1e-9, (
        f"the score now round-trips to f64 ({worst:.3g}); the reported score may have "
        "become f64, and these tolerances can be tightened"
    )
    assert np.float32(scores).astype(np.float64).tolist() == scores.tolist(), (
        "scores are no longer f32"
    )


# --------------------------------------------------------------------------------
# 5. The fold change, which is not a rank statistic at all.
# --------------------------------------------------------------------------------


def test_log_fold_change_matches_scanpys_expm1_ratio():
    """scanpy's `log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))`, on
    means taken over log1p data. The 1e-9 is what stops an unexpressed gene dividing
    by zero, and it is inside both halves of the ratio, so a gene that is silent in
    both gives exactly 0 rather than a NaN.
    """
    matrix, labels = blobs(seed=23)
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    dense = np.log1p(dense)
    dense[:, 3] = 0.0  # silent everywhere
    matrix = sparse.csr_matrix(dense)

    ours = scrust_wilcoxon(matrix, labels, 3)
    theirs = _scanpy_table(matrix, labels, 3)
    folds = np.asarray(ours["log2_fold_changes"], dtype=np.float64)
    np.testing.assert_allclose(folds, theirs["logfoldchanges"], rtol=1e-4, atol=1e-5)
    assert np.all(folds[:, 3] == 0.0), "a gene silent in both halves is a ratio of 1"


def test_log_fold_change_ignores_the_log_base_scanpy_records():
    """DOCUMENTED DIVERGENCE, pinned rather than fixed.

    scanpy's `expm1_func` is `np.expm1(x * log(base))` when `adata.uns["log1p"]["base"]`
    is set, and plain `np.expm1` otherwise (`_rank_genes_groups.py:136-140`). The core
    has no way to know the base -- the binding takes a matrix, not an AnnData -- so it
    always uses the natural one.

    `sc.pp.log1p` leaves `base` at None unless asked, so the default path agrees. This
    test states the size of the gap when it does not: it is a rescaling of the means
    before the ratio, so it does not cancel.
    """
    matrix, labels = blobs(seed=25)
    dense = np.log1p(np.asarray(matrix.todense(), dtype=np.float32)) / np.float32(np.log(2.0))
    matrix = sparse.csr_matrix(dense)

    names = [f"g{i}" for i in range(dense.shape[1])]
    adata = AnnData(
        matrix,
        obs={"group": [str(label) for label in labels]},
        var={"name": names},
    )
    adata.var_names = names
    adata.obs["group"] = adata.obs["group"].astype("category")
    adata.uns["log1p"] = {"base": 2.0}
    sc.tl.rank_genes_groups(
        adata, "group", method="wilcoxon", use_raw=False, n_genes=dense.shape[1]
    )

    order = {name: index for index, name in enumerate(names)}
    theirs = np.zeros(dense.shape[1])
    columns = [order[name] for name in adata.uns["rank_genes_groups"]["names"]["0"]]
    theirs[columns] = adata.uns["rank_genes_groups"]["logfoldchanges"]["0"]

    ours = np.asarray(scrust_wilcoxon(matrix, labels, 3)["log2_fold_changes"])[0]

    # The ranks, and so the scores, are untouched by the base: only the fold changes move.
    assert not np.allclose(ours, theirs, rtol=1e-2, atol=1e-2), (
        "the base no longer changes the fold change; this divergence can be closed"
    )
