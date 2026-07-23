"""Adversarial cross-checks for the differential expression path and its preprocessing.

These target the cases the existing suites do not reach: the tied zero block that
dominates every single-cell column, negative values left behind by `scale`, the
`tie_correct=True` branch nothing else exercises, the exact placement of the fold
change pseudocount, and the boundaries of `normalize_total`, `scale` and
`highly_variable_genes`.

The tests call the compiled core directly rather than through `scrust.pp` /
`scrust.tl`, because the Python layer hard-codes `tie_correct=False` and drops the
explicitly stored zeros a CSR is allowed to carry — both of which are exactly what
is under test here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

from scrust_call import DEVICE

REPO_ROOT = Path(__file__).resolve().parents[1]

# `cargo build -p scrust-py --release` puts a fresh cdylib in the work tree; prefer it
# over whatever is installed in site-packages, which may be built from another branch.
_WORKTREE_EXTENSION = REPO_ROOT / "target" / "pyext" / "_scrust.so"


def _load_extension():
    if _WORKTREE_EXTENSION.exists():
        # The name has to be `_scrust`: CPython derives the init symbol it looks
        # for (`PyInit__scrust`) from it.
        spec = importlib.util.spec_from_file_location("_scrust", _WORKTREE_EXTENSION)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_scrust", module)
        spec.loader.exec_module(module)
        return module
    try:
        from scrust import _scrust
    except ImportError as exc:  # pragma: no cover - environment without a build
        pytest.skip(f"the compiled core is not available: {exc}")
    return _scrust


ext = _load_extension()
sc = pytest.importorskip("scanpy")
anndata = pytest.importorskip("anndata")


# --------------------------------------------------------------------------- helpers


def csr_args(dense: np.ndarray) -> tuple:
    """The four arguments the core takes, with the structural zeros dropped."""
    matrix = sp.csr_matrix(np.asarray(dense, np.float32))
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def fully_stored_csr_args(dense: np.ndarray) -> tuple:
    """The same, but storing *every* entry, zeros included.

    scipy drops zeros on construction and scanpy calls `eliminate_zeros()` on the
    way in, so a stored zero is a shape the reference never sees. The core has to
    put it in the tied zero block anyway, which this makes testable.
    """
    dense = np.asarray(dense, np.float32)
    n_cells, n_genes = dense.shape
    indptr = np.arange(0, n_cells * n_genes + 1, n_genes, dtype=np.uint32)
    indices = np.tile(np.arange(n_genes, dtype=np.uint32), n_cells)
    return indptr, indices, dense.ravel().copy(), n_genes


def wilcoxon(args: tuple, labels, n_groups, reference=None, tie_correct=False):
    return ext.rank_genes_groups_wilcoxon(
        *args,
        np.asarray(labels, np.uint32),
        n_groups,
        reference,
        tie_correct,
        DEVICE,
    )


def scanpy_wilcoxon_rest(dense, masks, *, tie_correct):
    """scanpy's `_RankGenes.wilcoxon` with `ireference is None`, transcribed."""
    x = np.asarray(dense, np.float64)
    n_cells = x.shape[0]
    ranks = np.apply_along_axis(stats.rankdata, 0, x)
    coefficients = (
        np.array([stats.tiecorrect(ranks[:, j]) for j in range(x.shape[1])]) if tie_correct else 1
    )
    scores = []
    for mask in masks:
        n_active = int(mask.sum())
        rank_sum = ranks[mask, :].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std_dev = np.sqrt(coefficients * n_active * (n_cells - n_active) * (n_cells + 1) / 12.0)
            row = (rank_sum - n_active * (n_cells + 1) / 2.0) / std_dev
        row[np.isnan(row)] = 0
        scores.append(row)
    return np.array(scores)


def scanpy_wilcoxon_reference(dense, mask_obs, mask_rest, *, tie_correct):
    """scanpy's `_RankGenes.wilcoxon` with a reference group, transcribed."""
    x = np.asarray(dense, np.float64)
    sub = np.vstack([x[mask_obs], x[mask_rest]])
    ranks = np.apply_along_axis(stats.rankdata, 0, sub)
    n_active, m_active = int(mask_obs.sum()), int(mask_rest.sum())
    coefficients = (
        np.array([stats.tiecorrect(ranks[:, j]) for j in range(x.shape[1])]) if tie_correct else 1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        std_dev = np.sqrt(coefficients * n_active * m_active * (n_active + m_active + 1) / 12.0)
        row = (
            ranks[:n_active, :].sum(axis=0) - n_active * (n_active + m_active + 1) / 2.0
        ) / std_dev
    row[np.isnan(row)] = 0
    return row


@pytest.fixture
def adversarial_expression() -> tuple[np.ndarray, np.ndarray]:
    """A column layout built to break the closed-form zero block.

    Mostly zeros, negative values on both sides of it (what `scale` leaves
    behind), a gene that is zero in every cell, a gene that is a single negative
    constant, a gene stored as a zero in one whole group, and a `-0.0`.
    """
    rng = np.random.default_rng(20240722)
    n_cells, n_genes = 41, 12
    dense = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    dense[dense > 0.2] = 0.0  # a realistic pile of zeros, and a tied block
    dense[:, 2] = 0.0  # zero in every cell
    dense[:, 5] = -1.5  # one negative constant
    dense[:, 7] = np.abs(dense[:, 7])  # no negatives at all
    dense[:14, 8] = 0.0  # a whole group stored as zero
    dense[3, 9] = -0.0  # negative zero belongs in the zero block
    labels = np.array([0] * 14 + [1] * 15 + [2] * 12, np.uint32)
    return dense, labels


# ------------------------------------------------------------------- 1. Wilcoxon ties


@pytest.mark.parametrize("tie_correct", [False, True])
@pytest.mark.parametrize("stored", ["structural", "explicit"])
def test_wilcoxon_zero_block_matches_scipy_ranks(adversarial_expression, tie_correct, stored):
    """The closed-form average rank of the zero block has to equal scipy's.

    The core never materialises the zeros: it ranks only the stored entries and
    gives every unstored cell the constant `n_negative + (n_zero + 1) / 2`. That
    is only right if the zeros really do form one tied block sitting between the
    negatives and the positives, which is what this checks against a full
    `scipy.stats.rankdata` on the dense column.
    """
    dense, labels = adversarial_expression
    args = csr_args(dense) if stored == "structural" else fully_stored_csr_args(dense)

    got = np.asarray(wilcoxon(args, labels, 3, None, tie_correct)["scores"])
    masks = [labels == group for group in range(3)]
    expected = scanpy_wilcoxon_rest(dense, masks, tie_correct=tie_correct)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("tie_correct", [False, True])
def test_wilcoxon_against_a_reference_group_matches_scipy(adversarial_expression, tie_correct):
    """The reference path ranks only the two groups' cells, and nothing else."""
    dense, labels = adversarial_expression
    args = fully_stored_csr_args(dense)
    got = np.asarray(wilcoxon(args, labels, 3, 1, tie_correct)["scores"])
    masks = [labels == group for group in range(3)]

    for group in (0, 2):
        expected = scanpy_wilcoxon_reference(dense, masks[group], masks[1], tie_correct=tie_correct)
        np.testing.assert_allclose(got[group], expected, rtol=0, atol=1e-5)

    # scanpy omits the reference group; the core reports the neutral result.
    assert np.all(got[1] == 0.0)
    result = wilcoxon(args, labels, 3, 1, tie_correct)
    assert np.all(np.asarray(result["p_values"])[1] == 1.0)
    assert np.all(np.asarray(result["log2_fold_changes"])[1] == 0.0)


@pytest.mark.parametrize("tie_correct", [False, True])
def test_wilcoxon_gene_that_is_zero_everywhere_scores_zero(tie_correct):
    """One tied block spanning every cell.

    With `tie_correct=True` the coefficient is exactly 0, so the standard
    deviation is 0 next to a numerator that is exactly 0. scanpy maps the NaN to
    0 and so must the core — an inf here would rank a dead gene at the top.
    """
    dense = np.zeros((30, 3), np.float32)
    dense[:, 1] = 1.0  # constant, so also one tied block
    dense[:15, 2] = 1.0  # a real signal, to prove the test can move
    labels = np.array([0] * 15 + [1] * 15, np.uint32)

    result = wilcoxon(fully_stored_csr_args(dense), labels, 2, None, tie_correct)
    scores = np.asarray(result["scores"])

    assert scores[0, 0] == 0.0 and scores[1, 0] == 0.0
    assert scores[0, 1] == 0.0 and scores[1, 1] == 0.0
    assert np.isfinite(scores).all()
    assert abs(scores[0, 2]) > 1.0


def test_wilcoxon_tie_correction_changes_the_score_when_it_should():
    """`tie_correct=True` is the less travelled branch; make sure it does something.

    A column with a large tied block has a tie coefficient well below 1, which
    shrinks the standard deviation and inflates the score. If the flag were
    ignored the two calls would agree.
    """
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 3, (60, 4)).astype(np.float32)  # values 0, 1, 2 only
    labels = np.array([0] * 30 + [1] * 30, np.uint32)
    args = csr_args(dense)

    plain = np.asarray(wilcoxon(args, labels, 2, None, False)["scores"])
    corrected = np.asarray(wilcoxon(args, labels, 2, None, True)["scores"])

    interesting = np.abs(plain[0]) > 1e-6
    assert interesting.any()
    assert np.all(np.abs(corrected[0][interesting]) > np.abs(plain[0][interesting]))

    ranks = np.apply_along_axis(stats.rankdata, 0, dense.astype(np.float64))
    coefficients = np.array([stats.tiecorrect(ranks[:, j]) for j in range(4)])
    np.testing.assert_allclose(corrected[0], plain[0] / np.sqrt(coefficients), rtol=1e-5)


# ------------------------------------------------------------ 2. log2 fold change


def test_log2_fold_change_matches_scanpys_expm1_definition():
    """scanpy: `log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))`.

    Note on "where": adding the pseudocount to the mean instead of to `expm1` of
    the mean is not separable numerically — `expm1(m + 1e-9) - expm1(m)` is
    `1e-9 * exp(m)`, which for any mean a log-normalised matrix produces is far
    below the `f32` the fold change is reported in. What *is* observable is the
    magnitude of the pseudocount and the fact that it is added to both sides;
    both are pinned below.
    """
    rng = np.random.default_rng(3)
    dense = np.abs(rng.normal(0.4, 0.3, (40, 6))).astype(np.float32)
    dense[dense < 0.25] = 0.0
    dense[:, 4] = 0.0  # unexpressed both sides: the pseudocount's whole purpose
    dense[:20, 5] = 0.0  # unexpressed in one group only
    labels = np.array([0] * 20 + [1] * 20, np.uint32)

    got = np.asarray(wilcoxon(csr_args(dense), labels, 2, None, False)["log2_fold_changes"])

    masks = [labels == group for group in range(2)]
    mean_group = np.array([dense[m].mean(axis=0, dtype=np.float64) for m in masks])
    mean_rest = np.array([dense[~m].mean(axis=0, dtype=np.float64) for m in masks])

    scanpy_definition = np.log2((np.expm1(mean_group) + 1e-9) / (np.expm1(mean_rest) + 1e-9))
    np.testing.assert_allclose(got, scanpy_definition.astype(np.float32), rtol=2e-6)

    # A gene unexpressed on both sides must come back at exactly 0, not NaN: this
    # is what the pseudocount in the *denominator* buys.
    assert got[0, 4] == 0.0 and got[1, 4] == 0.0

    # Rejects "pseudocount only in the numerator", which would give +/-inf here.
    assert np.isfinite(got).all()


def test_log2_fold_change_pseudocount_is_exactly_1e_minus_9():
    """Pin the magnitude, on a gene whose `expm1(mean)` is the pseudocount itself.

    One cell of twenty carries `1e-8`, so the group mean is `1e-9` and
    `expm1(1e-9) == 1e-9`. The reported fold change is then
    `log2((1e-9 + p) / p)`, which is exactly 1.0 when `p == 1e-9` and nowhere
    near it for any other plausible pseudocount.
    """
    dense = np.zeros((20, 1), np.float32)
    dense[0, 0] = np.float32(1e-8)
    labels = np.array([0] * 10 + [1] * 10, np.uint32)

    got = np.asarray(wilcoxon(csr_args(dense), labels, 2, None, False)["log2_fold_changes"])

    assert got[0, 0] == pytest.approx(1.0, abs=1e-5)
    for wrong in (1.0, 1e-6, 1e-12):
        assert got[0, 0] != pytest.approx(np.log2((1e-9 + wrong) / wrong), abs=1e-3)
    # And the other way round for the group without the count.
    assert got[1, 0] == pytest.approx(-1.0, abs=1e-5)


# ----------------------------------------------- 3. Welch's t-test / overestim_var


def test_t_test_methods_are_not_implemented():
    """`de::parametric` is still `todo!()`, and the Python layer refuses the method.

    Recorded rather than skipped: the audit's finding is that `t-test` and
    `t-test_overestim_var` do not exist, so nothing about their degrees of
    freedom can be right or wrong yet.
    """
    scrust_tl = pytest.importorskip("scrust.tl")
    from anndata import AnnData

    adata = AnnData(
        sp.csr_matrix(np.eye(6, 4, dtype=np.float32)),
        obs={"group": ["a", "a", "a", "b", "b", "b"]},
    )
    adata.obs["group"] = adata.obs["group"].astype("category")
    for method in ("t-test", "t-test_overestim_var", "logreg"):
        with pytest.raises(ValueError, match="method must be one of"):
            scrust_tl.rank_genes_groups(adata, "group", method=method)


def test_overestim_var_changes_the_statistic_not_only_the_degrees_of_freedom():
    """A property the eventual implementation must satisfy.

    scanpy substitutes `ns_rest = ns_group` into `ttest_ind_from_stats`, which
    feeds `vn2 = var_rest / ns_rest`. That term is in the Welch denominator as
    well as in the Welch-Satterthwaite d.o.f., so `t-test_overestim_var` reports
    a *different t*, not merely a different p-value. An implementation that
    changes only the d.o.f. would be wrong.
    """
    plain = stats.ttest_ind_from_stats(2.0, 1.0, 10, 1.0, 3.0, 100, equal_var=False)
    overestimated = stats.ttest_ind_from_stats(2.0, 1.0, 10, 1.0, 3.0, 10, equal_var=False)
    assert plain.statistic != overestimated.statistic
    assert plain.pvalue != overestimated.pvalue


# ------------------------------------------------------- 4. Benjamini-Hochberg / f64


def test_de_p_values_below_f32_min_survive_as_f64():
    """A rank-sum p-value routinely falls under `f32`'s smallest normal value.

    A perfectly separating gene over a few hundred cells already lands below
    `1.2e-38`. If the p-values were carried in `f32` they would all be exactly
    zero, the ordering would be lost, and the Benjamini-Hochberg step-up would
    have nothing to work with.
    """
    n_per_group = 160
    n_cells = 2 * n_per_group
    dense = np.zeros((n_cells, 3), np.float32)
    dense[:n_per_group, 0] = np.linspace(1.0, 2.0, n_per_group)  # separating
    dense[n_per_group:, 0] = np.linspace(3.0, 4.0, n_per_group)
    dense[:, 1] = np.linspace(1.0, 2.0, n_cells)  # not separating
    labels = np.array([0] * n_per_group + [1] * n_per_group, np.uint32)

    result = wilcoxon(csr_args(dense), labels, 2, None, False)
    p_values = np.asarray(result["p_values"])
    adjusted = np.asarray(result["adjusted_p_values"])

    assert p_values.dtype == np.float64 and adjusted.dtype == np.float64
    assert 0.0 < p_values[0, 0] < np.finfo(np.float32).tiny
    assert np.float32(p_values[0, 0]) == 0.0  # what f32 would have done
    assert 0.0 < adjusted[0, 0] < np.finfo(np.float32).tiny

    # The score is reported in f32; recomputing the tail from it costs a
    # relative `z**2 * eps32` on the p-value, hence the loose tolerance. The
    # point of the check is that the p-value is the normal tail of the score at
    # all, not that it round-trips through f32.
    score = abs(np.float64(np.asarray(result["scores"])[0, 0]))
    np.testing.assert_allclose(p_values[0, 0], 2 * stats.norm.sf(score), rtol=score**2 * 1.2e-7)


def test_benjamini_hochberg_is_applied_per_group_over_all_genes():
    """scanpy corrects each group's p-values on their own, across every gene."""
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(19)
    dense = rng.poisson(0.8, (90, 40)).astype(np.float32)
    dense[:30, :6] *= 5  # something to actually find
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30, np.uint32)

    result = wilcoxon(csr_args(dense), labels, 3, None, False)
    p_values = np.asarray(result["p_values"])
    adjusted = np.asarray(result["adjusted_p_values"])

    for group in range(3):
        _, expected, _, _ = multipletests(p_values[group], method="fdr_bh")
        np.testing.assert_allclose(adjusted[group], expected, rtol=1e-12, atol=0)

    from scipy.stats import false_discovery_control

    for group in range(3):
        expected = false_discovery_control(p_values[group], method="bh")
        np.testing.assert_allclose(adjusted[group], expected, rtol=1e-12, atol=0)


# ------------------------------------------------------------------ 5. normalize_total


def test_normalize_total_uses_the_csr_median_including_empty_cells():
    """scanpy's CSR path takes the median over *all* cells; its dense path does not.

    The core has to match the CSR path, which is what a real AnnData with sparse
    `X` goes through. With enough zero-count cells the two medians differ, and
    every value in the matrix differs with them.
    """
    from anndata import AnnData

    rng = np.random.default_rng(31)
    dense = (rng.random((12, 5)) < 0.5) * rng.integers(1, 9, (12, 5))
    dense = dense.astype(np.float32)
    dense[8:] = 0.0  # four cells with no counts at all

    totals = dense.sum(axis=1)
    csr_median = np.median(totals)
    dense_median = np.median(totals[totals > 0])
    assert csr_median != dense_median  # otherwise the test proves nothing

    sparse_reference = AnnData(sp.csr_matrix(dense.copy()))
    sc.pp.normalize_total(sparse_reference)
    dense_reference = AnnData(dense.copy())
    sc.pp.normalize_total(dense_reference)

    indptr, indices, values, _n_cols = ext.normalize_total(*csr_args(dense), None, DEVICE)
    got = sp.csr_matrix((values, indices, indptr), shape=dense.shape).toarray()

    np.testing.assert_allclose(got, sparse_reference.X.toarray(), rtol=1e-6, atol=1e-6)
    assert not np.allclose(got, np.asarray(dense_reference.X), rtol=1e-3)

    # Zero-count cells are left alone rather than divided by zero.
    assert np.all(got[8:] == 0.0)
    assert np.isfinite(got).all()


def test_normalize_total_refuses_a_zero_median_instead_of_erasing_the_matrix():
    """A documented, deliberate divergence: worth pinning down, because it is loud.

    When more than half the cells are empty the median count is 0. scanpy divides
    by it, the size factors become inf, and every stored value silently becomes
    zero — the whole matrix, wiped, with no warning. The core raises instead.
    """
    from anndata import AnnData

    dense = np.zeros((7, 4), np.float32)
    dense[0] = [1, 2, 0, 0]
    dense[1] = [0, 0, 3, 4]
    dense[2] = [5, 0, 0, 0]

    reference = AnnData(sp.csr_matrix(dense.copy()))
    sc.pp.normalize_total(reference)
    assert np.all(reference.X.toarray() == 0.0)  # scanpy's behaviour, for the record

    with pytest.raises(ValueError, match="target_sum"):
        ext.normalize_total(*csr_args(dense), None, DEVICE)


# -------------------------------------------------------------------------- 6. scale


def test_scale_uses_the_corrected_variance():
    """ddof = 1, as scanpy's `mean_var(..., correction=1)`. ddof = 0 is visibly off."""
    dense = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
    got = np.asarray(ext.scale(*csr_args(dense), True, None, DEVICE)).ravel()

    column = dense.ravel().astype(np.float64)
    corrected = (column - column.mean()) / column.std(ddof=1)
    uncorrected = (column - column.mean()) / column.std(ddof=0)

    np.testing.assert_allclose(got, corrected, rtol=1e-6)
    assert not np.allclose(got, uncorrected, rtol=1e-3)


@pytest.mark.parametrize("n_cells", [2, 5, 30, 301])
@pytest.mark.parametrize("value", [0.001, 0.1, 0.30103, 0.75, 1.3, 3.5, 1234.5])
def test_scale_leaves_a_constant_gene_at_exactly_zero(n_cells, value):
    """A gene with no variance must standardise to zero, not to a whole s.d.

    scanpy substitutes 1 for a zero standard deviation. That substitution only
    fires if the variance of a constant gene really is zero, which needs the mean
    reduced in `f64` and rounded back to `f32` before the deviations are taken.
    Reduce in `f32` and the mean lands one ulp out: every centred value is the
    same tiny `-d`, the deviation is `|d| * sqrt(n / (n - 1))`, and the gene
    comes back as a constant `-sqrt((n - 1) / n)` — for 301 cells, `-0.9983`.
    Without zero-centering the same slip returns `value / |d|`, of order `1e7`.
    """
    dense = np.empty((n_cells, 2), np.float32)
    dense[:, 0] = np.float32(value)
    dense[:, 1] = np.linspace(1.0, 2.0, n_cells)  # a gene that does vary

    centred = np.asarray(ext.scale(*csr_args(dense), True, None, DEVICE))
    plain = np.asarray(ext.scale(*csr_args(dense), False, None, DEVICE))

    assert np.all(centred[:, 0] == 0.0)
    assert np.all(plain[:, 0] == np.float32(value))
    assert np.abs(centred[:, 1]).max() > 0.5  # the varying gene still moves

    # scanpy returns 0 here, or NaN when its `E[x^2] - mean^2` variance lands a
    # hair below zero and the square root gives up. Zero is the documented
    # behaviour ("variables that do not display any variation ... are set to 0");
    # the core has to deliver it, and must never return a finite non-zero.
    from anndata import AnnData

    reference = AnnData(np.array(dense))
    sc.pp.scale(reference)
    scanpy_column = np.asarray(reference.X)[:, 0]
    assert np.all(np.isnan(scanpy_column) | (scanpy_column == 0.0))


def test_scale_all_zero_gene_and_clip_order():
    """The clip happens after the division, and is one-sided without centering."""
    from anndata import AnnData

    rng = np.random.default_rng(41)
    dense = rng.random((25, 5)).astype(np.float32)
    dense[dense < 0.35] = 0.0
    dense[:, 3] = 0.0  # never expressed

    for zero_center in (True, False):
        for max_value in (None, 1.0, 0.5):
            reference = AnnData(sp.csr_matrix(dense.copy()))
            sc.pp.scale(reference, zero_center=zero_center, max_value=max_value)
            expected = reference.X
            expected = expected.toarray() if sp.issparse(expected) else np.asarray(expected)
            got = np.asarray(ext.scale(*csr_args(dense), zero_center, max_value, DEVICE))
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"zero_center={zero_center} max_value={max_value}",
            )
            assert np.all(got[:, 3] == 0.0)

    # Clipping after the division, not before: a value under the limit before
    # scaling can exceed it after, and must come back at the limit.
    unclipped = np.asarray(ext.scale(*csr_args(dense), True, None, DEVICE))
    clipped = np.asarray(ext.scale(*csr_args(dense), True, 0.5, DEVICE))
    assert unclipped.max() > 0.5
    assert clipped.max() == pytest.approx(0.5)
    assert clipped.min() == pytest.approx(-0.5)

    # Without centering the values stay non-negative and only the top is bound.
    plain = np.asarray(ext.scale(*csr_args(dense), False, 0.5, DEVICE))
    assert plain.min() == 0.0 and plain.max() == pytest.approx(0.5)


# ---------------------------------------------------------- 7. highly_variable_genes


def logged(counts: np.ndarray) -> np.ndarray:
    from anndata import AnnData

    adata = AnnData(sp.csr_matrix(np.asarray(counts, np.float32)))
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return np.asarray(adata.X.toarray(), np.float32)


def hvg_reference(dense: np.ndarray, n_top_genes: int, flavor: str):
    from anndata import AnnData

    adata = AnnData(sp.csr_matrix(np.asarray(dense, np.float32)))
    return sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, inplace=False)


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_hvg_matches_scanpy_end_to_end(flavor):
    """The whole pipeline: expm1, dispersion, binning, normalisation, selection."""
    rng = np.random.default_rng(11)
    counts = rng.poisson(1.2, (300, 60)).astype(np.float32)
    dense = logged(counts) if flavor == "seurat" else counts

    reference = hvg_reference(dense, 15, flavor)
    got = ext.highly_variable_genes(*csr_args(dense), 15, flavor, DEVICE)

    np.testing.assert_allclose(np.asarray(got["means"]), reference["means"].to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(got["dispersions"]), reference["dispersions"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(got["normalised_dispersions"]),
        reference["dispersions_norm"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(got["highly_variable"]), reference["highly_variable"].to_numpy()
    )


def test_hvg_seurat_undoes_the_log_before_taking_the_dispersion():
    """The `seurat` flavour computes mean and variance on `expm1(x)`, not on `x`.

    Skipping the `expm1` gives a different mean and a different dispersion for
    every gene, so the check is that the core agrees with the `expm1` definition
    and disagrees with the one that leaves the data logged.
    """
    rng = np.random.default_rng(13)
    dense = logged(rng.poisson(1.5, (200, 30)).astype(np.float32))
    got = ext.highly_variable_genes(*csr_args(dense), 8, "seurat", DEVICE)

    unlogged = np.expm1(dense.astype(np.float32)).astype(np.float64)
    mean = unlogged.mean(axis=0)
    variance = unlogged.var(axis=0, ddof=1)
    mean = np.where(mean == 0, 1e-12, mean)
    dispersion = variance / mean
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.log(np.where(dispersion == 0, np.nan, dispersion))

    np.testing.assert_allclose(np.asarray(got["means"]), np.log1p(mean), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(got["dispersions"]), dispersion, rtol=1e-5)

    still_logged = dense.astype(np.float64)
    assert not np.allclose(np.asarray(got["means"]), np.log1p(still_logged.mean(axis=0)), rtol=1e-3)


def test_hvg_binning_follows_pandas_cut_right_closed_intervals():
    """`pandas.cut(means, bins=20)`: right-closed, with the low edge nudged out.

    A mean landing exactly on an interior edge belongs to the bin *below* it, and
    the lowest mean has to land inside the first bin rather than on its open
    boundary — which is what the 0.1%-of-range widening of `bins[0]` is for.
    """
    import pandas as pd

    def rust_bins(means, n_bins):
        minimum, maximum = means.min(), means.max()
        if minimum == maximum:
            widen = lambda v: 0.001 * abs(v) if v != 0 else 0.001  # noqa: E731
            low, high = minimum - widen(minimum), maximum + widen(maximum)
            edges = np.linspace(low, high, n_bins + 1)
            edges[-1] = high
        else:
            edges = np.linspace(minimum, maximum, n_bins + 1)
            edges[-1] = maximum
            edges[0] -= (maximum - minimum) * 0.001
        assigned = np.searchsorted(edges, means, side="left") - 1
        assigned[(assigned < 0) | (assigned >= n_bins)] = -1
        return assigned

    means = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    for n_bins in (2, 3, 4, 8, 20):
        pandas_codes = pd.cut(pd.Series(means), bins=n_bins).cat.codes.to_numpy()
        np.testing.assert_array_equal(rust_bins(means, n_bins), pandas_codes)
        # Every value, including the extremes, must land in a bin.
        assert (pandas_codes >= 0).all()

    for constant in (0.0, 2.5, -3.0):
        repeated = np.repeat(constant, 5)
        pandas_codes = pd.cut(pd.Series(repeated), bins=4).cat.codes.to_numpy()
        np.testing.assert_array_equal(rust_bins(repeated, 4), pandas_codes)


def test_hvg_a_bin_holding_one_gene_normalises_to_exactly_one():
    """scanpy's `_postprocess_dispersions_seurat`: centre 0, scale by the bin mean.

    A single-gene bin has no ddof=1 standard deviation. scanpy replaces the
    missing deviation with the bin's mean and sets the centre to 0, so the lone
    gene comes out at exactly 1.0 — not NaN, and not dropped.
    """
    rng = np.random.default_rng(17)
    dense = logged(rng.poisson(1.5, (300, 50)).astype(np.float32))
    dense[:, 0] = 0.0
    dense[:5, 0] = 8.0  # a mean far above everything else: its own top bin

    reference = hvg_reference(dense, 10, "seurat")
    got = ext.highly_variable_genes(*csr_args(dense), 10, "seurat", DEVICE)

    normalised = np.asarray(got["normalised_dispersions"])
    reference_norm = reference["dispersions_norm"].to_numpy()

    lone = np.flatnonzero(reference_norm == 1.0)
    assert lone.size >= 1, "the fixture no longer produces a single-gene bin"
    np.testing.assert_allclose(normalised[lone], 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(normalised, reference_norm, rtol=1e-5, atol=1e-6)

    # The rule is `seurat`-only — `_postprocess_dispersions_seurat` is not called
    # for `cell_ranger`, whose statistics are a median and a MAD. Run the same
    # counts through it and the core must still follow scanpy, without borrowing
    # the seurat fix-up.
    counts = np.expm1(dense).astype(np.float32)
    cell_ranger = ext.highly_variable_genes(*csr_args(counts), 10, "cell_ranger", DEVICE)
    cell_ranger_reference = hvg_reference(counts, 10, "cell_ranger")
    np.testing.assert_allclose(
        np.asarray(cell_ranger["normalised_dispersions"]),
        cell_ranger_reference["dispersions_norm"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_hvg_returns_more_than_n_top_genes_when_the_cutoff_is_tied():
    """scanpy takes the n-th highest as a cut-off and keeps everything `>=` it.

    Duplicate a gene so its normalised dispersion is a hard tie, put the tie on
    the boundary, and both scanpy and the core must return more genes than were
    asked for. A `argsort`-and-take-n implementation would return exactly n.
    """
    rng = np.random.default_rng(23)
    dense = logged(rng.poisson(1.5, (200, 40)).astype(np.float32))
    duplicated = np.hstack([dense, dense[:, [0]], dense[:, [0]], dense[:, [0]]])

    over_requested = []
    for n_top_genes in range(1, 12):
        reference = hvg_reference(duplicated, n_top_genes, "seurat")
        got = ext.highly_variable_genes(*csr_args(duplicated), n_top_genes, "seurat", DEVICE)
        reference_flag = reference["highly_variable"].to_numpy()
        got_flag = np.asarray(got["highly_variable"])
        np.testing.assert_array_equal(got_flag, reference_flag)
        if got_flag.sum() > n_top_genes:
            over_requested.append((n_top_genes, int(got_flag.sum())))

    assert over_requested, "the fixture no longer straddles the cut-off"

    # `n_top_genes` above the gene count returns everything with a dispersion.
    got = ext.highly_variable_genes(*csr_args(dense), 10_000, "seurat", DEVICE)
    reference = hvg_reference(dense, 10_000, "seurat")
    np.testing.assert_array_equal(
        np.asarray(got["highly_variable"]), reference["highly_variable"].to_numpy()
    )


def test_hvg_cutoff_rule_holds_even_where_ulps_move_the_selected_set():
    """The `>= n-th highest` rule, on the range where the tie is not exact.

    A bin holding exactly two genes normalises them to `+/-1/sqrt(2)`, and
    pandas' `groupby.std` (a Welford recurrence) and the core's two-pass variance
    disagree on that value in the last bit. When the cut-off lands on such a
    block the two implementations select slightly different gene sets — not
    because either is wrong, but because the true values are a hard tie and no
    `>=` against a `f64` can break it consistently.

    So this pins the rule rather than the set: the core must return *at least*
    `n_top_genes`, must return more than that when the cut-off is tied, and must
    never flag a gene whose normalised dispersion is below the cut-off.
    """
    rng = np.random.default_rng(23)
    dense = logged(rng.poisson(1.5, (200, 40)).astype(np.float32))
    duplicated = np.hstack([dense, dense[:, [0]], dense[:, [0]], dense[:, [0]]])

    for n_top_genes in range(12, 25):
        got = ext.highly_variable_genes(*csr_args(duplicated), n_top_genes, "seurat", DEVICE)
        flag = np.asarray(got["highly_variable"])
        normalised = np.asarray(got["normalised_dispersions"], np.float64)

        assert flag.sum() >= n_top_genes
        selected = normalised[flag]
        rejected = normalised[~flag]
        assert not np.isnan(selected).any()
        assert selected.min() >= np.nan_to_num(rejected, nan=-np.inf).max()


def test_hvg_selection_agrees_with_scanpy_at_a_realistic_scale():
    """The ulp sensitivity above must not show up on data that looks like data.

    2500 genes over 20 mean bins leaves no two-gene bins and no exact ties, so
    the selected set has to match scanpy's exactly, at every `n_top_genes` a user
    would pick.
    """
    rng = np.random.default_rng(99)
    dense = logged(rng.poisson(0.35, (800, 2500)).astype(np.float32))

    for n_top_genes in (200, 500, 1000):
        reference = hvg_reference(dense, n_top_genes, "seurat")
        got = ext.highly_variable_genes(*csr_args(dense), n_top_genes, "seurat", DEVICE)
        np.testing.assert_array_equal(
            np.asarray(got["highly_variable"]),
            reference["highly_variable"].to_numpy(),
            err_msg=f"n_top_genes={n_top_genes}",
        )


def test_hvg_a_constant_gene_gives_a_nan_dispersion():
    """A documented divergence, pinned so it cannot drift unnoticed.

    A gene with no variance has `variance / mean == 0`, and scanpy's rule is to
    carry that as NaN rather than `log(0) == -inf`. The core reaches exactly zero
    and so reports NaN. scanpy's own `mean_var` computes `E[x^2] - mean^2` in a
    thread-partitioned reduction, which for the same gene lands on ~1e-17 instead
    of 0; its `dispersion == 0` test then misses, and `log(1e-17) ~ -39` is fed
    into the bin statistics, dragging the normalised dispersion of *every other
    gene in that bin* with it.

    This is not reproducible even between scanpy runs (the reduction depends on
    the numba thread count), so it is recorded rather than matched. NaN is the
    behaviour scanpy documents; the core is the one that delivers it.
    """
    rng = np.random.default_rng(29)
    dense = logged(rng.poisson(1.5, (300, 40)).astype(np.float32))
    dense[:, 5] = np.float32(1.0)  # exactly constant

    got = ext.highly_variable_genes(*csr_args(dense), 8, "seurat", DEVICE)
    dispersions = np.asarray(got["dispersions"])
    assert np.isnan(dispersions[5])
    assert not np.isnan(np.delete(dispersions, 5)).all()
    # A gene with no dispersion is never selected.
    assert not np.asarray(got["highly_variable"])[5]
