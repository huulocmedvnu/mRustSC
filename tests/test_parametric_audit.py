"""Cross-check the three non-Wilcoxon DE methods against scanpy.

`t-test`, `t-test_overestim_var` and `logreg` arrived on a branch whose first commit
says "incomplete, agent interrupted by session limit". The work was finished before the
branch was merged, and the tests that came with it are the author's own -- which this
session has repeatedly shown is not the same as having been checked. So these drive
scanpy on the same matrix and compare.

Two things deserve their own tests rather than a tolerance:

* `test_the_two_t_tests_differ_only_in_the_reference_sample_size`, because that single
  substitution is the whole difference between the two methods and it reaches the
  answer twice over -- through the standard error and through the degrees of freedom;
* `test_logreg_reports_no_p_values_because_scanpy_reports_none`, because inventing a
  p-value for a method that has none would look like extra information rather than a
  fabrication.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse, stats

from scrust_call import DEVICE, scrust_call

T_TESTS = ("t-test", "t-test_overestim_var")


def csr_args(matrix: sparse.csr_matrix):
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


_BINDING = {
    "t-test": "_scrust.rank_genes_groups_t_test",
    "t-test_overestim_var": "_scrust.rank_genes_groups_t_test_overestim_var",
}


def scrust_t_test(method, matrix, labels, n_groups, reference=None):
    return scrust_call(
        _BINDING[method],
        *csr_args(matrix),
        np.asarray(labels, np.uint32),
        n_groups,
        reference,
        DEVICE,
    )


def scrust_logreg(matrix, labels, n_groups, max_iterations=100):
    return scrust_call(
        "_scrust.rank_genes_groups_logreg",
        *csr_args(matrix),
        np.asarray(labels, np.uint32),
        n_groups,
        max_iterations,
        DEVICE,
    )


def blobs(n_cells=150, n_genes=30, n_groups=3, seed=0, sparsity=0.4):
    """Log-scale data with a real group effect, which is what these are run on."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups).astype(np.uint32)
    dense = rng.lognormal(0.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
    for group in range(n_groups):
        dense[labels == group, group * 4 : group * 4 + 4] *= np.float32(3.0)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(np.log1p(dense).astype(np.float32)), labels


def scanpy_table(matrix, labels, n_groups, method, reference=None):
    """scanpy's result put back into (group, gene) order, as it sorts per group."""
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    n_genes = dense.shape[1]
    names = [f"g{i}" for i in range(n_genes)]
    adata = AnnData(sparse.csr_matrix(dense), obs={"group": [str(x) for x in labels]})
    adata.var_names = names
    adata.obs["group"] = adata.obs["group"].astype("category")

    sc.tl.rank_genes_groups(
        adata,
        "group",
        method=method,
        use_raw=False,
        n_genes=n_genes,
        reference="rest" if reference is None else str(reference),
    )
    result = adata.uns["rank_genes_groups"]
    available = [k for k in ("scores", "pvals", "pvals_adj", "logfoldchanges") if k in result]

    out = {k: np.zeros((n_groups, n_genes)) for k in available}
    if reference is not None and "pvals" in out:
        out["pvals"][reference, :] = 1.0
        out["pvals_adj"][reference, :] = 1.0

    order = {name: i for i, name in enumerate(names)}
    for group in result["names"].dtype.names:
        row = int(group)
        columns = [order[name] for name in result["names"][group]]
        for key in available:
            out[key][row, columns] = result[key][group]
    return out


# --------------------------------------------------------------------------------
# 1. The two t-tests.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("method", T_TESTS)
def test_t_test_matches_scanpy_against_rest(method):
    matrix, labels = blobs()
    ours = scrust_t_test(method, matrix, labels, 3)
    theirs = scanpy_table(matrix, labels, 3, method)

    np.testing.assert_allclose(np.asarray(ours["scores"]), theirs["scores"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ours["p_values"]), theirs["pvals"], rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(ours["adjusted_p_values"]), theirs["pvals_adj"], rtol=1e-4, atol=1e-9
    )
    folds = np.asarray(ours["log2_fold_changes"], dtype=np.float64)
    finite = np.isfinite(folds) & np.isfinite(theirs["logfoldchanges"])
    np.testing.assert_allclose(
        folds[finite], theirs["logfoldchanges"][finite], rtol=1e-3, atol=1e-4
    )


@pytest.mark.parametrize("method", T_TESTS)
@pytest.mark.parametrize("reference", [0, 2])
def test_t_test_matches_scanpy_against_a_reference_group(method, reference):
    matrix, labels = blobs(seed=3)
    ours = scrust_t_test(method, matrix, labels, 3, reference)
    theirs = scanpy_table(matrix, labels, 3, method, reference)

    np.testing.assert_allclose(np.asarray(ours["scores"]), theirs["scores"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ours["p_values"]), theirs["pvals"], rtol=1e-4, atol=1e-9)
    assert np.all(np.asarray(ours["scores"])[reference] == 0.0)
    assert np.all(np.asarray(ours["p_values"])[reference] == 1.0)


def test_the_two_t_tests_differ_only_in_the_reference_sample_size():
    """The whole difference between the methods, isolated.

    `t-test_overestim_var` substitutes the tested group's own size for the reference
    side's. That reaches the answer twice: through `var_reference / n_reference` in the
    standard error, and through the Welch degrees of freedom. A small group therefore
    both inflates the reference variance term and loses degrees of freedom, which is
    the conservatism the variant is named for.

    So the overestimating variant must give a p-value no smaller than the plain one
    wherever the tested group is smaller than the rest -- and the two must not simply
    be identical, which is what a copy-paste of one into the other would produce.
    """
    matrix, labels = blobs(n_cells=180, seed=5)
    sizes = np.bincount(labels)
    assert sizes.min() * 2 < len(labels), "each group has to be smaller than its rest"

    plain = scrust_t_test("t-test", matrix, labels, 3)
    inflated = scrust_t_test("t-test_overestim_var", matrix, labels, 3)

    plain_p = np.asarray(plain["p_values"])
    inflated_p = np.asarray(inflated["p_values"])
    assert not np.allclose(plain_p, inflated_p), "the two methods produced the same thing"
    assert np.all(inflated_p >= plain_p - 1e-9), (
        "the overestimating variant must never be more confident than the plain test"
    )

    # And each still matches its own scanpy counterpart, so neither drifted to the other.
    np.testing.assert_allclose(
        plain_p, scanpy_table(matrix, labels, 3, "t-test")["pvals"], rtol=1e-4, atol=1e-9
    )
    np.testing.assert_allclose(
        inflated_p,
        scanpy_table(matrix, labels, 3, "t-test_overestim_var")["pvals"],
        rtol=1e-4,
        atol=1e-9,
    )


def test_a_gene_constant_everywhere_has_no_t_statistic():
    """Both variances are zero, so the standard error is zero and `t` is 0/0. scanpy
    maps the NaN statistic to 0 and the NaN p-value to 1; a p-value of 0 here would
    declare a gene that never varies to be the strongest marker in the data."""
    matrix, labels = blobs(seed=7)
    dense = matrix.toarray()
    dense[:, 0] = 0.0
    dense[:, 1] = 1.75
    matrix = sparse.csr_matrix(dense.astype(np.float32))

    for method in T_TESTS:
        ours = scrust_t_test(method, matrix, labels, 3)
        scores = np.asarray(ours["scores"])
        p_values = np.asarray(ours["p_values"])
        assert np.all(scores[:, :2] == 0.0), method
        assert np.all(p_values[:, :2] == 1.0), method
        theirs = scanpy_table(matrix, labels, 3, method)
        np.testing.assert_allclose(scores, theirs["scores"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(p_values, theirs["pvals"], rtol=1e-4, atol=1e-9)


def test_t_test_p_values_match_scipys_t_distribution():
    """Independent of scanpy: the p-value has to be `2 * t.sf(|score|, df)` for the
    Welch degrees of freedom the same moments imply. Recomputed here from the group
    moments rather than read back from the implementation."""
    matrix, labels = blobs(n_cells=120, n_genes=20, seed=11)
    dense = np.asarray(matrix.todense(), dtype=np.float64)
    ours = scrust_t_test("t-test", matrix, labels, 3)
    scores = np.asarray(ours["scores"], dtype=np.float64)
    p_values = np.asarray(ours["p_values"], dtype=np.float64)

    for group in range(3):
        inside = labels == group
        a, b = dense[inside], dense[~inside]
        va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
        na, nb = a.shape[0], b.shape[0]
        se_a, se_b = va / na, vb / nb
        with np.errstate(invalid="ignore", divide="ignore"):
            df = (se_a + se_b) ** 2 / (se_a**2 / (na - 1) + se_b**2 / (nb - 1))
            expected = 2.0 * stats.t.sf(np.abs(scores[group]), np.where(np.isnan(df), 1.0, df))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(p_values[group][finite], expected[finite], rtol=1e-4, atol=1e-9)


# --------------------------------------------------------------------------------
# 2. logreg
# --------------------------------------------------------------------------------


def test_logreg_scores_match_scanpy():
    """scanpy fits `sklearn.linear_model.LogisticRegression` and reports its
    coefficients as the scores. Compared by ranking rather than by value: the fit is
    an iterative optimum, so the coefficients agree to the solver's tolerance, not to
    f32, and what a caller reads off is the order."""
    matrix, labels = blobs(n_cells=180, n_genes=25, seed=13, sparsity=0.2)
    ours = np.asarray(scrust_logreg(matrix, labels, 3, 1000)["scores"], dtype=np.float64)
    theirs = scanpy_table(matrix, labels, 3, "logreg")["scores"]

    assert ours.shape == theirs.shape
    for group in range(3):
        top_ours = set(np.argsort(-ours[group])[:5])
        top_theirs = set(np.argsort(-theirs[group])[:5])
        overlap = len(top_ours & top_theirs)
        assert overlap >= 4, (
            f"group {group}: only {overlap} of the top 5 genes agree with scanpy\n"
            f"  ours   {sorted(top_ours)}\n  scanpy {sorted(top_theirs)}"
        )
        correlation = np.corrcoef(ours[group], theirs[group])[0, 1]
        assert correlation > 0.95, f"group {group}: coefficient correlation {correlation:.3f}"


def test_logreg_reports_no_p_values_because_scanpy_reports_none():
    """scanpy's `uns` entry for `logreg` carries `names` and `scores` and nothing else
    -- no p-values, no fold changes, because a coefficient is not a test.

    The core returns NaN in those fields rather than a number. That matters: a 0, or a
    Benjamini-Hochberg run over fabricated p-values, would look like extra information
    instead of an invention.
    """
    matrix, labels = blobs(n_cells=120, n_genes=15, seed=17)
    ours = scrust_logreg(matrix, labels, 3, 500)

    assert np.isnan(np.asarray(ours["p_values"])).all()
    assert np.isnan(np.asarray(ours["adjusted_p_values"])).all()
    assert np.isnan(np.asarray(ours["log2_fold_changes"])).all()
    assert np.isfinite(np.asarray(ours["scores"])).all()

    dense = np.asarray(matrix.todense(), dtype=np.float32)
    adata = AnnData(sparse.csr_matrix(dense), obs={"group": [str(x) for x in labels]})
    adata.obs["group"] = adata.obs["group"].astype("category")
    sc.tl.rank_genes_groups(adata, "group", method="logreg", use_raw=False)
    assert "pvals" not in adata.uns["rank_genes_groups"], (
        "scanpy now reports p-values for logreg; the core should follow"
    )


def test_logreg_separates_a_planted_marker():
    """Independent of scanpy: a gene expressed only in one group has to come out as
    that group's largest coefficient."""
    rng = np.random.default_rng(19)
    n_cells, n_genes = 150, 12
    labels = np.repeat(np.arange(3), n_cells // 3).astype(np.uint32)
    dense = rng.lognormal(0.0, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    for group in range(3):
        dense[labels != group, group] = 0.0
        dense[labels == group, group] += np.float32(5.0)
    matrix = sparse.csr_matrix(np.log1p(dense).astype(np.float32))

    scores = np.asarray(scrust_logreg(matrix, labels, 3, 1000)["scores"], dtype=np.float64)
    for group in range(3):
        assert int(np.argmax(scores[group])) == group, (
            f"group {group}'s planted marker is not its top coefficient"
        )
