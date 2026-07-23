"""t-test and logistic-regression differential expression, against scanpy.

scanpy is the reference, but for two of these methods it is not the *oracle*, and the
difference is the whole design of this file:

- The t-test p-value scanpy reports is `scipy.stats.ttest_ind_from_stats` fed `f32`
  moments, so on a real marker gene it sits ~1e-4 to ~4e-4 away from the value the same
  test gives on the exact `f64` data. We compute the moments in `f64` and land within
  ~1e-10 of exact. Holding our p-values to scanpy's to 1e-6 is therefore unreachable
  *because scanpy is the less accurate one*; the assertion is instead that we are no
  further from the exact `scipy` value than scanpy is, and both distances are recorded.

- The logreg score scanpy reports is a sklearn coefficient from an **unconverged** fit
  (sklearn's default `max_iter=100`, `tol=1e-4`): on PBMC 3k it stops ~3.7e-2 from the
  objective's unique minimiser, which reshuffles ~8% of the top hundred genes. The
  objective is strictly convex, so — as the contract says of any algorithm that
  minimises something — the minimiser is what is determined, and that is what we test
  against, requiring only that we reach it at least as well as scanpy's default does.

The gene-level comparisons all join on gene name, never on rank: scanpy's `argpartition`
top-n leaves tied scores in an undefined order, so rank *i* holds different genes in the
two results and a positional comparison invents a large error where there is none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse, stats

from reference_metrics import de_comparison, set_overlap
from scrust_call import scrust_call

TOP_N_GENES = 100

# Only the deterministic fields are held to scanpy directly. scanpy's p-values are the
# less accurate side of the comparison and are checked against exact `scipy` instead.
_STATISTIC_TOLERANCES = {"scores": 1e-3, "logfoldchanges": 1e-3}


def _run(adata: AnnData, method: str) -> dict[str, Any]:
    ours = adata.copy()
    scrust_call("tl.rank_genes_groups", ours, "group", method=method)
    return ours.uns["rank_genes_groups"]


def _scanpy(adata: AnnData, method: str) -> dict[str, Any]:
    expected = adata.copy()
    sc.tl.rank_genes_groups(expected, "group", method=method)
    return expected.uns["rank_genes_groups"]


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_t_test_matches_scanpy(
    lognorm: AnnData, method: str, record_property: Callable[[str, object], None]
) -> None:
    """Top-100 set and the deterministic per-gene fields, joined by gene name."""
    ours = _run(lognorm, method)
    expected = _scanpy(lognorm, method)
    top = min(TOP_N_GENES, lognorm.n_vars)
    dataset = lognorm.uns["dataset_id"]

    for group in expected["names"].dtype.names:
        problems, deviations = de_comparison(
            ours, expected, group, top=top, tolerances=_STATISTIC_TOLERANCES
        )
        for field, worst in deviations.items():
            record_property(f"de.{method}.{dataset}.{group}.{field}", f"{worst:.2e}")
        assert not problems, f"group {group}: {problems}"


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_t_test_p_values_are_no_worse_than_scanpy(
    lognorm: AnnData, method: str, record_property: Callable[[str, object], None]
) -> None:
    """Against exact `f64` scipy, our p-values must beat scanpy's `f32` ones.

    scanpy substitutes the tested group's own size for the reference on the overestim
    variant, so the exact reference is computed the same way.
    """
    ours = _run(lognorm, method)
    expected = _scanpy(lognorm, method)
    dense = np.asarray(lognorm.X.todense(), dtype=np.float64)
    top = min(TOP_N_GENES, lognorm.n_vars)
    dataset = lognorm.uns["dataset_id"]

    worst_ours = worst_scanpy = 0.0
    for group in expected["names"].dtype.names:
        mask = (lognorm.obs["group"] == group).to_numpy()
        genes = [str(name) for name in expected["names"][group][:top]]
        columns = [lognorm.var_names.get_loc(name) for name in genes]

        group_side = dense[mask][:, columns]
        rest_side = dense[~mask][:, columns]
        mean_g, var_g, n_g = group_side.mean(0), group_side.var(0, ddof=1), mask.sum()
        mean_r, var_r = rest_side.mean(0), rest_side.var(0, ddof=1)
        # t-test_overestim_var overestimates the variance of a small group by lending it
        # its own, smaller sample size on the reference side.
        n_r = n_g if method == "t-test_overestim_var" else (~mask).sum()
        _, exact = stats.ttest_ind_from_stats(
            mean_g, np.sqrt(var_g), n_g, mean_r, np.sqrt(var_r), n_r, equal_var=False
        )
        exact = np.nan_to_num(exact, nan=1.0)

        ours_by_gene = _join(ours, group, "pvals")
        scanpy_by_gene = _join(expected, group, "pvals")
        for gene, reference in zip(genes, exact, strict=True):
            scale = max(reference, np.finfo(float).tiny)
            worst_ours = max(worst_ours, abs(ours_by_gene[gene] - reference) / scale)
            worst_scanpy = max(worst_scanpy, abs(scanpy_by_gene[gene] - reference) / scale)

    record_property(f"de.{method}.{dataset}.pvals_vs_exact", f"{worst_ours:.2e}")
    record_property(f"de.{method}.{dataset}.scanpy_pvals_vs_exact", f"{worst_scanpy:.2e}")
    print(
        f"\n{method} on {dataset}: p-values vs exact scipy — ours {worst_ours:.2e}, "
        f"scanpy {worst_scanpy:.2e}"
    )
    assert worst_ours <= max(worst_scanpy, 1e-9), (
        f"our p-values ({worst_ours:.2e}) are further from exact than scanpy's ({worst_scanpy:.2e})"
    )


def test_logreg_reaches_the_optimum_scanpy_only_approaches(
    lognorm: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """logreg minimises a convex objective, so it is judged on the minimiser it reaches.

    scanpy's score is sklearn's *default* fit, which stops well short of that minimiser;
    the bar is that we match the converged optimum at least as well as scanpy does.
    """
    from sklearn.linear_model import LogisticRegression

    ours = _run(lognorm, "logreg")
    codes = lognorm.obs["group"].cat.codes.to_numpy()
    optimum = LogisticRegression(max_iter=10_000, tol=1e-10).fit(lognorm.X, codes).coef_
    scanpy = _scanpy(lognorm, "logreg")

    # scanpy reports no p-values and no fold changes for logreg, only names and scores.
    assert "pvals" not in ours and "logfoldchanges" not in ours
    assert "pvals" not in scanpy and "logfoldchanges" not in scanpy

    dataset = lognorm.uns["dataset_id"]
    gene_index = {name: i for i, name in enumerate(lognorm.var_names)}
    top = min(TOP_N_GENES, lognorm.n_vars)

    for position, group in enumerate(ours["names"].dtype.names):
        optimal_scores = optimum[position]
        optimal_top = set(np.argsort(optimal_scores)[::-1][:top])

        ours_top = {gene_index[str(n)] for n in ours["names"][group][:top]}
        scanpy_top = {gene_index[str(n)] for n in scanpy["names"][group][:top]}
        ours_overlap = set_overlap(ours_top, optimal_top)
        scanpy_overlap = set_overlap(scanpy_top, optimal_top)
        record_property(f"de.logreg.{dataset}.{group}.top{top}_vs_optimum", round(ours_overlap, 3))
        record_property(f"de.logreg.{dataset}.{group}.scanpy_vs_optimum", round(scanpy_overlap, 3))

        assert ours_overlap >= scanpy_overlap, (
            f"group {group}: our top {top} overlaps the optimum {ours_overlap:.3f}, "
            f"worse than scanpy's {scanpy_overlap:.3f}"
        )

        # And the coefficients themselves, joined by gene name over the genes both agree
        # are the strongest: an absolute floor because a coefficient near zero has no
        # relative accuracy.
        ours_by_gene = _join(ours, group, "scores")
        worst = max(
            abs(ours_by_gene[str(lognorm.var_names[i])] - optimal_scores[i])
            for i in ours_top & optimal_top
        )
        record_property(f"de.logreg.{dataset}.{group}.coef_dev", f"{worst:.2e}")
        assert worst <= 1e-2, f"group {group}: coefficient deviates {worst:.2e} from the optimum"


def test_rejects_an_unknown_method(lognorm: AnnData) -> None:
    """The wrapper names the supported methods rather than letting the core panic."""
    import scrust

    with pytest.raises(ValueError, match=r"wilcoxon.*t-test.*logreg|method must be one of"):
        scrust.tl.rank_genes_groups(lognorm.copy(), "group", method="deseq2")


# The constant-in-both gene makes scipy warn about cancellation; that gene is the point.
@pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation")
def test_t_test_matches_scipy_on_named_genes() -> None:
    """A hand-built matrix where every gene's Welch statistic is a scipy constant.

    Two groups of five cells over four genes chosen to exercise the branches: an ordinary
    gene, an implicit-zero gene, a gene constant in both groups (scipy NaN -> neutral),
    and a gene separated with zero within-group variance (statistic infinite).
    """
    group_a = np.array(
        [
            [3.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [4.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
        ]
    )
    group_b = np.array(
        [
            [0.0, 0.0, 2.0, 5.0],
            [1.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 2.0, 5.0],
            [1.0, 0.0, 2.0, 5.0],
        ]
    )
    dense = np.vstack([group_a, group_b]).astype(np.float32)
    adata = AnnData(sparse.csr_matrix(dense))
    adata.obs["group"] = ["a"] * 5 + ["b"] * 5
    adata.obs["group"] = adata.obs["group"].astype("category")

    ours = _run(adata, "t-test")
    scores = _join(ours, "a", "scores")
    p_values = _join(ours, "a", "pvals")

    for gene in range(dense.shape[1]):
        name = str(adata.var_names[gene])
        statistic, p_value = stats.ttest_ind(group_a[:, gene], group_b[:, gene], equal_var=False)
        if np.isnan(statistic):  # constant in both groups: scanpy's neutral result
            statistic, p_value = 0.0, 1.0
        if np.isinf(statistic):  # zero within-group variance, different means
            assert np.isinf(scores[name]) and np.sign(scores[name]) == np.sign(statistic)
            assert p_values[name] == pytest.approx(0.0, abs=1e-12)
            continue
        assert scores[name] == pytest.approx(statistic, rel=1e-4, abs=1e-6)
        assert p_values[name] == pytest.approx(p_value, rel=1e-9)


def test_two_variants_differ_only_in_the_substituted_reference_size() -> None:
    """The one knob between the variants is the reference sample size.

    It is not *only* the degrees of freedom: substituting the group's own size for the
    reference reaches the statistic through the standard error too. So the faithful check
    is that reproducing scipy with `nobs2 = ns_group` gives the overestim variant and
    `nobs2 = ns_rest` gives the plain one, from one and the same set of moments.
    """
    rng = np.random.default_rng(3)
    # A deliberately lopsided split, 8 against 32: the two variants coincide only when the
    # group is exactly half the cells, so an even split would test nothing.
    dense = rng.poisson(1.5, size=(40, 6)).astype(np.float32)
    adata = AnnData(sparse.csr_matrix(dense))
    adata.obs["group"] = ["a"] * 8 + ["b"] * 32
    adata.obs["group"] = adata.obs["group"].astype("category")

    plain = _run(adata, "t-test")
    overestim = _run(adata, "t-test_overestim_var")

    group = dense[:8]
    rest = dense[8:]
    stats_kwargs = dict(
        mean1=group.mean(0),
        std1=group.std(0, ddof=1),
        nobs1=8,
        mean2=rest.mean(0),
        std2=rest.std(0, ddof=1),
        equal_var=False,
    )
    plain_ref, _ = stats.ttest_ind_from_stats(nobs2=32, **stats_kwargs)
    overestim_ref, _ = stats.ttest_ind_from_stats(nobs2=8, **stats_kwargs)

    plain_scores = _join(plain, "a", "scores")
    overestim_scores = _join(overestim, "a", "scores")
    differ = False
    for gene in range(dense.shape[1]):
        name = str(adata.var_names[gene])
        assert plain_scores[name] == pytest.approx(plain_ref[gene], rel=1e-4, abs=1e-6)
        assert overestim_scores[name] == pytest.approx(overestim_ref[gene], rel=1e-4, abs=1e-6)
        differ = differ or abs(plain_scores[name] - overestim_scores[name]) > 1e-3
    assert differ, "a lopsided split must make the two variants disagree"


def test_a_single_cell_group_is_handled() -> None:
    """scanpy raises on a singlet group; the core leaves its variance uncorrected.

    A one-cell group has no unbiased variance, so scanpy refuses it outright. The core
    keeps it — the moment is well defined, the Welch denominator is not zero because the
    reference still has spread — and must not produce a NaN score or a NaN p-value.
    """
    dense = np.array(
        [
            [5.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(sparse.csr_matrix(dense))
    adata.obs["group"] = ["solo", "rest", "rest", "rest"]
    adata.obs["group"] = adata.obs["group"].astype("category")

    ours = _run(adata, "t-test")
    scores = np.asarray(ours["scores"]["solo"], dtype=np.float64)
    p_values = np.asarray(ours["pvals"]["solo"], dtype=np.float64)
    assert not np.isnan(scores).any()
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))


def test_a_gene_with_no_variance_in_either_group_is_neutral() -> None:
    """A gene constant in both groups is scipy's NaN, which scanpy maps to a neutral result."""
    dense = np.array(
        [
            [7.0, 3.0],
            [7.0, 1.0],
            [7.0, 8.0],
            [7.0, 9.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(sparse.csr_matrix(dense))
    adata.obs["group"] = ["a", "a", "b", "b"]
    adata.obs["group"] = adata.obs["group"].astype("category")

    for method in ("t-test", "t-test_overestim_var"):
        ours = _run(adata, method)
        for group in ("a", "b"):
            values = _join(ours, group, "scores")
            p_values = _join(ours, group, "pvals")
            constant_gene = str(adata.var_names[0])
            assert values[constant_gene] == 0.0
            assert p_values[constant_gene] == 1.0


def _join(result: Mapping[str, Any], group: str, field: str) -> dict[str, float]:
    """Field values keyed by gene name, the only join that survives a tie reorder."""
    names = [str(name) for name in result["names"][group]]
    values = np.asarray(result[field][group], dtype=np.float64)
    return dict(zip(names, values, strict=True))
