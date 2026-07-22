"""One test per algorithm, asserting exactly the criterion in `docs/API_CONTRACT.md`.

Each test gives scrust and scanpy the same input, prepared by scanpy, so the only
difference between the two runs is the single step under test. Every test runs twice:
on a small synthetic matrix, and on real PBMC 3k under the `reference` marker.

The scrust call always comes first, so an unimplemented step skips before the reference
is computed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from conftest import (
    CEILING_FRACTION,
    K_CAND,
    K_REF,
    N_COMPS,
    N_NEIGHBORS,
    STRICT_PRESERVATION,
    TARGET_SUM,
    check_pca_agreement,
    n_top_genes,
)
from reference_metrics import (
    as_dense,
    de_comparison,
    neighbor_sets,
    neighborhood_preservation,
    per_row_overlap,
    preservation_band,
    set_overlap,
)
from scrust_call import scrust_call

ELEMENTWISE = {"rtol": 1e-5, "atol": 1e-6}
TOP_N_GENES = 100


def _min_genes(adata: AnnData) -> int:
    """A threshold that drops roughly the worst tenth of cells on any dataset."""
    per_cell = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    return int(np.percentile(per_cell, 10))


def test_normalize_total(counts: AnnData) -> None:
    ours = counts.copy()
    scrust_call("pp.normalize_total", ours, target_sum=TARGET_SUM)

    expected = counts.copy()
    sc.pp.normalize_total(expected, target_sum=TARGET_SUM)
    assert_allclose(as_dense(ours.X), as_dense(expected.X), **ELEMENTWISE)


def test_log1p(counts: AnnData) -> None:
    base = counts.copy()
    sc.pp.normalize_total(base, target_sum=TARGET_SUM)

    ours = base.copy()
    scrust_call("pp.log1p", ours)

    expected = base.copy()
    sc.pp.log1p(expected)
    assert_allclose(as_dense(ours.X), as_dense(expected.X), **ELEMENTWISE)


def test_filter_cells(counts: AnnData) -> None:
    min_genes = _min_genes(counts)
    ours = counts.copy()
    scrust_call("pp.filter_cells", ours, min_genes=min_genes)

    expected = counts.copy()
    sc.pp.filter_cells(expected, min_genes=min_genes)
    assert expected.n_obs < counts.n_obs, "the threshold must actually filter something"
    assert_array_equal(ours.obs_names, expected.obs_names)
    assert_allclose(as_dense(ours.X), as_dense(expected.X), **ELEMENTWISE)


def test_filter_genes(counts: AnnData) -> None:
    ours = counts.copy()
    scrust_call("pp.filter_genes", ours, min_cells=3)

    expected = counts.copy()
    sc.pp.filter_genes(expected, min_cells=3)
    assert expected.n_vars < counts.n_vars, "the threshold must actually filter something"
    assert_array_equal(ours.var_names, expected.var_names)
    assert_allclose(as_dense(ours.X), as_dense(expected.X), **ELEMENTWISE)


def test_scale(lognorm: AnnData) -> None:
    ours = lognorm.copy()
    scrust_call("pp.scale", ours, zero_center=True, max_value=10)

    expected = lognorm.copy()
    sc.pp.scale(expected, zero_center=True, max_value=10)
    assert_allclose(as_dense(ours.X), as_dense(expected.X), **ELEMENTWISE)


def test_highly_variable_genes(lognorm: AnnData) -> None:
    n_top = n_top_genes(lognorm)
    ours = lognorm.copy()
    scrust_call("pp.highly_variable_genes", ours, n_top_genes=n_top, flavor="seurat")

    expected = lognorm.copy()
    sc.pp.highly_variable_genes(expected, n_top_genes=n_top, flavor="seurat")

    ours_selected = ours.var_names[ours.var["highly_variable"].to_numpy()]
    expected_selected = expected.var_names[expected.var["highly_variable"].to_numpy()]
    overlap = set_overlap(ours_selected, expected_selected)
    assert overlap >= 0.95, f"HVG set overlaps scanpy by {overlap:.3f}, contract wants >= 0.95"


def test_pca(scaled: AnnData, record_property: Callable[[str, object], None]) -> None:
    ours = scaled.copy()
    scrust_call("pp.pca", ours, n_comps=N_COMPS, zero_center=True, random_state=0)

    expected = scaled.copy()
    sc.pp.pca(expected, n_comps=N_COMPS, zero_center=True, random_state=0)

    check_pca_agreement(
        scaled, ours, expected, label=scaled.uns["dataset_id"], record=record_property
    )


def test_neighbors(embedded: AnnData) -> None:
    ours = embedded.copy()
    scrust_call("pp.neighbors", ours, n_neighbors=N_NEIGHBORS, use_rep="X_pca")

    expected = embedded.copy()
    sc.pp.neighbors(expected, n_neighbors=N_NEIGHBORS, use_rep="X_pca", random_state=0)

    overlaps = per_row_overlap(
        neighbor_sets(ours.obsp["distances"]),
        neighbor_sets(expected.obsp["distances"]),
    )
    # The contract's "each cell" read as the mean over cells; the per-cell distribution
    # is reported so a localised disagreement is still visible in the failure.
    assert overlaps.mean() >= 0.90, (
        f"mean neighbour overlap {overlaps.mean():.3f} < 0.90 "
        f"(worst cell {overlaps.min():.3f}, {np.mean(overlaps < 0.9):.1%} of cells below)"
    )


def _check_band(
    record_property: Callable[[str, object], None],
    algorithm: str,
    adata: AnnData,
    ours: NDArray[np.floating],
    reference: NDArray[np.floating],
    reseeded: NDArray[np.floating],
) -> None:
    """Assert our layout is not meaningfully worse than the reference is against itself.

    The numbers are recorded whether or not the test passes: the ceiling is the finding,
    pass/fail alone hides it.
    """
    dataset = adata.uns["dataset_id"]
    score, ceiling = preservation_band(reference, reseeded, ours, k_ref=K_REF, k_cand=K_CAND)
    record_property(f"{algorithm}.{dataset}.preservation", round(score, 4))
    record_property(f"{algorithm}.{dataset}.ceiling", round(ceiling, 4))
    print(
        f"\n{algorithm} on {dataset}: preservation {score:.3f}, "
        f"reference-vs-itself ceiling {ceiling:.3f} "
        f"(floor {CEILING_FRACTION * ceiling:.3f})"
    )
    assert score >= CEILING_FRACTION * ceiling, (
        f"{algorithm} preservation {score:.3f} is below {CEILING_FRACTION:.0%} of the "
        f"{ceiling:.3f} that {algorithm} reaches against itself on {dataset}"
    )


def test_umap(neighbored: AnnData, record_property: Callable[[str, object], None]) -> None:
    ours = neighbored.copy()
    scrust_call("tl.umap", ours, n_components=2, min_dist=0.5, spread=1.0, random_state=0)

    expected = neighbored.copy()
    sc.tl.umap(expected, min_dist=0.5, spread=1.0, random_state=0)
    reseeded = neighbored.copy()
    sc.tl.umap(reseeded, min_dist=0.5, spread=1.0, random_state=1)

    _check_band(
        record_property,
        "umap",
        neighbored,
        ours.obsm["X_umap"],
        expected.obsm["X_umap"],
        reseeded.obsm["X_umap"],
    )


def test_tsne(embedded: AnnData, record_property: Callable[[str, object], None]) -> None:
    ours = embedded.copy()
    scrust_call("tl.tsne", ours, n_pcs=N_COMPS, perplexity=30.0, random_state=0)

    expected = embedded.copy()
    sc.tl.tsne(expected, n_pcs=N_COMPS, perplexity=30.0, random_state=0)
    reseeded = embedded.copy()
    sc.tl.tsne(reseeded, n_pcs=N_COMPS, perplexity=30.0, random_state=1)

    _check_band(
        record_property,
        "tsne",
        embedded,
        ours.obsm["X_tsne"],
        expected.obsm["X_tsne"],
        reseeded.obsm["X_tsne"],
    )


@pytest.mark.parametrize(
    ("algorithm", "key", "kwargs"),
    [
        ("umap", "X_umap", {"n_components": 2, "min_dist": 0.5, "spread": 1.0}),
        # Perplexity below the default: with 18-cell clusters, 30 would smear them
        # together and destroy the separation the absolute threshold relies on.
        ("tsne", "X_tsne", {"n_pcs": 30, "perplexity": 10.0}),
    ],
)
def test_embedding_on_separated_clusters(
    blobs: AnnData,
    algorithm: str,
    key: str,
    kwargs: dict[str, object],
    record_property: Callable[[str, object], None],
) -> None:
    """The absolute contract threshold, on data where it is reachable.

    scanpy agrees with itself here across seeds, so anything short of 0.80 is a broken
    layout rather than the stochasticity of the method.
    """
    ours = blobs.copy()
    scrust_call(f"tl.{algorithm}", ours, random_state=0, **kwargs)

    expected = blobs.copy()
    getattr(sc.tl, algorithm)(expected, random_state=0, **kwargs)

    preserved = neighborhood_preservation(
        expected.obsm[key], ours.obsm[key], k_ref=K_REF, k_cand=K_CAND
    )
    record_property(f"{algorithm}.blobs.preservation", round(preserved, 4))
    assert preserved >= STRICT_PRESERVATION, (
        f"{algorithm} preservation {preserved:.3f} < {STRICT_PRESERVATION} on well "
        f"separated clusters, where scanpy reproduces itself exactly"
    )


def test_rank_genes_groups(
    lognorm: AnnData, record_property: Callable[[str, object], None]
) -> None:
    ours = lognorm.copy()
    scrust_call("tl.rank_genes_groups", ours, "group", method="wilcoxon")

    expected = lognorm.copy()
    sc.tl.rank_genes_groups(expected, "group", method="wilcoxon")

    ours_result = ours.uns["rank_genes_groups"]
    expected_result = expected.uns["rank_genes_groups"]
    top = min(TOP_N_GENES, lognorm.n_vars)
    dataset = lognorm.uns["dataset_id"]
    for group in expected_result["names"].dtype.names:
        problems, deviations = de_comparison(ours_result, expected_result, group, top=top)
        for field, worst in deviations.items():
            record_property(f"de.{dataset}.{group}.{field}", f"{worst:.2e}")
        print(f"\nde on {dataset}, group {group}: worst relative deviation per field {deviations}")
        assert not problems, f"group {group}: {problems}"


@pytest.mark.parametrize("path", ["pp.does_not_exist", "definitely.not.here"])
def test_scrust_call_skips_on_missing_names(path: str) -> None:
    """The skip helper must fire on a name that was never bound.

    Guarding the guard: if this ever fails, every skipped test below is meaningless.
    """
    with pytest.raises(pytest.skip.Exception, match="unavailable"):
        scrust_call(path)
