"""Dendrograms, ForceAtlas2 layouts and embedding densities, against scanpy.

The three differ in what agreement can mean, and the file is organised by that:

* `dendrogram` is deterministic, so it is held to `scipy.cluster.hierarchy`
  element-wise and to `scanpy.tl.dendrogram` on the leaf order it produces.
* `draw_graph` is stochastic. Per `docs/API_CONTRACT.md` the bar is the band the
  reference reaches against *itself* reseeded, never an absolute number, and
  never a coordinate comparison.
* `embedding_density` is deterministic again — both implementations are
  `scipy.stats.gaussian_kde` with Scott's bandwidth — so it is held to a
  correlation that only round-off should move.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec

import numpy as np
import pytest
import scanpy as sc
import scipy.cluster.hierarchy as sch
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

from reference_metrics import knn_indices
from scrust_call import scrust_call

# Neighbourhood preservation, at the sizes `conftest` fixes for every embedding:
# how much of a cell's K_REF nearest neighbours in one layout stays inside its
# K_CAND nearest in the other.
K_REF = 15
K_CAND = 30

# Iterations for the layout under test. scanpy's default is 500; 200 is where
# the ForceAtlas2 speed controller has already settled on PBMC 3k, and keeps the
# stochastic test to a few seconds a seed.
N_ITERATIONS = 200

# Both implementations evaluate the same estimator, so anything below this would
# mean a real difference rather than f32 round-off. The measured deviation on
# 3000 points is ~1e-6; see `test_embedding_density_matches_scanpy`.
DENSITY_CORRELATION = 0.999


def _preservation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean fraction of each cell's reference neighbours kept by `candidate`."""
    close = knn_indices(reference, K_REF)
    wide = knn_indices(candidate, K_CAND)
    return float(
        np.mean([len(set(close[i]) & set(wide[i])) / K_REF for i in range(close.shape[0])])
    )


def _within_between(positions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean distance between cells sharing a label, and between cells that do not."""
    distances = squareform(pdist(np.asarray(positions, dtype=np.float64)))
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    return float(distances[same].mean()), float(distances[~same].mean())


def _cliques(n_cliques: int, per_clique: int) -> AnnData:
    """Fully connected cliques joined in a chain by one weak edge each.

    The bridges keep the graph connected — an unconnected component only feels
    gravity — while leaving the cliques unambiguously separate.
    """
    n_obs = n_cliques * per_clique
    graph = np.zeros((n_obs, n_obs), dtype=np.float32)
    for clique in range(n_cliques):
        block = slice(clique * per_clique, (clique + 1) * per_clique)
        graph[block, block] = 1.0
        if clique:
            left, right = clique * per_clique - 1, clique * per_clique
            graph[left, right] = graph[right, left] = 0.01
    np.fill_diagonal(graph, 0.0)

    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsp["connectivities"] = sparse.csr_matrix(graph)
    adata.obs["group"] = np.repeat([f"clique{i}" for i in range(n_cliques)], per_clique)
    adata.obs["group"] = adata.obs["group"].astype("category")
    return adata


# ---------------------------------------------------------------------------
# dendrogram
# ---------------------------------------------------------------------------


def _hand_built_centroids() -> np.ndarray:
    """Six group centroids over twelve dimensions, with a known shape structure.

    Rows 0-1 are proportional (correlation 1), rows 2-3 are noisy copies of each
    other, and rows 4-5 are unrelated to everything. Correlation ignores scale,
    so no merge here can be explained by the magnitudes.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 12))
    return np.ascontiguousarray(
        np.vstack(
            [
                base[0],
                base[0] * 2.5,
                base[1],
                base[1] + rng.normal(scale=0.3, size=12),
                base[2],
                rng.normal(size=12),
            ]
        ),
        dtype=np.float32,
    )


def test_dendrogram_matches_scipy_linkage() -> None:
    """Element-wise against `scipy.cluster.hierarchy.linkage`, and on the leaf order.

    scanpy's distance is `1 - pearson correlation` between group means, which is
    scipy's `correlation` metric; the linkage method is `average`.
    """
    centroids = _hand_built_centroids()
    linkage, leaves = scrust_call("_scrust.dendrogram", centroids)

    reference = sch.linkage(
        pdist(centroids.astype(np.float64), metric="correlation"), method="average"
    )
    assert_allclose(linkage, reference, rtol=1e-6, atol=1e-9)
    assert list(leaves) == sch.dendrogram(reference, no_plot=True)["leaves"]
    # The pair that is one row scaled by another must merge first, at distance 0.
    assert set(linkage[0][:2]) == {0.0, 1.0}
    assert linkage[0][2] == pytest.approx(0.0, abs=1e-9)


def test_dendrogram_leaf_order_matches_scanpy(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Same tree and same leaves left to right as `scanpy.tl.dendrogram`, on real data.

    The comparison is against scanpy running `average` linkage, which is what this
    module implements; scanpy's own default is `complete`. The two methods share
    the pearson distance and differ only in how a merged cluster's distance to the
    rest is recomputed, so they often produce the same leaf order — but not always,
    and which datasets they agree on is measured here rather than assumed.
    """
    dataset = neighbored.uns["dataset_id"]
    ours = neighbored.copy()
    scrust_call("tl.dendrogram", ours, "group")
    slot = ours.uns["dendrogram_group"]

    reference = sc.tl.dendrogram(
        neighbored, "group", use_rep="X_pca", n_pcs=50, linkage_method="average", inplace=False
    )
    scanpy_default = sc.tl.dendrogram(neighbored, "group", use_rep="X_pca", n_pcs=50, inplace=False)
    agrees_with_default = slot["categories_idx_ordered"] == list(
        scanpy_default["categories_idx_ordered"]
    )
    record_property(f"dendrogram.{dataset}.leaf_order", list(slot["categories_idx_ordered"]))
    record_property(f"dendrogram.{dataset}.agrees_with_complete_linkage", agrees_with_default)
    print(
        f"\ndendrogram on {dataset}: leaf order {slot['categories_idx_ordered']}; "
        f"scanpy's default complete linkage {'agrees' if agrees_with_default else 'does not agree'}"
    )

    assert slot["categories_idx_ordered"] == list(reference["categories_idx_ordered"])
    assert slot["categories_ordered"] == list(reference["categories_ordered"])
    assert_allclose(slot["linkage"], reference["linkage"], rtol=1e-6, atol=1e-9)
    assert_allclose(
        slot["correlation_matrix"], reference["correlation_matrix"], rtol=1e-5, atol=1e-6
    )


def test_dendrogram_writes_the_slots_scanpy_plotting_reads(neighbored: AnnData) -> None:
    """`pl.dendrogram` and `pl.correlation_matrix` read these keys by name."""
    scrust_call("tl.dendrogram", neighbored, "group")
    slot = neighbored.uns["dendrogram_group"]
    assert slot["groupby"] == ["group"]
    assert slot["cor_method"] == "pearson"
    assert slot["linkage_method"] == "average"
    n_groups = len(neighbored.obs["group"].cat.categories)
    assert np.asarray(slot["linkage"]).shape == (n_groups - 1, 4)
    assert slot["correlation_matrix"].shape == (n_groups, n_groups)
    for key in ("icoord", "dcoord", "ivl", "leaves"):
        assert key in slot["dendrogram_info"]
    assert slot["dendrogram_info"]["ivl"] == slot["categories_ordered"]


def test_dendrogram_rejects_a_single_group(neighbored: AnnData) -> None:
    """One group has no pair to merge, so there is no tree to build."""
    neighbored.obs["only"] = "everyone"
    neighbored.obs["only"] = neighbored.obs["only"].astype("category")
    with pytest.raises(ValueError, match="2 are needed"):
        scrust_call("tl.dendrogram", neighbored, "only")


def test_dendrogram_rejects_a_non_categorical_groupby(neighbored: AnnData) -> None:
    """A continuous column has no groups to average, as scanpy also insists."""
    neighbored.obs["depth"] = np.arange(neighbored.n_obs, dtype=np.float64)
    with pytest.raises(ValueError, match="categorical"):
        scrust_call("tl.dendrogram", neighbored, "depth")


# ---------------------------------------------------------------------------
# draw_graph
# ---------------------------------------------------------------------------


def test_draw_graph_keeps_cliques_apart() -> None:
    """Three cliques joined by single weak edges stay three clumps."""
    adata = _cliques(3, 20)
    scrust_call("tl.draw_graph", adata, n_iterations=N_ITERATIONS)
    positions = adata.obsm["X_draw_graph_fa"]
    assert positions.shape == (60, 2)
    assert np.isfinite(positions).all()

    within, between = _within_between(positions, adata.obs["group"].cat.codes.to_numpy())
    assert between > 3.0 * within, f"within={within:.3g}, between={between:.3g}"


def test_draw_graph_keeps_real_groups_apart(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """The same claim on the real neighbour graph and scanpy's own labels."""
    dataset = neighbored.uns["dataset_id"]
    scrust_call("tl.draw_graph", neighbored, n_iterations=N_ITERATIONS)
    within, between = _within_between(
        neighbored.obsm["X_draw_graph_fa"], neighbored.obs["group"].cat.codes.to_numpy()
    )
    record_property(f"draw_graph.{dataset}.between_over_within", round(between / within, 3))
    print(f"\ndraw_graph on {dataset}: between/within distance ratio {between / within:.2f}")
    assert between > 1.5 * within


def test_draw_graph_reaches_the_band_the_reference_reaches_against_itself(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """The contract's criterion for a stochastic embedding, with its ceiling measured.

    A force-directed layout is not reproducible across seeds even by the
    reference, so the reference is first run against itself reseeded to find the
    band it can reach, and ours is judged against that band.

    One caveat is measured rather than assumed: `scanpy.tl.draw_graph` only runs
    ForceAtlas2 when `fa2-modified` is installed, and otherwise silently falls
    back to igraph's Fruchterman-Reingold — a *different* algorithm, which no
    ForceAtlas2 can be expected to land on top of. Where the real reference is
    absent, this asserts the claim that still holds across algorithms: our layout
    must be at least as reproducible across seeds as scanpy's is. The
    cross-implementation number is recorded either way.
    """
    dataset = neighbored.uns["dataset_id"]
    reference_is_forceatlas2 = find_spec("fa2_modified") is not None

    reference = {}
    for seed in (0, 1):
        run = neighbored.copy()
        sc.tl.draw_graph(run, random_state=seed)
        key = next(k for k in run.obsm if k.startswith("X_draw_graph_"))
        reference[seed] = np.asarray(run.obsm[key])

    ours = {}
    for seed in (0, 1):
        run = neighbored.copy()
        scrust_call("tl.draw_graph", run, n_iterations=N_ITERATIONS, random_state=seed)
        ours[seed] = np.asarray(run.obsm["X_draw_graph_fa"])

    band = _preservation(reference[0], reference[1])
    ours_band = _preservation(ours[0], ours[1])
    cross = _preservation(reference[0], ours[0])
    for name, value in (("reference_band", band), ("our_band", ours_band), ("cross", cross)):
        record_property(f"draw_graph.{dataset}.{name}", round(value, 4))
    print(
        f"\ndraw_graph on {dataset}: scanpy reproduces itself across seeds at "
        f"{band:.1%}, we reproduce ourselves at {ours_band:.1%}, and we preserve "
        f"{cross:.1%} of its neighbourhoods "
        f"({'ForceAtlas2' if reference_is_forceatlas2 else 'Fruchterman-Reingold fallback'})"
    )

    if reference_is_forceatlas2:
        # Same algorithm on both sides: the contract's 85% of the reference band.
        assert cross >= 0.85 * band, f"{cross:.3f} is below 85% of the {band:.3f} band"
    else:
        assert ours_band >= band, (
            f"our layout reproduces itself at {ours_band:.3f}, less than the "
            f"{band:.3f} scanpy's fallback manages"
        )


def test_draw_graph_is_deterministic_at_a_fixed_seed() -> None:
    """Same seed, same bytes; a different seed, a different layout."""
    adata = _cliques(2, 15)
    runs = []
    for seed in (3, 3, 4):
        run = adata.copy()
        scrust_call("tl.draw_graph", run, n_iterations=50, random_state=seed)
        runs.append(run.obsm["X_draw_graph_fa"])
    assert_allclose(runs[0], runs[1], rtol=0, atol=0)
    assert not np.array_equal(runs[0], runs[2])


def test_draw_graph_writes_the_slots_scanpy_plotting_reads() -> None:
    adata = _cliques(2, 10)
    scrust_call("tl.draw_graph", adata, n_iterations=20, random_state=5)
    assert adata.obsm["X_draw_graph_fa"].dtype == np.float32
    assert adata.uns["draw_graph"]["params"] == {"layout": "fa", "random_state": 5}


def test_draw_graph_rejects_an_empty_graph() -> None:
    adata = AnnData(np.zeros((3, 1), dtype=np.float32))
    adata.obsp["connectivities"] = sparse.csr_matrix((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="empty graph"):
        scrust_call("tl.draw_graph", adata, n_iterations=10)


def test_draw_graph_rejects_an_unsupported_layout() -> None:
    with pytest.raises(ValueError, match="ForceAtlas2"):
        scrust_call("tl.draw_graph", _cliques(2, 5), layout="fr")


# ---------------------------------------------------------------------------
# embedding_density
# ---------------------------------------------------------------------------


def test_embedding_density_matches_scanpy(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Per-cell agreement with `scanpy.tl.embedding_density`, grouped and ungrouped.

    Both compute `scipy.stats.gaussian_kde` with Scott's factor `n^(-1/(d+4))`
    over the *data* covariance, evaluated at the cells themselves and rescaled to
    [0, 1]; the only expected difference is that we accumulate in f32.
    """
    dataset = neighbored.uns["dataset_id"]
    for groupby in (None, "group"):
        ours, theirs = neighbored.copy(), neighbored.copy()
        scrust_call("tl.embedding_density", ours, basis="pca", groupby=groupby)
        sc.tl.embedding_density(theirs, basis="pca", groupby=groupby)

        key = "pca_density" if groupby is None else f"pca_density_{groupby}"
        mine = np.asarray(ours.obs[key], dtype=np.float64)
        reference = np.asarray(theirs.obs[key], dtype=np.float64)
        correlation = float(np.corrcoef(mine, reference)[0, 1])
        deviation = float(np.abs(mine - reference).max())

        label = f"embedding_density.{dataset}.{groupby or 'all'}"
        record_property(f"{label}.correlation", round(correlation, 6))
        record_property(f"{label}.max_deviation", f"{deviation:.3e}")
        print(
            f"\nembedding_density on {dataset} (groupby={groupby}): "
            f"correlation {correlation:.6f}, largest deviation {deviation:.3e}"
        )
        assert correlation >= DENSITY_CORRELATION
        assert ours.uns[f"{key}_params"] == theirs.uns[f"{key}_params"]
        assert (mine >= 0.0).all() and (mine <= 1.0).all()


def test_embedding_density_ranks_a_tight_cluster_above_an_isolated_point() -> None:
    """The analytic case: density is high where the points are, low where they are not."""
    rng = np.random.default_rng(0)
    embedding = np.vstack([rng.normal(scale=0.1, size=(60, 2)), [[8.0, 8.0]]]).astype(np.float32)
    adata = AnnData(np.zeros((61, 1), dtype=np.float32))
    adata.obsm["X_umap"] = embedding
    scrust_call("tl.embedding_density", adata)

    density = np.asarray(adata.obs["umap_density"], dtype=np.float64)
    assert density.argmin() == 60, "the isolated point must be the sparsest"
    assert density[:60].mean() > density[60] + 0.5
    # The same ranking scipy's own estimator gives.
    reference = gaussian_kde(embedding.T.astype(np.float64))(embedding.T.astype(np.float64))
    assert np.corrcoef(density, reference)[0, 1] >= DENSITY_CORRELATION


def test_embedding_density_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    adata = AnnData(np.zeros((200, 1), dtype=np.float32))
    adata.obsm["X_umap"] = rng.normal(size=(200, 2)).astype(np.float32)
    first, second = adata.copy(), adata.copy()
    scrust_call("tl.embedding_density", first)
    scrust_call("tl.embedding_density", second)
    assert_allclose(first.obs["umap_density"], second.obs["umap_density"], rtol=0, atol=0)


def test_embedding_density_rejects_a_mismatched_embedding() -> None:
    """A basis that is not there, and one that is not two-dimensional."""
    adata = AnnData(np.zeros((10, 1), dtype=np.float32))
    with pytest.raises(KeyError, match="X_umap"):
        scrust_call("tl.embedding_density", adata)

    adata.obsm["X_umap"] = np.arange(10, dtype=np.float32).reshape(10, 1)
    with pytest.raises(ValueError, match="2 are needed"):
        scrust_call("tl.embedding_density", adata)
