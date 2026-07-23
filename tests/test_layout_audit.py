"""Audit of `scrust_core::layout` against scanpy: dendrogram, draw_graph, density.

This file is a *second opinion* on `tests/test_layout.py`. It deliberately goes at the
places that file leaves open:

* `dendrogram` — `tests/test_layout.py` compares against `scipy` run with `average`
  linkage, which is what the core implements, and merely *records* whether scanpy's own
  default (`complete`) happens to agree. Here the divergence is pinned instead: an input
  is constructed on which the two methods provably disagree, and the size of the gap is
  asserted.
* `draw_graph` — the existing tests assert clique separation, determinism and shape.
  None of them can fail if edge *weights* are dropped, and none of them check that the
  layout recovers graph geometry beyond "two clumps". Both are tested here.
* `embedding_density` — the existing test correlates against `scanpy` on real PCA
  coordinates. Here the estimator's defining properties are pinned directly: the full
  covariance bandwidth (via affine equivariance), Scott's exponent beyond two
  dimensions, and the behaviour on degenerate input where scanpy returns NaN.

Every test calls into the crate through `scrust_call`; nothing here compares scipy to
scipy. `fa2-modified` is not installed in this environment, so `sc.tl.draw_graph`
cannot be run as a reference at all — see the module note above `test_draw_graph_*`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scanpy as sc
import scipy.cluster.hierarchy as sch
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, spearmanr

from scrust_call import DEVICE, scrust_call

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _csr_args(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The four arguments `_scrust.draw_graph` takes a CSR matrix as."""
    matrix = sparse.csr_matrix(dense)
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def _clique_block(size: int, weight: float) -> np.ndarray:
    """A fully connected, symmetric, self-loop-free block of `size` nodes."""
    block = np.full((size, size), weight, dtype=np.float32)
    np.fill_diagonal(block, 0.0)
    return block


def _path_graph(n_nodes: int) -> np.ndarray:
    """Symmetric adjacency of the path `0 - 1 - ... - (n_nodes - 1)`."""
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for node in range(n_nodes - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1.0
    return adjacency


def _divergent_centroids() -> np.ndarray:
    """Six centroids on which `average` and `complete` linkage disagree.

    Found by search over `default_rng(3)`; the fifth draw is the first whose two leaf
    orders differ. Regenerated rather than pasted so the provenance stays visible.
    """
    rng = np.random.default_rng(3)
    for _ in range(5):
        centroids = rng.normal(size=(6, 8))
    return np.ascontiguousarray(centroids, dtype=np.float32)


def _correlation_linkage(centroids: np.ndarray, method: str) -> np.ndarray:
    """scanpy's tree: `1 - pearson` between group means, fed to `sch.linkage`."""
    return sch.linkage(pdist(centroids.astype(np.float64), metric="correlation"), method=method)


def _scaled_kde(points: np.ndarray) -> np.ndarray:
    """scanpy's `_calc_density`, generalised to `d` columns instead of exactly two."""
    values = gaussian_kde(points.astype(np.float64).T)(points.astype(np.float64).T)
    return (values - values.min()) / (values.max() - values.min())


def _mean_pairwise(points: np.ndarray) -> float:
    return float(np.mean(pdist(points)))


# ---------------------------------------------------------------------------
# dendrogram
# ---------------------------------------------------------------------------


def test_dendrogram_uses_average_linkage_where_scanpy_defaults_to_complete() -> None:
    """Pins the one real divergence from scanpy in this module, and its size.

    `sc.tl.dendrogram` defaults to `linkage_method="complete"`; the core hard-codes
    average linkage (`crates/scrust-core/src/layout.rs`, `agglomerate`, whose
    Lance-Williams update is the size-weighted average one) and the Python wrapper
    records `linkage_method="average"` in the `uns` slot. On these centroids the two
    methods produce different leaf orders *and* different merge heights, so a caller
    who takes scanpy's default and this function to be interchangeable is wrong.

    The test asserts the core reproduces `average` element-wise and that it is *not*
    `complete`, which is what makes it fail if the core were ever switched.
    """
    centroids = _divergent_centroids()
    linkage, leaves = scrust_call("_scrust.dendrogram", centroids)
    linkage = np.asarray(linkage)

    average = _correlation_linkage(centroids, "average")
    complete = _correlation_linkage(centroids, "complete")

    assert_allclose(linkage, average, rtol=1e-6, atol=1e-7)
    assert list(map(int, leaves)) == sch.dendrogram(average, no_plot=True)["leaves"]

    # The divergence is real on this input, not a tolerance artefact.
    complete_leaves = sch.dendrogram(complete, no_plot=True)["leaves"]
    assert list(map(int, leaves)) != complete_leaves
    height_gap = float(np.max(np.abs(linkage[:, 2] - complete[:, 2])))
    assert height_gap > 0.5, height_gap


def test_dendrogram_breaks_distance_ties_the_way_scipy_does() -> None:
    """Groups with identical means give distance-0 ties; the merge order must still match.

    Three centroids are identical and two more are identical to each other, so three of
    the four merges happen at distance 0 and the tree is decided entirely by how ties
    are broken. The core scans the upper triangle row-major and keeps the strict
    minimum; scipy uses the nearest-neighbour chain. This asserts the two agree, which
    is the only thing that keeps `categories_ordered` reproducible against scanpy on
    data with duplicated group means (a real case: a group of one cell that happens to
    sit on another group's centroid).
    """
    centroids = np.array(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [4, 1, 3, 2], [4, 1, 3, 2]],
        dtype=np.float32,
    )
    linkage, leaves = scrust_call("_scrust.dendrogram", centroids)
    linkage = np.asarray(linkage)
    reference = _correlation_linkage(centroids, "average")

    assert_allclose(linkage, reference, rtol=0, atol=1e-12)
    assert list(map(int, leaves)) == sch.dendrogram(reference, no_plot=True)["leaves"]
    # Three merges at zero, one at the distance between the two distinct shapes.
    assert_allclose(linkage[:3, 2], 0.0, atol=1e-12)
    assert linkage[3, 2] > 1.0


def test_dendrogram_leaf_order_is_scipys_traversal_and_not_a_sort() -> None:
    """The leaf order must be scipy's depth-first walk, not the identity.

    `leaf_order` in the core reimplements `sch.dendrogram`'s default traversal
    (`count_sort=False, distance_sort=False`). A reimplementation that returned
    `0..n-1`, or that swapped the two children, would pass any test that only checks
    "every leaf appears once", so this pins the order against scipy on an input whose
    correct order is provably neither sorted nor reversed.
    """
    centroids = _divergent_centroids()
    _, leaves = scrust_call("_scrust.dendrogram", centroids)
    order = list(map(int, leaves))

    reference = sch.dendrogram(_correlation_linkage(centroids, "average"), no_plot=True)["leaves"]
    assert order == reference
    assert sorted(order) == list(range(len(order)))
    assert order != sorted(order)
    assert order != sorted(order, reverse=True)


def test_dendrogram_rejects_a_constant_centroid_that_scanpy_turns_into_nan() -> None:
    """A group whose mean is flat has undefined correlation; the core refuses it.

    scanpy's path (`mean_df.T.corr()` then `sch.linkage`) produces NaN and dies inside
    scipy with a message about non-finite values. The core raises a `ValueError` naming
    the offending group instead. Both refuse — this pins that the core does not
    silently return a tree built from a NaN distance.
    """
    centroids = np.array([[1, 1, 1], [1, 2, 3], [3, 2, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="constant"):
        scrust_call("_scrust.dendrogram", centroids)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distances = pdist(centroids.astype(np.float64), metric="correlation")
    assert np.isnan(distances).any(), "scanpy's own distance is NaN here"


def test_dendrogram_handles_a_group_of_a_single_cell() -> None:
    """A category with one cell has that cell as its centroid, exactly as scanpy.

    The mean of a one-row group is a degenerate case for the pandas `groupby` in
    `scrust.tl.dendrogram`; it is also the case where the correlation distance is at
    its noisiest. Compared against `sc.tl.dendrogram` run with `average` linkage, so
    the only thing under test is the group-mean and clustering path, not the linkage
    divergence pinned above.
    """
    rng = np.random.default_rng(11)
    n_obs, n_pcs = 61, 10
    pcs = rng.normal(size=(n_obs, n_pcs))
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20 + ["lonely"])
    # Push the singleton somewhere no group mean sits, so it cannot merge by accident.
    pcs[-1] = rng.normal(size=n_pcs) * 5.0 + 20.0

    adata = AnnData(
        np.zeros((n_obs, 2), dtype=np.float32),
        obs={"group": pd_categorical(labels)},
    )
    adata.obsm["X_pca"] = pcs

    ours = adata.copy()
    scrust_call("tl.dendrogram", ours, "group", n_pcs=n_pcs)
    slot = ours.uns["dendrogram_group"]

    reference = sc.tl.dendrogram(
        adata.copy(),
        "group",
        use_rep="X_pca",
        n_pcs=n_pcs,
        linkage_method="average",
        inplace=False,
    )
    assert slot["categories_idx_ordered"] == list(reference["categories_idx_ordered"])
    assert slot["categories_ordered"] == list(reference["categories_ordered"])
    assert_allclose(slot["linkage"], reference["linkage"], rtol=1e-5, atol=1e-6)
    # The singleton is a leaf of its own, i.e. the tree really has four groups.
    assert np.asarray(slot["linkage"]).shape == (3, 4)


def pd_categorical(values: np.ndarray):
    """`values` as a pandas categorical column, which is what scanpy's dendrogram needs."""
    import pandas as pd

    return pd.Categorical(values)


# ---------------------------------------------------------------------------
# draw_graph
#
# `fa2-modified` is not installed here, so `sc.tl.draw_graph(layout="fa")` cannot run
# and there is no reference layout to compare against. ForceAtlas2 is stochastic
# anyway, so these test the structural claims a layout has to satisfy for the
# downstream plot to mean anything, chosen so that a plausible bug breaks them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_draw_graph_recovers_the_geometry_of_a_path_graph(seed: int) -> None:
    """Layout distance must track graph distance along a path, not merely cluster.

    A 20-node path has a one-dimensional geometry that a correct force layout has to
    reproduce: node 0 ends far from node 19 and adjacent nodes end close. The Spearman
    correlation between all-pairs shortest-path distance and all-pairs layout distance
    is the whole claim in one number.

    This is the test that would catch a sign error in `attract` or `repel`, or an
    `advance` step that never moves anything: a random initial scatter scores ~0 here,
    and pure repulsion (attraction dropped) scores well below the bound.
    """
    adjacency = _path_graph(20)
    positions = np.asarray(
        scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 300, seed, DEVICE)
    )
    graph_distance = shortest_path(sparse.csr_matrix(adjacency), unweighted=True)
    upper = np.triu_indices(adjacency.shape[0], 1)

    correlation = spearmanr(pdist(positions), graph_distance[upper]).statistic
    assert correlation > 0.9, correlation


def test_draw_graph_honours_edge_weights() -> None:
    """Edge weight scales attraction, so a heavy clique lands tighter than a light one.

    Two disjoint cliques of the same size in the *same* layout, one wired at weight 6.0
    and the other at 0.4. `attract` multiplies the pull by the stored value, so the
    heavy clique must contract much further. Nothing else in the run differs — same
    node count, same degree, same masses — so if edge values were ignored (a binarised
    graph, or `edgeWeightInfluence` dropped) the two mean spreads would coincide.

    Measured ratio is ~3.8x; the bound is set at 2x so f32 and seed noise cannot reach it.
    """
    size = 8
    adjacency = np.zeros((2 * size, 2 * size), dtype=np.float32)
    adjacency[:size, :size] = _clique_block(size, 6.0)
    adjacency[size:, size:] = _clique_block(size, 0.4)

    positions = np.asarray(scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 200, 0, DEVICE))
    heavy = _mean_pairwise(positions[:size])
    light = _mean_pairwise(positions[size:])
    assert light > 2.0 * heavy, (heavy, light)


def test_draw_graph_is_reproducible_from_its_seed_and_moves_with_it() -> None:
    """Same seed, identical bytes; a different seed, a materially different layout.

    The core seeds its own splitmix64 rather than numpy, so this is the only guarantee
    a caller has that `random_state` means anything. The second half matters as much as
    the first: a layout that ignored the seed would still be "deterministic".
    """
    adjacency = _path_graph(12)
    first = np.asarray(scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 60, 7, DEVICE))
    again = np.asarray(scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 60, 7, DEVICE))
    other = np.asarray(scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 60, 8, DEVICE))

    assert np.array_equal(first, again)
    assert not np.allclose(first, other, atol=1e-3)


def test_draw_graph_ignores_edges_stored_below_the_diagonal() -> None:
    """Documents a real fragility: attraction reads only the upper triangle.

    `undirected_edges` keeps an entry only when `column > row`, so a graph handed over
    in lower-triangular storage contributes *no* attraction at all and the result is
    pure repulsion plus gravity. Degrees, and therefore masses, are still taken from
    the row counts, so the call succeeds and returns a plausible-looking layout.

    scanpy's `connectivities` are always symmetric, so no scanpy-shaped caller hits
    this; the core also never validates symmetry, which is why it is worth pinning.
    This test asserts the *current* behaviour — it is not the desired behaviour, and
    the fix would be to symmetrise (or to reject an asymmetric graph) in
    `crates/scrust-core/src/layout.rs`.
    """
    size = 8
    adjacency = np.zeros((2 * size, 2 * size), dtype=np.float32)
    adjacency[:size, :size] = _clique_block(size, 1.0)
    adjacency[size:, size:] = _clique_block(size, 1.0)
    labels = np.array([0] * size + [1] * size)

    def within_between(positions: np.ndarray) -> tuple[float, float]:
        distance = squareform(pdist(positions))
        same = labels[:, None] == labels[None, :]
        np.fill_diagonal(same, False)
        return float(distance[same].mean()), float(distance[~same].mean())

    full = np.asarray(scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 100, 0, DEVICE))
    lower = np.asarray(
        scrust_call("_scrust.draw_graph", *_csr_args(np.tril(adjacency)), 100, 0, DEVICE)
    )

    full_within, full_between = within_between(full)
    lower_within, lower_between = within_between(lower)

    # Symmetric storage: the cliques are an order of magnitude tighter than they are apart.
    assert full_within < 0.2 * full_between, (full_within, full_between)
    # Lower-triangular storage: no attraction survives, so the cliques do not form.
    assert lower_within > 0.7 * lower_between, (lower_within, lower_between)


def test_draw_graph_rejects_input_no_layout_exists_for() -> None:
    """Zero iterations, an edgeless graph and a non-square graph are all errors.

    Each of the three would otherwise return silently: zero iterations gives back the
    random scatter, an edgeless graph gives a pure repulsion cloud, and a non-square
    graph would index out of the position array.
    """
    adjacency = _path_graph(6)
    with pytest.raises(ValueError, match="n_iterations"):
        scrust_call("_scrust.draw_graph", *_csr_args(adjacency), 0, 0, DEVICE)

    empty = sparse.csr_matrix((6, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        scrust_call(
            "_scrust.draw_graph",
            empty.indptr.astype(np.uint32),
            empty.indices.astype(np.uint32),
            empty.data.astype(np.float32),
            6,
            10,
            0,
            DEVICE,
        )

    with pytest.raises(ValueError, match="square"):
        scrust_call("_scrust.draw_graph", *_csr_args(adjacency[:, :5]), 10, 0, DEVICE)


# ---------------------------------------------------------------------------
# embedding_density
# ---------------------------------------------------------------------------


def test_embedding_density_matches_scipy_gaussian_kde_and_lands_exactly_on_zero_and_one() -> None:
    """Element-wise against the estimator scanpy calls, plus the rescaling endpoints.

    `scanpy.tl.embedding_density` is `scipy.stats.gaussian_kde` with Scott's bandwidth
    followed by `(z - min) / (max - min)`. The per-cell agreement is the primary check;
    the endpoints are asserted separately because a rescaling that used, say, `z / max`
    would still correlate perfectly with the reference and would still be wrong.
    """
    rng = np.random.default_rng(0)
    embedding = np.ascontiguousarray(rng.normal(size=(400, 2)), dtype=np.float32)

    density = np.asarray(scrust_call("_scrust.embedding_density", embedding, DEVICE))
    assert_allclose(density, _scaled_kde(embedding), rtol=0, atol=1e-5)
    assert density.min() == 0.0
    assert density.max() == 1.0


@pytest.mark.parametrize("n_dims", [3, 4])
def test_embedding_density_uses_scotts_exponent_in_higher_dimensions(n_dims: int) -> None:
    """Pins the bandwidth exponent `n ** (-1 / (d + 4))` away from the 2-D case.

    The core accepts any number of columns even though the Python wrapper always slices
    the first two, so the `d`-dependence of Scott's factor is otherwise untested: a
    hard-coded `n ** (-1/6)` (the 2-D value) would pass every 2-D test in the suite and
    fail here.
    """
    rng = np.random.default_rng(1)
    embedding = np.ascontiguousarray(rng.normal(size=(300, n_dims)), dtype=np.float32)

    density = np.asarray(scrust_call("_scrust.embedding_density", embedding, DEVICE))
    assert_allclose(density, _scaled_kde(embedding), rtol=0, atol=1e-5)


def test_embedding_density_is_invariant_under_an_affine_map_of_the_embedding() -> None:
    """The bandwidth is the full data covariance, not an isotropic scalar.

    `gaussian_kde` scales the *covariance* of the data by Scott's factor, which makes
    the normalised density exactly equivariant: stretch, shear and translate the
    embedding and every cell keeps its density. An implementation that used a scalar or
    diagonal bandwidth would look fine on isotropic test data and drift here — the
    isotropic variant of this estimator deviates by ~0.09 on this input, against the
    1e-5 bound below, so the test has ~4 orders of magnitude of headroom.
    """
    rng = np.random.default_rng(7)
    embedding = np.ascontiguousarray(rng.normal(size=(300, 2)), dtype=np.float32)
    transform = np.array([[2.0, 0.7], [-0.3, 1.5]], dtype=np.float32)
    mapped = np.ascontiguousarray(
        embedding @ transform.T + np.array([5.0, -3.0], dtype=np.float32), dtype=np.float32
    )

    plain = np.asarray(scrust_call("_scrust.embedding_density", embedding, DEVICE))
    affine = np.asarray(scrust_call("_scrust.embedding_density", mapped, DEVICE))
    assert_allclose(affine, plain, rtol=0, atol=1e-5)


def test_embedding_density_flattens_a_symmetric_layout_where_scanpy_amplifies_roundoff() -> None:
    """Divergence: a perfectly uniform density gives 0 here and 0/0 in scanpy.

    The four corners of a square all have the same true density, so the rescaling
    denominator `max - min` is zero. scanpy divides anyway: in float64 the four kernel
    sums differ by ~3e-17, and that noise is stretched across the whole `[0, 1]` range,
    so `sc.tl.embedding_density` reports one corner at 1.0 and three at 0.0 — an
    artefact with no meaning. The core detects the empty range in `scale_to_unit` and
    returns zeros.

    Both answers are round-off; the core's is the defensible one. It matters because a
    `groupby` category whose cells sit on a symmetric grid is not exotic, and the two
    implementations then disagree by a full unit.
    """
    square = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    density = np.asarray(scrust_call("_scrust.embedding_density", square, DEVICE))
    assert float(density.max() - density.min()) < 1e-3, density
    assert_allclose(density, 0.0, atol=1e-3)

    reference = _scaled_kde(square)
    # scanpy's spread is the full range, entirely from a ~3e-17 difference in the sums.
    raw = gaussian_kde(square.astype(np.float64).T)(square.astype(np.float64).T)
    assert float(raw.max() - raw.min()) < 1e-15
    assert float(reference.max() - reference.min()) == 1.0


def test_embedding_density_rejects_degenerate_input_that_scanpy_returns_nan_for() -> None:
    """Too few cells, and collinear cells: the core errors where scanpy does not always.

    * Two cells in two dimensions: `gaussian_kde` does *not* raise — it builds a
      singular covariance, returns two identical values, and scanpy's rescaling turns
      them into `nan`. The core requires `n_cells > n_dims` and raises. This is a
      behavioural divergence for a `groupby` category with two cells: scanpy writes
      NaN into `obs`, scrust raises.
    * Four collinear cells: both refuse, scipy with `LinAlgError` and the core with a
      `ValueError` about a non-invertible covariance.
    """
    two_cells = np.array([[0, 0], [1, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="cells"):
        scrust_call("_scrust.embedding_density", two_cells, DEVICE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(_scaled_kde(two_cells)).all(), "scanpy silently yields NaN here"

    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"collinear|degenerate"):
        scrust_call("_scrust.embedding_density", collinear, DEVICE)
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_kde(collinear.astype(np.float64).T)
