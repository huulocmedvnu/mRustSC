"""Audit of the neighbour graph against umap-learn itself, not against scanpy's wrapper.

Everything downstream — UMAP, Leiden, PAGA, diffusion maps, Moran's I — consumes
`obsp["connectivities"]`. A defect here is invisible in those algorithms' own tests,
because each of them would be measured against a reference that got a *different*
graph. So these tests go to the source of truth: `umap.umap_.fuzzy_simplicial_set`,
called directly with the same neighbour lists our core produced, laid out the way
`scanpy.neighbors._connectivity.umap` lays them out (self first, at distance zero,
row width `n_neighbors`).

Two exactness bars are used, and the difference matters:

* the *sparsity pattern* must be identical, entry for entry. Nothing about it is
  floating point: it is the union of the knn pattern and its transpose, minus the
  diagonal. A divergence there is a bug, never noise.
* the *weights* are compared with `WEIGHT_TOLERANCE`. umap's bandwidth search stops
  as soon as the fuzzy cardinality is within `SMOOTH_K_TOLERANCE = 1e-5` of
  `log2(n_neighbors)`, and it evaluates that cardinality with a numba `parallel=True`
  reduction whose summation order is neither a plain float32 fold nor float64. Two
  faithful implementations therefore stop one bisection step apart on a few rows per
  thousand, which moves sigma by about 2e-6 relative and a weight by about 2e-5
  relative. That is the noise floor; anything above it is a real divergence. Getting
  the row width, rho, the floor or the symmetrisation wrong all move weights by
  percent, orders of magnitude above this bar.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from scrust_call import DEVICE, scrust_call

umap_ = pytest.importorskip("umap.umap_", reason="the audit compares against umap-learn")

N_NEIGHBORS = 15
# scanpy counts the cell itself; our core does not.
K = N_NEIGHBORS - 1

WEIGHT_TOLERANCE = 1e-4
# Below this fraction of bit-identical weights something structural has changed,
# even if every entry still sits inside WEIGHT_TOLERANCE.
MIN_BITWISE_FRACTION = 0.90


def embedding(n_cells: int = 240, n_dims: int = 10, seed: int = 11) -> np.ndarray:
    """A generic point cloud with no exact ties and no duplicates."""
    return np.random.default_rng(seed).normal(size=(n_cells, n_dims)).astype(np.float32)


def exact_knn(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The float64 ground truth: k non-self neighbours, ties broken by smaller index."""
    x64 = np.asarray(x, dtype=np.float64)
    d = np.sqrt(np.maximum(((x64[:, None, :] - x64[None, :, :]) ** 2).sum(-1), 0.0))
    n = d.shape[0]
    indices = np.zeros((n, k), dtype=np.int64)
    for cell in range(n):
        order = sorted((j for j in range(n) if j != cell), key=lambda j: (d[cell, j], j))
        indices[cell] = order[:k]
    return indices, np.take_along_axis(d, indices, 1)


def our_knn(x: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    indices, distances = scrust_call("_scrust.knn", np.ascontiguousarray(x), k, DEVICE)
    return np.asarray(indices), np.asarray(distances, dtype=np.float32)


def our_connectivities(indices: np.ndarray, distances: np.ndarray) -> sparse.csr_matrix:
    n_obs = indices.shape[0]
    indptr, columns, values, _ = scrust_call("_scrust.connectivities", indices, distances)
    matrix = sparse.csr_matrix(
        (np.asarray(values), np.asarray(columns), np.asarray(indptr)), shape=(n_obs, n_obs)
    )
    matrix.sort_indices()
    return matrix


def umap_rows(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same neighbours in umap's own layout: the cell itself first, at distance zero.

    This is exactly what `scanpy.neighbors._common._get_indices_distances_from_sparse_matrix`
    hands to `fuzzy_simplicial_set`, so the row width is `k + 1 == n_neighbors`.
    """
    n_obs = indices.shape[0]
    self_column = np.arange(n_obs, dtype=np.int64)[:, None]
    return (
        np.hstack([self_column, indices.astype(np.int64)]),
        np.hstack([np.zeros((n_obs, 1), dtype=np.float32), distances.astype(np.float32)]),
    )


def umap_reference(
    indices: np.ndarray, distances: np.ndarray, n_neighbors: int | None = None
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """`umap.umap_.fuzzy_simplicial_set` on our neighbours, as scanpy calls it."""
    knn_indices, knn_dists = umap_rows(indices, distances)
    n_obs = indices.shape[0]
    graph, sigmas, rhos = umap_.fuzzy_simplicial_set(
        sparse.coo_matrix((n_obs, 1)),
        knn_indices.shape[1] if n_neighbors is None else n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    graph = sparse.csr_matrix(graph)
    graph.sort_indices()
    return graph, sigmas, rhos


def assert_same_pattern(ours: sparse.csr_matrix, reference: sparse.csr_matrix) -> None:
    assert ours.shape == reference.shape
    np.testing.assert_array_equal(ours.indptr, reference.indptr)
    np.testing.assert_array_equal(ours.indices, reference.indices)


def assert_same_weights(ours: sparse.csr_matrix, reference: sparse.csr_matrix) -> None:
    assert_same_pattern(ours, reference)
    error = np.abs(ours.data - reference.data)
    worst = int(np.argmax(error / np.maximum(np.abs(reference.data), 1e-30)))
    relative = error / np.maximum(np.abs(reference.data), 1e-30)
    assert relative.max() <= WEIGHT_TOLERANCE, (
        f"weight {worst} of {len(error)}: umap {reference.data[worst]!r}, "
        f"ours {ours.data[worst]!r}, relative error {relative.max():.3e}"
    )
    bitwise = np.count_nonzero(error == 0) / len(error)
    assert bitwise >= MIN_BITWISE_FRACTION, (
        f"only {bitwise:.1%} of weights are bit-identical to umap; the binary search "
        "noise floor alone accounts for a few tenths of a percent, so this is structural"
    )


# 1. The off-by-one, end to end.


def test_row_width_matches_scanpy() -> None:
    """`pp.neighbors(n_neighbors=15)` must store 14 entries per row, like scanpy."""
    anndata = pytest.importorskip("anndata")
    x = embedding()
    adata = anndata.AnnData(np.zeros((x.shape[0], 3), dtype=np.float32), obsm={"X_pca": x})
    scrust_call("pp.neighbors", adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca", device=DEVICE)

    distances = sparse.csr_matrix(adata.obsp["distances"])
    per_row = np.diff(distances.indptr)
    assert per_row.min() == per_row.max() == N_NEIGHBORS - 1

    # ... and they are the *same* 14 neighbours the ground truth picks, not merely 14.
    expected, _ = exact_knn(x, K)
    for cell in range(x.shape[0]):
        stored = distances.indices[distances.indptr[cell] : distances.indptr[cell + 1]]
        assert set(stored) == set(expected[cell]), f"cell {cell}"
    assert np.count_nonzero(sparse.csr_matrix(adata.obsp["connectivities"]).diagonal()) == 0


# 2. The fuzzy simplicial set, against umap-learn directly.


@pytest.mark.parametrize("k", [2, 5, K, 30])
def test_connectivities_match_umap_fuzzy_simplicial_set(k: int) -> None:
    x = embedding()
    indices, distances = our_knn(x, k)
    reference, _, _ = umap_reference(indices, distances)
    assert_same_weights(our_connectivities(indices, distances), reference)


def test_target_cardinality_is_log2_of_the_umap_row_width() -> None:
    """The binary search must target `log2(k + 1)`, not `log2(k)`.

    This is where the off-by-one hides: our rows exclude the self neighbour, so the
    target has to be `log2` of umap's row width. Targeting `log2(k)` still produces a
    perfectly plausible graph — symmetric, in (0, 1], right sparsity — so only a
    comparison against a reference told the *wrong* width can catch it.
    """
    x = embedding()
    indices, distances = our_knn(x, K)
    ours = our_connectivities(indices, distances)

    right, _, _ = umap_reference(indices, distances, n_neighbors=K + 1)
    wrong, _, _ = umap_reference(indices, distances, n_neighbors=K)

    assert_same_weights(ours, right)
    assert_same_pattern(wrong, right)
    gap = np.abs(ours.data - wrong.data).max()
    assert gap > 100 * WEIGHT_TOLERANCE, (
        "targeting log2(k) instead of log2(k + 1) is indistinguishable here, so this "
        f"test cannot detect the off-by-one it exists to detect (gap {gap:.3e})"
    )


# 3. rho, and duplicate points.


def test_rho_is_the_nearest_strictly_positive_distance() -> None:
    """Duplicate cells sit at distance zero; rho must skip them, as umap-learn does."""
    x = embedding(n_cells=180, n_dims=6, seed=5)
    x[20:30] = x[20]  # ten exact copies: nine zero distances per row, fewer than k
    indices, distances = our_knn(x, K)
    assert np.count_nonzero(distances == 0.0) == 10 * (10 - 1)

    reference, _, rhos = umap_reference(indices, distances)
    assert np.count_nonzero(rhos == 0.0) == 0, "every cell here has a positive-distance neighbour"
    assert_same_weights(our_connectivities(indices, distances), reference)


def test_rho_is_zero_when_every_neighbour_is_a_duplicate() -> None:
    """With more than k copies of a point, rho is zero and every weight from it is 1."""
    x = embedding(n_cells=200, n_dims=6, seed=9)
    x[:20] = x[0]  # 20 copies, but k == 14, so the whole row is at distance zero
    indices, distances = our_knn(x, K)
    reference, _, rhos = umap_reference(indices, distances)
    assert np.count_nonzero(rhos == 0.0) == 20
    assert_same_weights(our_connectivities(indices, distances), reference)


def test_duplicate_points_are_within_the_expansions_resolution_of_zero() -> None:
    """Two byte-identical cells need not come out at distance exactly zero.

    `|a|^2 + |b|^2 - 2 a.b` reaches `a == b` through two different summations — the
    gemm's dot product and `sqr().sum()` — so the cancellation is only good to
    `f32::EPSILON * |a|^2`, and the square root turns that into a separation of order
    `|a| * sqrt(f32::EPSILON)`. `rho` is defined as the first *strictly* positive
    distance, so a duplicate can supply `rho` itself rather than being skipped.

    scanpy's sklearn backend has the same shape of artefact one precision down
    (`|a| * sqrt(f64 eps)`), so this is a difference of degree, not of kind: the
    duplicate's own weight is 1.0 either way, and the offsets of the real neighbours
    move by `rho`, which stays far below their spread. The test pins the size of the
    artefact so that a regression that made it comparable to a real distance shows up.
    """
    x = embedding(n_cells=180, n_dims=6, seed=5)
    x[100] = x[7]
    indices, distances = our_knn(x, K)

    position = list(indices[7]).index(100)
    assert position == 0, "the duplicate must still be the nearest neighbour"
    spurious = float(distances[7][position])
    radius = float(np.linalg.norm(x - x.mean(axis=0), axis=1).max())
    assert spurious <= 10.0 * radius * np.sqrt(np.finfo(np.float32).eps)
    assert spurious < 1e-2 * float(distances[7][1]), "rho noise rivals a real distance"

    # Whatever it is, our graph and umap's agree, because umap is fed the same numbers.
    assert_same_weights(
        our_connectivities(indices, distances), umap_reference(indices, distances)[0]
    )


# 4. The floor rules: which mean is which.


def test_sigma_floor_uses_the_global_mean_when_rho_is_zero() -> None:
    x = embedding(n_cells=200, n_dims=6, seed=4) * 10
    x[:20] = x[0]
    indices, distances = our_knn(x, K)
    reference, sigmas, rhos = umap_reference(indices, distances)

    _, rows = umap_rows(indices, distances)
    global_floor = 1e-3 * rows.mean()
    row_floor = 1e-3 * rows.mean(axis=1)
    bound = np.where(rhos > 0.0, row_floor, global_floor)
    assert np.count_nonzero(np.isclose(sigmas, bound, rtol=1e-6) & (rhos == 0.0)) == 20, (
        "the fixture no longer exercises the global-mean floor"
    )
    # The two means differ here by more than the tolerance, so using the wrong one shows.
    assert not np.allclose(global_floor, row_floor[rhos == 0.0], rtol=1e-2)
    assert_same_weights(our_connectivities(indices, distances), reference)


def test_sigma_floor_uses_the_row_mean_when_rho_is_positive() -> None:
    """A cell whose whole neighbourhood is equidistant drives sigma to zero.

    Twenty exact copies of one point, plus a handful of satellites just off them: each
    satellite sees fourteen neighbours at one identical distance, so rho equals every
    distance in the row, the cardinality is stuck at k, and the bisection collapses.
    Only the per-row floor keeps sigma finite, and it differs from the global one by
    orders of magnitude here.
    """
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 6)).astype(np.float32)
    x[:20] = x[0]
    x[20:25] = x[0] + (rng.normal(size=(5, 6)) * 1e-2).astype(np.float32)
    indices, distances = our_knn(x, K)
    reference, sigmas, rhos = umap_reference(indices, distances)

    _, rows = umap_rows(indices, distances)
    row_floor = 1e-3 * rows.mean(axis=1)
    bound = np.isclose(sigmas, row_floor, rtol=1e-6) & (rhos > 0.0)
    assert np.count_nonzero(bound) >= 2, "the fixture no longer exercises the row-mean floor"
    global_floor = 1e-3 * rows.mean()
    assert not np.allclose(global_floor, row_floor[bound], rtol=1e-2)
    assert_same_weights(our_connectivities(indices, distances), reference)


# 5. Symmetrisation.


def test_symmetrisation_is_the_fuzzy_union_of_umaps_own_directed_graph() -> None:
    """`a + a^T - a * a^T`, rebuilt from umap's sigmas and rhos, entry for entry.

    Building the directed graph from umap's own bandwidths removes the binary search
    from the comparison, so this holds bit-exactly.
    """
    x = embedding()
    indices, distances = our_knn(x, K)
    reference, sigmas, rhos = umap_reference(indices, distances)
    knn_indices, knn_dists = umap_rows(indices, distances)
    n_obs = x.shape[0]

    offsets = knn_dists - rhos[:, None]
    strengths = np.where(
        offsets <= 0.0, np.float32(1.0), np.exp(-(offsets / sigmas[:, None]))
    ).astype(np.float32)
    strengths[:, 0] = 0.0  # a cell has no affinity to itself
    directed = sparse.csr_matrix(
        (
            strengths.ravel(),
            knn_indices.ravel(),
            np.arange(0, knn_indices.size + 1, knn_indices.shape[1]),
        ),
        shape=(n_obs, n_obs),
    )
    directed.eliminate_zeros()
    union = sparse.csr_matrix(directed + directed.T - directed.multiply(directed.T))
    union.sort_indices()

    np.testing.assert_array_equal(union.indptr, reference.indptr)
    np.testing.assert_array_equal(union.indices, reference.indices)
    np.testing.assert_array_equal(union.data, reference.data)

    ours = our_connectivities(indices, distances)
    assert_same_pattern(ours, union)
    assert np.count_nonzero(ours.diagonal()) == 0
    assert (ours != ours.T).nnz == 0
    assert ours.data.min() > 0.0 and ours.data.max() <= 1.0


# 6. Tie breaking.


def test_ties_break_towards_the_smaller_index() -> None:
    """On an exact lattice every candidate set is a valid kNN; we pin ours.

    scanpy's sklearn backend leaves tied neighbours in whatever order `argsort` gives,
    which is not a rule we can or should reproduce. What we *can* guarantee is that the
    set we pick is a genuine kNN — the multiset of distances is the ground truth's —
    and that the choice is deterministic.
    """
    side = 8
    lattice = np.array([[i, j] for i in range(side) for j in range(side)], dtype=np.float32)
    indices, distances = our_knn(lattice, K)
    _, exact_distances = exact_knn(lattice, K)
    np.testing.assert_allclose(distances, exact_distances, atol=1e-6)

    for cell in range(len(lattice)):
        row_indices, row_distances = indices[cell], distances[cell]
        for a, b in zip(range(K - 1), range(1, K), strict=True):
            if np.isclose(row_distances[a], row_distances[b], atol=1e-6):
                assert row_indices[a] < row_indices[b], f"cell {cell}, positions {a},{b}"

    # And the graph built from it is still a valid fuzzy simplicial set.
    assert_same_weights(
        our_connectivities(indices, distances), umap_reference(indices, distances)[0]
    )


# 7. The distance expansion.


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e4, 1e5])
def test_distances_survive_an_embedding_far_from_the_origin(offset: float) -> None:
    """`|a|^2 + |b|^2 - 2 a.b` cancels catastrophically on un-centred coordinates.

    At an offset of 1e4 with unit spread every squared distance underflows to a
    negative, is clamped to zero, and the graph silently degenerates to all-ones
    connectivities. Centring the embedding first costs one pass and makes the
    resolvable separation a fraction of the cloud's radius rather than of its distance
    to the origin.
    """
    x = (np.random.default_rng(2).normal(size=(300, 12)) + offset).astype(np.float32)
    indices, distances = our_knn(x, K)
    expected_indices, expected_distances = exact_knn(x, K)

    for cell in range(x.shape[0]):
        assert set(indices[cell]) == set(expected_indices[cell]), f"cell {cell}"
    np.testing.assert_allclose(
        distances, expected_distances, rtol=1e-4, atol=1e-4 * max(1.0, float(offset) * 1e-4)
    )
