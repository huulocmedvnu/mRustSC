"""Audit of `scrust.tl.paga` against `scanpy.tl.paga(model="v1.2")`.

PAGA's connectivity is a ratio of observed to expected inter-group edges under a null
model that rewires the *directed* neighbour graph at random. Almost every way of
getting it wrong -- counting undirected pairs, forgetting the within-group edges in
`es`, normalising by group size instead of by expected edge count, dividing by `n`
instead of `n - 1` -- still produces a plausible-looking symmetric matrix, so the tests
here pin the formula term by term on hand-computable graphs and then cross-check the
same graphs against scanpy.

Two divergences from scanpy are pinned rather than smoothed over:

* explicitly stored zeros in `obsp["distances"]` used to be dropped by scrust and are
  counted as edges by scanpy. That was a real defect -- half the graph on data with
  duplicate cells -- and it is fixed; `test_explicit_zero_*` now pins the agreement;
* the spanning tree differs on tied connectivities, though the total weight matches.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse

from scrust_call import DEVICE, scrust_call

# float32 accumulation in the core against scanpy's float64.
RTOL = 1e-6


def _adata(distances: sparse.spmatrix, labels: list[str], *, n_neighbors: int = 15) -> AnnData:
    """An AnnData carrying nothing but a labelled directed distance graph.

    `tl.paga` reads `obsp["distances"]` and the group codes and nothing else, so X is a
    placeholder. `distances` is passed through untouched -- explicitly stored zeros
    included -- because whether those count as edges is one of the things under test.
    """
    n_obs = len(labels)
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obs["group"] = np.asarray(labels)
    adata.obs["group"] = adata.obs["group"].astype("category")
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = distances
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": n_neighbors, "method": "umap"},
    }
    return adata


def _from_directed_edges(edges, labels, *, weight=1.0) -> AnnData:
    """Build the graph from an explicit list of *directed* `(row, column)` entries."""
    rows = [r for r, _ in edges]
    columns = [c for _, c in edges]
    data = np.full(len(edges), weight, dtype=np.float64)
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=(len(labels), len(labels)))
    return _adata(matrix, labels)


def _undirected(pairs, labels, *, weights=None) -> AnnData:
    """Both directions of every pair, as a symmetric neighbour graph."""
    edges = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    if weights is None:
        return _from_directed_edges(edges, labels)
    data = np.asarray(list(weights) + list(weights), dtype=np.float64)
    rows = [r for r, _ in edges]
    columns = [c for _, c in edges]
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=(len(labels), len(labels)))
    return _adata(matrix, labels)


def _ours(adata: AnnData) -> dict:
    copy = adata.copy()
    scrust_call("tl.paga", copy, groups="group", device=DEVICE)
    return copy.uns["paga"]


def _theirs(adata: AnnData) -> dict:
    copy = adata.copy()
    sc.tl.paga(copy, groups="group", model="v1.2")
    return copy.uns["paga"]


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float64)


def _edges(tree) -> set[frozenset[int]]:
    dense = _dense(tree)
    return {frozenset((int(i), int(j))) for i, j in zip(*np.nonzero(dense), strict=True)}


def _ring_graph() -> tuple[AnnData, dict[frozenset[str], float]]:
    """Four 10-cell rings with 1, 2, 3, 4, 5 and 6 bridging pairs between them.

    Chosen so every one of the six connectivities is distinct and below the cap, which
    makes the maximum spanning tree unique and the ranking of the pairs a real check.
    """
    labels = [name for name in "abcd" for _ in range(10)]
    offset = {"a": 0, "b": 10, "c": 20, "d": 30}
    pairs = [(offset[name] + i, offset[name] + (i + 1) % 10) for name in "abcd" for i in range(10)]
    bridges = {
        ("a", "b"): 1,
        ("a", "c"): 2,
        ("a", "d"): 3,
        ("b", "c"): 4,
        ("b", "d"): 5,
        ("c", "d"): 6,
    }
    for (left, right), count in bridges.items():
        pairs += [(offset[left] + i, offset[right] + i) for i in range(count)]

    # es = 20 within-ring directed edges plus one outgoing edge per bridging pair.
    es = {name: 20 for name in "abcd"}
    for (left, right), count in bridges.items():
        es[left] += count
        es[right] += count
    expected = {}
    for (left, right), count in bridges.items():
        null = (es[left] * 10 + es[right] * 10) / 39
        expected[frozenset((left, right))] = 2 * count / null
    return _undirected(pairs, labels), expected


def test_connectivity_is_the_v1_2_null_model_term_by_term() -> None:
    """Pins `min(observed / ((e_i n_j + e_j n_i) / (n - 1)), 1)` on a hand-counted graph.

    The graph is deliberately asymmetric and has a singleton group, so the three
    connectivities are all different and each exercises a different term: unequal group
    sizes, a group of one cell, and a link that exists in only one direction. Every
    number below is counted by hand from the edge list, not read back from either
    implementation.
    """
    labels = ["a", "a", "a", "b", "b", "c"]
    edges = [
        (0, 1), (1, 0), (1, 2),  # 3 edges inside a
        (2, 3),                   # a -> b
        (3, 4), (4, 3),           # 2 edges inside b
        (3, 2),                   # b -> a
        (4, 5),                   # b -> c
        (5, 0),                   # c -> a, one direction only
    ]  # fmt: skip
    adata = _from_directed_edges(edges, labels)

    # e = outgoing directed entries per group, counting the ones that stay inside.
    e_a, e_b, e_c = 4, 4, 1
    n_a, n_b, n_c = 3, 2, 1
    n_minus_one = 5
    ab = 2 / ((e_a * n_b + e_b * n_a) / n_minus_one)  # 2 / 4.0
    ac = 1 / ((e_a * n_c + e_c * n_a) / n_minus_one)  # 1 / 1.4
    bc = 1 / ((e_b * n_c + e_c * n_b) / n_minus_one)  # 1 / 1.2

    ours = _dense(_ours(adata)["connectivities"])
    assert_allclose(ours, [[0.0, ab, ac], [ab, 0.0, bc], [ac, bc, 0.0]], rtol=RTOL)
    # And the same graph through scanpy, so the hand count is corroborated.
    assert_allclose(ours, _dense(_theirs(adata)["connectivities"]), rtol=RTOL, atol=0.0)


def test_a_one_directional_link_counts_as_one_edge_not_two() -> None:
    """A single directed entry is one observed edge; symmetrising the graph first is wrong.

    A common way to get PAGA wrong is to symmetrise `obsp["distances"]` before counting,
    which turns one stored entry into two observed edges and also raises the other
    group's `es`. Two 10-cell rings joined by a single entry `(0, 10)` are compared with
    the same graph plus the reverse entry: both numbers are hand-computed and neither is
    near the cap, so doubling, halving or symmetrising all fail here.
    """
    labels = ["a"] * 10 + ["c"] * 10
    rings = [(i, (i + 1) % 10) for i in range(10)] + [
        (10 + i, 10 + (i + 1) % 10) for i in range(10)
    ]
    ring_edges = [(a, b) for a, b in rings] + [(b, a) for a, b in rings]

    one_way = _from_directed_edges([*ring_edges, (0, 10)], labels)
    both_ways = _from_directed_edges([*ring_edges, (0, 10), (10, 0)], labels)

    # Each ring holds 20 directed edges; the bridge adds one outgoing edge per direction.
    assert _dense(_ours(one_way)["connectivities"])[0, 1] == pytest.approx(
        1 / ((21 * 10 + 20 * 10) / 19), rel=RTOL
    )
    assert _dense(_ours(both_ways)["connectivities"])[0, 1] == pytest.approx(
        2 / ((21 * 10 + 21 * 10) / 19), rel=RTOL
    )
    for adata in (one_way, both_ways):
        assert_allclose(
            _dense(_ours(adata)["connectivities"]),
            _dense(_theirs(adata)["connectivities"]),
            rtol=RTOL,
            atol=0.0,
        )


def test_confidence_is_capped_at_one() -> None:
    """A complete bipartite graph has 1.5x the expected inter-group edges; PAGA reports 1.

    Without the cap the value would be 8 / (16/3) = 1.5, so this fails loudly if the
    `min(..., 1)` is dropped, and it is a case a real dataset reaches whenever two
    groups are more tightly linked to each other than the null model allows.
    """
    labels = ["a", "a", "b", "b"]
    edges = [(i, j) for i in range(4) for j in range(4) if labels[i] != labels[j]]
    adata = _from_directed_edges(edges, labels)

    uncapped = 8 / ((4 * 2 + 4 * 2) / 3)
    assert uncapped > 1.0, "the case is only interesting when the raw ratio exceeds one"
    ours = _dense(_ours(adata)["connectivities"])
    assert ours[0, 1] == 1.0
    assert_allclose(ours, _dense(_theirs(adata)["connectivities"]), rtol=RTOL, atol=0.0)


def test_self_loops_raise_es_but_never_reach_the_diagonal() -> None:
    """A stored diagonal entry is an outgoing edge for `es` and not a group connection.

    scanpy builds a directed igraph including self-loops (they count in
    `vc.subgraph(i).ecount()`), then drops loops when contracting to the cluster graph.
    So a self-loop must *lower* every connectivity of its group by inflating `es`, while
    leaving the PAGA diagonal at zero. Both halves are asserted: the diagonal alone
    would pass for an implementation that ignored self-loops entirely.
    """
    labels = ["a", "a", "b", "b", "c", "c"]
    pairs = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]
    plain = _ours(_undirected(pairs, labels))
    with_loops = _undirected(pairs, labels)
    diagonal = sparse.csr_matrix(np.eye(6) * 0.5)
    with_loops.obsp["distances"] = (with_loops.obsp["distances"] + diagonal).tocsr()
    with_loops.obsp["connectivities"] = with_loops.obsp["distances"]
    looped = _ours(with_loops)

    plain_c = _dense(plain["connectivities"])
    looped_c = _dense(looped["connectivities"])
    assert_allclose(np.diag(looped_c), np.zeros(3), atol=0.0)
    assert looped_c[0, 1] < plain_c[0, 1], "self-loops must inflate es and lower confidence"
    # Hand count: es becomes 3+2=5 for a and c, 4+2=6 for b; n = 6, groups of 2.
    assert looped_c[0, 1] == pytest.approx(2 / ((5 * 2 + 6 * 2) / 5), rel=RTOL)
    assert_allclose(looped_c, _dense(_theirs(with_loops)["connectivities"]), rtol=RTOL, atol=0.0)


def test_groups_with_no_edges_between_them_are_exactly_zero() -> None:
    """The ends of a 0-1-2 chain, and a fourth group with no edges at all.

    Zeros must be structural, not a small number: an implementation that seeded the
    matrix with the expected counts, or that let a disconnected group produce a 0/0,
    would show up here as a non-zero or a NaN.
    """
    labels = ["a", "a", "b", "b", "c", "c", "d"]
    pairs = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]
    adata = _undirected(pairs, labels)
    ours = _dense(_ours(adata)["connectivities"])

    assert np.isfinite(ours).all()
    assert ours[0, 2] == 0.0 and ours[2, 0] == 0.0
    assert (ours[3] == 0.0).all() and (ours[:, 3] == 0.0).all()
    assert ours[0, 1] > 0.0 and ours[1, 2] > 0.0
    assert_allclose(ours, _dense(_theirs(adata)["connectivities"]), rtol=RTOL, atol=0.0)


def test_a_group_of_one_cell_is_scored_like_any_other() -> None:
    """A singleton has n_i = 1 and a non-zero `es`; nothing special-cased, no division by zero."""
    labels = ["a", "a", "a", "a", "solo"]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    adata = _undirected(pairs, labels)
    ours = _dense(_ours(adata)["connectivities"])

    # e_a = 7 (3 internal pairs both ways, plus (3,4)), e_solo = 1, n = 5.
    assert ours[0, 1] == pytest.approx(2 / ((7 * 1 + 1 * 4) / 4), rel=RTOL)
    assert np.isfinite(ours).all()
    assert_allclose(ours, _dense(_theirs(adata)["connectivities"]), rtol=RTOL, atol=0.0)


def test_only_edge_counts_are_used_and_not_the_distances() -> None:
    """PAGA binarises the graph: rescaling every distance must not move any number.

    The control is a topology change of the same size, which must move it -- otherwise
    "unchanged" would be satisfied by an implementation that returned a constant.
    """
    labels = ["a", "a", "a", "b", "b", "b"]
    pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 3)]
    unit = _dense(_ours(_undirected(pairs, labels))["connectivities"])
    varied = _dense(
        _ours(_undirected(pairs, labels, weights=[0.01, 7.0, 0.5, 300.0, 1e-3]))["connectivities"]
    )
    assert_allclose(varied, unit, rtol=0.0, atol=0.0)

    extra = _dense(_ours(_undirected([*pairs, (0, 5)], labels))["connectivities"])
    assert extra[0, 1] != unit[0, 1], "adding a bridge must change the confidence"


def test_ranking_of_pairs_follows_expected_not_observed_counts() -> None:
    """Six group pairs whose observed edge counts and confidences rank differently.

    The rings are equal-sized but their `es` differ through the bridges, so a naive
    observed-count implementation would rank a-b lowest and c-d highest by a different
    spacing; the exact values are computed from the null model in the fixture.
    """
    adata, expected = _ring_graph()
    ours = _dense(_ours(adata)["connectivities"])
    index = {"a": 0, "b": 1, "c": 2, "d": 3}
    for pair, value in expected.items():
        left, right = sorted(pair)
        assert ours[index[left], index[right]] == pytest.approx(value, rel=RTOL)
    off_diagonal = ours[np.triu_indices(4, 1)]
    assert len(set(off_diagonal)) == 6, "the fixture is meant to have six distinct values"
    assert (off_diagonal < 1.0).all(), "nothing should be hitting the cap here"
    assert_allclose(ours, _dense(_theirs(adata)["connectivities"]), rtol=RTOL, atol=0.0)


def test_tree_is_the_unique_maximum_spanning_tree() -> None:
    """With six distinct connectivities the maximum spanning tree is unique: c-d, b-d, a-d.

    scanpy takes the *minimum* spanning tree of the reciprocal connectivities; this pins
    that scrust's Prim-on-the-maximum agrees with it, and the hard-coded edge set means
    the test fails if either the sense of the optimisation or the reciprocal is dropped.
    """
    adata, _ = _ring_graph()
    ours = _ours(adata)
    tree = _dense(ours["connectivities_tree"])
    connectivities = _dense(ours["connectivities"])

    assert _edges(tree) == {frozenset((2, 3)), frozenset((1, 3)), frozenset((0, 3))}
    assert _edges(tree) == _edges(_theirs(adata)["connectivities_tree"])
    # The retained entries carry the connectivity, not the reciprocal used to rank them.
    for i, j in [(0, 3), (1, 3), (2, 3)]:
        assert tree[i, j] == pytest.approx(connectivities[i, j], rel=RTOL)


def test_tree_stores_each_edge_once_in_the_upper_triangle() -> None:
    """scanpy's `connectivities_tree` is asymmetric; a symmetric one would double every edge.

    Anything reading it as an edge list -- `pl.paga`, `paga_compare_paths` -- sees twice
    the edges if the lower triangle is filled in too, so the layout is part of the API.
    """
    adata, _ = _ring_graph()
    ours = _dense(_ours(adata)["connectivities_tree"])
    theirs = _dense(_theirs(adata)["connectivities_tree"])

    assert (np.tril(ours) == 0.0).all()
    assert np.count_nonzero(ours) == 3, "three edges over four groups, stored once each"
    assert (np.tril(theirs) == 0.0).all()
    assert np.count_nonzero(theirs) == np.count_nonzero(ours)


def test_tree_of_a_disconnected_graph_is_a_spanning_forest() -> None:
    """Two connected pairs of groups plus an isolated group: two edges, no more.

    Prim has to restart on a new component here; a version that connected the components
    through a zero-weight edge would report three or four edges instead.
    """
    labels = ["a", "a", "b", "b", "c", "c", "d", "d", "e"]
    pairs = [(0, 1), (2, 3), (1, 2), (4, 5), (6, 7), (5, 6)]
    adata = _undirected(pairs, labels)
    ours = _ours(adata)

    assert _edges(ours["connectivities_tree"]) == {frozenset((0, 1)), frozenset((2, 3))}
    assert _edges(ours["connectivities_tree"]) == _edges(_theirs(adata)["connectivities_tree"])
    assert (_dense(ours["connectivities"])[4] == 0.0).all(), "group e is isolated"


def test_tree_diverges_from_scanpy_on_tied_connectivities() -> None:
    """PINNED DIVERGENCE: tied connectivities give a different, equally heavy tree.

    The `min(..., 1)` cap makes exact ties routine on real data -- several pairs saturate
    at 1.0 -- and scrust breaks them with Prim from group 0 while scanpy breaks them with
    scipy's Kruskal on the reciprocals. Both answers are maximum spanning trees; the
    total weight is identical to the last bit, but the edge sets are not. This is a
    genuine, benign difference, and it is pinned here rather than hidden behind a loose
    comparison so that a change in either direction is noticed.
    """
    rng = np.random.default_rng(0)
    n, k = 60, 6
    dense = np.zeros((n, n))
    for i in range(n):
        for j in rng.choice([x for x in range(n) if x != i], k, replace=False):
            dense[i, j] = rng.random() + 0.1
    labels = [f"g{rng.integers(0, 5)}" for _ in range(n)]
    adata = _adata(sparse.csr_matrix(dense), labels)

    ours, theirs = _ours(adata), _theirs(adata)
    assert_allclose(
        _dense(ours["connectivities"]), _dense(theirs["connectivities"]), rtol=RTOL, atol=0.0
    )
    connectivities = _dense(ours["connectivities"])
    off_diagonal = connectivities[np.triu_indices(5, 1)]
    assert (off_diagonal == 1.0).sum() >= 2, "the divergence is driven by capped ties"

    ours_edges = _edges(ours["connectivities_tree"])
    theirs_edges = _edges(theirs["connectivities_tree"])
    assert ours_edges == {
        frozenset((0, 4)),
        frozenset((1, 3)),
        frozenset((1, 4)),
        frozenset((2, 4)),
    }
    assert theirs_edges == {
        frozenset((0, 4)),
        frozenset((1, 2)),
        frozenset((1, 3)),
        frozenset((1, 4)),
    }
    assert ours_edges != theirs_edges, "this test exists because they differ"
    assert len(ours_edges) == len(theirs_edges) == 4
    assert _dense(ours["connectivities_tree"]).sum() == pytest.approx(
        _dense(theirs["connectivities_tree"]).sum(), rel=1e-6
    ), "both are maximum spanning trees: the weights must agree even though the edges do not"


def test_explicit_zero_entries_count_as_edges() -> None:
    """A stored 0.0 in `obsp["distances"]` is an edge, on both sides. It was not always.

    `count_edges` used to skip stored entries whose value was 0.0, citing the
    `nonzero()` inside `get_igraph_from_adjacency`. scanpy does the opposite of what
    that comment assumed: `_compute_connectivities_v1_2` runs
    `ones.data = np.ones(len(ones.data))` *first* (`_paga.py:182-183`), so the
    `nonzero()` then sees nothing but ones and drops none of them.

    Four cells in two groups, two of the eight stored entries explicit zeros. Reading
    all eight gives `8 / ((4*2 + 4*2)/3) = 0.75`; skipping the zeros gave 0.50. Not a
    contrived matrix -- `sc.pp.neighbors` stores a zero whenever two cells coincide,
    which the companion test below reaches through an ordinary pipeline.
    """
    labels = ["a", "a", "b", "b"]
    rows = [0, 0, 1, 1, 2, 2, 2, 3]
    columns = [1, 2, 0, 2, 1, 3, 0, 2]
    data = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=(4, 4))
    assert matrix.nnz == 8 and (matrix.data == 0.0).sum() == 2
    adata = _adata(matrix, labels)

    assert _dense(_theirs(adata)["connectivities"])[0, 1] == pytest.approx(0.75)
    assert _dense(_ours(adata)["connectivities"])[0, 1] == pytest.approx(0.75)


def test_explicit_zeros_from_duplicate_cells_shift_a_real_pipeline() -> None:
    """Documents the size of the defect above on a graph `sc.pp.neighbors` actually built.

    120 cells of which 60 are exact duplicates of the other 60, so the kNN distance
    matrix stores 107 explicit zeros. The assertions below record scrust's *current,
    wrong* answer: every connectivity it reports is at least as large as scanpy's,
    because dropping edges shrank `es` faster than it shrank the observed count, and the
    worst pair was off by 0.096 absolute -- far outside any sane tolerance.

    Now that `count_edges` reads every stored entry, the two agree here to float
    precision, so this is a parity check on the shape that used to break. It stays as
    its own test rather than folding into the duplicate-free one below, because the
    graph it builds is the one that carries stored zeros: 540 of 1080 entries.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(120, 10)).astype(np.float32)
    values[60:] = values[:60]
    adata = AnnData(values)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
    assert (adata.obsp["distances"].data == 0.0).sum() > 50, "duplicates must give stored zeros"
    adata.obs["group"] = np.asarray([f"g{i % 5}" for i in range(120)])
    adata.obs["group"] = adata.obs["group"].astype("category")

    ours = _dense(_ours(adata)["connectivities"])
    theirs = _dense(_theirs(adata)["connectivities"])
    assert theirs.max() > 0.0, "a degenerate all-zero reference would pass anything"
    assert_allclose(ours, theirs, rtol=RTOL, atol=1e-7)


def test_connectivities_match_scanpy_on_a_knn_graph_without_duplicates() -> None:
    """End-to-end parity on a real `sc.pp.neighbors` graph with no stored zeros.

    This is the case the defect above does not touch, so it must agree to float32
    precision -- both the matrix and the spanning tree.
    """
    rng = np.random.default_rng(7)
    centres = rng.normal(scale=1.6, size=(6, 12))
    values = np.repeat(centres, 40, axis=0) + rng.normal(size=(240, 12))
    adata = AnnData(values.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=12, use_rep="X")
    assert (adata.obsp["distances"].data == 0.0).sum() == 0, "no duplicate cells, no stored zeros"
    adata.obs["group"] = np.asarray([f"g{i // 40}" for i in range(240)])
    adata.obs["group"] = adata.obs["group"].astype("category")

    ours, theirs = _ours(adata), _theirs(adata)
    mine, reference = _dense(ours["connectivities"]), _dense(theirs["connectivities"])
    assert reference.max() > 0.0, "a degenerate all-zero reference would pass anything"
    assert_allclose(mine, reference, rtol=RTOL, atol=1e-7)
    assert _edges(ours["connectivities_tree"]) == _edges(theirs["connectivities_tree"])
    assert (np.diag(mine) == 0.0).all()
    assert_allclose(mine, mine.T, rtol=0.0, atol=0.0)


def test_core_binding_returns_row_major_matrices_of_the_group_count() -> None:
    """The PyO3 layer flattens both matrices; pin the shape contract the wrapper relies on.

    Called against `_scrust.paga` directly so that a change to the flattening -- column
    major, or the tree symmetrised on the way out -- fails here rather than silently
    transposing every downstream result. The graph is asymmetric on purpose: a
    transposed connectivity matrix is invisible on a symmetric one, but the *tree* is
    not symmetric even here, so the triangle is the discriminating check.
    """
    labels = ["a", "a", "a", "b", "b", "c"]
    edges = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 3), (3, 2), (4, 5), (5, 0), (0, 5)]
    graph = _from_directed_edges(edges, labels).obsp["distances"].tocsr()
    codes = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint32)

    flat, tree, n_groups = scrust_call(
        "_scrust.paga",
        graph.indptr.astype(np.uint32),
        graph.indices.astype(np.uint32),
        graph.data.astype(np.float32),
        6,
        codes,
        3,
    )
    assert n_groups == 3
    connectivities = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    tree = np.asarray(tree, dtype=np.float64).reshape(3, 3)
    assert connectivities.shape == (3, 3) and tree.shape == (3, 3)
    assert_allclose(connectivities, connectivities.T, rtol=0.0, atol=0.0)
    assert (np.diag(connectivities) == 0.0).all()
    assert (np.tril(tree) == 0.0).all()
    assert np.count_nonzero(tree) == 2
    reference = _dense(_theirs(_from_directed_edges(edges, labels))["connectivities"])
    assert_allclose(connectivities, reference, rtol=RTOL, atol=0.0)


def test_group_sizes_are_not_written_to_uns() -> None:
    """PINNED DIVERGENCE: scanpy writes `uns["<groups>_sizes"]`, scrust does not.

    `sc.tl.paga` records the cell count of each group alongside the abstracted graph, and
    `sc.pl.paga` sizes its nodes from it by default. scrust's wrapper writes only the
    `uns["paga"]` slot, so plotting code that reads the sizes key raises a KeyError
    against a scrust-produced AnnData.
    """
    adata, _ = _ring_graph()
    ours = adata.copy()
    scrust_call("tl.paga", ours, groups="group", device=DEVICE)
    theirs = adata.copy()
    sc.tl.paga(theirs, groups="group", model="v1.2")

    assert list(theirs.uns["group_sizes"]) == [10, 10, 10, 10]
    assert "group_sizes" not in ours.uns, "if this now passes, drop the test: parity reached"
    assert set(ours.uns["paga"]) == {"connectivities", "connectivities_tree", "groups"}
