"""PAGA: the abstracted graph over a partition, against `scanpy.tl.paga` with `model="v1.2"`.

The connectivity of two groups is a ratio of observed to expected edges between them,
so the tests that matter are the ones a naive observed-only count would fail: unequal
group sizes, and the real neighbour graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse

from scrust_call import scrust_call

# The contract's element-wise bar for this algorithm.
CONNECTIVITY_RTOL = 1e-4
N_NEIGHBORS = 15


def _adata_from_edges(
    edges: Iterable[tuple[int, int]], labels: list[str], *, n_neighbors: int = N_NEIGHBORS
) -> AnnData:
    """An AnnData carrying nothing but a labelled, undirected neighbour graph.

    PAGA reads `obsp["distances"]` and the group labels, so the expression matrix is a
    placeholder. Every pair is stored in both directions, matching what `pp.neighbors`
    produces for mutual neighbours.
    """
    n_obs = len(labels)
    rows, columns = [], []
    for source, target in edges:
        rows += [source, target]
        columns += [target, source]
    # Distances are arbitrary but must not be zero: scanpy binarises the graph by
    # `nonzero()`, so a stored zero would not be an edge on either side.
    distances = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)), shape=(n_obs, n_obs)
    )
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


def _dense(matrix: object) -> np.ndarray:
    return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float64)


def _edge_set(tree: object) -> set[frozenset[int]]:
    """The tree's edges as unordered pairs; scanpy stores each one in one triangle only."""
    dense = _dense(tree)
    return {frozenset((int(i), int(j))) for i, j in zip(*np.nonzero(dense), strict=True)}


def _run_both(adata: AnnData) -> tuple[dict, dict]:
    ours = adata.copy()
    scrust_call("tl.paga", ours, groups="group")
    theirs = adata.copy()
    sc.tl.paga(theirs, groups="group", model="v1.2")
    return ours.uns["paga"], theirs.uns["paga"]


def test_paga_matches_scanpy(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Element-wise on the connectivities, and the same spanning tree edges."""
    dataset = neighbored.uns["dataset_id"]
    ours, theirs = _run_both(neighbored)

    mine, reference = _dense(ours["connectivities"]), _dense(theirs["connectivities"])
    deviation = np.abs(mine - reference) / np.where(reference != 0, np.abs(reference), 1.0)
    record_property(f"paga.{dataset}.max_relative_deviation", f"{deviation.max():.3e}")
    print(f"\npaga on {dataset}: largest relative deviation {deviation.max():.3e}")

    assert sparse.issparse(ours["connectivities"])
    assert ours["connectivities"].dtype == reference.dtype
    assert ours["groups"] == theirs["groups"] == "group"
    assert_allclose(mine, reference, rtol=CONNECTIVITY_RTOL, atol=0.0)
    assert _edge_set(ours["connectivities_tree"]) == _edge_set(theirs["connectivities_tree"])


def test_chain_of_three_groups() -> None:
    """The hand-checkable case: 0-1-2, with the ends unconnected.

    Two cells per group, each linked to its group mate, and one bridging pair per
    neighbouring pair of groups. Every group then holds 3 or 4 directed edges over
    2 cells, so with n = 6 the expected count between two neighbours is
    (3*2 + 4*2)/5 = 2.8 against the 2 edges observed.
    """
    labels = ["a", "a", "b", "b", "c", "c"]
    edges = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]
    adata = _adata_from_edges(edges, labels)
    scrust_call("tl.paga", adata, groups="group")

    connectivities = _dense(adata.uns["paga"]["connectivities"])
    assert_allclose(
        connectivities,
        [[0.0, 2 / 2.8, 0.0], [2 / 2.8, 0.0, 2 / 2.8], [0.0, 2 / 2.8, 0.0]],
        rtol=CONNECTIVITY_RTOL,
    )
    assert _edge_set(adata.uns["paga"]["connectivities_tree"]) == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }


def test_unequal_group_sizes_normalise_by_expected_edges() -> None:
    """Sizes chosen so that observed counts and connectivities rank the pairs oppositely.

    A 60-cell ring, a 20-cell ring and a 6-cell ring, bridged by 6 and 2 pairs. Counting
    stored entries by row: the groups hold 128, 46 and 14 directed edges over 60, 20 and
    6 cells, with n = 86. The big-mid pair then has 12 observed edges against
    (128*20 + 46*60)/85 expected, and the big-small pair 4 against (128*6 + 14*60)/85 —
    three times the edges for a *lower* confidence, which an unnormalised count inverts.
    """
    big, mid, small = 60, 20, 6
    labels = ["big"] * big + ["mid"] * mid + ["small"] * small
    rings = [range(0, big), range(big, big + mid), range(big + mid, big + mid + small)]
    edges = [
        (nodes[i], nodes[(i + 1) % len(nodes)])
        for nodes in map(list, rings)
        for i in range(len(nodes))
    ]
    edges += [(i, big + i) for i in range(6)]  # big-mid bridges
    edges += [(i, big + mid + i) for i in range(2)]  # big-small bridges

    adata = _adata_from_edges(edges, labels)
    ours, theirs = _run_both(adata)

    # Categories are sorted, so the rows are big, mid, small.
    connectivities = _dense(ours["connectivities"])
    big_mid, big_small = 12 / ((128 * 20 + 46 * big) / 85), 4 / ((128 * small + 14 * big) / 85)
    assert big_small > big_mid, "the case is only interesting if the ranking inverts"
    assert_allclose(connectivities[0, 1], big_mid, rtol=CONNECTIVITY_RTOL)
    assert_allclose(connectivities[0, 2], big_small, rtol=CONNECTIVITY_RTOL)
    assert_allclose(
        connectivities, _dense(theirs["connectivities"]), rtol=CONNECTIVITY_RTOL, atol=0.0
    )
    assert _edge_set(ours["connectivities_tree"]) == _edge_set(theirs["connectivities_tree"])


def test_rejects_a_label_out_of_range() -> None:
    """The core is given raw labels, so it cannot assume the wrapper produced them."""
    graph = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="groups"):
        scrust_call(
            "_scrust.paga",
            graph.indptr.astype(np.uint32),
            graph.indices.astype(np.uint32),
            graph.data.astype(np.float32),
            2,
            np.array([0, 7], dtype=np.uint32),
            2,
        )


def test_rejects_a_group_with_no_cells() -> None:
    """An unused category still counts towards `n_groups` and has no expected edges."""
    adata = _adata_from_edges([(0, 1), (2, 3), (1, 2)], ["a", "a", "b", "b"])
    adata.obs["group"] = adata.obs["group"].cat.add_categories(["ghost"])
    with pytest.raises(ValueError, match="no cells"):
        scrust_call("tl.paga", adata, groups="group")


def test_rejects_an_empty_graph() -> None:
    adata = _adata_from_edges([], [])
    with pytest.raises(ValueError, match="empty graph"):
        scrust_call("tl.paga", adata, groups="group")
