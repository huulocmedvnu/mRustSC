"""Clustering: what `tl.leiden` and `tl.louvain` must satisfy. Owned by feat/leiden.

Community labels are arbitrary integers, so nothing here compares them element by
element. Three kinds of claim are made instead:

* structural — a graph of disconnected cliques has exactly one right answer;
* objective — modularity is what both algorithms maximise, so it is asserted
  directly, against a hand computed value and between the two algorithms;
* agreement — against `scanpy.tl.leiden` by adjusted Rand index and normalised
  mutual information, measured against the band scanpy reaches against *itself*
  across seeds, because Leiden is a stochastic local search and that band, not
  1.0, is the ceiling.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from conftest import CEILING_FRACTION
from scrust_call import scrust_call

# Seeds per implementation for the agreement measurement. Six gives fifteen
# reference-against-itself pairs, enough for a median that does not swing on one
# unlucky run, and still costs under a second per implementation.
N_SEEDS = 6


def core():
    """The compiled extension, or a skip while the bindings are unregistered."""
    try:
        from scrust import _scrust
    except ImportError as exc:  # pragma: no cover - only without a built wheel
        pytest.skip(f"scrust is not installed: {exc}")
    if not hasattr(_scrust, "leiden"):
        pytest.skip("the clustering bindings are not registered in scrust-py/src/lib.rs yet")
    return _scrust


def csr_args(matrix: sparse.spmatrix) -> tuple:
    csr = sparse.csr_matrix(matrix)
    return (
        csr.indptr.astype(np.uint32),
        csr.indices.astype(np.uint32),
        csr.data.astype(np.float32),
        csr.shape[1],
    )


def graph_adata(connectivities: sparse.spmatrix) -> AnnData:
    """An AnnData carrying nothing but the graph, as `pp.neighbors` leaves it."""
    n_obs = connectivities.shape[0]
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsp["connectivities"] = sparse.csr_matrix(connectivities)
    adata.uns["neighbors"] = {"connectivities_key": "connectivities"}
    return adata


def cliques(n_cliques: int, size: int, *, bridge: float = 0.0) -> sparse.csr_matrix:
    """`n_cliques` complete graphs, optionally joined in a ring by weak edges."""
    n = n_cliques * size
    dense = np.zeros((n, n), dtype=np.float32)
    for clique in range(n_cliques):
        block = slice(clique * size, (clique + 1) * size)
        dense[block, block] = 1.0
    np.fill_diagonal(dense, 0.0)
    if bridge:
        for clique in range(n_cliques):
            a, b = clique * size, ((clique + 1) % n_cliques) * size
            dense[a, b] = dense[b, a] = bridge
    return sparse.csr_matrix(dense)


def labels_of(adata: AnnData, key: str) -> np.ndarray:
    return adata.obs[key].to_numpy().astype(str)


def cluster(adata: AnnData, algorithm: str, **kwargs) -> np.ndarray:
    scrust_call(f"tl.{algorithm}", adata, **kwargs)
    return labels_of(adata, kwargs.get("key_added", algorithm))


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["leiden", "louvain"])
def test_disconnected_cliques_are_recovered_exactly(algorithm: str) -> None:
    """Five disjoint cliques admit exactly one modularity optimum."""
    n_cliques, size = 5, 9
    adata = graph_adata(cliques(n_cliques, size))
    labels = cluster(adata, algorithm)

    assert len(set(labels)) == n_cliques
    for clique in range(n_cliques):
        block = labels[clique * size : (clique + 1) * size]
        assert len(set(block)) == 1, f"clique {clique} was split into {set(block)}"
    # A perfect recovery is the one case where an index of 1.0 is the right bar.
    truth = np.repeat(np.arange(n_cliques), size)
    assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)


@pytest.mark.parametrize("algorithm", ["leiden", "louvain"])
def test_labels_are_a_categorical_of_strings(algorithm: str) -> None:
    """scanpy's plotting reads a categorical of `'0'`, `'1'`, ... and its params."""
    adata = graph_adata(cliques(3, 8))
    scrust_call(f"tl.{algorithm}", adata, 0.8, random_state=7)

    column = adata.obs[algorithm]
    assert str(column.dtype) == "category"
    assert list(column.cat.categories) == [str(i) for i in range(3)]
    assert all(isinstance(value, str) for value in column)

    params = adata.uns[algorithm]["params"]
    assert params["resolution"] == 0.8
    assert params["random_state"] == 7
    if algorithm == "leiden":
        assert params["n_iterations"] == 2


def test_key_added_is_honoured() -> None:
    adata = graph_adata(cliques(3, 8))
    scrust_call("tl.leiden", adata, key_added="clusters")
    assert "clusters" in adata.obs
    assert "clusters" in adata.uns
    assert "leiden" not in adata.obs


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


def barbell() -> sparse.csr_matrix:
    """Two triangles joined by one edge: seven edges, small enough for pencil."""
    dense = np.zeros((6, 6), dtype=np.float32)
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
        dense[a, b] = dense[b, a] = 1.0
    return sparse.csr_matrix(dense)


def test_modularity_matches_a_hand_computed_value() -> None:
    """The barbell has 2m = 14. Split into its two triangles, each side holds three
    edges, of which the bridge is cut, so its internal weight is 2 * 3 = 6 and its
    summed strength is 2 + 2 + 3 = 7. Q = 2 * (6/14 - (7/14)^2)."""
    graph = barbell()
    split = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert core().modularity(*csr_args(graph), split, 1.0) == pytest.approx(
        6 / 7 - 0.5, abs=1e-12
    )

    # One community: every edge is internal, so the two terms cancel exactly.
    one = np.zeros(6, dtype=np.uint32)
    assert core().modularity(*csr_args(graph), one, 1.0) == pytest.approx(0.0, abs=1e-12)

    # Singletons: no internal weight at all, only the null model.
    alone = np.arange(6, dtype=np.uint32)
    strengths = np.array([2, 2, 3, 3, 2, 2], dtype=np.float64)
    assert core().modularity(*csr_args(graph), alone, 1.0) == pytest.approx(
        -((strengths / 14) ** 2).sum(), abs=1e-12
    )

    # resolution scales the null model term and nothing else.
    assert core().modularity(*csr_args(graph), split, 2.0) == pytest.approx(
        6 / 7 - 2 * 0.5, abs=1e-12
    )


def test_leiden_scores_no_worse_than_louvain(neighbored: AnnData) -> None:
    """The refinement pass is meant to buy quality, never cost it."""
    graph = neighbored.obsp["connectivities"]
    worse = []
    for seed in range(N_SEEDS):
        leiden = scrust_call("tl.leiden", neighbored, random_state=seed, key_added="l")
        del leiden
        louvain = scrust_call("tl.louvain", neighbored, random_state=seed, key_added="v")
        del louvain
        q_leiden = neighbored.uns["l"]["modularity"]
        q_louvain = neighbored.uns["v"]["modularity"]
        if q_leiden < q_louvain - 1e-9:
            worse.append((seed, q_leiden, q_louvain))
    del graph
    assert not worse, f"leiden scored below louvain on seeds {worse}"


def test_reported_modularity_is_the_partitions_modularity(neighbored: AnnData) -> None:
    """`uns[key]["modularity"]` must be the score of the labels actually written."""
    labels = cluster(neighbored, "leiden")
    ids = np.unique(labels, return_inverse=True)[1].astype(np.uint32)
    recomputed = core().modularity(*csr_args(neighbored.obsp["connectivities"]), ids, 1.0)
    assert recomputed == pytest.approx(neighbored.uns["leiden"]["modularity"], abs=1e-9)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["leiden", "louvain"])
def test_higher_resolution_never_finds_fewer_communities(
    neighbored: AnnData, algorithm: str
) -> None:
    counts = []
    for resolution in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        labels = cluster(neighbored, algorithm, resolution=resolution, key_added="k")
        counts.append((resolution, len(set(labels))))
    for (low, fewer), (high, more) in itertools.pairwise(counts):
        assert more >= fewer, f"resolution {high} found {more} communities, {low} found {fewer}"


@pytest.mark.parametrize("algorithm", ["leiden", "louvain"])
def test_a_fixed_seed_is_reproducible(neighbored: AnnData, algorithm: str) -> None:
    first = cluster(neighbored, algorithm, random_state=3, key_added="a")
    second = cluster(neighbored, algorithm, random_state=3, key_added="b")
    assert np.array_equal(first, second)
    assert neighbored.uns["a"]["modularity"] == neighbored.uns["b"]["modularity"]


def test_a_different_seed_may_differ_but_scores_comparably(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Leiden is a local search, so a reseeded run may land elsewhere. What must not
    change is the quality: the spread of modularity over seeds is what tells us the
    search is finding the same optimum basin, not wandering."""
    scores = []
    for seed in range(N_SEEDS):
        cluster(neighbored, "leiden", random_state=seed, key_added="s")
        scores.append(neighbored.uns["s"]["modularity"])
    spread = max(scores) - min(scores)
    record_property(f"leiden.{neighbored.uns['dataset_id']}.modularity_spread", round(spread, 5))
    assert min(scores) > 0.0
    assert spread <= 0.05 * abs(max(scores)), f"modularity across seeds ranged over {scores}"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_rejects_degenerate_input() -> None:
    extension = core()
    graph = cliques(2, 4)
    ok = csr_args(graph)

    with pytest.raises(ValueError, match="resolution"):
        extension.leiden(*ok, 0.0, 2, 0, "cpu")
    with pytest.raises(ValueError, match="resolution"):
        extension.louvain(*ok, 0.0, 0, "cpu")
    with pytest.raises(ValueError, match="resolution"):
        extension.modularity(*ok, np.zeros(8, dtype=np.uint32), 0.0)

    empty = csr_args(sparse.csr_matrix((0, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="square graph"):
        extension.leiden(*empty, 1.0, 2, 0, "cpu")

    negative = sparse.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        extension.leiden(*csr_args(negative), 1.0, 2, 0, "cpu")

    with pytest.raises(ValueError, match="labels"):
        extension.modularity(*ok, np.zeros(3, dtype=np.uint32), 1.0)


def test_missing_graph_is_reported() -> None:
    adata = AnnData(np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(KeyError, match="connectivities"):
        scrust_call("tl.leiden", adata)


# --------------------------------------------------------------------------
# Agreement with scanpy
# --------------------------------------------------------------------------


def scanpy_leiden(adata: AnnData, seed: int) -> np.ndarray:
    """`scanpy.tl.leiden` at its defaults, which is `leidenalg` with
    `RBConfigurationVertexPartition` — the objective our core maximises."""
    reference = adata.copy()
    with warnings.catch_warnings():
        # scanpy warns that the igraph backend will become the default; the
        # objective is the same either way and this test pins today's default.
        warnings.simplefilter("ignore", FutureWarning)
        try:
            sc.tl.leiden(reference, key_added="reference", random_state=seed)
        except ImportError as exc:
            pytest.skip(f"scanpy.tl.leiden needs leidenalg: {exc}")
    return labels_of(reference, "reference")


def median_agreement(
    left: list[np.ndarray], right: list[np.ndarray], *, same: bool
) -> tuple[float, float]:
    """Median ARI and NMI over all pairs, or over the distinct pairs when comparing
    a set of runs with itself."""
    pairs = (
        itertools.combinations(range(len(left)), 2)
        if same
        else itertools.product(range(len(left)), range(len(right)))
    )
    scores = [
        (
            adjusted_rand_score(left[i], right[j]),
            normalized_mutual_info_score(left[i], right[j]),
        )
        for i, j in pairs
    ]
    return float(np.median([a for a, _ in scores])), float(np.median([n for _, n in scores]))


@pytest.mark.reference
def test_agrees_with_scanpy_leiden(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Measured against the ceiling scanpy reaches against itself.

    Leiden is a randomised local search over a rugged objective, so two runs of the
    *same* implementation at different seeds disagree substantially: on PBMC 3k
    `leidenalg` reproduces itself to a median ARI of only ~0.72. Demanding more of a
    second implementation than the reference demands of itself would be measuring
    the seed, not the algorithm, so the bar is `CEILING_FRACTION` of that band, the
    same rule the contract already fixes for UMAP. The modularity comparison below
    is the assertion that carries the real weight: it is the quantity both
    implementations optimise, and it is compared without any tolerance for luck.
    """
    dataset = neighbored.uns["dataset_id"]
    ours = [cluster(neighbored, "leiden", random_state=seed, key_added="o") for seed in range(N_SEEDS)]
    our_scores = [
        core().modularity(
            *csr_args(neighbored.obsp["connectivities"]),
            np.unique(labels, return_inverse=True)[1].astype(np.uint32),
            1.0,
        )
        for labels in ours
    ]
    reference = [scanpy_leiden(neighbored, seed) for seed in range(N_SEEDS)]
    reference_scores = [
        core().modularity(
            *csr_args(neighbored.obsp["connectivities"]),
            np.unique(labels, return_inverse=True)[1].astype(np.uint32),
            1.0,
        )
        for labels in reference
    ]

    ceiling_ari, ceiling_nmi = median_agreement(reference, reference, same=True)
    ari, nmi = median_agreement(reference, ours, same=False)
    for name, value in [
        ("ari", ari),
        ("nmi", nmi),
        ("ari_ceiling", ceiling_ari),
        ("nmi_ceiling", ceiling_nmi),
        ("modularity", float(np.median(our_scores))),
        ("modularity_reference", float(np.median(reference_scores))),
    ]:
        record_property(f"leiden.{dataset}.{name}", round(value, 4))
    print(
        f"\nleiden on {dataset}: ARI {ari:.4f} (scanpy against itself {ceiling_ari:.4f}), "
        f"NMI {nmi:.4f} (ceiling {ceiling_nmi:.4f}); "
        f"modularity {np.median(our_scores):.5f} against {np.median(reference_scores):.5f}, "
        f"our range {min(our_scores):.5f}-{max(our_scores):.5f}, "
        f"scanpy's {min(reference_scores):.5f}-{max(reference_scores):.5f}"
    )

    assert ari >= CEILING_FRACTION * ceiling_ari, (
        f"ARI {ari:.4f} against scanpy, which reaches {ceiling_ari:.4f} against itself"
    )
    assert nmi >= CEILING_FRACTION * ceiling_nmi, (
        f"NMI {nmi:.4f} against scanpy, which reaches {ceiling_nmi:.4f} against itself"
    )
    # The objective is not stochastic in the way the labels are: whatever basin a
    # seed lands in, a correct implementation reaches the same quality band.
    assert min(our_scores) >= min(reference_scores) - 1e-3, (
        f"our modularity {min(our_scores):.5f} falls below scanpy's worst seed "
        f"{min(reference_scores):.5f}"
    )
