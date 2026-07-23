"""Line-by-line audit of `crates/scrust-core/src/cluster.rs` against the reference.

These tests are the falsifiable half of an audit written against three references:

* ``leidenalg`` 0.12.0, whose optimiser is the C++ ``libleidenalg``
  (``src/Optimiser.cpp``, ``src/MutableVertexPartition.cpp``,
  ``src/RBConfigurationVertexPartition.cpp``);
* Traag, Waltman & van Eck 2019, *From Louvain to Leiden*, algorithms 1-3;
* ``scanpy/tools/_leiden.py``, which drives ``leidenalg.find_partition`` with
  ``RBConfigurationVertexPartition`` on a **directed** igraph built from the
  symmetric connectivities.

Each test names the reference line it holds us to. Nothing here compares labels
elementwise — Leiden is a randomised local search — so every claim is either an
identity of the objective, a structural guarantee the algorithm promises, or a
monotonicity that follows from the algorithm only accepting improving moves.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scrust_call import DEVICE

REPO_ROOT = Path(__file__).resolve().parents[1]


def core():
    """The compiled extension, or a skip while the bindings are unregistered."""
    try:
        from scrust import _scrust
    except ImportError as exc:  # pragma: no cover - only without a built wheel
        pytest.skip(f"scrust is not installed: {exc}")
    if not hasattr(_scrust, "leiden"):
        pytest.skip("the clustering bindings are not registered in scrust-py/src/lib.rs yet")
    return _scrust


def csr_args(matrix) -> tuple:
    csr = sparse.csr_matrix(matrix)
    return (
        csr.indptr.astype(np.uint32),
        csr.indices.astype(np.uint32),
        csr.data.astype(np.float32),
        csr.shape[1],
    )


def random_graph(rng: np.random.Generator, n: int, density: float) -> sparse.csr_matrix:
    """A connected weighted graph: G(n, p) plus a path so nothing is isolated."""
    weights = (rng.random((n, n)) * 0.9 + 0.1) * (rng.random((n, n)) < density)
    upper = np.triu(weights, 1)
    dense = upper + upper.T
    for node in range(n - 1):
        if dense[node, node + 1] == 0.0:
            dense[node, node + 1] = dense[node + 1, node] = 0.5
    return sparse.csr_matrix(dense.astype(np.float32))


def is_connected(graph: sparse.csr_matrix, nodes: np.ndarray) -> bool:
    """Is the subgraph induced on `nodes` connected?"""
    inside = set(int(node) for node in nodes)
    csr = sparse.csr_matrix(graph)
    start = int(nodes[0])
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour in csr.indices[csr.indptr[node] : csr.indptr[node + 1]]:
            neighbour = int(neighbour)
            if neighbour in inside and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == len(inside)


def communities(labels: np.ndarray, n_communities: int):
    for community in range(n_communities):
        nodes = np.flatnonzero(labels == community)
        if nodes.size:
            yield community, nodes


# ---------------------------------------------------------------------------
# 1. The objective
# ---------------------------------------------------------------------------


def igraph_partition(graph: sparse.csr_matrix, labels, resolution: float):
    """`labels` as `scanpy.tl.leiden` would hand them to leidenalg.

    scanpy builds the igraph with ``directed=True`` (``_leiden.py``: ``directed =
    True if directed is None else directed``), so this does too. leidenalg's
    ``RBConfigurationVertexPartition.quality()`` is the *unnormalised* objective:
    ``sum_c (2 m_c - gamma K_c^2 / 2m)``, which is ``2m`` times ours.
    """
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")
    dense = np.asarray(sparse.csr_matrix(graph).todense(), dtype=np.float64)
    sources, targets = dense.nonzero()
    built = ig.Graph(directed=True)
    built.add_vertices(dense.shape[0])
    built.add_edges(list(zip(sources.tolist(), targets.tolist(), strict=True)))
    built.es["weight"] = dense[sources, targets].tolist()
    return la.RBConfigurationVertexPartition(
        built,
        initial_membership=[int(label) for label in labels],
        weights=built.es["weight"],
        resolution_parameter=resolution,
    )


@pytest.mark.parametrize("resolution", [0.25, 1.0, 2.5])
def test_the_objective_is_leidenalgs_rbconfiguration(resolution):
    """Our modularity is leidenalg's `quality()` divided by `2m`, exactly.

    This pins down every part of the null model at once: that the degrees are the
    *weighted* ones, that `2m` is the sum of every matrix entry rather than the
    edge count, and that the resolution multiplies only `k_i k_j / 2m`. Getting
    the factor of two wrong anywhere moves the answer by a factor of two or
    four, far outside the tolerance below.
    """
    extension = core()
    rng = np.random.default_rng(0)
    for _ in range(5):
        graph = random_graph(rng, 24, 0.3)
        labels = rng.integers(0, 5, 24).astype(np.uint32)
        ours = extension.modularity(*csr_args(graph), labels, resolution)
        reference = igraph_partition(graph, labels, resolution)
        # In float64: the weights are float32, but summing 2m in float32 would
        # itself cost more than the agreement being asserted.
        two_m = float(np.asarray(graph.todense(), dtype=np.float64).sum())
        assert ours == pytest.approx(reference.quality() / two_m, abs=1e-9)


def test_the_resolution_multiplies_only_the_null_model():
    """`Q(gamma) = sum_c internal_c / 2m - gamma * sum_c (K_c / 2m)^2`.

    Two resolutions determine the line, so a third must fall on it. A quality
    function that scaled the internal term as well, or that used unweighted
    degrees in the null model, would not be linear in `gamma` with this slope.
    """
    extension = core()
    rng = np.random.default_rng(1)
    graph = random_graph(rng, 30, 0.25)
    labels = rng.integers(0, 4, 30).astype(np.uint32)
    dense = np.asarray(sparse.csr_matrix(graph).todense(), dtype=np.float64)
    two_m = dense.sum()
    strengths = dense.sum(axis=1)
    null = sum((strengths[labels == community].sum() / two_m) ** 2 for community in range(4))
    internal = (
        sum(dense[np.ix_(labels == community, labels == community)].sum() for community in range(4))
        / two_m
    )
    for resolution in (0.1, 1.0, 3.0, 7.5):
        ours = extension.modularity(*csr_args(graph), labels, resolution)
        assert ours == pytest.approx(internal - resolution * null, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. The guarantee: communities are internally connected
# ---------------------------------------------------------------------------


def two_cliques_on_a_hinge() -> sparse.csr_matrix:
    """The shape a Louvain-style algorithm splits badly: two cliques on one node.

    Node 0 is the only link between clique A (1-6) and clique B (7-12). Local
    moving can pull node 0 into B while A still carries the label it inherited
    from node 0, which is how Louvain returns a community whose members do not
    reach each other. Leiden's refinement is precisely the pass that stops this,
    so the partition it returns must have every community connected.
    """
    n = 13
    dense = np.zeros((n, n), dtype=np.float32)
    for block in (range(1, 7), range(7, 13)):
        for a in block:
            for b in block:
                if a != b:
                    dense[a, b] = 1.0
    for hinge in (1, 7):
        dense[0, hinge] = dense[hinge, 0] = 1.0
    return sparse.csr_matrix(dense)


def test_every_leiden_community_is_internally_connected_on_the_hinge():
    """Leiden's headline guarantee, on the graph built to break it."""
    extension = core()
    graph = two_cliques_on_a_hinge()
    for seed in range(12):
        for resolution in (0.5, 1.0, 2.0):
            labels, _, n_communities = extension.leiden(
                *csr_args(graph), resolution, 3, seed, DEVICE
            )
            for community, nodes in communities(labels, n_communities):
                assert is_connected(graph, nodes), (
                    f"seed {seed}, resolution {resolution}: community {community} "
                    f"= {nodes.tolist()} is not connected"
                )


def test_every_leiden_community_is_internally_connected_under_fuzzing():
    """The same guarantee over 3600 random runs, and a bound on the run time.

    Run out of process so that a failure to *terminate* — which is what a
    non-monotone objective in the level loop produces — fails the test instead of
    hanging the suite. Before the free-list fix in `move_nodes` this both
    returned disconnected communities and ran forever on a 37-node graph.
    """
    core()
    script = textwrap.dedent(
        """
        import numpy as np, sys
        from scipy import sparse
        from scrust import _scrust
        sys.path.insert(0, %r)
        from test_cluster_audit import csr_args, random_graph, is_connected, communities
        from scrust_call import DEVICE  # this runs in a fresh interpreter, so it needs its own

        rng = np.random.default_rng(12345)
        bad = []
        for trial in range(200):
            n = int(rng.integers(8, 40))
            graph = random_graph(rng, n, float(rng.uniform(0.08, 0.4)))
            for resolution in (1.0, 2.0, 4.0):
                for seed in range(6):
                    labels, _, k = _scrust.leiden(*csr_args(graph), resolution, 3, seed, DEVICE)
                    for community, nodes in communities(labels, k):
                        if not is_connected(graph, nodes):
                            bad.append((trial, resolution, seed, nodes.tolist()))
        print("DISCONNECTED", len(bad))
        print(bad[:3])
        """
    ) % str(Path(__file__).parent)
    try:
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "leiden did not terminate within 300s on the random-graph fuzz: the "
            "level loop only ends when a pass makes no move, so an objective "
            "that can fall inside move_nodes lets it run forever"
        )
    assert finished.returncode == 0, finished.stderr[-4000:]
    assert "DISCONNECTED 0" in finished.stdout, finished.stdout[-4000:]


def test_a_zero_degree_node_is_never_merged_into_a_community():
    """A node with no edges has zero gain everywhere, so it must stay alone.

    leidenalg only moves a node when the improvement beats `10 * DBL_EPSILON`
    (`Optimiser.cpp:668`), and `diff_move` of an isolated node is identically
    zero, so leidenalg leaves it in its own community. A returned community that
    pairs an isolated node with anything else is both a divergence and a
    disconnected community.
    """
    extension = core()
    rng = np.random.default_rng(3)
    n = 12
    dense = np.asarray(random_graph(rng, n, 0.35).todense(), dtype=np.float32)
    padded = np.zeros((n + 1, n + 1), dtype=np.float32)
    padded[:n, :n] = dense
    graph = sparse.csr_matrix(padded)
    for seed in range(16):
        for resolution in (1.0, 2.0, 4.0):
            labels, _, _ = extension.leiden(*csr_args(graph), resolution, 3, seed, DEVICE)
            alone = labels[n]
            assert (labels == alone).sum() == 1, (
                f"seed {seed}, resolution {resolution}: the isolated node {n} was "
                f"put in community {alone} together with "
                f"{np.flatnonzero(labels == alone).tolist()}"
            )


# ---------------------------------------------------------------------------
# 5. Termination and n_iterations
# ---------------------------------------------------------------------------


def test_more_iterations_never_lower_the_objective():
    """Each pass restarts from the last partition, so quality cannot fall.

    `leidenalg` relies on the same property: `optimise_partition` calls the C++
    routine repeatedly on the partition it already holds (`Optimiser.py`), and
    the negative-`n_iterations` mode stops exactly when a pass fails to improve.
    Since every accepted move strictly improves the objective and aggregation
    preserves it, `Q(k + 1) >= Q(k)`. A move scored against the wrong community
    breaks this.
    """
    extension = core()
    rng = np.random.default_rng(7)
    for _ in range(12):
        graph = random_graph(rng, int(rng.integers(20, 45)), float(rng.uniform(0.1, 0.4)))
        for resolution in (1.0, 2.0, 4.0):
            for seed in range(4):
                previous = None
                for n_iterations in (1, 2, 3, 5):
                    _, quality, _ = extension.leiden(
                        *csr_args(graph), resolution, n_iterations, seed, DEVICE
                    )
                    if previous is not None:
                        assert quality >= previous - 1e-9, (
                            f"resolution {resolution}, seed {seed}: {n_iterations} "
                            f"iterations scored {quality} after {previous}"
                        )
                    previous = quality


# ---------------------------------------------------------------------------
# 3. The refinement phase is randomised, and only merges within a community
# ---------------------------------------------------------------------------


def test_the_refinement_is_randomised():
    """Different seeds must be able to give different partitions.

    A deterministic refinement is not Leiden; both the paper (choose a target
    with probability proportional to `exp(gain / theta)`) and `leidenalg`
    (`refine_consider_comms = RAND_NEIGH_COMM`, `Optimiser.cpp:20`) randomise it.
    Local moving is seeded too, so this alone does not isolate the refinement —
    but a run whose output never varied with the seed would certainly not be
    randomised anywhere.
    """
    extension = core()
    rng = np.random.default_rng(11)
    graph = random_graph(rng, 60, 0.12)
    seen = {
        tuple(extension.leiden(*csr_args(graph), 2.0, 2, seed, DEVICE)[0].tolist())
        for seed in range(16)
    }
    assert len(seen) > 1, "every seed gave the same partition; nothing is randomised"


def test_leiden_never_scores_below_louvain_on_the_hinge():
    """Refinement is not allowed to cost quality on the graph it exists for."""
    extension = core()
    graph = two_cliques_on_a_hinge()
    for seed in range(12):
        leiden = extension.leiden(*csr_args(graph), 1.0, 3, seed, DEVICE)[1]
        louvain = extension.louvain(*csr_args(graph), 1.0, seed, DEVICE)[1]
        assert leiden >= louvain - 1e-9, f"seed {seed}: leiden {leiden} < louvain {louvain}"


# ---------------------------------------------------------------------------
# 4. Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_preserves_the_objective():
    """Collapsing a partition to one node per community must not move `Q`.

    `Graph::collapse_graph` (`GraphHelper.cpp:703`) keeps the total edge weight
    and folds each community's internal weight into a self loop, so the quality
    of the collapsed partition equals the quality of the original — leidenalg
    asserts exactly this in its debug build (`Optimiser.cpp:369`). Here it is
    checked through the public surface: the modularity of a labelling of the
    graph must equal the modularity of the singleton labelling of its collapse.
    """
    extension = core()
    rng = np.random.default_rng(5)
    graph = random_graph(rng, 40, 0.2)
    labels = rng.integers(0, 6, 40).astype(np.uint32)
    dense = np.asarray(sparse.csr_matrix(graph).todense(), dtype=np.float64)
    n_communities = int(labels.max()) + 1
    onehot = np.zeros((40, n_communities))
    onehot[np.arange(40), labels] = 1.0
    collapsed = onehot.T @ dense @ onehot
    before = extension.modularity(*csr_args(graph), labels, 1.0)
    after = extension.modularity(
        *csr_args(sparse.csr_matrix(collapsed.astype(np.float32))),
        np.arange(n_communities, dtype=np.uint32),
        1.0,
    )
    assert before == pytest.approx(after, abs=1e-6)
    assert collapsed.sum() == pytest.approx(dense.sum(), rel=1e-9)
