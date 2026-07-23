"""Line-by-line audit of the UMAP layout against umap-learn.

The neighbourhood-preservation check in `test_reference.py` cannot fail usefully:
umap-learn agrees with *itself* across seeds only ~45% of the time on real data, so
any roughly-similar optimiser clears the bar. Everything here is built to fail
instead.

The centrepiece is `test_matches_a_transcription_of_layouts_py`: the module-level
`_reference_layout` is a direct transcription of
`umap/layouts.py::_optimize_layout_euclidean_single_epoch` plus the surrounding
`optimize_layout_euclidean` bookkeeping, and it is run against the compiled core on
the same graph from the same start. With `negative_sample_rate=0` there is no
randomness left, so the comparison is term by term. A second run turns negative
sampling back on and reproduces the tau88 stream as well, which pins the sampling
count, the self-sample rejection and the fact that a negative sample never moves the
vertex it was drawn from.

The transcription is only worth what its fidelity to layouts.py is worth, so
`test_transcribed_schedule_matches_umap_learn` and
`test_transcribed_epochs_per_sample_match_umap_learn` check it against the installed
umap-learn directly.

Known, deliberate divergences are pinned with `xfail(strict=True)` so they announce
themselves the day they are closed; see `test_global_arrangement_needs_spectral_init`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from scrust_call import scrust_call

umap_learn = pytest.importorskip("umap", reason="the audit is against umap-learn")
from umap.layouts import clip as umap_clip  # noqa: E402
from umap.umap_ import find_ab_params, make_epochs_per_sample  # noqa: E402
from umap.utils import tau_rand_int  # noqa: E402

# scanpy's defaults, which are what `scrust.tl.umap` has to reproduce.
MIN_DIST = 0.5
SPREAD = 1.0
LEARNING_RATE = 1.0
NEGATIVE_SAMPLE_RATE = 5

# f32 arithmetic on both sides, reached by different compilers and different `powf`
# implementations, so the comparison is to single-precision accumulation over the run
# rather than to the bit.
ATOL = 2e-3
RTOL = 2e-3

_GOLDEN = 0x9E3779B97F4A7C15
_U64 = (1 << 64) - 1
_U32 = (1 << 32) - 1


# --------------------------------------------------------------------------------
# Transcriptions of the scrust side. Every one of these mirrors a named function in
# `crates/scrust-core/src/umap.rs`; if the Rust changes, the comparison tests fail.
# --------------------------------------------------------------------------------


class TauRng:
    """`umap.rs::TauRng`: tau88, seeded through splitmix64."""

    def __init__(self, seed: int) -> None:
        splitmix = seed & _U64
        words = []
        for _ in range(3):
            splitmix = (splitmix + _GOLDEN) & _U64
            z = splitmix
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
            words.append((z ^ (z >> 31)) & _U32)
        self.state = [words[0] | 0x2, words[1] | 0x8, words[2] | 0x10]

    def next_u32(self) -> int:
        s = self.state
        # Rust's `<<` on u32 truncates, which is the `& 0xFFFFFFFF` umap-learn writes
        # out by hand; it has to happen *inside* the xor, not once at the end.
        s[0] = (((s[0] & 4294967294) << 12) & _U32) ^ ((((s[0] << 13) & _U32) ^ s[0]) >> 19)
        s[1] = (((s[1] & 4294967288) << 4) & _U32) ^ ((((s[1] << 2) & _U32) ^ s[1]) >> 25)
        s[2] = (((s[2] & 4294967280) << 17) & _U32) ^ ((((s[2] << 3) & _U32) ^ s[2]) >> 11)
        return s[0] ^ s[1] ^ s[2]

    def next_unit_f32(self) -> np.float32:
        return np.float32(np.float32(self.next_u32()) / np.float32(_U32))


def scrust_initial_layout(n_cells: int, n_components: int, seed: int) -> np.ndarray:
    """`umap.rs::random_layout` followed by `rescale_to_init_range`.

    umap-learn's own `init="random"` (`umap_.py:1095`) draws uniformly in [-10, 10]
    and `umap_.py:1188` then squashes each column into [0, 10]; only the generator
    differs.
    """
    rng = TauRng(seed)
    flat = np.array(
        [
            np.float32((rng.next_unit_f32() * np.float32(2.0) - np.float32(1.0)) * np.float32(10.0))
            for _ in range(n_cells * n_components)
        ],
        dtype=np.float32,
    )
    embedding = flat.reshape(n_cells, n_components)
    low = embedding.min(0)
    span = embedding.max(0) - low
    span = np.where(span > 0, span, 1.0)
    return (np.float32(10.0) * (embedding - low) / span).astype(np.float32)


def scrust_edge_list(graph: sparse.csr_matrix, n_epochs: int):
    """`umap.rs::EdgeList::from_graph`, in CSR row-major order.

    The order matters: the reference epoch is sequential, so visiting the edges in a
    different order gives a different answer.
    """
    graph = graph.tocsr()
    max_weight = float(graph.data.max())
    threshold = np.float32(max_weight) / np.float32(n_epochs)
    head, tail, eps = [], [], []
    for row in range(graph.shape[0]):
        for entry in range(graph.indptr[row], graph.indptr[row + 1]):
            weight = graph.data[entry]
            if weight <= 0.0 or weight < threshold:
                continue
            head.append(row)
            tail.append(int(graph.indices[entry]))
            eps.append(np.float64(max_weight) / np.float64(weight))
    return np.array(head, np.int64), np.array(tail, np.int64), np.array(eps, np.float64)


def scrust_umap(graph: sparse.csr_matrix, **kwargs):
    """The compiled core, reached past the Python wrapper so every knob is settable."""

    graph = graph.tocsr()
    params = dict(
        n_components=2,
        n_epochs=200,
        min_dist=MIN_DIST,
        spread=SPREAD,
        learning_rate=LEARNING_RATE,
        negative_sample_rate=NEGATIVE_SAMPLE_RATE,
        seed=0,
        device="cpu",
    )
    params.update(kwargs)
    return np.asarray(
        scrust_call(
            "_scrust.umap",
            graph.indptr.astype(np.uint32),
            graph.indices.astype(np.uint32),
            graph.data.astype(np.float32),
            graph.shape[1],
            *params.values(),
        ),
        dtype=np.float32,
    )


# --------------------------------------------------------------------------------
# The reference: umap/layouts.py, transcribed.
# --------------------------------------------------------------------------------


def _reference_layout(
    embedding,
    head,
    tail,
    epochs_per_sample,
    n_epochs,
    a,
    b,
    *,
    n_vertices,
    negative_sample_rate,
    initial_alpha,
    gamma=1.0,
    rng_factory=None,
):
    """`optimize_layout_euclidean` with `move_other=True`, term for term.

    `rng_factory(vertex)` supplies the per-head-vertex generator umap-learn keeps in
    `rng_state_per_sample[j]` (`layouts.py:161`). Passing None disables negative
    sampling outright, which is the deterministic case.
    """
    embedding = np.array(embedding, dtype=np.float32, copy=True)
    dim = embedding.shape[1]
    a, b = np.float32(a), np.float32(b)

    # layouts.py:323-325
    if negative_sample_rate:
        epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
        epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    rngs = None if rng_factory is None else [rng_factory(v) for v in range(n_vertices)]

    alpha = np.float32(initial_alpha)  # layouts.py:321
    for n in range(n_epochs):
        for i in range(len(epochs_per_sample)):
            if epoch_of_next_sample[i] > n:  # layouts.py:93
                continue
            j, k = head[i], tail[i]
            current, other = embedding[j], embedding[k]

            dist_squared = np.float32(((current - other) ** 2).sum())
            if dist_squared > 0.0:  # layouts.py:136-140
                grad_coeff = (
                    np.float32(-2.0) * a * b * np.float32(dist_squared ** (b - np.float32(1.0)))
                )
                grad_coeff /= a * np.float32(dist_squared**b) + np.float32(1.0)
            else:
                grad_coeff = np.float32(0.0)

            for d in range(dim):  # layouts.py:142-152: clip first, alpha after
                grad_d = np.float32(umap_clip(np.float32(grad_coeff * (current[d] - other[d]))))
                current[d] = np.float32(current[d] + grad_d * alpha)
                other[d] = np.float32(other[d] - grad_d * alpha)

            epoch_of_next_sample[i] += epochs_per_sample[i]  # layouts.py:154
            if rngs is None:
                continue

            n_neg_samples = int(  # layouts.py:156
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for _ in range(max(n_neg_samples, 0)):
                k = rngs[j].next_u32() % n_vertices  # layouts.py:161
                other = embedding[k]
                dist_squared = np.float32(((current - other) ** 2).sum())
                if dist_squared > 0.0:  # layouts.py:167-175
                    grad_coeff = np.float32(2.0) * np.float32(gamma) * b
                    grad_coeff /= np.float32(np.float32(0.001) + dist_squared) * (
                        a * np.float32(dist_squared**b) + np.float32(1.0)
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = np.float32(0.0)
                for d in range(dim):  # layouts.py:177-182: the tail is never moved
                    grad_d = (
                        np.float32(umap_clip(np.float32(grad_coeff * (current[d] - other[d]))))
                        if grad_coeff > 0.0
                        else np.float32(0.0)
                    )
                    current[d] = np.float32(current[d] + grad_d * alpha)
            epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i]

        alpha = np.float32(initial_alpha * (1.0 - (float(n) / float(n_epochs))))  # layouts.py:431
    return embedding


# --------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------


def _blob_graph(n_blobs: int = 3, per_blob: int = 8, seed: int = 3) -> sparse.csr_matrix:
    """A weighted symmetric graph with a spread of weights, so the firing schedule is
    non-trivial and the weakest edges fall under the cutoff."""
    rng = np.random.default_rng(seed)
    n = n_blobs * per_blob
    dense = np.zeros((n, n), dtype=np.float32)
    for row in range(n):
        for col in range(row + 1, n):
            same = row // per_blob == col // per_blob
            w = rng.uniform(0.2, 1.0) if same else rng.uniform(0.0, 0.06)
            dense[row, col] = dense[col, row] = np.float32(w)
    return sparse.csr_matrix(dense)


# --------------------------------------------------------------------------------
# 1. The a/b curve fit. The grid comparison lives in the Rust unit test
#    `umap::tests::fitted_curve_matches_umap_learn`; this pins the two values
#    scanpy's own defaults produce, so a regression is visible from Python too.
# --------------------------------------------------------------------------------


def test_scanpy_default_curve_parameters():
    a, b = find_ab_params(SPREAD, MIN_DIST)
    assert (round(a, 5), round(b, 5)) == (0.58303, 1.33417)


# --------------------------------------------------------------------------------
# 3. The epoch schedule.
# --------------------------------------------------------------------------------


def _umap_learn_firing_epochs(eps: float, n_epochs: int) -> list[int]:
    """umap-learn's stateful counter, `layouts.py:93` and `:154`."""
    nxt, fires = eps, []
    for n in range(n_epochs):
        if nxt <= n:
            fires.append(n)
            nxt += eps
    return fires


def _stateless_firing_epochs(eps: float, n_epochs: int, dtype=np.float64) -> list[int]:
    """The rule the Metal kernel uses, `umap_sgd.rs`: fire when floor(epoch / eps)
    increments."""
    eps = dtype(eps)
    out = []
    for n in range(n_epochs):
        now = math.floor(dtype(n) / eps)
        before = 0 if n == 0 else math.floor(dtype(n - 1) / eps)
        if now != before:
            out.append(n)
    return out


def test_transcribed_epochs_per_sample_match_umap_learn():
    """`scrust_edge_list` must produce umap-learn's schedule, or the reference epoch
    below is measuring the wrong thing."""
    graph = _blob_graph()
    n_epochs = 200

    coo = graph.tocoo()
    coo.sum_duplicates()
    coo.data[coo.data < (coo.data.max() / float(n_epochs))] = 0.0
    coo.eliminate_zeros()
    theirs = make_epochs_per_sample(coo.data, n_epochs)

    head, tail, eps = scrust_edge_list(graph, n_epochs)
    ours_edges = list(zip(head.tolist(), tail.tolist(), strict=True))
    their_edges = list(zip(coo.row.tolist(), coo.col.tolist(), strict=True))
    assert ours_edges == their_edges
    np.testing.assert_allclose(eps, theirs, rtol=1e-6)
    assert eps.min() >= 1.0, "epochs_per_sample is max/weight, so it cannot go below 1"


@pytest.mark.parametrize("n_epochs", [10, 200, 500])
def test_stateless_schedule_equals_umap_learns_counter(n_epochs):
    """The kernel drops umap-learn's per-edge counter for a closed form. In exact
    arithmetic they must agree for every schedule, including the sub-unit ones that
    `make_epochs_per_sample` cannot produce but the kernel would still be handed."""
    rng = np.random.default_rng(0)
    schedules = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 99.0]
    schedules += list(rng.uniform(1.0, 60.0, 300))
    for eps in schedules:
        assert _umap_learn_firing_epochs(float(eps), n_epochs) == _stateless_firing_epochs(
            float(eps), n_epochs
        ), f"schedule {eps} fires on different epochs"


def test_the_kernels_f32_schedule_drift_stays_negligible():
    """The kernel evaluates the closed form in f32, so a multiple of the schedule that
    sits within a rounding error of an integer can move a firing by one epoch. This
    bounds how often, and fails if it ever becomes common."""
    rng = np.random.default_rng(1)
    n_epochs = 500
    schedules = np.clip(1.0 / rng.uniform(1e-3, 1.0, 2000), 1.0, n_epochs)
    moved = fired = 0
    for eps in schedules:
        want = set(_umap_learn_firing_epochs(float(eps), n_epochs))
        got = set(_stateless_firing_epochs(float(eps), n_epochs, np.float32))
        moved += len(want ^ got)
        fired += len(want)
    assert moved / fired < 1e-4, f"{moved} of {fired} firings moved"


# --------------------------------------------------------------------------------
# 6. Negative sampling: the tau88 port.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state", [[7, 11, 19], [3, 9, 17], [123456789, 362436069, 521288629], [2**31, 2**32 - 1, 12345]]
)
def test_tau88_port_matches_umap_learns_tau_rand_int(state):
    """`umap.rs::TauRng::next_u32` against `umap.utils.tau_rand_int` word for word.

    umap-learn declares the state `int64` and returns `int32`, so the two agree
    exactly only while the state stays inside [0, 2^32) -- which is where scrust's
    splitmix64 seeding puts it, and is the claim being checked.
    """
    theirs_state = np.array(state, dtype=np.int64)
    ours = TauRng.__new__(TauRng)
    ours.state = list(state)
    for step in range(64):
        theirs = int(tau_rand_int(theirs_state)) & _U32
        assert ours.next_u32() == theirs, f"diverged at step {step} from {state}"
        assert list(theirs_state) == ours.state, f"state diverged at step {step}"


def test_negative_sample_count_follows_umap_learns_counter():
    """umap-learn does not draw a fixed number of negatives per firing: the count is
    whatever the negative-sample clock has accrued, so it is 4 on the first firing and
    varies afterwards for a non-integer schedule, averaging the rate. The CPU core
    reproduces this; the Metal kernel deliberately does not, which is what this
    documents.
    """

    def counts(eps, n_epochs, rate=NEGATIVE_SAMPLE_RATE):
        epn = eps / rate
        nn, nxt, out = epn, eps, []
        for n in range(n_epochs):
            if nxt <= n:
                nxt += eps
                m = int((n - nn) / epn)
                out.append(m)
                nn += m * epn
        return out

    assert counts(1.0, 500)[:3] == [4, 5, 5]
    varied = counts(1.3, 500)
    assert min(varied) < NEGATIVE_SAMPLE_RATE < max(varied)
    assert abs(np.mean(varied) - NEGATIVE_SAMPLE_RATE) < 0.05


# --------------------------------------------------------------------------------
# 2, 4, 5. One run of the real thing against the transcription.
# --------------------------------------------------------------------------------


def test_initial_layout_transcription_matches_the_core():
    """A one-epoch run with nothing to do leaves the initial layout untouched, so the
    core's `random_layout` + `rescale_to_init_range` is observable from here. Every
    comparison below rests on this.
    """
    graph = _blob_graph()
    n_epochs = 200
    # A schedule no edge meets: the weakest surviving edge fires first at ceil(eps).
    _, _, eps = scrust_edge_list(graph, n_epochs)
    assert eps.min() >= 1.0
    ours = scrust_umap(graph, n_epochs=n_epochs, negative_sample_rate=0, seed=11)
    mine = _reference_layout(
        scrust_initial_layout(graph.shape[0], 2, 11),
        *scrust_edge_list(graph, n_epochs)[:2],
        scrust_edge_list(graph, n_epochs)[2],
        n_epochs,
        *find_ab_params(SPREAD, MIN_DIST),
        n_vertices=graph.shape[0],
        negative_sample_rate=0,
        initial_alpha=LEARNING_RATE,
    )
    # Both start from the same layout; this asserts the start, the run is next.
    np.testing.assert_allclose(ours, mine, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_epochs", [1, 2, 17, 200])
@pytest.mark.parametrize("seed", [0, 7])
def test_matches_a_transcription_of_layouts_py(n_epochs, seed):
    """The audit's load-bearing test.

    With `negative_sample_rate=0` there is no randomness after the initial layout, so
    the core and a transcription of `_optimize_layout_euclidean_single_epoch` must
    agree coordinate by coordinate. It fails on any change to the gradient, the clip,
    the order of clip and `alpha`, the decay schedule, whether the tail vertex moves,
    the firing schedule, or the edge cutoff.

    `n_epochs=1` is included on purpose: nothing fires, because umap-learn's counter
    starts at `epochs_per_sample >= 1`, so the run must return the initial layout.
    """
    graph = _blob_graph()
    a, b = find_ab_params(SPREAD, MIN_DIST)
    head, tail, eps = scrust_edge_list(graph, n_epochs)

    ours = scrust_umap(graph, n_epochs=n_epochs, negative_sample_rate=0, seed=seed)
    theirs = _reference_layout(
        scrust_initial_layout(graph.shape[0], 2, seed),
        head,
        tail,
        eps,
        n_epochs,
        a,
        b,
        n_vertices=graph.shape[0],
        negative_sample_rate=0,
        initial_alpha=LEARNING_RATE,
    )
    np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_epochs", [3, 25])
def test_matches_the_transcription_with_negative_sampling(n_epochs):
    """The same comparison with repulsion on.

    The core seeds one tau88 per head vertex from `seed ^ vertex * GOLDEN`, exactly
    the shape of umap-learn's `rng_state_per_sample[j]`, so the draws are
    reproducible here. This is what pins the number of negative samples, the
    rejection of a self-sample, the acceptance of a true neighbour as a negative
    sample, and that repulsion never moves the sampled vertex.
    """
    graph = _blob_graph()
    a, b = find_ab_params(SPREAD, MIN_DIST)
    head, tail, eps = scrust_edge_list(graph, n_epochs)
    seed = 5

    ours = scrust_umap(graph, n_epochs=n_epochs, seed=seed)
    theirs = _reference_layout(
        scrust_initial_layout(graph.shape[0], 2, seed),
        head,
        tail,
        eps,
        n_epochs,
        a,
        b,
        n_vertices=graph.shape[0],
        negative_sample_rate=NEGATIVE_SAMPLE_RATE,
        initial_alpha=LEARNING_RATE,
        rng_factory=lambda v: TauRng(seed ^ ((v * _GOLDEN) & _U64)),
    )
    np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=ATOL)


def test_the_clip_is_applied_before_the_learning_rate():
    """`layouts.py:143,150` clips `grad_coeff * (current[d] - other[d])` and only then
    multiplies by `alpha`, so a large learning rate scales an already-clipped move.
    Clipping the finished move instead would cap it at 4 regardless of `alpha`.

    Two vertices, one edge, one firing epoch: the coefficient is far past the clip, so
    the move is exactly `4 * alpha` per endpoint and the two orderings differ by a
    factor of `alpha`.
    """
    graph = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    alpha = 8.0
    n_epochs = 2  # epoch 0 fires nothing, epoch 1 fires once at the initial alpha
    ours = scrust_umap(
        graph,
        n_epochs=n_epochs,
        negative_sample_rate=0,
        learning_rate=alpha,
        n_components=1,
        seed=4,
    )
    start = scrust_initial_layout(2, 1, 4)
    a, b = find_ab_params(SPREAD, MIN_DIST)

    separation = float(start[0, 0] - start[1, 0])
    d2 = separation * separation
    coefficient = -2.0 * a * b * d2 ** (b - 1.0) / (a * d2**b + 1.0)
    assert abs(coefficient * separation) > 4.0, "the clip has to bite for this to test it"

    clipped_then_scaled = np.clip(coefficient * separation, -4.0, 4.0) * alpha
    scaled_then_clipped = np.clip(coefficient * separation * alpha, -4.0, 4.0)
    assert abs(clipped_then_scaled - scaled_then_clipped) > 1.0

    np.testing.assert_allclose(ours[0, 0], start[0, 0] + clipped_then_scaled, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ours[1, 0], start[1, 0] - clipped_then_scaled, rtol=1e-4, atol=1e-4)


def test_the_learning_rate_decays_after_an_epoch_not_before():
    """`layouts.py:321,431`: `alpha` starts at `initial_alpha` and is updated at the
    *end* of epoch n, so epochs 0 and 1 both run at the full rate. Computing
    `1 - n / n_epochs` up front would run epoch 1 at `1 - 1/n_epochs`.
    """
    graph = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    a, b = find_ab_params(SPREAD, MIN_DIST)
    n_epochs = 4
    ours = scrust_umap(graph, n_epochs=n_epochs, negative_sample_rate=0, n_components=1, seed=4)

    start = scrust_initial_layout(2, 1, 4)
    at_full_rate = _reference_layout(
        start,
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1.0, 1.0]),
        n_epochs,
        a,
        b,
        n_vertices=2,
        negative_sample_rate=0,
        initial_alpha=LEARNING_RATE,
    )
    np.testing.assert_allclose(ours, at_full_rate, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------------
# 2. Initialisation: the one divergence that is not being closed here.
# --------------------------------------------------------------------------------


def _trajectory(n_groups=12, per=100, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_groups)
    basis = rng.normal(size=(3, dim))
    centres = (np.stack([np.cos(4 * t), np.sin(4 * t), 3 * t], 1) @ basis) * 4.0
    x = np.concatenate([c + rng.normal(scale=0.6, size=(per, dim)) for c in centres])
    return x.astype(np.float32), np.repeat(np.arange(n_groups), per)


def _adjacency_of_consecutive_groups(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of consecutive groups that stay among each other's two nearest
    centroids: a measure of global arrangement, not of local structure."""
    groups = np.unique(labels)
    centroids = np.stack([embedding[labels == g].mean(0) for g in groups])
    d = ((centroids[:, None] - centroids[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.argsort(1)[:, :2]
    score = 0.0
    for g in range(len(groups)):
        want = {g - 1, g + 1} & set(range(len(groups)))
        score += len(want & set(nearest[g].tolist())) / len(want)
    return score / len(groups)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="known divergence: the core initialises at random, umap-learn spectrally. "
    "The objective and local structure are unaffected -- see "
    "test_random_init_reaches_the_same_objective -- but the arrangement of clusters "
    "relative to each other is not recovered. Closing this needs a Laplacian "
    "eigensolver; flip to a pass when one lands.",
)
def test_global_arrangement_needs_spectral_init():
    """Clusters strung along a trajectory: does the layout keep their order?

    umap-learn scores ~0.52 with its default spectral init and ~0.15 with
    `init="random"`. The core scores ~0.13, i.e. it behaves exactly like umap-learn
    with random init -- which is what it is.
    """
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set

    x, labels = _trajectory()
    graph, _, _ = fuzzy_simplicial_set(x, 15, check_random_state(0), "euclidean")
    scores = [
        _adjacency_of_consecutive_groups(
            scrust_umap(graph.tocsr(), n_epochs=500, seed=seed), labels
        )
        for seed in (0, 1, 2)
    ]
    assert np.mean(scores) > 0.45, f"consecutive-cluster adjacency {np.mean(scores):.2f}"


@pytest.mark.slow
def test_random_init_reaches_the_same_objective():
    """What random initialisation does *not* cost: the fuzzy-set cross entropy UMAP
    actually minimises, and local structure.

    Measured against umap-learn driven both ways on the same graph, so the only thing
    that varies is the initialisation.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set, simplicial_set_embedding

    x, labels = _trajectory()
    graph, _, _ = fuzzy_simplicial_set(x, 15, check_random_state(0), "euclidean")
    graph = graph.tocoo()
    a, b = find_ab_params(SPREAD, MIN_DIST)

    dense = np.asarray(graph.todense(), dtype=np.float64)
    np.fill_diagonal(dense, 0.0)

    def cross_entropy(embedding):
        e = np.asarray(embedding, dtype=np.float64)
        d2 = ((e[:, None] - e[None]) ** 2).sum(-1)
        w = np.clip(1.0 / (1.0 + a * d2**b), 1e-12, 1.0 - 1e-12)
        term = -(dense * np.log(w) + (1.0 - dense) * np.log(1.0 - w))
        np.fill_diagonal(term, 0.0)
        return float(term.sum())

    def local_purity(embedding):
        idx = (
            NearestNeighbors(n_neighbors=16)
            .fit(embedding)
            .kneighbors(embedding, return_distance=False)[:, 1:]
        )
        return float(np.mean(labels[idx] == labels[:, None]))

    def umap_learn(init, seed):
        embedding, _ = simplicial_set_embedding(
            data=x,
            graph=graph.copy(),
            n_components=2,
            initial_alpha=LEARNING_RATE,
            a=a,
            b=b,
            gamma=1.0,
            negative_sample_rate=NEGATIVE_SAMPLE_RATE,
            n_epochs=500,
            init=init,
            random_state=check_random_state(seed),
            metric="euclidean",
            metric_kwds={},
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=False,
        )
        return np.asarray(embedding)

    spectral = [umap_learn("spectral", s) for s in (0, 1, 2)]
    ours = [scrust_umap(graph.tocsr(), n_epochs=500, seed=s) for s in (0, 1, 2)]

    reference = np.mean([cross_entropy(e) for e in spectral])
    spread = np.std([cross_entropy(e) for e in spectral])
    mine = np.mean([cross_entropy(e) for e in ours])
    assert mine < reference + 10.0 * max(spread, 1.0), (
        f"cross entropy {mine:.0f} against umap-learn's {reference:.0f} +- {spread:.0f}"
    )
    assert np.mean([local_purity(e) for e in ours]) > 0.99
