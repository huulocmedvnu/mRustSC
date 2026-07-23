"""Line-by-line audit of `scrust`'s t-SNE against `sklearn.manifold._t_sne`.

Every test here compares against scikit-learn's *own code* rather than against a
property: `sklearn.manifold._utils._binary_search_perplexity` is imported
directly, and the gradient is checked against a transcription of
`sklearn.manifold._t_sne._kl_divergence` that is itself pinned to the real
function in `test_the_transcribed_gradient_is_faithful`. A divergence in the
affinities, the symmetrisation, the Student-t kernel, the gradient coefficient,
the `Z` normalisation, the gains, the momentum schedule or the initialisation
therefore fails a test rather than merely shifting a quality metric.

The trick that makes the optimiser observable through the binding: at iteration
zero `update` is zero, so `update * grad < 0` is false everywhere and every gain
decays once to `0.8`. The first step is exactly

    p1 = p0 - learning_rate * 0.8 * grad(p0)

so running the same input for `n_iterations=0` and `n_iterations=1` recovers one
exact gradient evaluation. The same identity holds at the phase boundary,
*because* scikit-learn restarts `update` and `gains` there — which is what
`test_the_phase_switch_restarts_momentum_and_gains` exploits.

These tests call the compiled extension directly, because `scrust.tl.tsne` fixes
`n_iterations` at 1000 and `n_components` at 2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from scrust_call import DEVICE

sklearn_tsne = pytest.importorskip("sklearn.manifold._t_sne")
sklearn_utils = pytest.importorskip("sklearn.manifold._utils")
from sklearn.metrics.pairwise import pairwise_distances  # noqa: E402

MACHINE_EPSILON = sklearn_tsne.MACHINE_EPSILON
binary_search_perplexity = sklearn_utils._binary_search_perplexity

PERPLEXITY = 30.0
EARLY_EXAGGERATION = 12.0
LEARNING_RATE = 200.0
# The gain every coordinate carries after a single step from a standing start.
FIRST_STEP_GAIN = 0.8

# Both implementations are f32 where scikit-learn is f64, so nothing is expected
# to agree to better than single precision; the divergences this file exists to
# catch are all four or more orders of magnitude larger than this.
GRADIENT_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def extension():
    """The compiled core, or a skip. Must be the build from this worktree."""
    module = pytest.importorskip("scrust.tl._embedding")
    try:
        return module._extension()
    except Exception as exc:
        pytest.skip(f"the scrust extension is unavailable: {exc}")


def gaussian(n_samples: int, n_features: int, seed: int, offset: float = 0.0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_samples, n_features)) + offset
    return np.ascontiguousarray(values.astype(np.float32))


# --------------------------------------------------------------------------- #
# References transcribed from scikit-learn.
# --------------------------------------------------------------------------- #


def sklearn_joint_probabilities(x, perplexity):
    """`_joint_probabilities`, sklearn/manifold/_t_sne.py:38-68, verbatim.

    Returned condensed, holding one entry per unordered pair, which is the same
    number `scrust` stores at both `(i, j)` and `(j, i)` of its dense matrix.
    """
    distances = pairwise_distances(x, squared=True).astype(np.float32)
    conditional_p = binary_search_perplexity(distances, perplexity, 0)
    p = conditional_p + conditional_p.T
    sum_p = np.maximum(np.sum(p), MACHINE_EPSILON)
    return np.maximum(squareform(p) / sum_p, MACHINE_EPSILON)


def sklearn_kl_gradient(layout, joint, degrees_of_freedom, n_samples, n_components):
    """`_kl_divergence`, sklearn/manifold/_t_sne.py:174-202, verbatim but for the
    `skip_num_points` and `compute_error` branches, which scanpy never uses."""
    embedded = layout.reshape(n_samples, n_components)

    dist = pdist(embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    grad = np.ndarray((n_samples, n_components), dtype=np.float64)
    pqd = squareform((joint - q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(pqd[i], order="K"), embedded[i] - embedded)
    coefficient = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    return grad.ravel() * coefficient


def scrust_binary_search_perplexity(distances, perplexity):
    """`conditional_affinities`, crates/scrust-core/src/tsne.rs:189-247, in the
    same f32 arithmetic, including the sequential accumulation of the two row
    sums (`np.cumsum` in f32 is what the Rust `for` loop does; `np.sum` is not,
    it reduces pairwise and is more accurate).

    scikit-learn does this in f64 and says so at sklearn/manifold/_utils.pyx:63.
    """
    f32 = np.float32
    epsilon = f32(1e-8)  # EPSILON_DBL, _utils.pyx:9
    tolerance = f32(1e-5)  # PERPLEXITY_TOLERANCE, _utils.pyx:10
    n_samples = distances.shape[0]
    desired_entropy = f32(np.log(f32(perplexity)))
    result = np.zeros_like(distances)

    def sequential_sum(values):
        return np.cumsum(values, dtype=np.float32)[-1]

    for i in range(n_samples):
        row = distances[i]
        beta = f32(1.0)
        beta_min = f32(-np.inf)
        beta_max = f32(np.inf)
        weights = np.zeros(n_samples, dtype=np.float32)
        for _ in range(100):  # n_steps, _utils.pyx:43
            weights = np.exp((-row * beta).astype(np.float32)).astype(np.float32)
            weights[i] = f32(0.0)
            mass = sequential_sum(weights)
            if mass == 0.0:
                mass = epsilon
            weights = (weights / mass).astype(np.float32)
            expected = sequential_sum((row * weights).astype(np.float32))
            entropy = f32(f32(np.log(mass)) + f32(beta * expected))
            excess = f32(entropy - desired_entropy)
            if abs(excess) <= tolerance:
                break
            if excess > 0.0:
                beta_min = beta
                beta = f32(beta * 2.0) if np.isinf(beta_max) else f32((beta + beta_max) / 2)
            else:
                beta_max = beta
                beta = f32(beta / 2.0) if np.isinf(beta_min) else f32((beta + beta_min) / 2)
        result[i] = weights
    return result


def realised_perplexity(rows):
    out = []
    for row in rows:
        positive = row[row > 0].astype(np.float64)
        out.append(np.exp(-(positive * np.log(positive)).sum()))
    return np.array(out)


# --------------------------------------------------------------------------- #
# The transcriptions above are only worth as much as their fidelity.
# --------------------------------------------------------------------------- #


def test_the_transcribed_gradient_is_faithful():
    """Pin `sklearn_kl_gradient` to scikit-learn's real `_kl_divergence`."""
    x = gaussian(120, 8, seed=41)
    joint = sklearn_joint_probabilities(x, PERPLEXITY)
    rng = np.random.default_rng(1)
    layout = rng.standard_normal(120 * 2) * 1e-4

    _, reference = sklearn_tsne._kl_divergence(layout, joint, 1, 120, 2)
    mine = sklearn_kl_gradient(layout, joint, 1, 120, 2)
    np.testing.assert_allclose(mine, reference, rtol=0, atol=1e-18)


# --------------------------------------------------------------------------- #
# A1: the perplexity binary search.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n_samples", "n_features", "perplexity", "seed"),
    [(150, 8, 30.0, 7), (150, 8, 5.0, 8), (150, 20, 45.0, 9)],
)
def test_perplexity_search_matches_sklearn(n_samples, n_features, perplexity, seed):
    """Tolerance, step cap, bracket widening and the `EPSILON_DBL` mass floor.

    The bar is scikit-learn's own accuracy: the search stops at an entropy
    tolerance of 1e-5, which is a relative perplexity error of 1e-5, so the
    reference itself misses the target by `3e-4` at perplexity 30. `scrust` runs
    the identical recurrence in f32 and must stay within an order of that.
    """
    x = gaussian(n_samples, n_features, seed)
    distances = pairwise_distances(x, squared=True).astype(np.float32)

    reference = np.asarray(binary_search_perplexity(distances, perplexity, 0))
    ours = scrust_binary_search_perplexity(distances, perplexity)

    theirs_error = np.abs(realised_perplexity(reference) - perplexity).max()
    ours_error = np.abs(realised_perplexity(ours) - perplexity).max()
    assert ours_error < 10 * max(theirs_error, 1e-6), (
        f"worst perplexity error {ours_error:.3e} against scikit-learn's {theirs_error:.3e}"
    )
    assert np.abs(ours - reference).max() < 1e-4


# --------------------------------------------------------------------------- #
# A2-A4: one gradient step, end to end through the binding.
# --------------------------------------------------------------------------- #


def one_gradient_step(extension, x, n_components=2):
    """The gradient `scrust` computed at its own starting point, recovered from
    two runs that differ by a single iteration."""
    n_samples = x.shape[0]
    args = (n_components, PERPLEXITY, EARLY_EXAGGERATION, LEARNING_RATE)
    start = extension.tsne(x, *args, 0, 0, DEVICE).astype(np.float64)
    stepped = extension.tsne(x, *args, 1, 0, DEVICE).astype(np.float64)
    gradient = (start - stepped).ravel() / (LEARNING_RATE * FIRST_STEP_GAIN)
    assert start.shape == (n_samples, n_components)
    return start, gradient


def worst_relative(actual, expected):
    return np.abs(actual - expected).max() / np.abs(expected).max()


@pytest.mark.parametrize(("n_samples", "n_features", "seed"), [(200, 10, 1), (300, 25, 2)])
def test_one_gradient_step_matches_sklearn(extension, n_samples, n_features, seed):
    """The whole per-iteration computation against scikit-learn's.

    This is the load-bearing test of the file. It closes over the perplexity
    search, `P = (P + P^T) / 2n`, the Student-t weights, the `Z` that normalises
    `Q`, the factor of `2 (v + 1) / v`, the early exaggeration applied to `P`
    alone, and the first gain update — any one of them wrong moves the answer by
    far more than single precision.
    """
    x = gaussian(n_samples, n_features, seed)
    start, ours = one_gradient_step(extension, x)

    joint = sklearn_joint_probabilities(x, PERPLEXITY) * EARLY_EXAGGERATION
    reference = sklearn_kl_gradient(start.ravel(), joint, 1, n_samples, 2)

    deviation = worst_relative(ours, reference)
    assert deviation < GRADIENT_TOLERANCE, f"worst relative gradient deviation {deviation:.3e}"


def test_the_gradient_survives_an_uncentred_input(extension):
    """`|x|^2 + |y|^2 - 2 x.y` cancels catastrophically in f32 on offset data.

    scikit-learn upcasts f32 input to f64 for the whole product
    (`_euclidean_distances_upcast`, sklearn/metrics/pairwise.py:403-406). Without
    an equivalent this deviates by 14 per cent, not by 3e-6.
    """
    x = gaussian(200, 10, seed=1, offset=500.0)
    start, ours = one_gradient_step(extension, x)

    joint = sklearn_joint_probabilities(x, PERPLEXITY) * EARLY_EXAGGERATION
    reference = sklearn_kl_gradient(start.ravel(), joint, 1, 200, 2)

    deviation = worst_relative(ours, reference)
    assert deviation < GRADIENT_TOLERANCE, f"worst relative gradient deviation {deviation:.3e}"


@pytest.mark.parametrize("n_components", [1, 2, 3, 4])
def test_degrees_of_freedom_follow_n_components(extension, n_components):
    """`degrees_of_freedom = max(n_components - 1, 1)`, sklearn/manifold/_t_sne.py:1024.

    It sets both the exponent of the Student-t kernel and the gradient
    coefficient, so hard-coding one degree of freedom is 25 per cent wrong in
    three dimensions and 33 per cent in four.
    """
    n_samples = 200
    x = gaussian(n_samples, 10, seed=9)
    start, ours = one_gradient_step(extension, x, n_components)

    joint = sklearn_joint_probabilities(x, PERPLEXITY) * EARLY_EXAGGERATION
    dof = max(n_components - 1, 1)
    reference = sklearn_kl_gradient(start.ravel(), joint, dof, n_samples, n_components)

    deviation = worst_relative(ours, reference)
    assert deviation < GRADIENT_TOLERANCE, f"worst relative gradient deviation {deviation:.3e}"


# --------------------------------------------------------------------------- #
# A4: the optimiser's phase schedule.
# --------------------------------------------------------------------------- #


EXPLORATION_ITERATIONS = 250  # TSNE._EXPLORATION_MAX_ITER, _t_sne.py:805


def test_the_phase_switch_restarts_momentum_and_gains(extension):
    """`_tsne` calls `_gradient_descent` twice (_t_sne.py:1078 and 1094), and each
    call rebuilds `update` and `gains` from scratch (_t_sne.py:388-389).

    Iteration 250 is therefore another standing start: no momentum carried over,
    every gain back at one and decaying once to 0.8, and the exaggeration gone.
    """
    n_samples = 200
    x = gaussian(n_samples, 10, seed=5)
    args = (2, PERPLEXITY, EARLY_EXAGGERATION, LEARNING_RATE)
    before = extension.tsne(x, *args, EXPLORATION_ITERATIONS, 0, DEVICE).astype(np.float64)
    after = extension.tsne(x, *args, EXPLORATION_ITERATIONS + 1, 0, DEVICE).astype(np.float64)
    step = (before - after).ravel()

    joint = sklearn_joint_probabilities(x, PERPLEXITY)
    restarted = (
        LEARNING_RATE
        * FIRST_STEP_GAIN
        * sklearn_kl_gradient(before.ravel(), joint, 1, n_samples, 2)
    )
    deviation = np.abs(step - restarted).max() / np.abs(step).max()
    assert deviation < GRADIENT_TOLERANCE, (
        f"iteration {EXPLORATION_ITERATIONS} is not a restarted, unexaggerated step: "
        f"deviation {deviation:.3e}"
    )

    # The same identity must fail in the middle of the second phase, or the test
    # above would pass on an optimiser that had no momentum at all.
    mid = extension.tsne(x, *args, 300, 0, DEVICE).astype(np.float64)
    mid_next = extension.tsne(x, *args, 301, 0, DEVICE).astype(np.float64)
    mid_step = (mid - mid_next).ravel()
    naive = (
        LEARNING_RATE * FIRST_STEP_GAIN * sklearn_kl_gradient(mid.ravel(), joint, 1, n_samples, 2)
    )
    assert np.abs(mid_step - naive).max() / np.abs(mid_step).max() > 0.1


def test_early_exaggeration_covers_exactly_the_first_250_iterations(extension):
    """Iteration 249 is still exaggerated; iteration 250 is not."""
    n_samples = 200
    x = gaussian(n_samples, 10, seed=5)
    args = (2, PERPLEXITY, EARLY_EXAGGERATION, LEARNING_RATE)
    joint = sklearn_joint_probabilities(x, PERPLEXITY)

    before = extension.tsne(x, *args, EXPLORATION_ITERATIONS, 0, DEVICE).astype(np.float64)
    plain = sklearn_kl_gradient(before.ravel(), joint, 1, n_samples, 2)
    exaggerated = sklearn_kl_gradient(before.ravel(), joint * EARLY_EXAGGERATION, 1, n_samples, 2)
    after = extension.tsne(x, *args, EXPLORATION_ITERATIONS + 1, 0, DEVICE).astype(np.float64)
    step = (before - after).ravel()

    plain_deviation = worst_relative(step, LEARNING_RATE * FIRST_STEP_GAIN * plain)
    exaggerated_deviation = worst_relative(step, LEARNING_RATE * FIRST_STEP_GAIN * exaggerated)
    assert plain_deviation < GRADIENT_TOLERANCE < exaggerated_deviation


# --------------------------------------------------------------------------- #
# A5: the initialisation.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_components", [2, 3])
def test_pca_initialisation_matches_sklearn(extension, n_components):
    """`init="pca"`, scaled so the *first* component has standard deviation 1e-4
    and every other component keeps its relative size (_t_sne.py:1002-1012).

    Compared up to a per-column sign, which is arbitrary in any SVD and which the
    optimiser is equivariant under: flipping an axis of the layout flips that
    axis of the gradient, so both trajectories mirror.
    """
    from sklearn.decomposition import PCA

    x = gaussian(200, 10, seed=13)
    args = (n_components, PERPLEXITY, EARLY_EXAGGERATION, LEARNING_RATE)
    ours = extension.tsne(x, *args, 0, 0, DEVICE).astype(np.float64)

    reference = PCA(n_components=n_components, random_state=0).fit_transform(x)
    reference = reference / np.std(reference[:, 0]) * 1e-4

    assert np.std(ours[:, 0]) == pytest.approx(1e-4, rel=1e-3)
    for column in range(n_components):
        theirs = reference[:, column]
        mine = ours[:, column]
        sign = np.sign(np.dot(mine, theirs))
        np.testing.assert_allclose(mine * sign, theirs, rtol=0, atol=2e-9)


# --------------------------------------------------------------------------- #
# The precondition on perplexity.
# --------------------------------------------------------------------------- #


def test_accepts_every_input_scikit_learn_accepts(extension):
    """scikit-learn's only precondition is `perplexity < n_samples`
    (`_check_params_vs_input`, sklearn/manifold/_t_sne.py:845-850), and scanpy adds
    none of its own.

    `scrust` used to refuse `n_cells < 3 * perplexity` as well, so a 60-cell
    subcluster at the default perplexity of 30 raised where `sc.tl.tsne` returns a
    layout. The one-third rule is advice about reading a t-SNE, not a precondition
    of the algorithm, and the comment carrying it credited scanpy, which has no
    such rule. The guard is now scikit-learn's.
    """
    x = gaussian(60, 10, seed=17)
    layout = extension.tsne(x, 2, PERPLEXITY, EARLY_EXAGGERATION, LEARNING_RATE, 250, 0, DEVICE)
    assert layout.shape == (60, 2)
    assert np.isfinite(layout).all()


def test_rejects_perplexity_at_or_above_the_cell_count(extension):
    """The boundary scikit-learn draws: `perplexity < n_samples`, so equality raises.

    Below it the bandwidth search still has a target it can reach; at or above it
    the requested neighbourhood is the whole data set and there is none.
    """
    x = gaussian(20, 5, seed=3)
    for perplexity in (20.0, 25.0):
        with pytest.raises(ValueError, match="perplexity"):
            extension.tsne(x, 2, perplexity, EARLY_EXAGGERATION, LEARNING_RATE, 250, 0, DEVICE)

    layout = extension.tsne(x, 2, 19.0, EARLY_EXAGGERATION, LEARNING_RATE, 250, 0, DEVICE)
    assert layout.shape == (20, 2)
    assert np.isfinite(layout).all()
