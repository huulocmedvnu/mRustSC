"""Line-by-line audit of `crates/scrust-core/src/pca.rs` against its references.

The references are, in order of authority:

* `sklearn/utils/extmath.py::_randomized_svd` / `_randomized_range_finder` — the
  algorithm,
* `sklearn/decomposition/_pca.py::PCA._fit_truncated` — what a PCA reports,
* `sklearn/decomposition/_truncated_svd.py::TruncatedSVD.fit_transform` — what an
  *uncentred* PCA reports, which is a different statistic,
* `scanpy/preprocessing/_pca/__init__.py::pca` — which of those two is used, and
  where the results are stored.

Every test here is written to fail on a *specific* divergence, not to confirm
agreement on the components the data determines strongly. Leading components of a
well-separated spectrum agree between almost any two SVD implementations; that
agreement proves very little, so nothing here rests on it.

ONE TEST IN THIS FILE IS EXPECTED TO FAIL:
`test_trailing_singular_values_survive_an_ill_conditioned_spectrum`. It documents
a real, quantified defect that cannot be fixed without replacing the final Gram
eigendecomposition. See its docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

# The audit runs in a git worktree whose freshly built extension lives beside this
# checkout, while the installed `scrust` .pth still points at the primary one.
# Prefer this tree so the tests exercise the code being audited.
_LOCAL_PYTHON = Path(__file__).resolve().parents[1] / "python"
if (_LOCAL_PYTHON / "scrust").is_dir():
    sys.path.insert(0, str(_LOCAL_PYTHON))

from scrust_call import scrust_call  # noqa: E402

sc = pytest.importorskip("scanpy")
sklearn_pca = pytest.importorskip("sklearn.decomposition")
extmath = pytest.importorskip("sklearn.utils.extmath")


# --------------------------------------------------------------------------- #
# fixtures: matrices whose exact spectrum we control
# --------------------------------------------------------------------------- #


def _orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def known_spectrum(
    *, n_cells: int = 400, n_genes: int = 300, n_comps: int = 20, ratio: float = 1e4
) -> tuple[np.ndarray, np.ndarray]:
    """A matrix with a known, slowly decaying spectrum.

    The first `n_comps` singular values are geometric from 100 down to
    `100 / ratio`; the tail keeps decaying so nothing is degenerate. Returned as
    float32 because that is the precision the Rust core works in — the point is to
    compare two float32 algorithms, not to blame float32.
    """
    rng = np.random.default_rng(0)
    rank = min(n_cells, n_genes)
    s = np.empty(rank)
    s[:n_comps] = 100.0 * np.exp(-np.log(ratio) * np.arange(n_comps) / (n_comps - 1))
    s[n_comps:] = s[n_comps - 1] * 0.9 ** np.arange(1, rank - n_comps + 1)
    u = _orthogonal(n_cells, rng)[:, :rank]
    v = _orthogonal(n_genes, rng)[:, :rank]
    x = ((u * s) @ v.T).astype(np.float32)
    return x, s


def degenerate_counts(n_cells: int = 300, n_genes: int = 200) -> np.ndarray:
    """Uniform sparse counts: adjacent singular values differ by well under 1%.

    This is what the noise components of real single-cell data look like, and it is
    the regime that decides how much oversampling is worth.
    """
    rng = np.random.default_rng(0)
    dense = (rng.random((n_cells, n_genes)) < 0.4) * rng.integers(1, 20, (n_cells, n_genes))
    return dense.astype(np.float32)


def run_scrust(x: np.ndarray, n_comps: int, *, zero_center: bool = True) -> AnnData:
    adata = AnnData(sp.csr_matrix(x))
    scrust_call("pp.pca", adata, n_comps=n_comps, zero_center=zero_center, random_state=0)
    return adata


def exact_singular_values(x: np.ndarray) -> np.ndarray:
    centred = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    return np.linalg.svd(centred, compute_uv=False)


def reported_singular_values(adata: AnnData) -> np.ndarray:
    """Invert `explained_variance_ = S**2 / (n - 1)` to recover S."""
    variance = np.asarray(adata.uns["pca"]["variance"], dtype=np.float64)
    return np.sqrt(np.maximum(variance, 0.0) * (adata.n_obs - 1))


def abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


# --------------------------------------------------------------------------- #
# 1. oversampling and iteration count
# --------------------------------------------------------------------------- #


def test_oversampling_is_earned_on_a_degenerate_spectrum():
    """`OVERSAMPLING = 24` must beat scikit-learn's default of 10, or go back to 10.

    A wider sketch than the reference is only defensible if it buys something. It
    does — but only where the spectrum is degenerate, and the gain is a property of
    randomised SVD rather than of this implementation: scikit-learn shows the same
    jump when its own `n_oversamples` is raised from 10 to 24.

    If this test ever fails, the constant has stopped paying for itself and should
    be dropped to 10 to match `randomized_svd`.
    """
    x = degenerate_counts()
    n_comps = 10
    dense = x.astype(np.float64)
    centred = dense - dense.mean(axis=0)

    reference = sklearn_pca.PCA(n_components=n_comps, svd_solver="arpack", random_state=0)
    reference_scores = reference.fit_transform(dense)

    ours = run_scrust(x, n_comps).obsm["X_pca"]

    sklearn_default = extmath.randomized_svd(centred, n_comps, n_oversamples=10, random_state=0)[0]

    worst_ours = min(abs_correlation(reference_scores[:, i], ours[:, i]) for i in range(n_comps))
    worst_sklearn_10 = min(
        abs_correlation(reference_scores[:, i], sklearn_default[:, i]) for i in range(n_comps)
    )

    # A degenerate spectrum is exactly where a narrow sketch collapses: at
    # n_oversamples=10 scikit-learn's tenth component correlates ~0.37 with arpack.
    assert worst_sklearn_10 < 0.95, (
        "the fixture is no longer degenerate enough to justify extra oversampling; "
        f"sklearn at n_oversamples=10 already reaches {worst_sklearn_10:.4f}"
    )
    assert worst_ours >= 0.99, f"worst component correlation {worst_ours:.4f}"


def test_iteration_count_follows_the_sklearn_rule_in_both_branches():
    """`n_iter = 7 if n_components < 0.1 * min(shape) else 4`.

    The count is not observable from Python, but its *effect* is: with the rule in
    force both branches must reach an exact SVD, whereas the textbook two
    iterations would not. `n_comps=8` of `min(shape)=300` takes the 7-iteration
    branch, `n_comps=60` takes the 4-iteration branch.
    """
    x, _ = known_spectrum(ratio=1e2)
    exact = exact_singular_values(x)
    for n_comps in (8, 60):
        ours = reported_singular_values(run_scrust(x, n_comps))
        error = np.max(np.abs(ours - exact[:n_comps]) / exact[:n_comps])
        assert error < 1e-3, f"n_comps={n_comps}: worst relative error {error:.2e}"


# --------------------------------------------------------------------------- #
# 2. the power iteration's re-orthonormalisation
# --------------------------------------------------------------------------- #


def test_range_finder_does_not_lose_rank():
    """The whitening normaliser must not delete basis directions.

    `whiten` inverts the square root of an f32 Gram matrix. Directions whose Gram
    eigenvalue falls below `RANK_TOLERANCE` used to be replaced by a zero column,
    and a zero column stays zero through every later power iteration — so the
    sketch quietly lost rank and the trailing components came back as *exact
    zeros*. On this matrix the basis collapsed from 44 columns to 16, at both
    oversampling 10 and 24, and no amount of oversampling recovered it.

    scikit-learn's LU (and QR) normalisers never do this: they are not rank
    revealing, so every direction survives however small it has become.
    """
    x, _ = known_spectrum(ratio=1e4)
    n_comps = 20
    ours = reported_singular_values(run_scrust(x, n_comps))
    exact = exact_singular_values(x)[:n_comps]

    assert np.all(ours > 0), f"components returned as exact zero: {ours}"
    # Nothing may be off by so much that it is not the right singular value at all.
    error = np.abs(ours - exact) / exact
    assert np.max(error) < 1e-2, f"worst relative error {np.max(error):.2e}\n{error}"


# --------------------------------------------------------------------------- #
# 3. the final decomposition   ***THIS TEST IS EXPECTED TO FAIL***
# --------------------------------------------------------------------------- #


def test_trailing_singular_values_survive_an_ill_conditioned_spectrum():
    """KNOWN FAILURE — the final Gram eigendecomposition halves the usable digits.

    scikit-learn forms `B = Q^T X` and takes an exact SVD of that small matrix
    (`extmath.py`, `Uhat, s, Vt = linalg.svd(B, full_matrices=False)`). We instead
    take a Jacobi eigendecomposition of `B B^T` (`pca.rs`, `let gram =
    b_transpose.t()?...matmul(&b_transpose)?`). Forming the Gram squares the
    condition number, so the relative error in `sigma_i` grows like
    `eps * (sigma_1 / sigma_i)**2` instead of `eps * (sigma_1 / sigma_i)`.

    Measured on this matrix (`sigma_1 / sigma_20 = 1e4`), both algorithms in
    float32:

        component   ours        scikit-learn
        0-8         ~1e-7       ~2e-7        (indistinguishable)
        9-16        1e-5 .. 4e-4 ~5e-7       (ours is 20x-500x worse)

    Nothing about the range finder explains this: it persists with a QR-normalised
    basis, and it is the *only* remaining gap once the rank loss of point 2 is
    fixed.

    The fix is not small. `B^T` is `(n_genes, sketch_width)`, so an exact SVD of it
    means either a one-sided Jacobi SVD applied to `B^T` directly — O(n_genes *
    width**2) per sweep, in f64, on the CPU, which gives up the GPU for the final
    step — or accumulating the Gram in f64, which candle cannot do on Metal at all.
    Left failing on purpose rather than loosened into a pass.
    """
    x, _ = known_spectrum(ratio=1e4)
    n_comps = 20
    exact = exact_singular_values(x)[:n_comps]
    centred = (x - x.mean(axis=0)).astype(np.float32)

    ours = reported_singular_values(run_scrust(x, n_comps))
    theirs = extmath.randomized_svd(centred, n_comps, random_state=0)[1]

    our_error = np.abs(ours - exact) / exact
    their_error = np.abs(theirs - exact) / exact

    report = "\n".join(
        f"  {i:3d}  sigma={exact[i]:10.5f}  ours={our_error[i]:8.2e}  sklearn={their_error[i]:8.2e}"
        for i in range(n_comps)
    )
    assert np.max(their_error) < 1e-5, f"the reference itself is inaccurate here\n{report}"
    assert np.max(our_error) < 1e-5, (
        "trailing singular values depart from the exact SVD where scikit-learn's "
        f"do not (see the docstring)\n{report}"
    )


# --------------------------------------------------------------------------- #
# 4. centring as a rank-one correction
# --------------------------------------------------------------------------- #


def test_implicit_centring_equals_explicit_centring():
    """`X_centred @ W == X @ W - ones @ (mean^T @ W)`, both directions.

    Running with `zero_center=True` on `X` must give what running with
    `zero_center=False` on an already-centred `X` gives: the first exercises the
    rank-one correction in both `times` (used for the initial sketch and every even
    power step) and `transpose_times` (every odd power step), the second exercises
    neither. If either correction had the wrong orientation or the wrong reduction
    axis, the two would diverge past f32 noise.

    The variances are comparable because for an already-centred matrix
    `sum(x**2) == sum((x - mean)**2)`, so the two denominators coincide -- except
    for the deliberate ddof difference, which is why only the *embeddings* and the
    ratios are compared here.
    """
    x = degenerate_counts()
    centred = (x - x.mean(axis=0)).astype(np.float32)
    n_comps = 8

    implicit = run_scrust(x, n_comps, zero_center=True)
    explicit = run_scrust(centred, n_comps, zero_center=False)

    scale = np.max(np.abs(implicit.obsm["X_pca"]))
    difference = np.max(np.abs(implicit.obsm["X_pca"] - explicit.obsm["X_pca"])) / scale
    assert difference < 1e-4, f"implicit and explicit centring differ by {difference:.2e}"

    assert np.allclose(
        implicit.uns["pca"]["variance_ratio"],
        explicit.uns["pca"]["variance_ratio"],
        rtol=2e-4,
    )


def test_loadings_are_right_singular_vectors_of_the_centred_matrix():
    """A one-sided check that the transposed correction is right.

    `transpose_times` is the only place the mean enters as `mean^T @ colsums(L)`,
    and it is what the *loadings* are built from. So verify the loadings directly:
    they must be orthonormal and must diagonalise the centred gene covariance.

    Uses the decaying spectrum rather than the degenerate one on purpose: where
    adjacent eigenvalues differ by under a percent, *any* randomised SVD mixes the
    eigenvectors and the residual is ~1e-2 for scikit-learn too, so the check would
    say nothing about centring.
    """
    x, _ = known_spectrum(ratio=1e2, n_comps=10)
    n_comps = 8
    adata = run_scrust(x, n_comps)
    loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)  # (n_genes, n_comps)
    dense = x.astype(np.float64)
    centred = dense - dense.mean(axis=0)

    gram = loadings.T @ loadings
    assert np.allclose(gram, np.eye(n_comps), atol=1e-4), "loadings are not orthonormal"

    covariance_action = centred.T @ (centred @ loadings)
    eigenvalues = np.asarray(adata.uns["pca"]["variance"], dtype=np.float64) * (adata.n_obs - 1)
    residual = covariance_action - loadings * eigenvalues
    relative = np.linalg.norm(residual) / np.linalg.norm(covariance_action)
    assert relative < 1e-4, f"loadings are not eigenvectors of X_c^T X_c: {relative:.2e}"


# --------------------------------------------------------------------------- #
# 5. what scanpy stores
# --------------------------------------------------------------------------- #


def test_stored_keys_and_orientation_match_scanpy():
    x = degenerate_counts()
    n_comps = 6
    ours = run_scrust(x, n_comps)
    theirs = AnnData(sp.csr_matrix(x))
    sc.pp.pca(theirs, n_comps=n_comps, svd_solver="arpack", random_state=0)

    assert ours.obsm["X_pca"].shape == theirs.obsm["X_pca"].shape == (x.shape[0], n_comps)
    # scanpy stores `pca_.components_.T`, i.e. (n_vars, n_comps), NOT (n_comps, n_vars).
    assert ours.varm["PCs"].shape == theirs.varm["PCs"].shape == (x.shape[1], n_comps)
    for key in ("variance", "variance_ratio", "params"):
        assert key in ours.uns["pca"], key
    assert ours.uns["pca"]["variance"].shape == (n_comps,)
    assert ours.uns["pca"]["variance_ratio"].shape == (n_comps,)
    assert ours.uns["pca"]["params"]["zero_center"] is True


def test_variance_ratio_denominator_is_the_total_variance_of_all_genes():
    """`explained_variance_ratio_ = explained_variance_ / total_var`.

    `total_var` is the variance of *every* gene with ddof=1 — scikit-learn's sparse
    path computes it as `mean_variance_axis(X, axis=0)[1].sum() * n / (n - 1)` — not
    the sum of the retained variances, and not the second moment.
    """
    x = degenerate_counts()
    n_comps = 6
    ours = run_scrust(x, n_comps)
    total = np.var(x.astype(np.float64), axis=0, ddof=1).sum()

    variance = np.asarray(ours.uns["pca"]["variance"], dtype=np.float64)
    ratio = np.asarray(ours.uns["pca"]["variance_ratio"], dtype=np.float64)
    assert np.allclose(ratio, variance / total, rtol=1e-4), (
        f"implied total {np.mean(variance / ratio):.6f} vs all-gene variance {total:.6f}"
    )
    assert ratio.sum() < 1.0

    theirs = AnnData(sp.csr_matrix(x))
    sc.pp.pca(theirs, n_comps=n_comps, svd_solver="arpack", random_state=0)
    assert np.allclose(ratio, theirs.uns["pca"]["variance_ratio"], rtol=2e-3)


def test_uncentred_reports_truncated_svd_statistics_not_pca_statistics():
    """`zero_center=False` sends scanpy to `TruncatedSVD`, which reports differently.

    `TruncatedSVD` sets `explained_variance_ = np.var(X @ V.T, axis=0)` — ddof=0,
    and with the mean of each score column removed — over
    `full_var = np.var(X, axis=0).sum()`, also ddof=0. It does *not* report
    `S**2 / (n - 1)` over the total second moment: doing that makes the first
    component, the one carrying the grand mean, read as by far the largest, when
    scanpy reports it as one of the smallest.
    """
    x = degenerate_counts()
    n_comps = 8
    ours = run_scrust(x, n_comps, zero_center=False)
    theirs = AnnData(sp.csr_matrix(x))
    sc.pp.pca(theirs, n_comps=n_comps, zero_center=False, svd_solver="arpack", random_state=0)

    ours_variance = np.asarray(ours.uns["pca"]["variance"], dtype=np.float64)
    ours_ratio = np.asarray(ours.uns["pca"]["variance_ratio"], dtype=np.float64)

    # The mean-carrying component must not dominate.
    assert ours_variance[0] < ours_variance[1]

    # It is the variance of the score column, ddof=0.
    scores = np.asarray(ours.obsm["X_pca"], dtype=np.float64)
    assert np.allclose(ours_variance, np.var(scores, axis=0), rtol=1e-3)

    # The denominator is the ddof=0 total gene variance, not the second moment.
    full_var = np.var(x.astype(np.float64), axis=0).sum()
    second_moment = (x.astype(np.float64) ** 2).sum() / (x.shape[0] - 1)
    assert np.allclose(ours_ratio, ours_variance / full_var, rtol=1e-3)
    assert not np.allclose(ours_ratio, ours_variance / second_moment, rtol=1e-2)

    assert np.allclose(ours_variance, theirs.uns["pca"]["variance"], rtol=1e-2)
    assert np.allclose(ours_ratio, theirs.uns["pca"]["variance_ratio"], rtol=1e-2)


# --------------------------------------------------------------------------- #
# 6. sign convention
# --------------------------------------------------------------------------- #


def test_sign_convention_is_svd_flip_on_the_loadings():
    """scanpy's PCA calls `svd_flip(U, Vt, u_based_decision=False)`.

    Both `_fit_full` and `_fit_truncated` in `_pca.py` pass
    `u_based_decision=False`, and `_fit_truncated` even passes `flip_sign=False`
    into `_randomized_svd` so that `randomized_svd`'s own u-based flip is skipped.
    So the rule is: the largest-magnitude entry of each *row of Vt* — each row of
    loadings — is positive. That is what `fix_component_signs` does, so the rules
    agree; this pins the rule rather than a sampled outcome.
    """
    x = degenerate_counts()
    n_comps = 10
    ours = run_scrust(x, n_comps)
    loadings = np.asarray(ours.varm["PCs"], dtype=np.float64)  # (n_genes, n_comps)
    for i in range(n_comps):
        column = loadings[:, i]
        assert column[np.argmax(np.abs(column))] > 0, f"component {i} violates svd_flip"

    theirs = AnnData(sp.csr_matrix(x))
    sc.pp.pca(theirs, n_comps=n_comps, svd_solver="arpack", random_state=0)
    reference = np.asarray(theirs.varm["PCs"], dtype=np.float64)
    for i in range(n_comps):
        column = reference[:, i]
        assert column[np.argmax(np.abs(column))] > 0


def test_sign_agrees_with_scanpy_where_the_argmax_is_unambiguous():
    """The rule agrees; the *outcome* only agrees when the largest loading is clear.

    `svd_flip` keys off a single argmax, so two implementations that both obey it
    can still disagree when the two largest loadings of a component are within
    float32 noise of each other. On a well-separated spectrum they are not, and the
    signs must match exactly. On degenerate noise components they can and do
    differ — which is a property of `svd_flip`, not of this port, but it is worth
    knowing before comparing two plots.
    """
    x, _ = known_spectrum(ratio=1e2, n_comps=10)
    n_comps = 10
    ours = run_scrust(x, n_comps)
    theirs = AnnData(sp.csr_matrix(x))
    sc.pp.pca(theirs, n_comps=n_comps, svd_solver="arpack", random_state=0)

    for i in range(n_comps):
        a = np.asarray(ours.obsm["X_pca"], dtype=np.float64)[:, i]
        b = np.asarray(theirs.obsm["X_pca"], dtype=np.float64)[:, i]
        correlation = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert correlation > 0.99, f"component {i} is sign-flipped or wrong: {correlation:+.4f}"
