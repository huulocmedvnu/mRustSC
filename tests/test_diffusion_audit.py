"""Audit of `crates/scrust-core/src/diffusion.rs` against `sc.tl.diffmap` / `sc.tl.dpt`.

Every test here drives scanpy on the *same* input rather than transcribing its code. The
two exceptions are marked in their own docstrings: `_dense_transition_spectrum`, which
needs the spectrum of scanpy's own `transitions_sym` matrix and so calls
`numpy.linalg.eigvalsh` on the matrix scanpy built, and `_extreme_signs`, which is a
one-line summary of an array.

Where scrust and scanpy genuinely differ the divergence is pinned with the size of the
gap, not hidden behind a loose tolerance. Four such divergences are pinned below:

* `tl.dpt` returns 0 where scanpy returns NaN when every cell coincides with the root;
* `tl.dpt` returns a finite pseudotime for cells with no path to the root, where scanpy
  returns `inf` (reachable by feeding a scanpy `X_diffmap` of a disconnected graph into
  `scrust.tl.dpt`, since scrust's own `tl.diffmap` refuses such a graph outright);
* `tl.diffmap` raises for `n_comps >= n_cells` where scanpy silently clamps to
  `n_cells - 1`, and accepts `n_comps <= 2` where scanpy refuses;
* `tl.diffmap` fixes the sign of every component; scanpy leaves it arbitrary.

One outright hole in the Rust is pinned by `test_an_explicit_zero_defeats_the
_connectivity_guard`: the guard that refuses a disconnected graph walks the CSR sparsity
pattern, so a stored-but-zero entry bridging two components is counted as an edge and the
guard passes. scanpy has the same hole, because `scipy.sparse.csgraph.connected_
components` also reads the pattern; it is recorded here rather than fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from numpy.testing import assert_allclose

from scrust_call import DEVICE, scrust_call

# scanpy stores the map as f32 and computes the DPT row sum in f32; scrust accumulates
# the same sum in f64 and rounds once at the end. On every fixture below the two agree to
# a few units in the last f32 place, so the pseudotime comparisons are held at f32 eps
# scaled by the [0, 1] range rather than at some negotiated tolerance.
F32_EPS = 1.2e-7

# scanpy's `Neighbors._get_dpt_row` gives a component with eigenvalue >= 0.9994 weight 1
# instead of `lambda / (1 - lambda)`. The constant is repeated here only to build test
# inputs that straddle it; the *behaviour* at the boundary is always read off scanpy.
SCANPY_STATIONARY_CUTOFF = 0.9994


def _neighbored(graph: sp.csr_matrix) -> AnnData:
    """An AnnData carrying `graph` as its connectivity graph, as `pp.neighbors` leaves it.

    scanpy's `DPT` reads `uns["neighbors"]`, so the key has to be present even though
    nothing in the diffusion map depends on the parameters it records.
    """
    n = graph.shape[0]
    adata = AnnData(np.zeros((n, 1), dtype=np.float32))
    adata.obsp["connectivities"] = graph
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": 5, "method": "umap"},
    }
    return adata


def _irregular_graph(n: int = 60, seed: int = 0) -> sp.csr_matrix:
    """A connected, symmetric, weighted graph whose degrees vary by an order of magnitude.

    The density normalisation `D^-1 W D^-1` is the step a uniform-degree graph cannot
    test: on a regular graph it is a global rescaling and cancels out of the spectrum.
    Here the degrees range over roughly 1 to 20, so dropping the normalisation moves the
    eigenvalues by a lot -- which is what `test_transition_spectrum_...` asserts.
    """
    rng = np.random.default_rng(seed)
    dense = np.zeros((n, n), dtype=np.float32)
    for cell in range(n - 1):  # a spanning path, so the graph is connected by construction
        weight = rng.uniform(0.1, 3.0)
        dense[cell, cell + 1] = dense[cell + 1, cell] = weight
    for _ in range(80):
        left, right = rng.integers(0, n, 2)
        if left != right:
            weight = rng.uniform(0.1, 3.0)
            dense[left, right] = dense[right, left] = weight
    return sp.csr_matrix(dense)


def _path_graph(n: int) -> sp.csr_matrix:
    """`n` cells in a line, both ends carrying a self loop so every degree is equal."""
    dense = np.zeros((n, n), dtype=np.float32)
    for cell in range(n - 1):
        dense[cell, cell + 1] = dense[cell + 1, cell] = 1.0
    dense[0, 0] = dense[n - 1, n - 1] = 1.0
    return sp.csr_matrix(dense)


def _knn_graph(n: int = 300, seed: int = 0) -> AnnData:
    """A real `pp.neighbors` graph over a one-dimensional trajectory of `n` cells.

    A genuine k-NN graph is what `tl.dpt` is for, and its spectrum is flat below the first
    few components -- the regime where a hand-built path graph says nothing.
    """
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n, 20)).astype(np.float32)
    coords[:, 0] += np.linspace(0.0, 8.0, n, dtype=np.float32)
    adata = AnnData(coords)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", random_state=0)
    return adata


def _dense_transition_spectrum(adata: AnnData, *, density_normalize: bool) -> np.ndarray:
    """Eigenvalues of scanpy's own `transitions_sym`, largest magnitude first.

    scanpy builds the matrix (`DPT.compute_transitions`); only the eigensolve is done here
    with `numpy.linalg.eigvalsh`, because scanpy exposes no way to ask for the spectrum of
    the *un*-normalised operator through `tl.diffmap`. This is the one place in the file
    where a reference number does not come out of a scanpy entry point end to end.
    """
    from scanpy.tools._dpt import DPT

    dpt = DPT(adata.copy())
    dpt.compute_transitions(density_normalize=density_normalize)
    matrix = dpt.transitions_sym
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = np.linalg.eigvalsh(np.asarray(dense, dtype=np.float64))
    return values[np.argsort(-np.abs(values))]


def _extreme_signs(embedding: np.ndarray) -> list[float]:
    """The sign of the largest-magnitude entry of each column."""
    columns = np.asarray(embedding, dtype=np.float64).T
    return [float(np.sign(column[np.argmax(np.abs(column))])) for column in columns]


def _with_explicit_zero(graph: sp.csr_matrix, row: int, column: int) -> sp.csr_matrix:
    """`graph` with entry `(row, column)` stored but set to zero."""
    out = graph.copy()
    for slot in range(out.indptr[row], out.indptr[row + 1]):
        if out.indices[slot] == column:
            out.data[slot] = 0.0
            return out
    raise AssertionError(f"({row}, {column}) is not stored, so it cannot be zeroed")


def test_transition_spectrum_matches_scanpy_on_an_irregular_weighted_graph() -> None:
    """The whole `diffmap` pipeline -- Coifman-Lafon normalisation, symmetrisation and
    the largest-magnitude eigenvalue selection -- against `sc.tl.diffmap` on a graph
    where each of those steps changes the answer.

    Two things make this test sharp. First, the degrees are far from uniform, so the
    `D^-1 W D^-1` step is not a no-op: the assertion at the end shows that turning it off
    inside scanpy moves the spectrum by more than 0.05, five hundred times the tolerance
    asserted against scanpy. Second, five of the ten leading eigenvalues by *magnitude*
    are negative here, so an implementation that took the ten algebraically largest
    instead of the ten largest in magnitude -- the `which="LM"` selection `eigsh` makes --
    would return a visibly different list.
    """
    graph = _irregular_graph()
    ours = _neighbored(graph)
    scrust_call("tl.diffmap", ours, n_comps=10, device=DEVICE)
    ours_evals = np.asarray(ours.uns["diffmap_evals"], dtype=np.float64)

    reference = _neighbored(graph)
    sc.tl.diffmap(reference, n_comps=10)
    reference_evals = np.asarray(reference.uns["diffmap_evals"], dtype=np.float64)

    assert (reference_evals < 0).sum() >= 4, (
        "this fixture is meant to have a negative tail inside the leading ten by "
        f"magnitude, got {reference_evals}"
    )
    assert_allclose(ours_evals, reference_evals, rtol=1e-4, err_msg="diffmap eigenvalues")

    # Power: the same ten eigenvalues without the density normalisation are a different
    # list, so matching scanpy above is a statement about the normalisation and not just
    # about "some symmetric operator built from this graph".
    plain = _dense_transition_spectrum(reference, density_normalize=False)[:10]
    normalised = _dense_transition_spectrum(reference, density_normalize=True)[:10]
    assert np.max(np.abs(plain - normalised)) > 0.05, (
        "the fixture no longer distinguishes a density-normalised operator from a plain "
        f"one, so the comparison above has no power: {plain} vs {normalised}"
    )


def test_the_leading_eigenvector_is_the_analytic_stationary_state() -> None:
    """`X_diffmap[:, 0]` must be the square root of the row sums of `D^-1 W D^-1` -- the
    exact stationary state of scanpy's symmetrised operator -- with eigenvalue 1.

    This is the one property of a diffusion map that is analytic rather than numerical:
    `T = Z^-1 (D^-1 W D^-1) Z^-1` with `Z = diag(z)` and `z_i` the square root of row `i`
    of `D^-1 W D^-1`, so `T z = z` exactly. `z` is read off scanpy's own `DPT.Z`, so a
    wrong normalisation on either side breaks the match. On this graph `z` is neither
    constant nor proportional to the degrees, and both of those are asserted, so the
    check cannot pass by accident on a plausible-but-wrong vector.
    """
    from scanpy.tools._dpt import DPT

    graph = _irregular_graph()
    ours = _neighbored(graph)
    scrust_call("tl.diffmap", ours, n_comps=10, device=DEVICE)

    dpt = DPT(_neighbored(graph))
    dpt.compute_transitions()
    z_matrix = dpt.Z.toarray() if sp.issparse(dpt.Z) else np.asarray(dpt.Z)
    z = 1.0 / np.diag(z_matrix)

    def unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        return vector / np.linalg.norm(vector)

    stationary = unit(z)
    degrees = unit(np.asarray(graph.sum(axis=0)).ravel())
    constant = unit(np.ones(graph.shape[0]))
    assert abs(float(stationary @ degrees)) < 0.999, "z coincides with the degrees here"
    assert abs(float(stationary @ constant)) < 0.999, "z is constant here"

    ours_leading = unit(np.asarray(ours.obsm["X_diffmap"])[:, 0])
    assert abs(float(ours_leading @ stationary)) > 1.0 - 1e-6, (
        "the leading diffusion component is not the stationary state "
        f"(|cos| = {abs(float(ours_leading @ stationary))})"
    )
    assert float(ours.uns["diffmap_evals"][0]) == pytest.approx(1.0, abs=1e-5)


def test_dpt_reproduces_scanpy_on_the_same_diffusion_basis() -> None:
    """`tl.dpt` computes Haghverdi Eq. 15 exactly as scanpy does, on a real k-NN graph.

    The eigenvectors of a k-NN diffusion map are only determined where the spectrum is
    not degenerate, so comparing two independently computed maps confuses a pseudotime
    bug with a harmless rotation. This test removes that: it takes *scrust's* map and
    hands the identical `X_diffmap` and `diffmap_evals` to both implementations, so any
    difference is the pseudotime formula and nothing else. It is asserted at f32 epsilon.

    Power: the last assertion recomputes scanpy's pseudotime from the same basis with
    every eigenvalue pushed above the stationary cutoff -- that is, with every weight set
    to 1 instead of `lambda / (1 - lambda)` -- and requires the answer to move by more
    than 0.05. So the `atol` above is small enough to see a missing or wrong weighting.
    """
    adata = _knn_graph()
    ours = adata.copy()
    scrust_call("tl.diffmap", ours, n_comps=15, device=DEVICE)
    basis = np.ascontiguousarray(ours.obsm["X_diffmap"], dtype=np.float32)
    evals = np.ascontiguousarray(ours.uns["diffmap_evals"], dtype=np.float32)

    ours.uns["iroot"] = 0
    scrust_call("tl.dpt", ours, n_dcs=10, device=DEVICE)
    ours_time = np.asarray(ours.obs["dpt_pseudotime"], dtype=np.float64)

    reference = adata.copy()
    reference.obsm["X_diffmap"] = basis
    reference.uns["diffmap_evals"] = evals
    reference.uns["iroot"] = 0
    sc.tl.dpt(reference, n_dcs=10)
    reference_time = np.asarray(reference.obs["dpt_pseudotime"], dtype=np.float64)

    assert_allclose(ours_time, reference_time, rtol=0, atol=F32_EPS, err_msg="dpt pseudotime")

    unweighted = adata.copy()
    unweighted.obsm["X_diffmap"] = basis
    unweighted.uns["diffmap_evals"] = np.full_like(evals, 0.9999)
    unweighted.uns["iroot"] = 0
    sc.tl.dpt(unweighted, n_dcs=10)
    moved = np.max(
        np.abs(np.asarray(unweighted.obs["dpt_pseudotime"], dtype=np.float64) - reference_time)
    )
    assert moved > 0.05, (
        f"dropping the lambda/(1-lambda) weighting only moves pseudotime by {moved:.2e} on "
        "this fixture, so the comparison above would not catch a wrong weighting"
    )


def test_dpt_uses_scanpys_stationary_cutoff_of_0_9994() -> None:
    """The exact constant at which a component switches from weight `lambda/(1-lambda)`
    to weight 1, pinned against scanpy on eigenvalues that straddle it.

    This boundary is worth an exact test because the two branches are not close: at
    `lambda = 0.99939` the weight is about 1638, at `lambda = 0.99941` it is 1. So a
    cutoff off by even 1e-5, or a `>` where scanpy has `>=`, changes the answer by three
    orders of magnitude rather than by a rounding error. The eigenvalues are chosen to
    sit either side of the cutoff *after* the f32 round trip both implementations do.

    Power: the second half re-runs scanpy with the sub-cutoff eigenvalue nudged above it
    and requires the pseudotime to change, so the comparison is known to be sensitive to
    which side of the boundary a component lands on.
    """
    rng = np.random.default_rng(1)
    n = 40
    basis = np.ascontiguousarray(rng.standard_normal((n, 3)), dtype=np.float32)
    below, above = np.float32(0.99939), np.float32(0.99941)
    assert below < SCANPY_STATIONARY_CUTOFF <= above, "the fixture must straddle the cutoff"
    evals = np.ascontiguousarray([1.0, below, above], dtype=np.float32)

    def pseudotime(values: np.ndarray, *, engine: str) -> np.ndarray:
        adata = _neighbored(_path_graph(n))
        adata.obsm["X_diffmap"] = basis.copy()
        adata.uns["diffmap_evals"] = np.ascontiguousarray(values, dtype=np.float32)
        adata.uns["iroot"] = 0
        if engine == "scrust":
            scrust_call("tl.dpt", adata, n_dcs=3, device=DEVICE)
        else:
            sc.tl.dpt(adata, n_dcs=3)
        return np.asarray(adata.obs["dpt_pseudotime"], dtype=np.float64)

    reference = pseudotime(evals, engine="scanpy")
    assert_allclose(
        pseudotime(evals, engine="scrust"),
        reference,
        rtol=0,
        atol=F32_EPS,
        err_msg="pseudotime across the stationary cutoff",
    )

    both_stationary = pseudotime(np.array([1.0, above, above], dtype=np.float32), engine="scanpy")
    assert np.max(np.abs(both_stationary - reference)) > 0.1, (
        "moving the middle eigenvalue across the cutoff barely changes scanpy's answer, "
        "so this test does not actually pin the cutoff"
    )


def test_dpt_ignores_the_sign_of_a_diffusion_component() -> None:
    """Flipping the sign of any column of `X_diffmap` must leave pseudotime unchanged.

    Eq. 15 squares the difference between the root and each cell within a component, so
    the arbitrary sign of an eigenvector cannot reach the answer. This is what makes the
    sign-convention divergence pinned in the next test harmless for `tl.dpt`, and it also
    catches an implementation that took `weight * (root - cell)` without squaring, or
    that summed signed contributions across components.
    """
    adata = _knn_graph(n=200, seed=3)
    scrust_call("tl.diffmap", adata, n_comps=8, device=DEVICE)
    basis = np.ascontiguousarray(adata.obsm["X_diffmap"], dtype=np.float32)
    evals = np.ascontiguousarray(adata.uns["diffmap_evals"], dtype=np.float32)

    def pseudotime(embedding: np.ndarray) -> np.ndarray:
        run = adata.copy()
        run.obsm["X_diffmap"] = np.ascontiguousarray(embedding, dtype=np.float32)
        run.uns["diffmap_evals"] = evals
        run.uns["iroot"] = 7
        scrust_call("tl.dpt", run, n_dcs=8, device=DEVICE)
        return np.asarray(run.obs["dpt_pseudotime"], dtype=np.float64)

    plain = pseudotime(basis)
    assert plain.max() == pytest.approx(1.0), "pseudotime is scaled onto [0, 1]"
    assert plain.std() > 1e-3, "a constant pseudotime would make the flip test vacuous"

    signs = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    assert_allclose(pseudotime(basis * signs), plain, rtol=0, atol=F32_EPS)


def test_component_signs_are_fixed_where_scanpy_leaves_them_arbitrary() -> None:
    """DIVERGENCE. scrust orients every diffusion component so that its largest-magnitude
    entry is positive; scanpy publishes whatever `eigsh` returned.

    This is deliberate in the Rust (`fix_component_signs`) and is invisible to `tl.dpt`,
    which squares the components -- the previous test pins that. It is *not* invisible to
    anyone who compares `obsm["X_diffmap"]` with scanpy's entry by entry, or who reads a
    plot axis direction, so it is recorded rather than absorbed into a tolerance: on this
    fixture scanpy leaves two of six components pointing the other way.
    """
    graph = _path_graph(40)
    ours = _neighbored(graph)
    scrust_call("tl.diffmap", ours, n_comps=6, device=DEVICE)
    reference = _neighbored(graph)
    sc.tl.diffmap(reference, n_comps=6)

    ours_signs = _extreme_signs(ours.obsm["X_diffmap"])
    reference_signs = _extreme_signs(reference.obsm["X_diffmap"])
    assert ours_signs == [1.0] * 6, f"scrust must orient every component, got {ours_signs}"
    assert reference_signs != ours_signs, (
        "scanpy happens to agree on every sign here, so this fixture no longer "
        f"demonstrates the divergence: {reference_signs}"
    )
    # Up to those signs the two maps are the same vectors, which is what makes the
    # difference a convention and not a different subspace. The tolerance is the measured
    # gap between f32 subspace iteration and f64 `eigsh` on this graph -- 6.9e-5 absolute
    # on entries of size 0.16, i.e. 4e-4 relative, inside the module's stated 1e-3 -- and
    # is loose enough only for that, not for a sign or a permutation.
    assert_allclose(
        np.abs(np.asarray(ours.obsm["X_diffmap"], dtype=np.float64)),
        np.abs(np.asarray(reference.obsm["X_diffmap"], dtype=np.float64)),
        rtol=0,
        atol=2e-4,
    )


def test_dpt_returns_zero_where_scanpy_returns_nan_for_a_coincident_basis() -> None:
    """DIVERGENCE. When every cell sits exactly on the root in the components used,
    scrust reports pseudotime 0 for all cells and scanpy reports NaN for all cells.

    scanpy's `_set_pseudotime` divides the distance row by its own maximum; when that
    maximum is exactly 0 the division is 0/0. The Rust guards it (`if farthest > 0.0`) and
    leaves the distances at zero. scrust's answer is the defensible one, but it is a
    difference in output that a caller checking `isnan` would notice, so it is pinned with
    both sides asserted rather than only ours.

    The input is degenerate on purpose: it is exactly the 0-0 case, reachable through the
    public `tl.dpt` because that function consumes whatever `X_diffmap` is stored.
    """
    n = 10
    basis = np.zeros((n, 2), dtype=np.float32)
    evals = np.array([1.0, 0.5], dtype=np.float32)

    ours = _neighbored(_path_graph(n))
    ours.obsm["X_diffmap"] = basis.copy()
    ours.uns["diffmap_evals"] = evals
    ours.uns["iroot"] = 0
    scrust_call("tl.dpt", ours, n_dcs=2, device=DEVICE)
    ours_time = np.asarray(ours.obs["dpt_pseudotime"], dtype=np.float64)

    reference = _neighbored(_path_graph(n))
    reference.obsm["X_diffmap"] = basis.copy()
    reference.uns["diffmap_evals"] = evals
    reference.uns["iroot"] = 0
    sc.tl.dpt(reference, n_dcs=2)
    reference_time = np.asarray(reference.obs["dpt_pseudotime"], dtype=np.float64)

    assert np.all(ours_time == 0.0), f"scrust changed its 0-0 answer: {ours_time}"
    assert np.all(np.isnan(reference_time)), (
        f"scanpy no longer returns NaN here, so the divergence has moved: {reference_time}"
    )


def test_dpt_gives_finite_pseudotime_to_unreachable_cells_where_scanpy_gives_inf() -> None:
    """DIVERGENCE, and the sharper edge of it. On a disconnected graph scanpy marks every
    cell outside the root's component with `inf`; scrust returns an ordinary finite
    number for them, indistinguishable from a genuinely reachable cell.

    scrust's own `tl.diffmap` refuses a disconnected graph outright (asserted below), so
    this cannot arise from a pure-scrust pipeline. It arises from a mixed one: scanpy
    computes the map, scrust computes the pseudotime, which is exactly what
    `scrust.tl.dpt` does when `X_diffmap` is already present. The Rust `dpt` never sees
    the graph and has no way to know, so the fix would have to be in the wrapper; it is
    documented here, not fixed.

    Within the root's own component the two agree to f32 epsilon, so this is purely about
    the unreachable half.
    """
    n = 40
    dense = np.zeros((n, n), dtype=np.float32)
    for cell in range(n - 1):
        if cell == n // 2 - 1:
            continue  # the cut
        dense[cell, cell + 1] = dense[cell + 1, cell] = 1.0
    for end in (0, n // 2 - 1, n // 2, n - 1):
        dense[end, end] = 1.0
    graph = sp.csr_matrix(dense)

    reference = _neighbored(graph)
    sc.tl.diffmap(reference, n_comps=6)
    reference.uns["iroot"] = 0
    sc.tl.dpt(reference, n_dcs=6)
    reference_time = np.asarray(reference.obs["dpt_pseudotime"], dtype=np.float64)
    unreachable = np.isinf(reference_time)
    assert unreachable.sum() == n // 2, (
        f"scanpy marks {unreachable.sum()} cells unreachable, expected {n // 2}"
    )

    ours = _neighbored(graph)
    ours.obsm["X_diffmap"] = np.ascontiguousarray(reference.obsm["X_diffmap"], dtype=np.float32)
    ours.uns["diffmap_evals"] = np.ascontiguousarray(
        reference.uns["diffmap_evals"], dtype=np.float32
    )
    ours.uns["iroot"] = 0
    scrust_call("tl.dpt", ours, n_dcs=6, device=DEVICE)
    ours_time = np.asarray(ours.obs["dpt_pseudotime"], dtype=np.float64)

    assert np.all(np.isfinite(ours_time)), "scrust started reporting inf; update this test"
    assert ours_time[unreachable].max() > 0.5, (
        "the unreachable cells are given a large finite pseudotime, not something a "
        f"caller could spot as a sentinel: {ours_time[unreachable].max()}"
    )
    assert_allclose(
        ours_time[~unreachable],
        reference_time[~unreachable],
        rtol=0,
        atol=F32_EPS,
        err_msg="the root's own component must still match scanpy",
    )

    with pytest.raises(ValueError, match="connected"):
        scrust_call("tl.diffmap", _neighbored(graph), n_comps=6, device=DEVICE)


def test_an_explicit_zero_defeats_the_connectivity_guard() -> None:
    """DEFECT (not fixed). `diffmap`'s refusal of a disconnected graph counts *stored*
    CSR entries as edges, so a pair of explicitly stored zeros bridging two components
    lets the graph through and the map is computed anyway.

    The guard exists, by its own comment in `diffusion.rs`, because a disconnected graph
    makes the leading eigenspace degenerate -- every component contributes its own
    eigenvalue 1 -- and calls that a "silent wrong answer". This test shows the silent
    wrong answer being produced: the returned spectrum starts with *two* eigenvalues equal
    to 1, whose eigenvectors span an arbitrary basis of a two-dimensional space, and no
    error is raised. Removing the two stored zeros from the same matrix makes the identical
    call raise.

    scanpy has the same hole -- `scipy.sparse.csgraph.connected_components` also reads
    only the pattern, asserted below -- so this is not a divergence from the reference; it
    is a gap between what the Rust promises in its own comment and what it checks. Fixing
    it means testing values, not just structure, which is a change to the Rust and is out
    of scope for this audit.
    """
    from scipy.sparse.csgraph import connected_components

    n = 8
    dense = np.zeros((n, n), dtype=np.float32)
    for cell in range(n - 1):
        dense[cell, cell + 1] = dense[cell + 1, cell] = 1.0
    dense[0, 0] = dense[n - 1, n - 1] = 1.0
    bridged = sp.csr_matrix(dense)
    bridged = _with_explicit_zero(bridged, 3, 4)
    bridged = _with_explicit_zero(bridged, 4, 3)
    assert int((bridged.data == 0).sum()) == 2

    truly_split = bridged.copy()
    truly_split.eliminate_zeros()
    assert connected_components(truly_split)[0] == 2
    # The hole, in scipy's own words: the same graph with the bridge stored as zero.
    assert connected_components(bridged)[0] == 1, "scipy now looks at values, not structure"

    with pytest.raises(ValueError, match="connected"):
        scrust_call("tl.diffmap", _neighbored(truly_split), n_comps=4, device=DEVICE)

    ours = _neighbored(bridged)
    scrust_call("tl.diffmap", ours, n_comps=4, device=DEVICE)
    evals = np.asarray(ours.uns["diffmap_evals"], dtype=np.float64)
    assert evals[0] == pytest.approx(1.0, abs=1e-5)
    assert evals[1] == pytest.approx(1.0, abs=1e-5), (
        "the guard now catches this; if the Rust was fixed, delete this test's xfail wording"
    )


def test_n_comps_at_the_cell_count_is_refused_where_scanpy_clamps() -> None:
    """DIVERGENCE in validation. `n_comps >= n_cells` raises in scrust; scanpy silently
    returns `n_cells - 1` components instead of the number asked for.

    Both are defensible -- the last eigenpair of an `n x n` operator is not determined by
    an `n`-dimensional subspace -- but the shapes differ, so a caller who sizes an array
    from `n_comps` gets an exception from one and a short array from the other. Pinned
    with the exact shape scanpy produces.
    """
    graph = _path_graph(12)

    reference = _neighbored(graph)
    sc.tl.diffmap(reference, n_comps=13)
    assert reference.obsm["X_diffmap"].shape == (12, 11)
    assert len(reference.uns["diffmap_evals"]) == 11

    with pytest.raises(ValueError, match="n_comps"):
        scrust_call("tl.diffmap", _neighbored(graph), n_comps=13, device=DEVICE)
    with pytest.raises(ValueError, match="n_comps"):
        scrust_call("tl.diffmap", _neighbored(graph), n_comps=12, device=DEVICE)

    # And the largest count both accept still agrees, so the boundary is the only issue.
    ours = _neighbored(graph)
    scrust_call("tl.diffmap", ours, n_comps=11, device=DEVICE)
    assert_allclose(
        np.asarray(ours.uns["diffmap_evals"], dtype=np.float64),
        np.asarray(reference.uns["diffmap_evals"], dtype=np.float64),
        rtol=1e-4,
    )


def test_small_n_comps_is_accepted_where_scanpy_refuses_it() -> None:
    """DIVERGENCE in validation, and the one place the documented eigenvalue ordering
    breaks. scanpy rejects `n_comps <= 2` outright; scrust accepts `n_comps = 1`, and on a
    bipartite graph it can then return `-1` as the leading eigenvalue.

    `diffusion.rs` documents `eigenvalues` as "descending, starting at 1 for a connected
    graph". An even cycle is connected and bipartite, so its operator has both `+1` and
    `-1` in the spectrum with equal magnitude; `leading_by_magnitude` breaks that tie by
    sort order, and at `n_comps = 1` it can keep `-1` and drop the stationary state
    entirely -- so `X_diffmap[:, 0]` is the alternating vector, not the steady state that
    scanpy's docstring promises at index 0. It is a genuine tie, not an arithmetic error,
    and scanpy cannot be asked for a second opinion because it refuses the call; that is
    why this pins the behaviour rather than calling it wrong.

    Asking for three components resolves the tie -- both `+1` and `-1` are then kept and
    the descending sort puts `+1` first -- which is asserted so that the failure is
    localised to the small-`n_comps` regime scanpy never enters.
    """
    n = 20
    dense = np.zeros((n, n), dtype=np.float32)
    for cell in range(n):
        dense[cell, (cell + 1) % n] = dense[(cell + 1) % n, cell] = 1.0
    graph = sp.csr_matrix(dense)

    with pytest.raises(ValueError, match="greater than 2"):
        sc.tl.diffmap(_neighbored(graph), n_comps=1)

    one = _neighbored(graph)
    scrust_call("tl.diffmap", one, n_comps=1, device=DEVICE)
    leading = float(np.asarray(one.uns["diffmap_evals"])[0])
    assert abs(leading) == pytest.approx(1.0, abs=1e-5)
    assert leading == pytest.approx(-1.0, abs=1e-5), (
        "the magnitude tie now resolves in favour of the stationary state; if the Rust "
        f"was changed to prefer the algebraically larger eigenvalue, got {leading}"
    )

    three = _neighbored(graph)
    scrust_call("tl.diffmap", three, n_comps=3, device=DEVICE)
    evals = np.asarray(three.uns["diffmap_evals"], dtype=np.float64)
    assert evals[0] == pytest.approx(1.0, abs=1e-5), evals
    assert evals[-1] == pytest.approx(-1.0, abs=1e-5), evals


def test_unsorted_csr_indices_give_the_same_map() -> None:
    """A connectivity graph whose column indices are not sorted within a row must give
    the same diffusion map as the sorted one.

    `symmetric_transitions`, `column_sums` and the ELLPACK `SparseOperator` all index the
    CSR arrays directly and none of them sorts, so this checks an assumption the Rust
    makes silently. scipy produces sorted indices from most constructors but not from all
    of them (`csr_matrix((data, indices, indptr))` is taken as given), so a caller can
    reach this. The permutation is asserted to have actually unsorted the matrix, which is
    what stops the test from comparing a matrix against itself.
    """
    graph = _irregular_graph(n=40, seed=5)
    shuffled = graph.copy()
    rng = np.random.default_rng(0)
    for row in range(shuffled.shape[0]):
        start, stop = shuffled.indptr[row], shuffled.indptr[row + 1]
        order = rng.permutation(stop - start)
        shuffled.indices[start:stop] = shuffled.indices[start:stop][order]
        shuffled.data[start:stop] = shuffled.data[start:stop][order]
    shuffled.has_sorted_indices = False
    assert not shuffled.has_canonical_format, "the fixture did not end up unsorted"
    assert np.any(shuffled.indices != graph.indices), "the permutation was a no-op"

    sorted_run = _neighbored(graph)
    scrust_call("tl.diffmap", sorted_run, n_comps=6, device=DEVICE)
    shuffled_run = _neighbored(shuffled)
    scrust_call("tl.diffmap", shuffled_run, n_comps=6, device=DEVICE)

    assert_allclose(
        np.asarray(shuffled_run.uns["diffmap_evals"], dtype=np.float64),
        np.asarray(sorted_run.uns["diffmap_evals"], dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    assert_allclose(
        np.abs(np.asarray(shuffled_run.obsm["X_diffmap"], dtype=np.float64)),
        np.abs(np.asarray(sorted_run.obsm["X_diffmap"], dtype=np.float64)),
        rtol=0,
        atol=1e-5,
    )
