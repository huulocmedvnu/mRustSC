"""Diffusion maps and diffusion pseudotime, against scanpy and against a path graph.

The eigenvalues are the well determined half of a diffusion map and are asserted to the
contract's `rtol=1e-3`. The eigenvectors are not always determined: a diffusion spectrum
is nearly flat below the leading few components, and where two eigenvalues coincide any
rotation of their plane is an equally valid pair of eigenvectors. So the per-component
comparison is made only where scanpy reproduces *itself* from a different random start,
exactly as `check_pca_agreement` does for PCA, and both counts are recorded either way.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.stats import spearmanr

from reference_metrics import component_correlations
from scrust_call import scrust_call

N_COMPS = 15
N_DCS = 10

# A diffusion map is deterministic up to the arbitrary basis of a degenerate eigenspace,
# and pseudotime is a weighted distance in that basis — so rotations inside a degenerate
# plane move it slightly. Ordering is what pseudotime means, and on a connected neighbour
# graph the measured Spearman correlation against scanpy is above 0.999; 0.99 leaves room
# for f32 and for those rotations without admitting a genuinely different ordering.
PSEUDOTIME_SPEARMAN = 0.99


def _path_graph(n: int) -> AnnData:
    """`n` cells in a line, with the connectivity graph a `pp.neighbors` run would give.

    The ends carry a self loop so that every cell has the same degree: the transition
    matrix divides by the degree, and without it the two ends sit off the cosine that the
    interior follows. With it the first non-trivial component is the exact discrete
    cosine, which is monotone, and that is what makes this test analytic.
    """
    rows, cols = [], []
    for cell in range(n - 1):
        rows += [cell, cell + 1]
        cols += [cell + 1, cell]
    rows += [0, n - 1]
    cols += [0, n - 1]
    graph = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    adata = AnnData(np.zeros((n, 1), dtype=np.float32))
    adata.obsp["connectivities"] = graph
    return adata


def _disconnected_graph(n: int) -> AnnData:
    adata = _path_graph(n)
    graph = adata.obsp["connectivities"].tolil()
    graph[n // 2 - 1, n // 2] = 0.0
    graph[n // 2, n // 2 - 1] = 0.0
    adata.obsp["connectivities"] = graph.tocsr()
    adata.obsp["connectivities"].eliminate_zeros()
    return adata


def _scanpy_diffmap(adata: AnnData, random_state: int = 0) -> AnnData:
    reference = adata.copy()
    sc.tl.diffmap(reference, n_comps=N_COMPS, random_state=random_state)
    return reference


def test_diffmap_matches_scanpy(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    label = neighbored.uns["dataset_id"]
    ours = neighbored.copy()
    scrust_call("tl.diffmap", ours, n_comps=N_COMPS)

    reference = _scanpy_diffmap(neighbored)
    ceiling_run = _scanpy_diffmap(neighbored, random_state=7)

    assert ours.obsm["X_diffmap"].shape == reference.obsm["X_diffmap"].shape, (
        "scanpy keeps the trivial first component in X_diffmap; so must we"
    )

    ours_evals = np.asarray(ours.uns["diffmap_evals"], dtype=np.float64)
    reference_evals = np.asarray(reference.uns["diffmap_evals"], dtype=np.float64)
    worst = float(np.max(np.abs(ours_evals - reference_evals) / np.abs(reference_evals)))
    record_property(f"diffmap.{label}.eigenvalue_rel_error", f"{worst:.2e}")
    assert_allclose(ours_evals, reference_evals, rtol=1e-3, err_msg="eigenvalues")

    ours_corr = component_correlations(ours.obsm["X_diffmap"], reference.obsm["X_diffmap"])
    ceiling_corr = component_correlations(
        ceiling_run.obsm["X_diffmap"], reference.obsm["X_diffmap"]
    )
    determined = ceiling_corr >= 0.99
    record_property(f"diffmap.{label}.determined_components", int(determined.sum()))
    record_property(f"diffmap.{label}.components_over_0.99", int((ours_corr >= 0.99).sum()))
    print(
        f"\ndiffmap on {label}: {int(determined.sum())}/{N_COMPS} components are determined "
        f"(scanpy reproduces itself there), we match {int((ours_corr >= 0.99).sum())}; "
        f"worst eigenvalue error {worst:.2e}"
    )

    bad = np.flatnonzero(determined & (ours_corr < 0.99))
    assert bad.size == 0, (
        f"components {bad.tolist()} correlate {ours_corr[bad].round(4).tolist()} < 0.99 "
        f"although scanpy reaches {ceiling_corr[bad].round(4).tolist()} there"
    )


def test_dpt_orders_cells_like_scanpy(
    neighbored: AnnData, record_property: Callable[[str, object], None]
) -> None:
    label = neighbored.uns["dataset_id"]
    neighbored.uns["iroot"] = 0

    ours = neighbored.copy()
    scrust_call("tl.diffmap", ours, n_comps=N_COMPS)
    scrust_call("tl.dpt", ours, n_dcs=N_DCS)

    reference = _scanpy_diffmap(neighbored)
    sc.tl.dpt(reference, n_dcs=N_DCS)

    ours_time = np.asarray(ours.obs["dpt_pseudotime"], dtype=np.float64)
    reference_time = np.asarray(reference.obs["dpt_pseudotime"], dtype=np.float64)
    correlation = float(spearmanr(ours_time, reference_time).statistic)
    record_property(f"dpt.{label}.spearman", round(correlation, 5))
    print(f"\ndpt on {label}: Spearman {correlation:.5f} against scanpy")

    assert ours_time.min() == 0.0, "the root is at pseudotime zero"
    assert 0.0 <= ours_time.max() <= 1.0, "scanpy scales pseudotime onto [0, 1]"
    assert correlation >= PSEUDOTIME_SPEARMAN, (
        f"pseudotime ordering correlates {correlation:.4f} with scanpy's"
    )


def test_path_graph_gives_a_monotone_component_and_pseudotime() -> None:
    """The one test here that does not consult scanpy: on a path, the first non-trivial
    diffusion component is the discrete cosine and pseudotime from an end is its rank."""
    n = 120
    adata = _path_graph(n)
    scrust_call("tl.diffmap", adata, n_comps=6)

    component = np.asarray(adata.obsm["X_diffmap"][:, 1], dtype=np.float64)
    steps = np.diff(component)
    assert np.all(steps > 0) or np.all(steps < 0), (
        f"the first diffusion component of a path must be monotone, got {component}"
    )

    adata.uns["iroot"] = 0
    scrust_call("tl.dpt", adata, n_dcs=6)
    pseudotime = np.asarray(adata.obs["dpt_pseudotime"], dtype=np.float64)
    assert pseudotime[0] == 0.0
    assert np.all(np.diff(pseudotime) > 0), (
        f"pseudotime from an endpoint must increase along the path, got {pseudotime}"
    )
    assert pseudotime[-1] == pytest.approx(1.0)


def test_diffmap_is_deterministic() -> None:
    first, second = _path_graph(60), _path_graph(60)
    scrust_call("tl.diffmap", first, n_comps=8)
    scrust_call("tl.diffmap", second, n_comps=8)
    np.testing.assert_array_equal(first.obsm["X_diffmap"], second.obsm["X_diffmap"])
    np.testing.assert_array_equal(first.uns["diffmap_evals"], second.uns["diffmap_evals"])


def test_cpu_and_gpu_agree() -> None:
    if not scrust_call("gpu_available"):
        pytest.skip("no Metal device on this machine")
    cpu, gpu = _path_graph(90), _path_graph(90)
    scrust_call("tl.diffmap", cpu, n_comps=8, device="cpu")
    scrust_call("tl.diffmap", gpu, n_comps=8, device="gpu")
    # f32 accumulation orders differ between the backends, so the map agrees to f32
    # precision rather than bit for bit.
    assert_allclose(cpu.uns["diffmap_evals"], gpu.uns["diffmap_evals"], rtol=1e-4)
    assert_allclose(
        np.abs(cpu.obsm["X_diffmap"]), np.abs(gpu.obsm["X_diffmap"]), rtol=1e-3, atol=1e-5
    )

    for adata in (cpu, gpu):
        adata.uns["iroot"] = 0
        scrust_call("tl.dpt", adata, n_dcs=8)
    assert_allclose(
        cpu.obs["dpt_pseudotime"], gpu.obs["dpt_pseudotime"], rtol=1e-3, atol=1e-5
    )


def test_a_disconnected_graph_is_refused() -> None:
    adata = _disconnected_graph(40)
    with pytest.raises(ValueError, match="connected"):
        scrust_call("tl.diffmap", adata, n_comps=4)


def test_too_many_components_are_refused() -> None:
    adata = _path_graph(12)
    with pytest.raises(ValueError, match="n_comps"):
        scrust_call("tl.diffmap", adata, n_comps=13)


def test_a_root_outside_the_data_is_refused() -> None:
    adata = _path_graph(30)
    scrust_call("tl.diffmap", adata, n_comps=5)
    adata.uns["iroot"] = 30
    with pytest.raises(ValueError, match="root"):
        scrust_call("tl.dpt", adata, n_dcs=5)


def test_more_components_than_the_map_holds_are_refused() -> None:
    adata = _path_graph(30)
    scrust_call("tl.diffmap", adata, n_comps=5)
    adata.uns["iroot"] = 0
    with pytest.raises(ValueError, match="n_dcs"):
        scrust_call("tl.dpt", adata, n_dcs=6)
