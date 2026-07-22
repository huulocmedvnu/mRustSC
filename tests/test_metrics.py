"""`scrust.metrics` against scanpy. Owned by feat/metrics.

The two autocorrelation statistics are deterministic reductions, so agreement is
element wise to the contract's tolerance on every gene, not a set overlap. The
analytic cases pin the ends of each scale, which a comparison against a reference
cannot: an implementation that is wrong in the same way as its reference would
still pass the cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from scrust_call import scrust_call

# The contract's tolerance for a deterministic transform. Both statistics divide
# a graph-wide reduction by a per-gene one, so an f32 path drifts here where
# scanpy's f64 one does not; see the report for the measured margin.
RTOL = 1e-4
# Moran's I of a gene with no signal is ~1/n, so a purely relative bar would
# measure rounding around zero rather than correctness.
ATOL = 1e-6

STATISTICS = ["morans_i", "gearys_c"]


def worst_deviation(ours: np.ndarray, reference: np.ndarray, names: object) -> tuple[float, str]:
    """Largest relative deviation and the gene it falls on, ignoring `nan` pairs."""
    ours, reference = np.asarray(ours, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    comparable = np.isfinite(ours) & np.isfinite(reference)
    relative = np.zeros_like(reference)
    scale = np.maximum(np.abs(reference[comparable]), ATOL / RTOL)
    relative[comparable] = np.abs(ours[comparable] - reference[comparable]) / scale
    worst = int(np.argmax(relative))
    return float(relative[worst]), str(np.asarray(names)[worst])


@pytest.mark.parametrize("statistic", STATISTICS)
def test_matches_scanpy_on_every_gene(neighbored: AnnData, statistic: str) -> None:
    ours = scrust_call(f"metrics.{statistic}", neighbored)
    reference = getattr(sc.metrics, statistic)(neighbored)

    assert np.isnan(ours).tolist() == np.isnan(reference).tolist()
    deviation, gene = worst_deviation(ours, reference, neighbored.var_names)
    print(
        f"\n{statistic} on {neighbored.uns['dataset_id']} ({neighbored.n_vars} genes): "
        f"worst relative deviation {deviation:.3e} on {gene}"
    )
    assert deviation <= RTOL, f"{statistic} differs by {deviation:.3e} on {gene}"


@pytest.fixture
def ring() -> AnnData:
    """64 cells in a cycle, each joined to its two neighbours, carrying three genes.

    `smooth` varies as slowly as anything can around the ring, `alternating` as
    fast, and `constant` not at all — the two ends of both scales and the
    degenerate case, all on a graph whose statistics can be worked out by hand.
    """
    n = 64
    edges = sparse.diags_array(  # type: ignore[attr-defined]
        [np.ones(n - 1), np.ones(n - 1)], offsets=[1, -1], shape=(n, n)
    ).tolil()
    edges[0, n - 1] = edges[n - 1, 0] = 1.0
    cells = np.arange(n)
    values = np.column_stack(
        [
            np.cos(2 * np.pi * cells / n),
            np.where(cells % 2 == 0, 1.0, -1.0),
            np.full(n, 2.5),
        ]
    ).astype(np.float32)

    adata = AnnData(sparse.csr_matrix(values))
    adata.var_names = ["smooth", "alternating", "constant"]
    adata.obsp["connectivities"] = sparse.csr_matrix(edges)
    adata.uns["neighbors"] = {"connectivities_key": "connectivities"}
    adata.obs["half"] = pd.Categorical(np.where(cells < n // 2, "left", "right"))
    return adata


def test_smooth_and_alternating_signals_reach_the_ends_of_both_scales(ring: AnnData) -> None:
    n = ring.n_obs
    morans = scrust_call("metrics.morans_i", ring)
    gearys = scrust_call("metrics.gearys_c", ring)

    # A cosine over 64 cells barely changes between neighbours.
    assert morans[0] > 0.99
    assert gearys[0] < 0.01
    # Alternating +-1 is the exact opposite, and exactly so: every edge crosses
    # the sign, which is Moran's I of -1 and Geary's C at its maximum 2(n-1)/n.
    assert morans[1] == pytest.approx(-1.0, abs=1e-5)
    assert gearys[1] == pytest.approx(2 * (n - 1) / n, abs=1e-5)


def test_a_constant_gene_returns_what_scanpy_returns(ring: AnnData) -> None:
    ours_morans = scrust_call("metrics.morans_i", ring)
    ours_gearys = scrust_call("metrics.gearys_c", ring)
    with pytest.warns(UserWarning, match="constant"):
        reference_morans = sc.metrics.morans_i(ring)
    with pytest.warns(UserWarning, match="constant"):
        reference_gearys = sc.metrics.gearys_c(ring)

    assert np.isnan(reference_morans[2]) and np.isnan(reference_gearys[2])
    assert np.isnan(ours_morans[2]) and np.isnan(ours_gearys[2])
    np.testing.assert_allclose(ours_morans[:2], reference_morans[:2], rtol=RTOL)
    np.testing.assert_allclose(ours_gearys[:2], reference_gearys[:2], rtol=RTOL)


@pytest.mark.parametrize("statistic", STATISTICS)
def test_vals_accepts_a_name_an_obs_column_or_an_array(ring: AnnData, statistic: str) -> None:
    per_gene = scrust_call(f"metrics.{statistic}", ring)
    by_name = scrust_call(f"metrics.{statistic}", ring, vals="smooth")
    assert by_name == pytest.approx(per_gene[0], rel=RTOL)

    # scanpy's explicit array layout is (n_features, n_cells).
    explicit = scrust_call(f"metrics.{statistic}", ring, vals=ring.X.toarray().T)
    np.testing.assert_allclose(explicit, per_gene, rtol=RTOL)

    ring.obs["smooth_obs"] = np.asarray(ring[:, "smooth"].X.todense()).ravel()
    from_obs = scrust_call(f"metrics.{statistic}", ring, vals="smooth_obs")
    assert from_obs == pytest.approx(per_gene[0], rel=RTOL)

    several = scrust_call(f"metrics.{statistic}", ring, vals=["smooth", "alternating"])
    np.testing.assert_allclose(several, per_gene[:2], rtol=RTOL)


@pytest.mark.parametrize("statistic", STATISTICS)
def test_gpu_agrees_with_cpu(neighbored: AnnData, statistic: str) -> None:
    from scrust import _scrust

    if not _scrust.gpu_available():
        pytest.skip("no Metal device on this machine")
    cpu = scrust_call(f"metrics.{statistic}", neighbored, device="cpu")
    gpu = scrust_call(f"metrics.{statistic}", neighbored, device="gpu")
    # f32 either way, and the scatter-add on the GPU accumulates in a different
    # order, so the bar is f32 rounding rather than equality.
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("statistic", STATISTICS)
def test_rejects_mismatched_shapes_and_an_empty_graph(ring: AnnData, statistic: str) -> None:
    with pytest.raises(ValueError):
        scrust_call(f"metrics.{statistic}", ring, vals=np.zeros((2, ring.n_obs + 1)))
    with pytest.raises(KeyError):
        scrust_call(f"metrics.{statistic}", ring, use_graph="distances")

    ring.obsp["connectivities"] = sparse.csr_matrix((ring.n_obs, ring.n_obs), dtype=np.float32)
    with pytest.raises(ValueError):
        scrust_call(f"metrics.{statistic}", ring)


@pytest.mark.parametrize("normalize", [True, False])
def test_confusion_matrix_matches_scanpy(neighbored: AnnData, normalize: bool) -> None:
    labels = pd.DataFrame(
        {
            "group": neighbored.obs["group"].to_numpy(),
            # A relabelling that merges and renames, so the table is not diagonal
            # and the two axes do not carry the same label set.
            "coarse": pd.Categorical(
                np.where(neighbored.obs["group"].cat.codes < 1, "a", "b"),
            ),
        },
        index=neighbored.obs_names,
    )
    ours = scrust_call(
        "metrics.confusion_matrix", "group", "coarse", labels, normalize=normalize
    )
    reference = sc.metrics.confusion_matrix("group", "coarse", labels, normalize=normalize)
    pd.testing.assert_frame_equal(ours, reference, check_dtype=False)


def test_confusion_matrix_accepts_bare_arrays() -> None:
    orig = ["b", "b", "a", "a", "c"]
    new = ["2", "1", "1", "1", "1"]
    ours = scrust_call("metrics.confusion_matrix", orig, new, normalize=False)
    pd.testing.assert_frame_equal(
        ours, sc.metrics.confusion_matrix(orig, new, normalize=False), check_dtype=False
    )


def test_modularity_scores_the_labelling_on_the_neighbour_graph(ring: AnnData) -> None:
    """Wired to `cluster::modularity`, which `feat/leiden` still owes us.

    The call is asserted against igraph's answer for the same partition, so it
    starts passing the moment that branch lands; until then `scrust_call` reports
    it as the `todo!()` stub it is.
    """
    ours = scrust_call("metrics.modularity", ring, "half")
    assert ours == pytest.approx(sc.metrics.modularity(ring, "half"), rel=1e-4)
