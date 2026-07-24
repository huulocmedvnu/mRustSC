"""Audit native Harmony (`pp.harmony_integrate`) by integration metrics, not bit-for-bit.

Harmony is iterative and k-means seeded, so it does not reproduce `harmonypy` (a compiled
C++ backend) to the bit. Correctness for a batch-integration method is instead:

* the objective decreases and converges, and
* batch mixing improves -- measured by iLISI, the integration Local Inverse Simpson's
  Index: the effective number of batches in each cell's neighbourhood, averaged. It runs
  from 1 (batches separated) to `n_batches` (perfectly mixed).

A synthetic batch effect is injected into the PCA embedding (a shift on one batch), so a
correct method raises iLISI from ~1 back towards `n_batches`. `harmonypy` is run on the
same data as a reference; its iLISI and the cosine correlation are recorded, and scrust is
asserted to integrate at least as well as it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

import scrust as sr

N_BATCHES = 2


def _ilisi(embedding: np.ndarray, batch: np.ndarray, k: int = 30) -> float:
    """Mean inverse-Simpson index of batch labels over each cell's `k` neighbours."""
    neighbours = NearestNeighbors(n_neighbors=k).fit(embedding)
    _, idx = neighbours.kneighbors(embedding)
    scores = []
    for i in range(embedding.shape[0]):
        _, counts = np.unique(batch[idx[i, 1:]], return_counts=True)
        proportions = counts / counts.sum()
        scores.append(1.0 / np.sum(proportions**2))
    return float(np.mean(scores))


def _mean_abs_column_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over PCs of |cosine| between the two centred embeddings, column by column."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    numerator = (a * b).sum(0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12
    return float(np.mean(np.abs(numerator / denominator)))


@pytest.fixture(scope="module")
def _batched(_pbmc3k_labelled: AnnData) -> AnnData:
    """PBMC 3k PCA with two batches and a real batch shift injected on one of them."""
    adata = _pbmc3k_labelled.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    rng = np.random.default_rng(0)
    batch = rng.integers(0, N_BATCHES, adata.n_obs)
    adata.obs["batch"] = batch.astype(str)
    adata.obsm["X_pca"][batch == 1] += rng.normal(0, 1, 50) * 3.0  # separate the batches
    adata.uns["batch_codes"] = batch
    return adata


@pytest.fixture
def batched(_batched: AnnData) -> AnnData:
    return _batched.copy()


def test_harmony_raises_batch_mixing(
    batched: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """iLISI climbs from ~1 (separated) back towards 2 (mixed) after correction."""
    batch = batched.uns["batch_codes"]
    before = _ilisi(batched.obsm["X_pca"], batch)

    sr.pp.harmony_integrate(batched, key="batch", device="cpu")
    after = _ilisi(batched.obsm["X_pca_harmony"], batch)

    record_property("harmony.ilisi_before", round(before, 4))
    record_property("harmony.ilisi_after", round(after, 4))
    assert before < 1.2, f"batches were not separated to begin with (iLISI {before:.3f})"
    assert after > before + 0.4, (
        f"correction did not mix batches (iLISI {before:.3f} -> {after:.3f})"
    )
    assert after > 1.5


def test_objective_converges(batched: AnnData) -> None:
    """The harmony objective drops substantially and its tail stabilises.

    Harmony's objective is not strictly monotone across outer iterations -- the M-step
    changes the embedding, so re-clustering can nudge it up by a hair near convergence --
    so the test asserts a large overall drop and a small final wobble, not monotonicity.
    """
    sr.pp.harmony_integrate(batched, key="batch", device="cpu")
    objective = batched.uns["harmony"]["objective"]
    assert len(objective) >= 2

    total_drop = objective[0] - objective[-1]
    assert total_drop > 0.02 * objective[0], "objective did not converge downward"
    final_wobble = abs(objective[-1] - objective[-2])
    assert final_wobble < 0.05 * total_drop, "objective had not stabilised at the end"


def test_integrates_at_least_as_well_as_harmonypy(
    batched: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """Reference check against harmonypy on the same data: iLISI and cosine correlation.

    Not a bit-for-bit assertion -- harmonypy is a C++ backend with its own k-means seed and
    lambda schedule. scrust is only required to mix batches at least as well; the
    correlation is recorded for the record.
    """
    harmonypy = pytest.importorskip("harmonypy")
    batch = batched.uns["batch_codes"]
    x = np.ascontiguousarray(batched.obsm["X_pca"], dtype=np.float64)

    sr.pp.harmony_integrate(batched, key="batch", device="cpu")
    ours = batched.obsm["X_pca_harmony"]

    reference = harmonypy.run_harmony(x, batched.obs[["batch"]], ["batch"], verbose=False)
    theirs = np.asarray(reference.Z_corr)
    if theirs.shape[0] != len(batch):
        theirs = theirs.T

    ilisi_ours = _ilisi(ours, batch)
    ilisi_theirs = _ilisi(np.ascontiguousarray(theirs), batch)
    correlation = _mean_abs_column_cosine(ours, theirs)

    record_property("harmony.ilisi_scrust", round(ilisi_ours, 4))
    record_property("harmony.ilisi_harmonypy", round(ilisi_theirs, 4))
    record_property("harmony.cosine_correlation", round(correlation, 4))
    assert ilisi_ours >= ilisi_theirs - 0.1
