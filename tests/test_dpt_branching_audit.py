"""Audit native DPT branch detection against scanpy by adjusted Rand index.

`tl.dpt(n_branchings>0)` used to raise `NotImplementedError`. It now runs a port of
scanpy's Haghverdi 2016 branch detection (`scrust.tl._dpt_branching`) and writes
`obs["dpt_groups"]`. Branch labels are arbitrary, so parity is the adjusted Rand index of
the two partitions, per `docs/API_CONTRACT.md` (clustering labels are compared by ARI).

The algorithm reads the diffusion map, so a fair test of the *port* feeds scrust's and
scanpy's branching the **same** diffmap: any difference is then the branching logic alone,
and it must be none (ARI = 1.0). A second test runs the whole scrust path end to end and
only asserts it produces the field without raising, recording its ARI for the record.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score

from scrust.tl._dpt_branching import dpt_groups
from scrust_call import scrust_call


@pytest.fixture(scope="module")
def _prepared(_pbmc3k_labelled: AnnData) -> AnnData:
    """PBMC 3k through to a diffusion map, prepared with scanpy, with a root cell set."""
    adata = _pbmc3k_labelled.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.diffmap(adata, n_comps=15)
    adata.uns["iroot"] = int(np.argmin(adata.obsm["X_diffmap"][:, 1]))
    adata.obs["dpt_pseudotime"] = 0.0  # branching does not use it; set so the port can read it
    return adata


@pytest.fixture
def prepared(_prepared: AnnData) -> AnnData:
    return _prepared.copy()


@pytest.mark.parametrize("n_branchings", [1, 2])
def test_branch_partition_matches_scanpy_exactly(
    prepared: AnnData, n_branchings: int, record_property: Callable[[str, object], None]
) -> None:
    """On the same diffmap, the native partition equals scanpy's: ARI = 1.0."""
    reference = prepared.copy()
    sc.tl.dpt(reference, n_branchings=n_branchings, n_dcs=10)
    labels = dpt_groups(prepared, n_branchings=n_branchings, min_group_size=0.01, n_dcs=10)

    ours = np.asarray(labels)
    theirs = reference.obs["dpt_groups"].to_numpy()
    score = adjusted_rand_score(theirs, ours)
    record_property(f"dpt_branching.n{n_branchings}.ari", round(float(score), 6))

    assert reference.obs["dpt_groups"].nunique() == len(set(ours.tolist()))
    # Group sizes as multisets: the partition is identical, not merely similar.
    assert sorted(np.bincount(ours).tolist()) == sorted(
        reference.obs["dpt_groups"].value_counts().tolist()
    )
    assert score == pytest.approx(1.0)


def test_dpt_end_to_end_writes_dpt_groups_without_error(
    prepared: AnnData, record_property: Callable[[str, object], None]
) -> None:
    """The full scrust path (its own pseudotime + branching) produces the field.

    End-to-end ARI against scanpy also folds in scrust's diffmap, so it is recorded, not
    asserted; the branching port itself is pinned to ARI = 1.0 by the test above.
    """
    del prepared.obs["dpt_pseudotime"]
    scrust_call("tl.dpt", prepared, n_branchings=1, n_dcs=10)

    assert "dpt_groups" in prepared.obs
    assert prepared.obs["dpt_groups"].nunique() >= 2
    assert "dpt_pseudotime" in prepared.obs

    reference = prepared.copy()
    sc.tl.dpt(reference, n_branchings=1, n_dcs=10)
    score = adjusted_rand_score(
        reference.obs["dpt_groups"].to_numpy(), prepared.obs["dpt_groups"].to_numpy()
    )
    record_property("dpt_branching.end_to_end.ari", round(float(score), 6))


def test_n_branchings_zero_still_skips_branching(prepared: AnnData) -> None:
    """`n_branchings=0` writes pseudotime only, never `dpt_groups`."""
    del prepared.obs["dpt_pseudotime"]
    scrust_call("tl.dpt", prepared, n_branchings=0)
    assert "dpt_pseudotime" in prepared.obs
    assert "dpt_groups" not in prepared.obs
