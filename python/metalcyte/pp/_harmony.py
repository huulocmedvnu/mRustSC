"""Harmony batch-effect correction, as `scanpy.external.pp.harmony_integrate`.

A native port of Harmony (Korsunsky et al. 2019): a soft k-means E-step that clusters the
PCA embedding while penalising batch-imbalanced clusters, and a ridge-regression M-step
that removes the batch shift within each cluster, alternated to convergence. The heavy
matmuls run in the Rust core on the GPU; see `crates/metalcyte-core/src/harmony.rs`.

Harmony is iterative and k-means seeded, so this does not reproduce `harmonypy` bit for
bit. Correctness is judged by batch mixing (iLISI rising after correction) and by cosine
correlation with harmonypy, pinned in `tests/test_harmony_audit.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from metalcyte._shared import _default_device, _dense, _extension

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["harmony_integrate"]


def harmony_integrate(
    adata: AnnData,
    key: str = "batch",
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    theta: float = 2.0,
    sigma: float = 0.1,
    lamb: float | None = None,
    alpha: float = 0.2,
    batch_prop_cutoff: float = 1e-5,
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_kmeans: int = 20,
    random_state: int = 0,
    device: str | None = None,
) -> None:
    """Integrate batches in `obsm[basis]`, writing corrected coordinates to `obsm[adjusted_basis]`.

    `key` names the `obs` column of batch labels. The harmony objective at each iteration
    (the convergence curve) is stored in `uns["harmony"]["objective"]`. `lamb=None` estimates
    the ridge penalty per cluster and batch as `alpha` times the batch's soft count in the
    cluster, as Harmony 1.2 and harmonypy 2 do; a number fixes it for every batch, the
    original Harmony. A batch whose share of a cluster is below `batch_prop_cutoff` is not
    corrected in that cluster.
    """
    if basis not in adata.obsm:
        raise KeyError(f"adata.obsm has no {basis!r}; run metalcyte.pp.pca first")
    if key not in adata.obs:
        raise KeyError(f"adata.obs has no {key!r}")

    column = adata.obs[key]
    if not isinstance(column.dtype, pd.CategoricalDtype):
        column = column.astype("category")
    codes = column.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"adata.obs[{key!r}] has unlabelled cells")
    n_batches = len(column.cat.categories)

    embedding = _dense(adata.obsm[basis])
    corrected, objective = _extension().harmony_integrate(
        embedding,
        codes.astype(np.uint32),
        n_batches,
        float(theta),
        float(sigma),
        None if lamb is None else float(lamb),
        0 if n_clusters is None else int(n_clusters),
        int(max_iter_harmony),
        int(max_iter_kmeans),
        int(random_state),
        device if device is not None else _default_device(),
        float(alpha),
        float(batch_prop_cutoff),
    )
    adata.obsm[adjusted_basis] = np.asarray(corrected, dtype=np.float32)
    adata.uns["harmony"] = {
        "objective": [float(value) for value in objective],
        "key": key,
        "basis": basis,
        "adjusted_basis": adjusted_basis,
    }
