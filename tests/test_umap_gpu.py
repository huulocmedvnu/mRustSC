"""`tl.umap(parallel=True)` on the GPU: the Hogwild kernel behind the Metal path.

The layout is not reproducible run to run (like the CPU `parallel=True` path), so the tests
check structure: finite coordinates, separated clusters, and agreement in quality with the
multi-core CPU optimiser on the same graph. `device="cpu"` must keep the call on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

import metalcyte as mc

pytestmark = pytest.mark.skipif(not mc.gpu_available(), reason="needs a usable Metal GPU")


def _clustered(n_per=400, n_clusters=4, dims=10, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=12.0, size=(n_clusters, dims))
    x = np.vstack([c + rng.normal(size=(n_per, dims)) for c in centres]).astype(np.float32)
    labels = np.repeat(np.arange(n_clusters), n_per)
    adata = AnnData(X=sp.csr_matrix((x.shape[0], 1), dtype=np.float32), obsm={"X_pca": x})
    mc.pp.neighbors(adata, n_neighbors=15, method="exact")
    return adata, labels


def _vote(layout, labels, k=15):
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(layout).kneighbors(layout)[1][:, 1:]
    return float(
        np.mean([np.bincount(labels[r]).argmax() == y for r, y in zip(idx, labels, strict=True)])
    )


def test_gpu_layout_is_finite_and_separates_clusters():
    adata, labels = _clustered()
    mc.tl.umap(adata, parallel=True, random_state=0)
    layout = adata.obsm["X_umap"]
    assert layout.shape == (adata.n_obs, 2)
    assert np.isfinite(layout).all()
    assert _vote(layout, labels) > 0.99


def test_gpu_layout_matches_cpu_parallel_quality():
    adata, labels = _clustered(seed=1)
    mc.tl.umap(adata, parallel=True, random_state=0)
    gpu = _vote(adata.obsm["X_umap"], labels)
    mc.tl.umap(adata, parallel=True, random_state=0, device="cpu")
    cpu = _vote(adata.obsm["X_umap"], labels)
    assert abs(gpu - cpu) < 0.02


def test_cpu_device_keeps_parallel_umap_off_the_gpu(monkeypatch):
    adata, _ = _clustered(n_per=100)
    from metalcyte._shared import _extension

    ext = _extension()
    called = []
    monkeypatch.setattr(ext, "umap_metal", lambda *a, **k: called.append(1), raising=True)
    mc.tl.umap(adata, parallel=True, device="cpu")
    assert called == []
    assert np.isfinite(adata.obsm["X_umap"]).all()
