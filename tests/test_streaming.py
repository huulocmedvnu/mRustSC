"""`pp.preprocess_backed` gives the in-memory pipeline's answer without holding the matrix.

The streamed head of the pipeline is held to the in-memory one on a small synthetic
counts file: the same cells pass the filter, the same genes are flagged variable, and
the PCA embedding spans the same space. The PCA reference is scanpy's exact
`covariance_eigh` solver when scanpy is installed (the streamed path is that algorithm),
and the in-memory `metalcyte` PCA otherwise, held to its leading components only because
its randomised tail is not exact.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")
mc = pytest.importorskip("metalcyte")


def _counts_file(tmp_path, n_cells=3000, n_genes=800, seed=0):
    rng = np.random.default_rng(seed)
    # Three cell types with their own gene programmes, on a Poisson background,
    # so there are variable genes to find and structure for PCA to recover.
    types = rng.integers(0, 3, n_cells)
    base = rng.gamma(2.0, 0.5, n_genes)
    dense = np.empty((n_cells, n_genes), dtype=np.float32)
    for t in range(3):
        mask = types == t
        programme = base.copy()
        programme[t * 100 : (t + 1) * 100] *= 6.0
        depth = rng.lognormal(0.0, 0.3, mask.sum())[:, None]
        dense[mask] = rng.poisson(programme[None, :] * depth).astype(np.float32)
    # A few cells with almost nothing in them, so the cell filter has work to do.
    dense[:20] = (dense[:20] > 8).astype(np.float32)
    adata = anndata.AnnData(sp.csr_matrix(dense))
    adata.obs["kind"] = [f"t{t}" for t in types]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    path = tmp_path / "counts.h5ad"
    adata.write_h5ad(path)
    return path, adata


def _subspace_agreement(x, y, k):
    qx, _ = np.linalg.qr(x[:, :k] - x[:, :k].mean(0))
    qy, _ = np.linalg.qr(y[:, :k] - y[:, :k].mean(0))
    return np.linalg.svd(qx.T @ qy, compute_uv=False).min()


def test_streamed_head_matches_the_in_memory_pipeline(tmp_path):
    path, adata = _counts_file(tmp_path)
    streamed = mc.pp.preprocess_backed(
        path, n_top_genes=200, n_comps=20, min_genes=50, block_size=700, obs_columns=("kind",)
    )

    whole = adata.copy()
    mc.pp.filter_cells(whole, min_genes=50)
    mc.pp.filter_genes(whole, min_cells=3)
    mc.pp.normalize_total(whole, target_sum=1e4)
    mc.pp.log1p(whole)
    mc.pp.highly_variable_genes(whole, n_top_genes=200)

    assert list(streamed.obs_names) == list(whole.obs_names)
    assert streamed.uns["streaming"]["cells_dropped"] == adata.n_obs - whole.n_obs
    assert (streamed.obs["kind"].to_numpy() == whole.obs["kind"].to_numpy()).all()
    kept = sp.csr_matrix(adata[whole.obs_names].X)
    np.testing.assert_array_equal(streamed.obs["n_genes"].to_numpy(), np.diff(kept.indptr))

    flagged_streamed = set(streamed.var_names[streamed.var["highly_variable"].to_numpy()])
    flagged_whole = set(whole.var_names[whole.var["highly_variable"].to_numpy()])
    overlap = len(flagged_streamed & flagged_whole) / len(flagged_whole)
    assert overlap >= 0.99, f"HVG sets overlap only {overlap:.3f}"

    # PCA on the streamed gene set, against the exact solver when available.
    subset = whole[:, whole.var_names.isin(flagged_streamed)].copy()
    try:
        import scanpy as sc

        sc.pp.scale(subset, max_value=10)
        sc.pp.pca(subset, n_comps=20, svd_solver="covariance_eigh")
        k_exact = 20
    except ImportError:
        mc.pp.scale(subset, max_value=10)
        mc.pp.pca(subset, n_comps=20, random_state=0)
        k_exact = 10
    agreement = _subspace_agreement(streamed.obsm["X_pca"], subset.obsm["X_pca"], k_exact)
    assert agreement > 0.995, f"leading {k_exact} components span different spaces: {agreement:.4f}"
    np.testing.assert_allclose(
        streamed.uns["pca"]["variance_ratio"][:k_exact],
        subset.uns["pca"]["variance_ratio"][:k_exact],
        rtol=2e-3,
    )


def test_streamed_result_is_independent_of_the_block_size(tmp_path):
    path, _ = _counts_file(tmp_path, n_cells=1500, n_genes=400, seed=3)
    a = mc.pp.preprocess_backed(path, n_top_genes=100, n_comps=10, min_genes=50, block_size=1500)
    b = mc.pp.preprocess_backed(path, n_top_genes=100, n_comps=10, min_genes=50, block_size=137)
    np.testing.assert_array_equal(a.var["highly_variable"], b.var["highly_variable"])
    assert _subspace_agreement(a.obsm["X_pca"], b.obsm["X_pca"], 10) > 0.9999
    np.testing.assert_allclose(a.obsm["X_pca"], b.obsm["X_pca"], rtol=1e-3, atol=1e-3)
