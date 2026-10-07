#!/usr/bin/env python3
"""Cluster the PBMC 3k dataset (2 700 cells) with metalcyte.

    MPLBACKEND=Agg PYTHONPATH=$PWD/python .venv/bin/python examples/pbmc3k.py

The script runs a standard single-cell workflow from raw counts to clusters and
marker genes. It downloads the PBMC 3k count matrix (10x Genomics) once and keeps it in
`.cache/`. Each step prints the call that ran and how long it took.

The steps are quality control, filtering, normalisation, gene selection, regression
of technical covariates, scaling, PCA, the neighbour graph, UMAP, t-SNE, Leiden
clustering, marker genes and PAGA. The script then saves three figures to
`.cache/figures/`.

On an M3 Pro the script takes about 25 seconds. Most of it is the exact t-SNE of 2 643 cells.
"""

from __future__ import annotations

import shutil
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any

import anndata as ad

import metalcyte as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / ".cache"
DATASET_PATH = CACHE / "pbmc3k_raw.h5ad"
DATASET_URL = "https://falexwolf.de/data/pbmc3k_raw.h5ad"
FIGURE_DIR = CACHE / "figures"

TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_COMPS = 50
N_NEIGHBORS = 15
MAX_VALUE = 10.0
RESOLUTION = 1.0


def step(call: str) -> None:
    """Print the call that the next line runs."""
    print(f"\n  {call}")


def timed(label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one call and print how long it took."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    print(f"          {label} in {time.perf_counter() - start:.3f} s")
    return result


def _ssl_context() -> ssl.SSLContext:
    """Use the certifi certificates when installed.

    The python.org installers for macOS ship without system certificates, so the
    default context can fail to verify the download server.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def load_pbmc3k() -> ad.AnnData:
    """Download the raw PBMC 3k counts if they are missing, then read them."""
    if not DATASET_PATH.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"          downloading {DATASET_URL}")
        partial = DATASET_PATH.with_suffix(".part")
        with (
            urllib.request.urlopen(DATASET_URL, context=_ssl_context()) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        partial.rename(DATASET_PATH)
    return ad.read_h5ad(DATASET_PATH)


def main() -> int:
    print(f"metalcyte {mc.__version__}, GPU available: {mc.gpu_available()}")

    # ---------------------------------------------------------------- load the data
    print("\n=== 1. Read PBMC 3k")
    step(f'adata = anndata.read_h5ad("{DATASET_PATH.relative_to(REPO_ROOT)}")')
    adata = load_pbmc3k()
    adata.var_names_make_unique()
    print(f"          {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------------------------------------- quality control
    print("\n=== 2. Quality control")
    step('mc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])')
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    timed("mc.pp.calculate_qc_metrics", mc.pp.calculate_qc_metrics, adata, qc_vars=["mt"])

    print("\n=== 3. Filter cells and genes")
    step("mc.pp.filter_cells(adata, min_genes=200)")
    timed("mc.pp.filter_cells", mc.pp.filter_cells, adata, min_genes=200)
    step("mc.pp.filter_genes(adata, min_cells=3)")
    timed("mc.pp.filter_genes", mc.pp.filter_genes, adata, min_cells=3)

    # Cells with 5 % or more mitochondrial counts are removed with a boolean mask.
    adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
    print(f"          {adata.n_obs} cells x {adata.n_vars} genes after filtering")

    # ------------------------------------------------------------- normalise, log
    print("\n=== 4. Normalise and logarithmise")
    step("mc.pp.normalize_total(adata, target_sum=1e4)")
    timed("mc.pp.normalize_total", mc.pp.normalize_total, adata, target_sum=TARGET_SUM)
    step("mc.pp.log1p(adata)")
    timed("mc.pp.log1p", mc.pp.log1p, adata)
    adata.raw = adata

    # --------------------------------------------------------- highly variable genes
    print("\n=== 5. Highly variable genes")
    step('mc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")')
    timed(
        "mc.pp.highly_variable_genes",
        mc.pp.highly_variable_genes,
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
    )
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    print(f"          kept {adata.n_vars} genes")

    print("\n=== 6. Regress out total counts and mitochondrial percentage")
    step('mc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])')
    timed("mc.pp.regress_out", mc.pp.regress_out, adata, ["total_counts", "pct_counts_mt"])

    # ------------------------------------------------------------------ scale, PCA
    print("\n=== 7. Scale and PCA")
    step("mc.pp.scale(adata, max_value=10)")
    timed("mc.pp.scale", mc.pp.scale, adata, zero_center=True, max_value=MAX_VALUE)
    step("mc.pp.pca(adata, n_comps=50)")
    timed("mc.pp.pca", mc.pp.pca, adata, n_comps=N_COMPS, random_state=0)
    print(f"          obsm['X_pca'] {adata.obsm['X_pca'].shape}")

    # ---------------------------------------------------------- neighbours and UMAP
    print("\n=== 8. Neighbour graph, UMAP and t-SNE")
    step('mc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")')
    timed("mc.pp.neighbors", mc.pp.neighbors, adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    step("mc.tl.umap(adata)")
    timed("mc.tl.umap", mc.tl.umap, adata, random_state=0)
    print(f"          obsm['X_umap'] {adata.obsm['X_umap'].shape}")

    step("mc.tl.tsne(adata, n_pcs=50, perplexity=30)")
    print("          NOTE: metalcyte's t-SNE is exact, so its cost grows with the square")
    print("          of the cell count. It refuses more than 20 000 cells.")
    timed("mc.tl.tsne", mc.tl.tsne, adata, n_pcs=N_COMPS, perplexity=30.0, random_state=0)

    # -------------------------------------------------------------------- clustering
    print("\n=== 9. Cluster the graph")
    step("mc.tl.leiden(adata, resolution=1.0)")
    timed("mc.tl.leiden", mc.tl.leiden, adata, resolution=RESOLUTION, random_state=0)
    print(f"          {adata.obs['leiden'].nunique()} clusters")

    # ------------------------------------------------------- differential expression
    print("\n=== 10. Rank marker genes")
    step('mc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")')
    # Marker genes are ranked on the log-normalised matrix, not the scaled one.
    ranked = adata.raw.to_adata()[:, adata.var_names].copy()
    ranked.obs["leiden"] = adata.obs["leiden"]
    timed(
        "mc.tl.rank_genes_groups",
        mc.tl.rank_genes_groups,
        ranked,
        "leiden",
        method="wilcoxon",
    )
    adata.uns["rank_genes_groups"] = ranked.uns["rank_genes_groups"]

    step('mc.get.rank_genes_groups_df(adata, group="0").head()')
    markers = mc.get.rank_genes_groups_df(adata, group="0")
    print(markers.head(5).to_string(index=False))

    # ------------------------------------------------------------------------- paga
    print("\n=== 11. Cluster connectivity (PAGA)")
    step('mc.tl.paga(adata, groups="leiden")')
    timed("mc.tl.paga", mc.tl.paga, adata, "leiden")
    print(f"          uns['paga']['connectivities'] {adata.uns['paga']['connectivities'].shape}")

    # --------------------------------------------------------------------- accessors
    print("\n=== 12. Tables")
    step('mc.get.obs_df(adata, keys=["CST3", "NKG7", "leiden"]).head()')
    keys = [gene for gene in ("CST3", "NKG7", "PPBP") if gene in adata.var_names][:2]
    print(mc.get.obs_df(adata, keys=[*keys, "leiden"]).head(3).to_string())
    step('mc.get.aggregate(adata, by="leiden", func="mean")')
    means = mc.get.aggregate(adata, "leiden", "mean")
    print(f"          aggregated to {means.shape[0]} groups x {means.shape[1]} genes")

    # ------------------------------------------------------------------------ plots
    print("\n=== 13. Figures")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "umap_leiden.png": lambda path: mc.pl.umap(adata, color=["leiden", *keys], save=path),
        "pca_variance_ratio.png": lambda path: mc.pl.pca_variance_ratio(adata, save=path),
        "rank_genes_groups.png": lambda path: mc.pl.rank_genes_groups(adata, save=path),
    }
    for name, draw in figures.items():
        path = FIGURE_DIR / name
        timed(f"saved {path.relative_to(REPO_ROOT)}", draw, path)

    print("\n=== Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
