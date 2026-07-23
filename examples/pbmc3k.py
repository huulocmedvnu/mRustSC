#!/usr/bin/env python3
"""scanpy's PBMC 3k tutorial, run through scrust as far as scrust goes today.

    PYTHONPATH=$PWD/python .venv/bin/python examples/pbmc3k.py

Every step prints the scanpy call it replaces next to the scrust call that ran, so
the two APIs can be read against each other. Nothing is silently substituted: a step
that still goes through scanpy says so on its own line.

The timings below still call scanpy for every step, so what the script measures is
scanpy, not scrust — it is a tutorial transcript and a timing baseline, not a
benchmark of this crate. `docs/BENCHMARKS.md` has the side-by-side numbers.

The whole script takes about a minute, most of it in the one step that is not from
the tutorial: an exact t-SNE of 2 600 cells.

Plotting is scanpy's job in both worlds. The results land in the AnnData slots
`sc.pl` reads, so `sc.pl.umap(adata, color="leiden")` works on the object this
script leaves behind; it is not called here only because a script has no display.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import scanpy as sc

import scrust as sr

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE = REPO_ROOT / ".cache" / "scanpy"

TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_COMPS = 50
N_NEIGHBORS = 15
MAX_VALUE = 10.0
RESOLUTION = 1.0


def step(scanpy_call: str, scrust_call: str) -> None:
    """Print one step of the tutorial as the pair of calls it is."""
    print(f"\n  scanpy: {scanpy_call}")
    print(f"  scrust: {scrust_call}")


def timed(label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one call and print what it cost, so the script doubles as a small timing."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    print(f"          {label} in {time.perf_counter() - start:.3f} s")
    return result


def main() -> int:
    print(f"scrust {sr.__version__}, GPU available: {sr.gpu_available()}")
    print(f"scanpy {sc.__version__}")

    # ---------------------------------------------------------------- load the data
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = DATASET_CACHE
    print("\n=== 1. Read PBMC 3k")
    step(
        'adata = sc.read_10x_mtx("filtered_gene_bc_matrices/hg19/")',
        "reading is scanpy's job; scrust writes no readers",
    )
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    print(f"          {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------------------------------------- quality control
    print("\n=== 2. Quality control")
    step(
        'sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)',
        'sr.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)',
    )
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    timed(
        "sc.pp.calculate_qc_metrics",
        sc.pp.calculate_qc_metrics,
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    print("\n=== 3. Filter cells and genes")
    step(
        "sc.pp.filter_cells(adata, min_genes=200)",
        "sr.pp.filter_cells(adata, min_genes=200)",
    )
    timed("sr.pp.filter_cells", sr.pp.filter_cells, adata, min_genes=200)
    step(
        "sc.pp.filter_genes(adata, min_cells=3)",
        "sr.pp.filter_genes(adata, min_cells=3)",
    )
    timed("sr.pp.filter_genes", sr.pp.filter_genes, adata, min_cells=3)

    # The tutorial's mitochondrial cut, done with plain AnnData indexing in both
    # worlds: it is a boolean mask, not a library function.
    adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
    print(f"          {adata.n_obs} cells x {adata.n_vars} genes after filtering")

    # ------------------------------------------------------------- normalise, log
    print("\n=== 4. Normalise and logarithmise")
    step(
        "sc.pp.normalize_total(adata, target_sum=1e4)",
        "sr.pp.normalize_total(adata, target_sum=1e4)",
    )
    timed("sr.pp.normalize_total", sr.pp.normalize_total, adata, target_sum=TARGET_SUM)
    step("sc.pp.log1p(adata)", "sr.pp.log1p(adata)")
    timed("sr.pp.log1p", sr.pp.log1p, adata)
    adata.raw = adata

    # --------------------------------------------------------- highly variable genes
    print("\n=== 5. Highly variable genes")
    step(
        "sc.pp.highly_variable_genes(adata, n_top_genes=2000)",
        "sr.pp.highly_variable_genes(adata, n_top_genes=2000)",
    )
    timed(
        "sr.pp.highly_variable_genes",
        sr.pp.highly_variable_genes,
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
    )
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    print(f"          kept {adata.n_vars} genes")

    print("\n=== 6. Regress out confounders")
    step(
        'sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])',
        'sr.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])',
    )
    print("          skipped: the tutorial's own text calls this step optional, and")
    print("          running it with scanpy here would dominate the timings below")

    # ------------------------------------------------------------------ scale, PCA
    print("\n=== 7. Scale and PCA")
    step(
        "sc.pp.scale(adata, max_value=10)",
        "sr.pp.scale(adata, max_value=10)",
    )
    timed("sr.pp.scale", sr.pp.scale, adata, zero_center=True, max_value=MAX_VALUE)
    step(
        "sc.pp.pca(adata, n_comps=50, svd_solver='arpack')",
        "sr.pp.pca(adata, n_comps=50)",
    )
    timed("sr.pp.pca", sr.pp.pca, adata, n_comps=N_COMPS, random_state=0)
    print(f"          obsm['X_pca'] {adata.obsm['X_pca'].shape}")

    # ---------------------------------------------------------- neighbours and UMAP
    print("\n=== 8. Neighbourhood graph and UMAP")
    step(
        "sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)",
        'sr.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")',
    )
    timed("sr.pp.neighbors", sr.pp.neighbors, adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    step("sc.tl.umap(adata)", "sr.tl.umap(adata)")
    timed("sr.tl.umap", sr.tl.umap, adata, random_state=0)
    print(f"          obsm['X_umap'] {adata.obsm['X_umap'].shape}")

    step("sc.tl.tsne(adata)", "sr.tl.tsne(adata)  # not in the tutorial; scrust has it")
    print("          NOTE: scrust's t-SNE is exact O(n^2). It is ~4x slower than")
    print("          scanpy's Barnes-Hut at this size and 17x slower at 10 000 cells,")
    print("          and refuses above 20 000. Use sc.tl.tsne at scale.")
    timed("sr.tl.tsne", sr.tl.tsne, adata, n_pcs=N_COMPS, perplexity=30.0, random_state=0)

    # -------------------------------------------------------------------- clustering
    print("\n=== 9. Cluster the graph")
    step(
        "sc.tl.leiden(adata, resolution=1.0, key_added='leiden')",
        "sr.tl.leiden(adata, resolution=1.0, key_added='leiden')",
    )
    timed(
        "sc.tl.leiden",
        sc.tl.leiden,
        adata,
        resolution=RESOLUTION,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    print(f"          {adata.obs['leiden'].nunique()} clusters")

    # ------------------------------------------------------- differential expression
    print("\n=== 10. Rank marker genes")
    step(
        'sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")',
        'sr.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")',
    )
    # The tutorial ranks on the log-normalised matrix, not the scaled one.
    ranked = adata.raw.to_adata()[:, adata.var_names].copy()
    ranked.obs["leiden"] = adata.obs["leiden"]
    timed(
        "sr.tl.rank_genes_groups",
        sr.tl.rank_genes_groups,
        ranked,
        "leiden",
        method="wilcoxon",
    )
    adata.uns["rank_genes_groups"] = ranked.uns["rank_genes_groups"]

    step(
        'sc.get.rank_genes_groups_df(adata, group="0").head()',
        'sr.get.rank_genes_groups_df(adata, group="0").head()',
    )
    markers = sr.get.rank_genes_groups_df(adata, group="0")
    print(markers.head(5).to_string(index=False))

    # ------------------------------------------------------------------------- paga
    print("\n=== 11. Abstracted graph")
    step('sc.tl.paga(adata, groups="leiden")', 'sr.tl.paga(adata, groups="leiden")')
    timed("sr.tl.paga", sr.tl.paga, adata, "leiden")
    print(f"          uns['paga']['connectivities'] {adata.uns['paga']['connectivities'].shape}")

    # --------------------------------------------------------------------- accessors
    print("\n=== 12. Accessors")
    step(
        'sc.get.obs_df(adata, keys=["CST3", "leiden"]).head()',
        'sr.get.obs_df(adata, keys=["CST3", "leiden"]).head()',
    )
    keys = [gene for gene in ("CST3", "NKG7", "PPBP") if gene in adata.var_names][:2]
    print(sr.get.obs_df(adata, keys=[*keys, "leiden"]).head(3).to_string())
    step(
        'sc.get.aggregate(adata, by="leiden", func="mean")',
        'sr.get.aggregate(adata, by="leiden", func="mean")',
    )
    means = sr.get.aggregate(adata, "leiden", "mean")
    print(f"          aggregated to {means.shape[0]} groups x {means.shape[1]} genes")

    print("\n=== Done")
    print("Steps run by scrust: filter_cells, filter_genes, normalize_total, log1p,")
    print("highly_variable_genes, scale, pca, neighbors, umap, tsne, rank_genes_groups,")
    print("paga, and the get.* accessors.")
    print("Steps that had to fall back to scanpy: calculate_qc_metrics, leiden.")
    print("Steps skipped entirely: regress_out.")
    print("\nsc.pl.umap(adata, color=['leiden', 'CST3']) works on this object.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
