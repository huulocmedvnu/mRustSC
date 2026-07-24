# %% [markdown]
# # PBMC 3k end-to-end, on the pure scrust backend
#
# This is the scanpy PBMC 3k clustering tutorial with **every computational step run by
# `scrust`** on the Apple GPU (Metal via candle, plus the hand-written `knn` kernel behind
# `pp.neighbors`). `scanpy` appears only to
#
# * **load** the dataset (`sc.datasets.pbmc3k`), and
# * **plot** the results (`sc.pl.*`), because plotting is scanpy's job and the results land
#   in the AnnData slots `sc.pl` reads.
#
# There is not a single `sc.pp.*` or `sc.tl.*` call below — that is enforced by an AST
# check (`_assert_no_scanpy_compute`) the script runs on itself before any work.
#
# Run it as a script:
#
# ```bash
# PYTHONPATH=$PWD/python .venv/bin/python docs/tutorials/pbmc3k_clustering.py
# ```
#
# or open the notebook built from it with `jupytext --to notebook
# docs/tutorials/pbmc3k_clustering.py`. The cells run top to bottom, so a kernel with
# `scrust` installed executes the whole tutorial with *Run All*.

# %% [markdown]
# ## Step 1 — Imports and Metal GPU availability
#
# `scrust` mirrors scanpy's module layout (`pp`, `tl`), so the calls read the same; the
# arithmetic runs in Rust. `sr.gpu_available()` reports whether Metal came up — when it
# did, `device="auto"` puts every device-aware step on the GPU without the caller choosing.

# %%
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc  # loading + plotting ONLY — never sc.pp.* / sc.tl.*

import scrust as sr  # every computational step

plt.switch_backend("Agg")  # headless: render figures to files, never open a window

# Resolve the repo root whether this runs as a script (`__file__` exists) or inside a
# notebook kernel (it does not), by walking up to the directory holding pyproject.toml.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
REPO_ROOT = next((p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists()), _HERE)
FIGURE_DIR = REPO_ROOT / "docs" / "tutorials" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the dataset the test suite already cached, so this runs offline.
sc.settings.datasetdir = REPO_ROOT / ".cache" / "scanpy"
sc.settings.figdir = FIGURE_DIR
sc.settings.verbosity = 1

print(f"scrust {sr.__version__} | Metal GPU available: {sr.gpu_available()}")

# %% [markdown]
# ## Compliance check — zero scanpy computation
#
# Parse this file and fail if any `sc.pp.*` or `sc.tl.*` attribute appears: the guarantee
# that no computation touched scanpy. It runs only as a script (a notebook kernel has no
# `__file__`), so *Run All* in Jupyter skips it rather than erroring.


# %%
def _assert_no_scanpy_compute(path: Path) -> None:
    """Fail if the source calls into `sc.pp.*` or `sc.tl.*` (scanpy computation)."""
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        # Match attribute chains rooted at `sc`, e.g. sc.pp.pca / sc.tl.leiden.
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sc"
            and node.value.attr in {"pp", "tl"}
        ):
            offenders.append(f"line {node.lineno}: sc.{node.value.attr}.{node.attr}")
    if offenders:
        raise AssertionError("scanpy computation found:\n  " + "\n  ".join(offenders))
    print("compliance: 0 calls to sc.pp.* or sc.tl.* in the pipeline")


if "__file__" in globals():
    _assert_no_scanpy_compute(Path(__file__).resolve())

# %% [markdown]
# ## Step 2 — Load PBMC 3k
#
# Reading is scanpy's job; scrust ships no readers. `var_names_make_unique` avoids the
# duplicate-gene warning downstream.

# %%
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
print(f"loaded {adata.n_obs} cells x {adata.n_vars} genes")

# %% [markdown]
# ## Step 3 — Quality control and filtering
#
# Flag mitochondrial genes (a boolean column of `var`), then `sr.pp.calculate_qc_metrics`
# fills the per-cell QC columns scanpy uses. Cells and genes are filtered with
# `sr.pp.filter_cells` / `sr.pp.filter_genes`; the mitochondrial-fraction cut is plain
# AnnData boolean indexing (numpy), not a library call.

# %%
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sr.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

sr.pp.filter_cells(adata, min_genes=200)
sr.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
print(f"after QC: {adata.n_obs} cells x {adata.n_vars} genes")

# %% [markdown]
# ## Step 4 — Normalisation and log transform
#
# Counts-per-10k normalisation followed by `log1p`. `adata.raw` keeps the full
# log-normalised matrix, which the differential-expression step reads back later.

# %%
sr.pp.normalize_total(adata, target_sum=1e4)
sr.pp.log1p(adata)
adata.raw = adata

# %% [markdown]
# ## Step 5 — Highly variable genes and scaling
#
# Select the 2000 most variable genes (Seurat flavour), subset to them, then z-score each
# gene with `sr.pp.scale`. Scaling densifies, so subsetting to HVGs first keeps it small.

# %%
sr.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
sr.pp.scale(adata, zero_center=True, max_value=10)
print(f"kept {adata.n_vars} highly variable genes")

# %% [markdown]
# ## Step 6 — PCA
#
# Truncated PCA (randomised SVD, on the GPU). Writes `obsm["X_pca"]`, `varm["PCs"]` and
# `uns["pca"]`, so `sc.pl.pca_variance_ratio` can read the spectrum.

# %%
sr.pp.pca(adata, n_comps=50, random_state=0)
print(f"obsm['X_pca'] {adata.obsm['X_pca'].shape}")

# %% [markdown]
# ## Step 7 — k-NN graph, with the Metal GPU kernel
#
# `device="auto"` routes the k-NN search to the hand-written `knn` Metal kernel on Apple
# silicon (the first scrust GPU kernel on the call path), and to the candle CPU path
# otherwise. Both produce the same neighbour graph — `tests/test_device_parity.py` pins
# them equal. Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.

# %%
sr.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", device="auto")
print(f"neighbour graph on {adata.n_obs} cells (device='auto')")

# %% [markdown]
# ## Step 8 — Clustering and UMAP embedding
#
# Leiden clustering over the neighbour graph, then a UMAP layout for visualisation. Both
# read `obsp["connectivities"]`; `sr.tl.leiden` writes `obs["leiden"]` and `sr.tl.umap`
# writes `obsm["X_umap"]`.

# %%
sr.tl.leiden(adata, resolution=1.0, key_added="leiden")
sr.tl.umap(adata, random_state=0)
n_clusters = adata.obs["leiden"].nunique()
print(f"{n_clusters} Leiden clusters; obsm['X_umap'] {adata.obsm['X_umap'].shape}")

# %% [markdown]
# ## Step 9 — Differential expression (marker genes)
#
# Rank marker genes per cluster with the Wilcoxon rank-sum test. The test runs on the
# **log-normalised** matrix (recovered from `adata.raw`), not the scaled one, because fold
# changes on z-scored data are meaningless — this mirrors the scanpy tutorial.

# %%
ranked = adata.raw.to_adata()[:, adata.var_names].copy()
ranked.obs["leiden"] = adata.obs["leiden"]
sr.tl.rank_genes_groups(ranked, "leiden", method="wilcoxon")
top = ranked.uns["rank_genes_groups"]["names"][0]
print(f"top marker per cluster: {list(top)}")

# %% [markdown]
# ## Step 10 — Visualisation (scanpy plotting only)
#
# scrust wrote every result into the slots `sc.pl.*` reads, so scanpy's plotting works
# unchanged. Figures are saved under `docs/tutorials/figures/`.

# %%
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False)
plt.savefig(FIGURE_DIR / "pca_variance_ratio.png", dpi=120, bbox_inches="tight")
plt.close()

sc.pl.umap(adata, color=["leiden"], show=False)
plt.savefig(FIGURE_DIR / "umap_leiden.png", dpi=120, bbox_inches="tight")
plt.close()

sc.pl.rank_genes_groups(ranked, n_genes=20, sharey=False, show=False)
plt.savefig(FIGURE_DIR / "rank_genes_groups.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"figures written to {FIGURE_DIR}")
