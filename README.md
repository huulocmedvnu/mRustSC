# Metalcyte

[![PyPI](https://img.shields.io/pypi/v/metalcyte.svg)](https://pypi.org/project/metalcyte/)
[![Python](https://img.shields.io/pypi/pyversions/metalcyte.svg)](https://pypi.org/project/metalcyte/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20silicon-lightgrey.svg)](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/INSTALL.md)
[![CI](https://github.com/huulocmedvnu/metalcyte/actions/workflows/ci.yml/badge.svg)](https://github.com/huulocmedvnu/metalcyte/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/huulocmedvnu/metalcyte/blob/main/LICENSE)

**Single-cell RNA-seq analysis at atlas scale on an Apple silicon laptop.**

Metalcyte is a Rust engine with a scanpy-compatible Python interface. It runs the standard
single-cell pipeline on every CPU core, on the matrix coprocessor through Apple Accelerate and on
the integrated GPU through Metal kernels, all within the chip's unified memory. An out-of-core mode
processes count matrices larger than the machine's memory.

![UMAP of 4 062 980 human embryonic cells coloured by cell type](https://raw.githubusercontent.com/huulocmedvnu/metalcyte/main/docs/figures/embryo4m_umap.png)

*The complete survey of human embryonic development, 4 062 980 cells by 45 676 genes, processed
from raw counts to clusters in 9 minutes on a laptop with 18 GB of memory.*

## Highlights

- **Fast.** A 117 308-cell bone-marrow atlas goes from counts to marker genes in 12 s, against
  213 s for scanpy at its defaults.
- **Memory-frugal.** A million cells run within 18 GB, and 4 million cells peak at about 9 GB.
- **Energy-efficient.** The same analysis uses one seventh of scanpy's energy.
- **Validated.** Every algorithm is tested against scanpy's output with a stated tolerance.
- **Drop-in.** Results go to the standard AnnData slots, so scanpy's plotting and downstream
  tools read them unchanged.

## Features

**Preprocessing.** Quality-control metrics, cell and gene filtering, library-size normalisation,
log and square-root transforms, highly variable gene selection (Seurat and Cell Ranger
flavours), scaling, and principal component analysis on the GPU.

**Out-of-core analysis.** Quality control, normalisation, feature selection, scaling and PCA in
four streamed passes over an h5ad file on disk, without loading the matrix.

**Batch correction and integration.** Regression of covariates, ComBat, and Harmony integration
on all cores.

**Neighbour graphs.** Exact k-nearest-neighbour search on the GPU and approximate search by
NN-descent for large datasets, selected automatically by dataset size.

**Embeddings.** UMAP with a parallel optimiser, t-SNE with FFT-accelerated interpolation on the
GPU, diffusion maps, force-directed layouts and PAGA.

**Clustering and trajectories.** Leiden and Louvain community detection, diffusion pseudotime
and dendrograms of clusters.

**Marker genes.** Wilcoxon rank-sum, t-test and logistic-regression marker tests, a Wilcoxon test
streamed from disk for datasets beyond memory, marker overlap, and gene-set and cell-cycle scoring.

**Spatial and clustering metrics.** Moran's I, Geary's C, modularity and confusion matrices.

**Plotting.** Embedding scatter plots rasterised on the GPU, which draw millions of points in
about a second.

The full reference is in [docs/API.md](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/API.md).

## Installation

On a Mac with Apple silicon (macOS 13 or later) and Python 3.11 to 3.13:

```bash
pip install metalcyte
pip install "metalcyte[plot]"   # with matplotlib for plotting
```

Neither Rust nor Xcode is required. To build from source, see [docs/INSTALL.md](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/INSTALL.md).

## Quick start

```python
import metalcyte as mc
import scanpy as sc  # example data only

adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

mc.pp.filter_cells(adata, min_genes=200)
mc.pp.filter_genes(adata, min_cells=3)
mc.pp.normalize_total(adata, target_sum=1e4)
mc.pp.log1p(adata)
mc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
mc.pp.scale(adata, max_value=10)
mc.pp.pca(adata, n_comps=50)
mc.pp.neighbors(adata, n_neighbors=15)
mc.tl.umap(adata, parallel=True)
mc.tl.leiden(adata)
mc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

mc.pl.umap(adata, color="leiden")
```

For a dataset larger than memory, start from the counts file on disk:

```python
adata = mc.pp.preprocess_backed("atlas_counts.h5ad", n_top_genes=2000, n_comps=50)
mc.pp.neighbors(adata)
mc.tl.umap(adata, parallel=True)
mc.tl.leiden(adata)
```

A step-by-step tutorial is in [docs/tutorials/pbmc3k_clustering.ipynb](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/tutorials/pbmc3k_clustering.ipynb).

## Benchmarks

Apple M3 Pro, 18 GB of memory, scanpy 1.12.4. Full methodology in
[docs/PERFORMANCE.md](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/PERFORMANCE.md).

| Dataset | Cells | scanpy | Metalcyte | Metalcyte memory |
|---|---:|---:|---:|---:|
| Bone marrow, counts to markers | 117 308 | 213 s | **12 s** | |
| Human embryo, counts to clusters | 1 001 288 | did not finish¹ | **103 s** | under 1.5 GB per step |
| Human embryo, full survey | 4 062 980 | not run | **538 s** | about 9 GB at peak |

¹ scanpy's scaling step needed 21.9 GB on the 18 GB machine, and the run was stopped after
15 minutes of swapping.

| Energy, 117 308 cells | Time | Energy |
|---|---:|---:|
| scanpy | 237 s | 1 228 J |
| Metalcyte | 13 s | **168 J** |

Agreement with scanpy on the bone-marrow atlas: the same highly variable genes, the same principal
subspace, Leiden clusterings at an adjusted Rand index of 0.95 and the same marker genes. Details in
[docs/VALIDATION.md](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/VALIDATION.md).

## Documentation

- [Installation](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/INSTALL.md)
- [API reference](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/API.md)
- [Performance and methodology](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/PERFORMANCE.md)
- [Validation against scanpy](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/VALIDATION.md)
- [How it works](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/HOW_IT_WORKS.md) and [architecture](https://github.com/huulocmedvnu/metalcyte/blob/main/docs/ARCHITECTURE.md)
- [Changelog](https://github.com/huulocmedvnu/metalcyte/blob/main/CHANGELOG.md)

## Citation

If you use Metalcyte in your research, please cite it as described in [CITATION.cff](https://github.com/huulocmedvnu/metalcyte/blob/main/CITATION.cff).
A manuscript is in preparation.

## Contributing

Bug reports and feature requests are welcome in the
[issue tracker](https://github.com/huulocmedvnu/metalcyte/issues). See
[CONTRIBUTING.md](https://github.com/huulocmedvnu/metalcyte/blob/main/CONTRIBUTING.md) for the development setup.

## License

Metalcyte is released under the [MIT License](https://github.com/huulocmedvnu/metalcyte/blob/main/LICENSE).
