# scrust

Single-cell analysis with a Rust core that runs the heavy numerics on the GPU of
Apple M-series chips. Python is the interface only: it holds the AnnData
plumbing and the defaults, and every matrix operation happens in Rust.

The API mirrors scanpy, so an existing script changes its import and keeps its
plotting:

```python
import scanpy as sc
import scrust as sr

adata = sc.read_10x_mtx("filtered_gene_bc_matrices/hg19/")
sr.pp.filter_cells(adata, min_genes=200)
sr.pp.normalize_total(adata)
sr.pp.log1p(adata)
sr.pp.highly_variable_genes(adata, n_top_genes=2000)
sr.pp.pca(adata, n_comps=50)
sr.pp.neighbors(adata, n_neighbors=15)
sr.tl.umap(adata)
sr.tl.rank_genes_groups(adata, "louvain")
sc.pl.umap(adata, color="louvain")     # scanpy plotting still works
```

## Why Rust and Metal

The expensive steps of a single-cell pipeline — PCA, the neighbour graph, UMAP
and t-SNE layouts, differential expression — are large batched matrix
operations. Apple silicon's unified memory means the count matrix is not copied
across a bus to reach the GPU, which is what usually makes GPU acceleration
uneconomic at single-cell matrix sizes.

Everything expressible as tensor algebra is written once against
[candle](https://github.com/huggingface/candle) and runs on either device, so
the CPU path is the same code and serves as the correctness oracle. Only the
irregular inner loops — nearest-neighbour selection, UMAP's negative sampling,
t-SNE's repulsive forces — have hand written Metal kernels.

## Correctness

scanpy is the reference. Every algorithm is cross-checked against it on real
data, with the form of agreement chosen per algorithm — element-wise for
deterministic transforms, neighbourhood preservation for stochastic embeddings.
`docs/API_CONTRACT.md` lists what is asserted, `docs/VALIDATION.md` what was
measured.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

Requires a Rust toolchain and, for the GPU path, Apple silicon with Xcode's
Metal compiler. Without a GPU everything still runs on the CPU.

## Layout

```
crates/
  scrust-core/    data types and every algorithm, written against candle
  scrust-gpu/     Metal context and hand written kernels
  scrust-py/      PyO3 bindings, conversion only
python/scrust/    the scanpy-shaped API and AnnData plumbing
tests/            cross-checks against scanpy
```

## Develop

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
.venv/bin/pytest
```
