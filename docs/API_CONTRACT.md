# API contract

Fifteen branches are developed in parallel against this document. The signatures
below already exist as stubs on `main`, so the workspace always compiles; a
branch fills in bodies and adds tests, and never renames anything here.

## Ownership

| Branch | Files | Delivers |
| --- | --- | --- |
| `feat/preprocess-basics` | `scrust-core/src/preprocess/{normalize,filter,scale}.rs` | `normalize_total`, `log1p`, `filter_cells`, `filter_genes`, `subset`, `scale` |
| `feat/hvg` | `scrust-core/src/preprocess/hvg.rs` | `highly_variable_genes` |
| `feat/pca` | `scrust-core/src/pca.rs` | `pca` by randomised SVD |
| `feat/neighbors` | `scrust-core/src/neighbors.rs` | `knn`, `connectivities` |
| `feat/knn-kernel` | `scrust-gpu/src/kernels/knn.rs` | `knn_metal`, fused distance + selection |
| `feat/umap` | `scrust-core/src/umap.rs` | `umap` |
| `feat/umap-kernel` | `scrust-gpu/src/kernels/umap_sgd.rs` | `umap_epoch` |
| `feat/tsne` | `scrust-core/src/tsne.rs` | `tsne` |
| `feat/tsne-kernel` | `scrust-gpu/src/kernels/tsne_gradient.rs` | `tsne_gradient` |
| `feat/wilcoxon` | `scrust-core/src/de/wilcoxon.rs` | `rank_genes_groups_wilcoxon` |
| `feat/glm` | `scrust-core/src/de/glm.rs` | `fit_negative_binomial` |
| `feat/dea-stats` | `scrust-core/src/de/{dispersion,hypothesis,multiple_testing}.rs` | size factors, dispersions, Wald test, BH |
| `feat/bindings` | `scrust-py/src/{convert,preprocess,embedding,de}.rs` | the `scrust._scrust` extension module |
| `feat/python-api` | `python/scrust/{pp,tl}.py` | the scanpy-shaped Python API |
| `feat/reference-tests` | `tests/`, `benches/`, `.github/workflows/` | scanpy cross-checks, benchmarks, CI |

Files owned by `main` and never edited on a branch: `Cargo.toml` (workspace and
crates), `.cargo/config.toml`, `pyproject.toml`, `scrust-core/src/{lib,error,device,sparse}.rs`,
`scrust-gpu/src/{lib,context}.rs`, every `mod.rs`, `python/scrust/__init__.py`.

## Core conventions

- Matrices are **cells by genes**, matching AnnData. `CsrMatrix` crosses the
  Python boundary; algorithms densify row blocks when they need a tensor.
- `f32` throughout. The Apple GPU has no `f64`, and scanpy's own results are
  `f32` after normalisation.
- Every algorithm takes `&candle_core::Device` and is written once against
  candle tensors, so the CPU path is the same code and acts as the oracle.
- Metal kernels in `scrust-gpu` are **optimisations, not separate algorithms**:
  each must return what its `scrust-core` counterpart returns, and its tests
  must assert that.
- Randomness takes an explicit seed. Two runs with the same seed are identical.
- Errors are `scrust_core::Error`; never `panic!` or `unwrap` on user input.

## Python API

Signatures mirror scanpy so that existing scripts change only their import.
Results go into the slots scanpy uses:

```python
pp.filter_cells(adata, *, min_genes=None, min_counts=None, inplace=True)
pp.filter_genes(adata, *, min_cells=None, min_counts=None, inplace=True)
pp.normalize_total(adata, *, target_sum=None, inplace=True)
pp.log1p(adata, *, inplace=True)
pp.highly_variable_genes(adata, *, n_top_genes=2000, flavor="seurat", inplace=True)
pp.scale(adata, *, zero_center=True, max_value=None, inplace=True)
pp.pca(adata, *, n_comps=50, zero_center=True, random_state=0, device="auto")
pp.neighbors(adata, *, n_neighbors=15, use_rep="X_pca", device="auto")

tl.umap(adata, *, n_components=2, min_dist=0.5, spread=1.0, n_epochs=None,
        random_state=0, device="auto")
tl.tsne(adata, *, n_pcs=50, perplexity=30.0, early_exaggeration=12.0,
        learning_rate=200.0, random_state=0, device="auto")
tl.rank_genes_groups(adata, groupby, *, groups="all", reference="rest",
                     method="wilcoxon", device="auto")
```

| Function | Writes |
| --- | --- |
| `pp.pca` | `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]["variance_ratio"]` |
| `pp.neighbors` | `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]` |

**Off-by-one, verified against scanpy:** scanpy's `n_neighbors` counts the cell
itself, while `KnnGraph` excludes it. `pp.neighbors(n_neighbors=15)` must call
`knn(embedding, 14)`. Getting this wrong shifts every neighbour set by one and
is invisible in a smoke test.
| `pp.highly_variable_genes` | `var["highly_variable"]`, `var["means"]`, `var["dispersions_norm"]` |
| `tl.umap` | `obsm["X_umap"]` |
| `tl.tsne` | `obsm["X_tsne"]` |
| `tl.rank_genes_groups` | `uns["rank_genes_groups"]` with `names`, `scores`, `pvals`, `pvals_adj`, `logfoldchanges` as structured arrays, one field per group |

## Extension module

`scrust._scrust` exposes flat, typed functions. The Python layer owns defaults
and AnnData plumbing; the Rust layer owns none of it.

```python
_scrust.gpu_available() -> bool
_scrust.normalize_total(indptr, indices, values, n_cols, target_sum, device) -> tuple
_scrust.log1p(indptr, indices, values, n_cols) -> tuple
_scrust.filter_cells(indptr, indices, values, n_cols, min_genes, min_counts) -> np.ndarray[bool]
_scrust.filter_genes(indptr, indices, values, n_cols, min_cells, min_counts) -> np.ndarray[bool]
_scrust.scale(indptr, indices, values, n_cols, zero_center, max_value, device) -> np.ndarray
_scrust.highly_variable_genes(indptr, indices, values, n_cols, n_top_genes, flavor, device) -> dict
_scrust.pca(indptr, indices, values, n_cols, n_components, zero_center, seed, device) -> dict
_scrust.knn(embedding, k, device) -> tuple[np.ndarray, np.ndarray]
_scrust.connectivities(indices, distances) -> tuple
_scrust.umap(indptr, indices, values, n_cols, params..., device) -> np.ndarray
_scrust.tsne(embedding, params..., device) -> np.ndarray
_scrust.rank_genes_groups_wilcoxon(indptr, indices, values, n_cols, labels,
                                   n_groups, reference, tie_correct, device) -> dict
```

Sparse matrices cross as the three CSR arrays plus `n_cols`; a `tuple` return of
the same shape is a sparse result. Dense results are 2-D `numpy` arrays.

## scanpy is the reference

Correctness is defined by agreement with scanpy on the same input, and every
branch must state what it measured. What agreement means differs per algorithm:

| Algorithm | Asserted against scanpy |
| --- | --- |
| `normalize_total`, `log1p`, `filter_*`, `scale` | element-wise equality, `rtol=1e-5` |
| `highly_variable_genes` | the selected gene set overlaps by >= 95% |
| `pca` | `abs(corr)` per component >= 0.99 (sign is arbitrary), variance ratios to `rtol=1e-3` |
| `neighbors` | >= 90% of each cell's neighbour set shared |
| `umap`, `tsne` | neighbourhood preservation: >= 80% of a cell's 15 nearest neighbours in the scanpy embedding are within its 30 nearest in ours. Coordinates are **not** comparable |
| `rank_genes_groups` | identical gene ranking for the top 100 per group, scores to `rtol=1e-3` |

Where a deviation is real and defensible, document it rather than loosening a
threshold silently.
