# scrust

Single-cell analysis with a Rust core that runs heavy numerics on the GPU of Apple
M-series chips. Python is the interface only: it holds the AnnData plumbing and default
settings, while matrix operations execute natively in Rust.

The API mirrors Scanpy, allowing a script to change its import and keep its plotting —
across the 40 functions `scrust` mirrors. All 40 are fully implemented as of `v0.2.0`.
Scanpy is much larger than 40 functions, so the status table below outlines what is
covered. All benchmark numbers are measured, not claimed.

```python
import scanpy as sc
import scrust as sr

adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

sr.pp.filter_cells(adata, min_genes=200)
sr.pp.filter_genes(adata, min_cells=3)
sr.pp.normalize_total(adata, target_sum=1e4)
sr.pp.log1p(adata)
sr.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

sr.pp.scale(adata, max_value=10)
sr.pp.pca(adata, n_comps=50)
sr.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
sr.tl.umap(adata)
sr.tl.leiden(adata)
sr.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

sc.pl.umap(adata, color="leiden")  # Scanpy plotting still works
```

`method=` supports all four Scanpy options: `"wilcoxon"`, `"t-test"`,
`"t-test_overestim_var"`, and `"logreg"`. As in Scanpy, `"logreg"` reports scores only
(no p-values or fold changes).

`examples/pbmc3k.py` demonstrates this pipeline end-to-end, printing each Scanpy call
alongside its scrust replacement. Every mirrored step has a scrust equivalent —
including `calculate_qc_metrics` and `leiden`. Note that timings in that script serve as
a tutorial transcript and Scanpy baseline, not a scrust benchmark. For true performance
measurements, see `benches/benchmark.py`.

## Why Rust and Metal

The computationally expensive steps of a single-cell pipeline — PCA, neighbor graph
construction, UMAP/t-SNE layouts, and differential expression — consist of large batched
matrix operations. Apple Silicon's Unified Memory Architecture (UMA) avoids copying count
matrices across a PCIe bus to reach the GPU, eliminating the overhead that traditionally
makes GPU acceleration uneconomic at single-cell matrix sizes.

Everything expressible as tensor algebra is written against candle and runs on either CPU
or Metal GPU. The CPU path uses the same algorithm and serves as the correctness oracle.

### Note on Floating-Point Parity

Same code does not mean identical bitwise outputs. `settings.device` defaults to
`"auto"`, resolving to Metal wherever available (`crates/scrust-core/src/device.rs`).
Because `f32` addition is non-associative, GPU parallel reductions land a few ULPs away
from sequential CPU execution.

This difference caused a real bug during development: the expansion
`|a-b|^2 = |a|^2 + |b|^2 - 2a.b` canceled to exactly zero for identical cells on CPU, but
left `9.5e-7` on Metal. Taking the square root amplified this to `9.8e-4`, breaking
duplicate cell connectivity (`=1.0`) in UMAP graphs. `tests/test_device_parity.py` now
enforces strict tolerance bounds between both backends.

## Benchmarks

Measured at 499, 2,638, and 10,000 cells. Speedup relative to Scanpy (values above 1.00x
mean scrust is faster):

| Operation | 499 cells | 2,638 cells | 10,000 cells |
| --- | ---: | ---: | ---: |
| `tl.paga` | 23.00x | 55.67x | 90.40x |
| `tl.rank_genes_groups` | 10.15x | 28.72x | 41.95x |
| `tl.umap` | 3.41x | 2.96x | 4.84x |
| `pp.neighbors` | 0.64x | 1.15x | 5.80x |
| `pp.pca` | 0.36x | 1.36x | 1.63x |
| `pp.log1p` | 0.43x | 0.44x | 0.45x |
| `pp.scale` | 0.17x | 0.35x | 0.42x |
| `pp.normalize_total` | 0.09x | 0.17x | 0.30x |
| `tl.tsne` | 1.42x | 0.25x | 0.06x |

### Performance Insights

- **The Wins:** Dense batched tensor operations and per-gene statistics show massive
  speedups.
- **The Losses:** Cheap elementwise operations are bandwidth-bound and too small to
  offset the Python/Rust FFI boundary crossing overhead.
- **`tl.tsne` Limitations:** Uses exact `O(N^2)` distance formulation — optimal for GPU
  tensor cores on smaller datasets, whereas Scanpy uses Barnes-Hut `O(N log N)`. It scales
  poorly beyond 10,000 cells (17x slower) and raises a `ValueError` above 20,000 cells to
  prevent OOM. Use `sc.tl.tsne` or `sr.tl.umap` at scale.

For peak memory consumption and complete benchmarks, see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Custom Metal Kernels

The `pp.neighbors` row incorporates hand-written Metal kernels. `crates/scrust-py` links
against `crates/scrust-gpu` to route k-NN graph queries to custom Metal kernels, achieving
~2-2.5x speedups over candle while maintaining bit-for-bit agreement with the CPU oracle
(`tests/test_device_parity.py`, 4/4 pass).

Three additional kernels (`spmm`, `tsne_gradient`, `umap_sgd`) are implemented and tested:
`umap_sgd` intentionally remains unwired (uses Hogwild asynchronous updates) to ensure
deterministic UMAP layouts regardless of GPU availability.

All other GPU ops execute via candle's Metal backend. See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Installation

Building from source on Apple Silicon (no PyPI wheels published yet):

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

Prerequisites: Rust toolchain and Xcode Metal compiler.

Without a GPU, operations gracefully fall back to the CPU path. For platform details and
build instructions, see [docs/INSTALL.md](docs/INSTALL.md).

**Verification:**

```bash
python -c "import scrust; print(scrust.__version__, scrust.gpu_available())"
```

## Status & Capabilities (v0.2.0)

Zero `todo!` stubs remain in `crates/`. `tl.dpt(n_branchings > 0)` now performs native
branch detection (Haghverdi 2016 port, verified ARI = 1.0000 against Scanpy).

`v0.2.0` introduces four core native capabilities verified by audit suites:

| Capability | Status | Verification |
| --- | --- | --- |
| `tl.score_genes_cell_cycle` | Native Rust | 4/4 parity tests, CPU & Metal (f64 accumulator matches Scanpy) |
| Out-of-Core Backed Streaming | Native Rust | Bit-for-bit vs in-memory; Peak RAM reduced from 1141 MB → 737 MB (0.65x) |
| `tl.dpt(n_branchings > 0)` | Native Binding | ARI = 1.0000 vs Scanpy for `n_branchings` = 1 and 2 |
| `pp.harmony_integrate` | Native Rust | iLISI 1.00 → 1.90; CPU execution reduced from 0.590s → 0.182s (3.2x) |

### Harmony Integration Usage

```python
import scrust as sr

sr.pp.pca(adata, n_comps=50)
sr.pp.harmony_integrate(adata, key="batch")      # Writes to obsm["X_pca_harmony"]
sr.pp.neighbors(adata, use_rep="X_pca_harmony")  # Downstream graph on integrated space
```

## API Coverage

| Area | Mirrored Functions |
| --- | --- |
| `pp` | `calculate_qc_metrics`, `combat`, `downsample_counts`, `filter_cells`, `filter_genes`, `filter_genes_dispersion`, `harmony_integrate`, `highly_variable_genes`, `log1p`, `neighbors`, `normalize_per_cell`, `normalize_total`, `pca`, `regress_out`, `sample`, `scale`, `sqrt`, `subsample` |
| `tl` | `dendrogram`, `diffmap`, `dpt`, `draw_graph`, `embedding_density`, `filter_rank_genes_groups`, `leiden`, `louvain`, `marker_gene_overlap`, `paga`, `rank_genes_groups`, `score_genes`, `score_genes_cell_cycle`, `tsne`, `umap` |
| `metrics` | `morans_i`, `gearys_c`, `confusion_matrix`, `modularity` |
| `get` | `obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate` |

For intentional behavior divergences from Scanpy (e.g., `tl.diffmap` clamping, `tl.dpt`
pseudotime values), see [docs/VALIDATION.md](docs/VALIDATION.md).

## Repository Layout

```text
crates/
  scrust-core/    Core data types and algorithms written against candle
  scrust-gpu/     Metal context & custom kernels (knn bound to Python)
  scrust-py/      PyO3 bindings for zero-copy FFI data transfer
python/scrust/    Scanpy-shaped API wrappers and AnnData integration
benches/          Performance benchmarks (benchmark.py, streaming.py)
examples/         Full PBMC 3k walkthrough pipeline
tests/            Comprehensive integration and audit test suite
```

## Development & Testing

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/pytest
```

`tests/` contains 843 Python tests across 40 files. Audits run against the device
specified by `SCRUST_TEST_DEVICE` (default `"cpu"`, set to `"auto"` for GPU testing).
