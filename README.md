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

Measured on an Apple M3 Pro (18 GB) after the Apple silicon optimisation pass, PBMC 3k bootstrapped to
10 000 and 50 000 cells. Speedup relative to Scanpy (values above 1.00x mean scrust is faster); full
before/after table and method in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Operation | 10,000 cells | 50,000 cells |
| --- | ---: | ---: |
| `tl.rank_genes_groups` | 43.3x | 59.7x |
| `pp.neighbors` | 20.0x | 4.5x |
| `pp.scale` | 5.5x | 17.8x |
| `pp.pca` | 5.7x | 6.7x |
| `tl.leiden` | 134x | 29x |
| `tl.umap(parallel=True)` | about 28x | 6x faster than sequential |
| `tl.umap` (default, deterministic) | 5.0x | not measured |
| `pp.log1p` | 3.1x | 2.1x |
| `pp.normalize_total` | 1.9x | 1.3x |
| `pp.highly_variable_genes` | 1.0x | 1.1x |
| `tl.tsne` | see below | refuses > 20,000 cells |

### Performance Insights

- **The Wins:** Dense batched tensor operations and per-gene statistics show massive
  speedups.
- **Elementwise steps:** `normalize_total`, `log1p` and `scale` now run zero-copy on numpy's
  buffers across all performance and efficiency cores, so they no longer lose to the FFI overhead.
- **`tl.tsne` Limitations:** Uses exact `O(N^2)` distance formulation — optimal for GPU
  tensor cores on smaller datasets, whereas Scanpy uses Barnes-Hut `O(N log N)`. It scales
  poorly beyond 10,000 cells (17x slower) and raises a `ValueError` above 20,000 cells to
  prevent OOM. Use `sc.tl.tsne` or `sr.tl.umap` at scale.

For peak memory consumption and complete benchmarks, see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Custom Metal Kernels

The `pp.neighbors` row runs a hand-written, tiled Metal k-NN kernel: 64 queries per threadgroup,
candidate tiles in threadgroup memory, and a shader specialised per `(n_dims, k)` so each query row
and its top-k list stay in registers. It returns the same neighbours as the CPU oracle
(`tests/test_device_parity.py`, and a brute-force test with duplicated points).

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

Prerequisites: a Rust toolchain. The Metal shaders are compiled at run time by the system Metal
framework, so the Xcode Metal toolchain is not needed. Apple Accelerate BLAS is linked by default.

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
