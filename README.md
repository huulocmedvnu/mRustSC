# Metalcyte

**Metalcyte** is a single-cell analysis engine for Apple silicon: a Rust core with a Python interface
that uses the whole package the chip offers, every performance and efficiency core, the AMX matrix units
through Accelerate, the GPU through hand-written Metal kernels, and one unified memory, so an
atlas-scale dataset runs on the laptop on the desk. A 117 308-cell atlas goes from counts to marker
genes in 20 s, a 1 001 288-cell atlas from counts to Leiden clusters in 210 s on 18 GB, with an
out-of-core head that never holds the matrix. Python is the interface only: it holds the AnnData
plumbing and defaults. Results land in the standard AnnData slots, so a script written for scanpy
runs on Metalcyte after changing its import and keeps its plotting.

All benchmark numbers are measured, not claimed.

```python
import scanpy as sc
import metalcyte as mc

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
mc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
mc.tl.umap(adata)
mc.tl.leiden(adata)
mc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

sc.pl.umap(adata, color="leiden")  # Scanpy plotting still works
```

`method=` supports all four Scanpy options: `"wilcoxon"`, `"t-test"`,
`"t-test_overestim_var"`, and `"logreg"`. As in Scanpy, `"logreg"` reports scores only
(no p-values or fold changes).

`examples/pbmc3k.py` demonstrates this pipeline end-to-end, printing each Scanpy call
alongside its metalcyte replacement. Every mirrored step has a metalcyte equivalent —
including `calculate_qc_metrics` and `leiden`. Note that timings in that script serve as
a tutorial transcript and Scanpy baseline, not a metalcyte benchmark. For true performance
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
`"auto"`, resolving to Metal wherever available (`crates/metalcyte-core/src/device.rs`).
Because `f32` addition is non-associative, GPU parallel reductions land a few ULPs away
from sequential CPU execution.

This difference caused a real bug during development: the expansion
`|a-b|^2 = |a|^2 + |b|^2 - 2a.b` canceled to exactly zero for identical cells on CPU, but
left `9.5e-7` on Metal. Taking the square root amplified this to `9.8e-4`, breaking
duplicate cell connectivity (`=1.0`) in UMAP graphs. `tests/test_device_parity.py` now
enforces strict tolerance bounds between both backends.

## Benchmarks

Measured on an Apple M3 Pro laptop (5 performance + 6 efficiency cores, 14-core GPU, 18 GB unified
memory), scanpy 1.12.4. Every number below is in `benches/results/` with the script that produced it;
the full write-up, including what was tried and did not work, is [docs/SCALE.md](docs/SCALE.md).

**A real atlas, start to finish.** 117 308 bone-marrow cells (CELLxGENE), QC → filters → normalise →
log1p → 2 000 variable genes → scale → PCA → 15-NN graph → UMAP → Leiden → Wilcoxon markers
(`benches/pipeline.py`):

| | scanpy (defaults) | scanpy (tuned) | metalcyte, CPU only | metalcyte, Metal |
|---|---:|---:|---:|---:|
| whole pipeline | 233 s | 86 s | 29 s | **20 s** |
| energy (powermetrics, idle subtracted) | 974 J | | | **195 J** |
| PCA / neighbours / UMAP / Leiden | 13.6 / 17.9 / 46.8 / 135 s | 3.7 / 18.2 / 46.1 / 2.5 s | 3.1 / 4.9 / 4.9 / 0.7 s | 1.1 / 2.0 / 4.8 / 0.6 s |

"Tuned" is scanpy with `covariance_eigh` PCA, `igraph` Leiden and an unseeded UMAP, the fastest
settings it offers; metalcyte uses `tl.umap(parallel=True)`. The two libraries find the same biology
(same variable genes, the same PCA subspace, Leiden clusterings that agree; `benches/agreement.py`).

**A million cells on 18 GB.** 1 001 288 human embryo cells (CELLxGENE), through
`mc.pp.preprocess_backed` (QC, filters, normalise, log1p, HVG, scale and PCA over row blocks of the
on-disk counts, never holding the matrix) and then the graph steps in memory
(`benches/pipeline_1m.py`):

| | scanpy | metalcyte, CPU only | metalcyte, Metal |
|---|---:|---:|---:|
| QC → Leiden, 953 436 cells kept | did not finish (`scale` needed 21.9 GB, PCA swapped) | 409 s | **210 s** |
| peak memory added per step | | < 1.5 GB | < 1.5 GB |

**Which part of the chip buys what** (`benches/ablation.py`, 117k atlas): the GPU is worth 2.5x on
the neighbour search and 2x on PCA and nothing elsewhere; all eleven cores are worth 6.6x on the UMAP
optimiser, with the efficiency cores carrying a third of it; the zero-copy numpy borrows are worth
4.5 s of a 20 s run; Accelerate (AMX) does not show at 2 000 genes.

**Per operation, bootstrapped PBMC 3k.** `benches/benchmark.py` times each step on its own at 10k,
50k and 100k cells; the table is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The short version:
PCA 6 to 80x, the neighbour graph 2.5 to 20x, Leiden 30 to 130x, Wilcoxon 40 to 73x, the elementwise
steps at parity, `tl.tsne` exact and capped at 20 000 cells.

### Where the time still goes

- The exact neighbour search is quadratic: 2 s at 117k, 120 s at 1M on the GPU. An approximate
  index is the next step above a million cells.
- The Metal neighbour kernel runs at about 15% of the chip's arithmetic peak. A
  `simdgroup_float8x8` version is in the tree, opt-in and documented as not yet faster.
- scanpy's `tl.umap` and `tl.leiden` at their defaults are single-threaded; the tuned column is
  the fair one to quote against.

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
python -c "import metalcyte; print(metalcyte.__version__, metalcyte.gpu_available())"
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
import metalcyte as mc

mc.pp.pca(adata, n_comps=50)
mc.pp.harmony_integrate(adata, key="batch")      # Writes to obsm["X_pca_harmony"]
mc.pp.neighbors(adata, use_rep="X_pca_harmony")  # Downstream graph on integrated space
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
  metalcyte-core/    Core data types and algorithms written against candle
  metalcyte-gpu/     Metal context & custom kernels (knn bound to Python)
  metalcyte-py/      PyO3 bindings for zero-copy FFI data transfer
python/metalcyte/    Scanpy-shaped API wrappers and AnnData integration
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
specified by `METALCYTE_TEST_DEVICE` (default `"cpu"`, set to `"auto"` for GPU testing).
