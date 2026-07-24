# scrust

Single-cell analysis with a Rust core that runs the heavy numerics on the GPU of
Apple M-series chips. Python is the interface only: it holds the AnnData plumbing
and the defaults, and every matrix operation happens in Rust.

The API mirrors scanpy, so a script changes its import and keeps its plotting — for
the 40 functions scrust mirrors. All 40 are implemented at 0.2.0; scanpy is much
larger than 40 functions, so the [status table](#status) below is the list of what
is covered, and the numbers are [measured, not claimed](docs/VALIDATION.md).

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
sc.pl.umap(adata, color="leiden")        # scanpy plotting still works
```

`method=` takes all four of scanpy's values: `"wilcoxon"`, `"t-test"`,
`"t-test_overestim_var"` and `"logreg"`. As in scanpy, `"logreg"` reports scores
only — no p-values and no fold changes.

`examples/pbmc3k.py` is this pipeline end to end, printing each scanpy call beside the
scrust call that replaces it. Every mirrored step has a scrust equivalent — including
`calculate_qc_metrics` and `leiden` — so the transcript shows the pairs in full. The
timings the script prints still call scanpy for every step: it is a tutorial transcript
and a scanpy baseline, not a scrust benchmark. The benchmark is `benches/benchmark.py`.

## Why Rust and Metal

The expensive steps of a single-cell pipeline — PCA, the neighbour graph, the UMAP
and t-SNE layouts, differential expression — are large batched matrix operations.
Apple silicon's unified memory means the count matrix is not copied across a bus to
reach the GPU, which is what usually makes GPU acceleration uneconomic at
single-cell matrix sizes.

Everything expressible as tensor algebra is written once against
[candle](https://github.com/huggingface/candle) and runs on either device, so the
CPU path is the same code and serves as the correctness oracle.

Same code is not the same bits. `settings.device` defaults to `"auto"`, which
resolves to Metal wherever Metal exists (`crates/scrust-core/src/device.rs`), so a
caller who names no device is on the GPU. f32 addition is not associative and a GPU
reduction lands a few ulps from a sequential one; the two devices agree to within
floating-point tolerance, not exactly. That gap has produced at least one real bug:
the expansion `|a-b|^2 = |a|^2 + |b|^2 - 2a.b` cancelled to exactly zero for two
identical cells on the CPU but left 9.5e-7 on Metal, which the square root amplified
to 9.8e-4, and duplicate cells stopped being at connectivity 1 in the UMAP graph.
`tests/test_device_parity.py` now holds the devices against each other.

This buys speed unevenly, and the benchmark reports both directions. Measured at
499, 2 638 and 10 000 cells, speedup against scanpy (above 1.00x scrust is faster):

| operation | 499 | 2 638 | 10 000 |
| --- | ---: | ---: | ---: |
| `tl.paga` | 23.00x | 55.67x | 90.40x |
| `tl.rank_genes_groups` | 10.15x | 28.72x | 41.95x |
| `tl.umap` | 3.41x | 2.96x | 4.84x |
| `pp.neighbors` | 0.64x | 1.15x | 5.80x |
| `pp.pca` | 0.36x | 1.36x | 1.63x |
| `pp.log1p` | 0.43x | 0.44x | 0.45x |
| `pp.scale` | 0.17x | 0.35x | 0.42x |
| `pp.normalize_total` | 0.09x | 0.17x | 0.30x |
| `tl.tsne` | 1.42x | 0.25x | **0.06x** |

The wins are the dense batched work and the per-gene statistics. The losses are the
cheap elementwise steps, which are bandwidth-bound work too small to repay crossing
the Python/Rust boundary, and which do not reach parity even at 10 000 cells.

**`tl.tsne` is a genuine limitation, not overhead.** It is exact `O(n^2)` — chosen
because a dense quadratic form is a matmul the GPU eats — where scanpy uses
Barnes-Hut `O(n log n)`. It wins on small data and loses badly on large: 17x slower
than scanpy at 10 000 cells, and it refuses more than 20 000 cells outright with a
`ValueError` rather than exhausting memory. Use `sc.tl.tsne` at scale, or `tl.umap`.

Full tables, peak memory, and what could not be measured are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md); run it yourself with
`benches/benchmark.py`.

The `pp.neighbors` row is the first hand-written Metal kernel on the call path.
`crates/scrust-py` now depends on `crates/scrust-gpu` and routes a Metal caller's k-NN
to the `knn` kernel, which is ~2-2.5x faster than the candle path and agrees with the
CPU oracle bit-for-bit (`tests/test_device_parity.py`, 4 of 4). That is what lifts the
row from `0.43x / 0.59x / 2.33x` to `0.64x / 1.15x / 5.80x`.

The other three kernels (`spmm`, `tsne_gradient`, `umap_sgd`) are written and tested but
**not on the call path**: `spmm` has no natural Python-reachable consumer, and
`umap_sgd` is Hogwild and left unwired on purpose so a UMAP layout does not depend on
whether the caller has a GPU. Everything else GPU still goes through candle's Metal
backend. See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Install

Apple silicon, from source (no PyPI wheel is published yet):

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

Needs a Rust toolchain and Xcode's Metal compiler. Without a GPU everything still
runs on the CPU path, which is the same algorithm — the same algorithm, not the
same bits; see [above](#why-rust-and-metal). Full detail, including why Linux and
Windows do not build, is in [docs/INSTALL.md](docs/INSTALL.md).

Verify:

```bash
python -c "import scrust; print(scrust.__version__, scrust.gpu_available())"
```

## Status

All 40 mirrored functions are implemented at 0.2.0. There are no stubs left:
`grep -rn 'todo!' crates/` returns nothing, and the one remaining
`NotImplementedError` in `python/scrust/` is `tl.dpt(n_branchings=)` above 0, which
is branch detection rather than a whole function. Cross-check figures are in
[docs/VALIDATION.md](docs/VALIDATION.md); per-function detail in
[docs/API.md](docs/API.md).

| area | mirrored |
| --- | --- |
| `pp` | `calculate_qc_metrics`, `combat`, `downsample_counts`, `filter_cells`, `filter_genes`, `filter_genes_dispersion`, `highly_variable_genes`, `log1p`, `neighbors`, `normalize_per_cell`, `normalize_total`, `pca`, `regress_out`, `sample`, `scale`, `sqrt`, `subsample` |
| `tl` | `dendrogram`, `diffmap`, `dpt`, `draw_graph`, `embedding_density`, `filter_rank_genes_groups`, `leiden`, `louvain`, `marker_gene_overlap`, `paga`, `rank_genes_groups`, `score_genes`, `score_genes_cell_cycle`, `tsne`, `umap` |
| `metrics` | `morans_i`, `gearys_c`, `confusion_matrix`, `modularity` |
| `get` | `obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate` |

Implemented is not identical. Several functions diverge from scanpy deliberately —
`tl.dendrogram` records the linkage it used, `tl.diffmap` raises where scanpy clamps,
`tl.dpt` gives unreachable cells a finite pseudotime where scanpy writes `inf`, and
`tl.paga` does not write `uns["<groups>_sizes"]`, which `sc.pl.paga` reads to size
nodes. Each divergence is listed and pinned by a test; see
[docs/API.md](docs/API.md) and [docs/VALIDATION.md](docs/VALIDATION.md).

Not everything that takes a `device` uses one. `tl.umap`, `tl.leiden`, `tl.louvain`,
`pp.normalize_total`, `pp.highly_variable_genes` and the differential-expression
methods take the argument and bind it to `_device`: they always run on the CPU.
`pp.pca`, `pp.neighbors` and `tl.tsne` do use it.

## Documentation

- [docs/API.md](docs/API.md) — every function, what it writes, and how it differs
  from scanpy's signature.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — the design as a user feels it.
- [docs/VALIDATION.md](docs/VALIDATION.md) — measured agreement with scanpy.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — timings, speedups and peak memory,
  wins and losses.
- [docs/INSTALL.md](docs/INSTALL.md) — install and platform support.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
  [docs/API_CONTRACT.md](docs/API_CONTRACT.md) — the internal contract the parallel
  feature branches were built against; all of them are merged.

## Layout

```
crates/
  scrust-core/    data types and every algorithm, written against candle
  scrust-gpu/     Metal context and hand written kernels (knn bound to Python; rest not)
  scrust-py/      PyO3 bindings, conversion only
python/scrust/    the scanpy-shaped API and AnnData plumbing
benches/          benchmark.py (vs scanpy) and streaming.py (out-of-core memory)
examples/         pbmc3k.py, the tutorial run through scrust
tests/            cross-checks against scanpy
```

## Develop

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/pytest
```

`tests/` holds 437 Python tests across 34 files, of which 16 are `test_*_audit.py`
cross-checks against scanpy, scikit-learn, umap-learn, scipy and statsmodels. Two
internal modules, `chunked` and `sparse`, have no such cross-check because they have
no equivalent to check against.

The audits run against one device at a time, chosen by `SCRUST_TEST_DEVICE`
(default `"cpu"`; set it to `"auto"` for the GPU). `tests/test_device_parity.py` is
the file that holds the two against each other, and it skips entirely where there is
no Metal device — which includes the `macos-14` hosted runners CI uses. **A green CI
is not evidence that the GPU path passes.** That leg runs on developer hardware.
