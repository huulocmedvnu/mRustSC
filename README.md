# scrust

Single-cell analysis with a Rust core that runs the heavy numerics on the GPU of
Apple M-series chips. Python is the interface only: it holds the AnnData plumbing
and the defaults, and every matrix operation happens in Rust.

The API mirrors scanpy, so a script changes its import and keeps its plotting — for
the steps scrust implements. It does not yet implement all of scanpy; the
[status table](#status) below is the honest list, and the numbers are
[measured, not claimed](docs/VALIDATION.md).

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

sc.tl.leiden(adata)                      # scrust has no clustering yet — see below
sr.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.pl.umap(adata, color="leiden")        # scanpy plotting still works
```

`examples/pbmc3k.py` is this pipeline end to end, printing each scanpy call beside
the scrust call that replaced it and marking the one step (`leiden`) it has to
borrow back from scanpy. It runs today.

## Why Rust and Metal

The expensive steps of a single-cell pipeline — PCA, the neighbour graph, the UMAP
and t-SNE layouts, differential expression — are large batched matrix operations.
Apple silicon's unified memory means the count matrix is not copied across a bus to
reach the GPU, which is what usually makes GPU acceleration uneconomic at
single-cell matrix sizes.

Everything expressible as tensor algebra is written once against
[candle](https://github.com/huggingface/candle) and runs on either device, so the
CPU path is the same code and serves as the correctness oracle.

This buys speed unevenly, and the benchmark reports both directions. Measured at
499, 2 638 and 10 000 cells, speedup against scanpy (above 1.00x scrust is faster):

| operation | 499 | 2 638 | 10 000 |
| --- | ---: | ---: | ---: |
| `tl.paga` | 23.00x | 55.67x | 90.40x |
| `tl.rank_genes_groups` | 10.15x | 28.72x | 41.95x |
| `tl.umap` | 3.41x | 2.96x | 4.84x |
| `pp.neighbors` | 0.43x | 0.59x | 2.33x |
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

Note also that the hand-written Metal kernels in `crates/scrust-gpu` (sparse SpMM
and friends) are written and tested but **not yet wired to any Python call** — today
the GPU you get is candle's Metal backend. See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Install

Apple silicon, from source (no PyPI wheel is published yet):

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

Needs a Rust toolchain and Xcode's Metal compiler. Without a GPU everything still
runs on the CPU path, which is the same algorithm. Full detail, including why
Linux and Windows do not build, is in [docs/INSTALL.md](docs/INSTALL.md).

Verify:

```bash
python -c "import scrust; print(scrust.__version__, scrust.gpu_available())"
```

## Status

16 of scanpy's 40 mirrored functions are implemented at 0.2.0. The other 24 exist,
import and type-check, and raise `NotImplementedError` naming the branch that owes
them — none is a silent no-op. Cross-check figures are in
[docs/VALIDATION.md](docs/VALIDATION.md); per-function detail in
[docs/API.md](docs/API.md).

| area | implemented | not yet |
| --- | --- | --- |
| `pp` | `filter_cells`, `filter_genes`, `normalize_total`, `log1p`, `highly_variable_genes`, `scale`, `pca`, `neighbors` | `calculate_qc_metrics`, `normalize_per_cell`, `sqrt`, `filter_genes_dispersion`, `regress_out`, `combat`, `subsample`, `sample`, `downsample_counts` |
| `tl` | `umap`, `tsne`, `rank_genes_groups`, `paga` | `leiden`, `louvain`, `diffmap`, `dpt`, `score_genes`, `score_genes_cell_cycle`, `marker_gene_overlap`, `filter_rank_genes_groups`, `dendrogram`, `draw_graph`, `embedding_density` |
| `metrics` | — | `morans_i`, `gearys_c`, `confusion_matrix`, `modularity` |
| `get` | `obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate` | — |

The gap you will hit first is clustering: there is no `leiden` or `louvain`, so a
tutorial run reaches the clustering step and stops. `examples/pbmc3k.py` bridges it
with `sc.tl.leiden` and says so.

## Documentation

- [docs/API.md](docs/API.md) — every function, what it writes, and how it differs
  from scanpy's signature.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — the design as a user feels it.
- [docs/VALIDATION.md](docs/VALIDATION.md) — measured agreement with scanpy.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — timings, speedups and peak memory,
  wins and losses.
- [docs/INSTALL.md](docs/INSTALL.md) — install and platform support.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
  [docs/API_CONTRACT.md](docs/API_CONTRACT.md) — the internal contract branches are
  built against.

## Layout

```
crates/
  scrust-core/    data types and every algorithm, written against candle
  scrust-gpu/     Metal context and hand written kernels (not yet bound to Python)
  scrust-py/      PyO3 bindings, conversion only
python/scrust/    the scanpy-shaped API and AnnData plumbing
benches/          benchmark.py (vs scanpy) and streaming.py (out-of-core memory)
examples/         pbmc3k.py, the tutorial as far as scrust supports it
tests/            cross-checks against scanpy
```

## Develop

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/pytest
```
