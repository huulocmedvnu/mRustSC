# Metalcyte

**Metalcyte** is a single-cell analysis engine for Apple silicon. A Rust core with a Python
interface runs the standard pipeline, from quality control to clustering and marker genes, on every
performance and efficiency core, on the AMX matrix units through Accelerate, and on the GPU through
hand-written Metal kernels, all in the chip's unified memory. An out-of-core head puts a million cells
through quality control, feature selection and PCA without holding the matrix. On an M3 Pro laptop
with 18 GB, a 117 308-cell atlas goes from counts to marker
genes in 12 s and a 1 001 288-cell atlas
from counts to Leiden clusters in 210 s.

Results are written to the standard AnnData slots, so an analysis script written for scanpy runs on
Metalcyte after changing its import, and scanpy's plotting still reads the results.

## Quick start

```python
import metalcyte as mc
import scanpy as sc  # for the example data and for plotting

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
mc.tl.umap(adata, parallel=True)
mc.tl.leiden(adata)
mc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

sc.pl.umap(adata, color="leiden")
```

A matrix that does not fit memory goes through the out-of-core head instead:

```python
adata = mc.pp.preprocess_backed("atlas_counts.h5ad", n_top_genes=2000, n_comps=50)
mc.pp.neighbors(adata, use_rep="X_pca")
mc.tl.umap(adata, parallel=True)
mc.tl.leiden(adata)
```

`examples/pbmc3k.py` walks through the first pipeline step by step; `docs/API.md` documents every
function and argument.

## Installation

Metalcyte builds from source on Apple silicon (macOS 13 or later, a Rust toolchain, Python 3.11 or
later). The Metal shaders are compiled at run time by the system Metal framework, so Xcode's Metal
toolchain is not needed.

```bash
git clone https://github.com/huulocmedvnu/metalcyte
cd metalcyte
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
.venv/bin/python -c "import metalcyte as mc; print(mc.__version__, mc.gpu_available())"
```

Metalcyte is built and tested on macOS on Apple silicon; other platforms are untested.
[docs/INSTALL.md](docs/INSTALL.md) has the details and the optional extras.

## What it does

| module | functions |
|---|---|
| `mc.pp` | `calculate_qc_metrics`, `filter_cells`, `filter_genes`, `normalize_total`, `normalize_per_cell`, `log1p`, `sqrt`, `highly_variable_genes`, `filter_genes_dispersion`, `scale`, `regress_out`, `combat`, `harmony_integrate`, `pca`, `neighbors`, `subsample`, `sample`, `downsample_counts`, `preprocess_backed` |
| `mc.tl` | `umap`, `tsne`, `leiden`, `louvain`, `paga`, `diffmap`, `dpt`, `draw_graph`, `embedding_density`, `dendrogram`, `rank_genes_groups` (Wilcoxon, t-test, t-test with overestimated variance, logistic regression), `filter_rank_genes_groups`, `marker_gene_overlap`, `score_genes`, `score_genes_cell_cycle` |
| `mc.metrics` | `morans_i`, `gearys_c`, `confusion_matrix`, `modularity` |
| `mc.get` | `obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate` |
| `mc.pl` | `umap`, `pca_variance_ratio`, `rank_genes_groups` (the rest of plotting is scanpy's, on the same AnnData) |

Every function is held to scanpy's output by a test in `tests/`, with the tolerance stated in the
test; [docs/VALIDATION.md](docs/VALIDATION.md) lists them and the few places where the two differ on
purpose. Randomness takes a seed and gives the same result on the same device, with one exception:
`tl.umap(parallel=True)` runs a lock-free optimiser whose layout is reproducible in structure, not in
coordinates.

## Performance

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores, 14-core GPU, 18 GB), scanpy 1.12.4.
Every number is in `benches/results/` next to the script that produced it, and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) has the full tables and how they were measured.

**A real atlas, start to finish.** 117 308 bone-marrow cells (CELLxGENE), QC to Wilcoxon markers,
mean of three runs (`benches/pipeline.py`):

| | scanpy (defaults) | scanpy (tuned) | Metalcyte, CPU | Metalcyte, Metal |
|---|---:|---:|---:|---:|
| whole pipeline | 213 s | 79 s | 16 s | **12 s** |
| energy (powermetrics, idle subtracted; measured before the marker-test rewrite, when the run took 20 s) | 974 J | | | **195 J** |
| PCA / neighbours / UMAP / Leiden / markers | 9.4 / 17.1 / 43.9 / 128.1 / 7.9 s | 3.5 / 16.7 / 42.7 / 2.2 / 7.2 s | 2.0 / 4.6 / 4.8 / 0.7 / 0.5 s | 0.8 / 2.0 / 4.8 / 0.7 / 0.5 s |

"Tuned" is scanpy with the `covariance_eigh` PCA solver, `igraph` Leiden and an unseeded UMAP, the
fastest settings it offers. The two libraries find the same biology on this atlas: the same variable
genes, the same leading principal subspace, Leiden clusterings at an adjusted Rand index of 0.95 and
the same marker genes (`benches/agreement.py`).

**A million cells on 18 GB.** 1 001 288 human embryo cells (CELLxGENE) through
`mc.pp.preprocess_backed` and then the graph steps (`benches/pipeline_1m.py`):

| | scanpy | Metalcyte, CPU | Metalcyte, Metal |
|---|---:|---:|---:|
| QC to Leiden, 953 436 cells kept | did not finish (the scaling step needed 21.9 GB) | 409 s | **210 s** |
| memory added per step | | under 1.5 GB | under 1.5 GB |

**Scaling.** On subsamples of that atlas Metalcyte is 21x faster than scanpy's defaults at 10 000 cells
and 27x at 250 000, 19x and 3.4x against scanpy tuned; scanpy's defaults did not finish 500 000 cells
in 40 minutes and scanpy tuned does not fit a million.

**Which part of the chip buys what.** Switching features off one at a time on the 117k atlas
(`benches/ablation.py`): the GPU is worth 2.3x on the neighbour search and 2.4x on PCA, all eleven
cores are worth 7x on the UMAP optimiser with the efficiency cores carrying a third of it, the
zero-copy numpy borrows are worth 10 s of a 13 s run, and Accelerate shows nothing at 2 000 genes.

## Devices and reproducibility

`mc.settings.device` is `"auto"`: the GPU where there is one, the CPU otherwise. Set it to `"cpu"`,
or export `METALCYTE_DEVICE=cpu`, to keep a whole session on the CPU; every function also takes a
`device` argument. The two devices give the same answer to `f32` precision, and the neighbour lists
they produce are identical, but they are not bitwise identical, because a parallel reduction adds in
a different order. For results that must match across machines bit for bit, run on the CPU.

## Limits

- The neighbour search is exact and quadratic in cells: 2 s at 117 000 cells, 120 s at a million on
  the GPU. scanpy's approximate index overtakes it near 400 000 cells, and an approximate index is
  the next item on the plan.
- The in-memory pipeline fits 18 GB up to about 250 000 cells; above that, use `preprocess_backed`.
- `tl.tsne` is exact and refuses more than 20 000 cells. `regress_out` and `combat` cap their dense
  working set at 8 GiB.
- `mc.pl` has three functions; plot with scanpy on the same AnnData for the rest.
- Metalcyte is built and tested on macOS on Apple silicon only. Continuous integration runs the
  whole suite on the CPU; the GPU tests run locally.

## Documentation

- [docs/API.md](docs/API.md): every function, argument and the AnnData slot it writes.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): the measurements above in full, with the methodology.
- [docs/VALIDATION.md](docs/VALIDATION.md): how each algorithm is held to scanpy.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the
  Rust core, the Metal kernels and the out-of-core head.
- [docs/development/](docs/development/): working notes, the optimisation history and what is planned.

## Development

```bash
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings && cargo test --workspace
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
METALCYTE_TEST_DEVICE=auto .venv/bin/pytest        # the audits against scanpy, on the GPU
```

The repository is `crates/` (the Rust core, the Metal kernels, the PyO3 bindings), `python/metalcyte/`
(the interface), `tests/`, `benches/` and `docs/`.

## License

MIT.
