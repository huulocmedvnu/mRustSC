# Performance

Every number on this page was measured on one laptop, an Apple MacBook Pro with the M3 Pro
chip (5 performance and 6 efficiency cores, 14-core GPU, 18 GB of unified memory), running
macOS 26.6, Python 3.12, scanpy 1.12.4 and Metalcyte 0.3.0. The data files, the scripts, the
result of every run and the figure code are in the repository under `benches/`, so each table
can be reproduced with one command, given in its section. The development notes under
`docs/development/` keep the longer account, including the measurements that did not come
out as expected.

## A real atlas, start to finish

117 308 bone-marrow cells from CZ CELLxGENE, through the standard pipeline: quality-control
metrics, the cell and gene filters, normalisation, the log transform, 2 000 variable genes,
scaling, 50 principal components, a 15-neighbour graph, UMAP, Leiden and a Wilcoxon marker test
per author cell type. Each library runs the whole pipeline in its own process, so every step sees
the previous step's output from the same library. Seconds per step, mean ± sd over three runs.

| step | scanpy (defaults) | scanpy (tuned) | Metalcyte, CPU | Metalcyte, Metal |
|---|--:|--:|--:|--:|
| QC, filters, normalise, log transform, variable genes, subset | 6.3 | 6.5 | 3.2 | 3.1 |
| scale | 0.6 ± 0.2 | 0.7 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PCA | 9.4 ± 0.2 | 3.5 ± 0.2 | 2.0 ± 0.1 | 0.8 ± 0.0 |
| neighbour graph | 17.1 ± 0.0 | 16.7 ± 0.2 | 4.6 ± 0.3 | 2.0 ± 0.0 |
| UMAP | 43.9 ± 0.4 | 42.7 ± 0.2 | 4.8 ± 0.1 | 4.8 ± 0.1 |
| Leiden | 128.1 ± 0.9 | 2.2 ± 0.2 | 0.7 ± 0.1 | 0.7 ± 0.0 |
| marker genes | 7.9 ± 0.4 | 7.2 ± 0.1 | 0.5 ± 0.0 | 0.5 ± 0.0 |
| **whole pipeline** | 213.2 ± 1.5 | 79.3 ± 1.5 | 15.8 ± 0.6 | 11.9 ± 0.2 |

"scanpy (tuned)" is scanpy with the covariance-eigendecomposition PCA solver, the igraph Leiden
backend with two iterations and an unseeded UMAP, the fastest settings it offers. Metalcyte uses
the parallel UMAP optimiser. Both libraries find 36 or 37 Leiden clusters on this atlas.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library scanpy --tuned
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library metalcyte --umap-parallel
```

## The same answer

The two libraries were compared on every intermediate of that pipeline, on the same cells with
the same seeds (`benches/agreement.py`).

| what is compared | agreement |
|---|---:|
| the 2 000 variable genes, Jaccard index of the two sets | 1.00 |
| the PCA embeddings, smallest canonical correlation of the leading 10 / 30 / 50 components | 0.9999 / 1.0000 / 0.876 |
| the 15-neighbour graphs, mean fraction of a cell's neighbours shared, each library's own PCA | 0.79 |
| the same, with Metalcyte's exact search on scanpy's PCA | 0.92 |
| the Leiden clusterings (37 and 36 clusters), adjusted Rand index / normalised mutual information | 0.94 / 0.96 |
| each Leiden clustering against the author's cell types, normalised mutual information | 0.57 / 0.57 |
| the Wilcoxon marker scores, median Spearman correlation over cell types | 1.00 |
| the top-50 marker lists, median overlap | 1.00 |

The two libraries select the same genes, span the same leading principal subspace and rank the
same marker genes. The neighbour graphs differ where scanpy's approximate index differs from an
exact search, and the clusterings built on them agree at an adjusted Rand index of 0.94.

## A million cells on 18 GB

1 001 288 human embryo cells from CZ CELLxGENE. Metalcyte runs the out-of-core head (quality
control, filters, normalisation, log transform, variable genes, scaling and PCA over row blocks of
the counts on disk, never holding the matrix), then the neighbour graph, UMAP and Leiden in memory.
953 436 cells pass the filters. Seconds per step, one run.

| step | Metalcyte, Metal | Metalcyte, CPU only | memory added, Metal |
|---|---:|---:|---:|
| out-of-core head, four passes | 32 | 39 | 1.0 GB |
| neighbour graph (approximate, the default above 200 000 cells) | 17 | 19 | 0.8 GB |
| UMAP (parallel) | 44 | 45 | 1.0 GB |
| Leiden | 10 | 9 | 1.3 GB |
| whole run | **103** | 112 | |

With the exact neighbour search instead (`method="exact"`), the graph takes 118 s on the GPU and
283 s on the CPU, and the whole run 210 s and 409 s.

scanpy on the same file and machine read the counts, filtered, normalised and selected genes in
about four minutes, then its scaling step added 21.9 GB of footprint and its PCA paged through
16 GB of swap until it was stopped at the 15-minute mark.

The in-memory pipeline also reaches a million cells on this laptop when the GPU is used: 308 s
from counts to Leiden clusters, with the marker test included. The out-of-core head is faster and
uses less memory, and is the recommended path above a quarter of a million cells.

```bash
.venv/bin/python benches/prepare_counts.py data/embryo_1m.h5ad data/embryo_1m_counts.h5ad
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline_1m.py data/embryo_1m_counts.h5ad --umap-parallel
```

## Scaling

Random subsamples of the embryo atlas, the same pipeline, one run per point, with a 40-minute cap
and a watchdog that stops a run adding more than 10 GB of swap. Whole-pipeline seconds. Up to
250 000 cells every configuration runs the in-memory pipeline; from 500 000 cells Metalcyte runs
the out-of-core head and the graph steps.

| cells | scanpy (defaults) | scanpy (tuned) | Metalcyte, CPU | Metalcyte, Metal |
|---:|---:|---:|---:|---:|
| 10 000 | 30 | 27 | 1 | 1 |
| 25 000 | 42 | 30 | 3 | 3 |
| 50 000 | 123 | 45 | 7 | 8 |
| 100 000 | 264 | 73 | 15 | 16 |
| 250 000 | 1 368 | 177 | 68 | 52 |
| 500 000 | over 40 minutes | 377 | 113 | 75 |
| 1 000 000 | not attempted | out of memory | 366 | 207 |

The gap widens with size: Metalcyte on Metal is 21 times faster than scanpy's defaults at 10 000
cells and 27 times at 250 000, and 19 and 3.4 times faster than scanpy tuned. The sweep above ran
the exact neighbour search at every size; the next section gives the approximate search that the
default now uses above 200 000 cells.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/scaling.py data/embryo_1m_counts.h5ad --source data/embryo_1m.h5ad --json benches/results/scaling_embryo.json
```

## Exact against approximate neighbour search

Subsamples of the embryo atlas's 50-dimensional PCA embedding, 15 neighbours, one run per point
(`benches/knn_methods.py`). The exact search compares every pair; the approximate search is
NN-descent seeded with a random-projection forest, on every core. Recall is the fraction of the
exact neighbours the approximate lists contain.

| cells | approximate | exact, GPU | exact, CPU | recall |
|---:|---:|---:|---:|---:|
| 10 000 | 0.1 s | 0.0 s | 0.1 s | 0.994 |
| 25 000 | 0.3 s | 0.1 s | 0.2 s | 0.991 |
| 50 000 | 0.4 s | 0.4 s | 1.3 s | 0.988 |
| 100 000 | 1.9 s | 1.5 s | 4.6 s | 0.984 |
| 250 000 | 3.7 s | 8.7 s | 23.5 s | 0.977 |
| 500 000 | 8.4 s | 33.9 s | 85.8 s | 0.970 |
| 953 436 | 15.9 s | 118.1 s | 282.5 s | 0.961 |

The exact search is faster on the GPU up to about 100 000 cells and on the CPU up to about
50 000. `pp.neighbors` switches to the approximate search above 200 000 cells by default, where
the exact graph, which matches scanpy's cell for cell, stops being cheap.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/knn_methods.py benches/results/embryo1m_metalcyte_metal.h5ad --json benches/results/knn_methods_embryo.json
```

## Beyond the standard pipeline at a million cells

Single runs on the 953 436-cell embryo embedding (50 principal components) on the M3 Pro.

| step | Metalcyte | note |
|---|---:|---|
| t-SNE, FFT-accelerated (1 000 iterations) | 65 s | exact up to 20 000 cells, FFT above: Accelerate's FFT on the cores, the attractive term on the GPU; 16 s on the 117 308-cell atlas |
| Harmony, 7 experiment batches | 9 s | 4 outer iterations; harmonypy 2.1 (compiled) on the same input: 3 s; per-cell cosine between the two results 0.999 |

`regress_out` and `combat` read the sparse matrix in gene blocks and hold only their dense
result, so their limit is the result's size against 60% of the machine's memory.

## Energy

The package's own power counters, sampled every 100 ms while the 117 308-cell pipeline ran, with
the idle draw measured just before each run subtracted.

| run | seconds | net energy | mean power |
|---|---:|---:|---:|
| scanpy (defaults) | 237 | 1 228 J | 6.1 W |
| Metalcyte, CPU only, parallel UMAP | 16 | 273 J | 15.8 W |
| Metalcyte, Metal, parallel UMAP | 13 | 168 J | 11.8 W |

Metalcyte on Metal does the same analysis for one seventh of the energy: 19 times sooner at twice the
power, because scanpy keeps one core busy and Metalcyte keeps eleven and the GPU. The CPU-only run
draws more power than the Metal run and takes longer, so the GPU saves energy as well as time. Per
million cells the Metal run is about 1.4 kJ. The published rapids-singlecell run on an NVIDIA L40S
takes 92 s for a million cells on a card rated at 300 W with a 200 W host, which puts that run at
20 to 45 kJ even at half load.

```bash
sudo sh benches/run_energy.sh data/bone_marrow_117k_counts.h5ad
```

## What each part of the chip is worth

The 117 308-cell pipeline again, with one feature switched off at a time (`benches/ablation.py`).
Seconds for the whole pipeline and for the steps that move; the run-to-run noise is about 2 s.

| configuration | what is off | whole | PCA | neighbours | UMAP | markers |
|---|---|---:|---:|---:|---:|---:|
| everything on | nothing | 12.9 | 1.0 | 2.0 | 4.9 | 0.5 |
| no GPU | Metal | 16.3 | 2.3 | 4.6 | 4.8 | 0.5 |
| no Accelerate | Apple's BLAS on the AMX units | 13.0 | 1.0 | 2.1 | 4.6 | 0.5 |
| performance cores only | the 6 efficiency cores | 15.4 | 0.9 | 2.0 | 7.7 | 0.6 |
| one core | every core but one | 44.8 | 1.0 | 2.1 | 34.6 | 1.8 |
| no zero-copy | the in-place numpy borrows | 23.2 | 6.5 | 2.1 | 4.9 | 0.9 |
| sequential UMAP | the lock-free optimiser | 40.4 | 0.9 | 2.2 | 32.5 | 0.6 |

- The GPU is worth 2.3 times on the neighbour search and 2.4 times on PCA, and nothing elsewhere at
  this size.
- All eleven cores are worth 7.0 times on the UMAP optimiser, and the efficiency cores carry a
  third of that work.
- The zero-copy borrows of numpy's buffers are worth 10 s of a 13 s run, most of it in PCA.
- Accelerate shows nothing at 2 000 genes. The matrix products are too small for the AMX units to
  make a difference.

## Limits

- The exact neighbour search is quadratic in cells. The approximate search covers the large
  regime with a recall of 0.96 at a million cells; a user who needs the exact graph above
  200 000 cells pays the quadratic cost.
- The Metal neighbour kernel runs at about 15% of the chip's arithmetic peak. A version on the
  GPU's matrix units is in the tree, opt-in, and is not yet faster.
- t-SNE is exact up to 20 000 cells and FFT-accelerated above: 16 s for the 117 308-cell atlas and
  65 s for the 953 436 embryo cells (1 000 iterations, from the 50 principal components).
- `regress_out` and `combat` produce a dense result that must fit within 60% of the machine's
  memory; the input is read in gene blocks.
- Metalcyte is built and tested on macOS on Apple silicon only. The continuous-integration runners
  have no usable GPU, so the GPU tests run locally.
