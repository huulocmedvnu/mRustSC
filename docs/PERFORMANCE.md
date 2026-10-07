# Performance

All numbers on this page come from one laptop. It is an Apple MacBook Pro with the M3 Pro chip.
The chip has 5 performance cores, 6 efficiency cores, a 14-core GPU and 18 GB of unified memory.
Unified memory means the CPU and the GPU share the same memory, so data does not have to be
copied between them. The software was macOS 26.6, Python 3.12, scanpy 1.12.4 and Metalcyte 0.3.0.

The data files, scripts, raw results and figure code are in the repository under `benches/`.
Each section gives the one command that reproduces its table. The notes under
`docs/development/` give the longer account, including measurements that did not turn out as
expected.

In the tables, "Metal" means Metalcyte used the Mac's GPU through Apple's Metal interface. "CPU"
means it used only the processor cores.

## A real atlas, start to finish

This test uses 117 308 bone-marrow cells from CZ CELLxGENE. Each library ran the standard
pipeline: quality-control metrics, cell and gene filters, normalisation, the log transform,
selection of 2 000 variable genes, scaling, 50 principal components, a 15-neighbour graph, UMAP,
Leiden clustering and a Wilcoxon marker test for each author cell type. Each library ran the
whole pipeline in its own process. So every step received the previous step's output from the
same library. The table gives seconds per step, as the mean ± sd of three runs.

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

"scanpy (tuned)" uses the fastest settings scanpy offers. These are the
covariance-eigendecomposition PCA solver, the igraph Leiden backend with two iterations, and an
unseeded UMAP. Metalcyte uses its parallel UMAP optimiser, which runs on all cores. Both
libraries find 36 or 37 Leiden clusters on this atlas.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library scanpy --tuned
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library metalcyte --umap-parallel
```

## Both libraries give the same answer

We compared the two libraries at every step of that pipeline. Both used the same cells and the
same random seeds (`benches/agreement.py`). A value of 1.00 means full agreement.

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

The two libraries select the same genes and find the same leading principal components. They
also rank the same marker genes. The neighbour graphs differ a little. scanpy finds neighbours
with an approximate search, and Metalcyte uses an exact one at this size. The two clusterings
still agree with an adjusted Rand index (ARI) of 0.94. ARI measures how well two clusterings
match, where 1 is a perfect match and 0 is chance level.

## A million cells on an 18 GB laptop

This test uses 1 001 288 human embryo cells from CZ CELLxGENE. Metalcyte runs the first steps
"out of core", so it never holds the whole count matrix in memory. It reads the
count matrix from disk in blocks of cells and keeps only one block in memory at a time. These
first steps are quality control, filters, normalisation, log transform, variable genes, scaling
and PCA. Metalcyte then builds the neighbour graph, UMAP and Leiden clusters in memory, from the
much smaller PCA result. 953 436 cells pass the filters. The table gives seconds per step from
one run.

| step | Metalcyte, Metal | Metalcyte, CPU only | memory added, Metal |
|---|---:|---:|---:|
| out-of-core head, four passes | 32 | 39 | 1.0 GB |
| neighbour graph (approximate, the default above 200 000 cells) | 17 | 19 | 0.8 GB |
| UMAP (parallel) | 44 | 45 | 1.0 GB |
| Leiden | 10 | 9 | 1.3 GB |
| whole run | **103** | 112 | |

You can ask for the exact neighbour search instead (`method="exact"`). Then the graph takes
118 s on the GPU and 283 s on the CPU. The whole run takes 210 s and 409 s.

scanpy ran on the same file and machine. It read the counts, filtered, normalised and selected
genes in about four minutes. Its scaling step then added 21.9 GB of memory use. Its PCA then
used 16 GB of swap, which is disk space the system uses when memory runs out. We stopped it at
15 minutes.

The in-memory pipeline of Metalcyte also handles a million cells on this laptop when it uses the
GPU. It takes 308 s from counts to Leiden clusters, including the marker test. The out-of-core
path is faster and uses less memory. We recommend it above a quarter of a million cells.

```bash
.venv/bin/python benches/prepare_counts.py data/embryo_1m.h5ad data/embryo_1m_counts.h5ad
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline_1m.py data/embryo_1m_counts.h5ad --umap-parallel
```

## How run time grows with the number of cells

We took random subsamples of the embryo atlas and ran the same pipeline once per size. Each run
had a 40-minute limit. A watchdog stopped any run that added more than 10 GB of swap. The table
gives whole-pipeline seconds. Up to 250 000 cells, every configuration runs the in-memory
pipeline. From 500 000 cells, Metalcyte runs the out-of-core first steps and then the graph
steps.

| cells | scanpy (defaults) | scanpy (tuned) | Metalcyte, CPU | Metalcyte, Metal |
|---:|---:|---:|---:|---:|
| 10 000 | 30 | 27 | 1 | 1 |
| 25 000 | 42 | 30 | 3 | 3 |
| 50 000 | 123 | 45 | 7 | 8 |
| 100 000 | 264 | 73 | 15 | 16 |
| 250 000 | 1 368 | 177 | 68 | 52 |
| 500 000 | over 40 minutes | 377 | 113 | 75 |
| 1 000 000 | not attempted | out of memory | 366 | 207 |

The gap grows with size. Metalcyte on Metal is 21 times faster than scanpy's defaults at 10 000
cells and 27 times faster at 250 000. Against tuned scanpy it is 19 and 3.4 times faster. This
sweep used the exact neighbour search at every size. The next section covers the approximate
search, which is now the default above 200 000 cells.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/scaling.py data/embryo_1m_counts.h5ad --source data/embryo_1m.h5ad --json benches/results/scaling_embryo.json
```

## Exact and approximate neighbour search

To build the neighbour graph, Metalcyte finds the 15 closest cells to each cell in PCA space.
The exact search compares every pair of cells. Its cost grows with the square of the number of
cells. The approximate search uses NN-descent. NN-descent starts from rough neighbour lists,
which here come from a random-projection forest. It then improves each list by checking the
neighbours of a cell's neighbours. It runs on all cores. Recall is the share of the true (exact)
neighbours that the approximate lists contain. A recall of 1 means every true neighbour was
found.

The test used subsamples of the embryo atlas's 50-dimensional PCA embedding, with one run per
size (`benches/knn_methods.py`).

| cells | approximate | exact, GPU | exact, CPU | recall |
|---:|---:|---:|---:|---:|
| 10 000 | 0.1 s | 0.0 s | 0.1 s | 0.994 |
| 25 000 | 0.3 s | 0.1 s | 0.2 s | 0.991 |
| 50 000 | 0.4 s | 0.4 s | 1.3 s | 0.988 |
| 100 000 | 1.9 s | 1.5 s | 4.6 s | 0.984 |
| 250 000 | 3.7 s | 8.7 s | 23.5 s | 0.977 |
| 500 000 | 8.4 s | 33.9 s | 85.8 s | 0.970 |
| 953 436 | 15.9 s | 118.1 s | 282.5 s | 0.961 |

The exact search is faster on the GPU up to about 100 000 cells. On the CPU it is faster up to
about 50 000 cells. The exact search finds the true nearest neighbours, but it becomes slow at
large sizes. So by default `pp.neighbors` switches to the approximate search above 200 000 cells.

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/knn_methods.py benches/results/embryo1m_metalcyte_metal.h5ad --json benches/results/knn_methods_embryo.json
```

## Four million cells

This test uses the full survey of human embryonic development from CZ CELLxGENE (dataset
f7c1c579-2dc0-47e2-ba19-8165c5a0e353). It has 4 062 980 cells and 45 676 genes. The file stores
2.38 billion counts and takes 28.9 GB on disk. We used the same laptop with 18 GB of memory,
single runs and the default neighbour search. Nothing was tuned for this file.

| step | seconds | memory added |
|---|---:|---:|
| out-of-core head, four passes (QC, normalise, variable genes, scale, PCA) | 155 | 1.8 GB |
| neighbour graph (approximate) | 85 | 4.1 GB |
| UMAP (parallel) | 237 | 4.9 GB |
| Leiden | 61 | 7.3 GB |
| **from counts to clusters** | **538** (9 minutes, 71 clusters) | peak about 9 GB |
| Wilcoxon markers on the 2 000 variable genes kept by the head (0.63 GB) | 1.3 | |
| UMAP scatter of all 4 062 980 cells, `pl.umap`, saved | 1.5 | |
| t-SNE, FFT-accelerated, on the GPU | 497 | |

The number of clusters changes between runs on this file. Three runs gave 71, 65 and 59. The
approximate neighbour search updates its lists from many threads at once. So two runs with the
same seed give nearly the same graph, with small differences. In a test on the embryo atlas, 5
of 300 000 rows differed. This atlas has many similar small clusters, and Leiden splits them
differently when the graph changes slightly.

Converting the CELLxGENE file to a counts file (`benches/prepare_counts.py`, read in a stream)
took 118 s and used 2.9 GB of memory. We also tested a size in between, 2 002 576 cells. We made
it by stacking the 1M atlas with a thinned copy of itself (`benches/double_counts.py`). It ran
in 202 s on Metal (first steps 62, neighbours 38, UMAP 83, Leiden 20) and 222 s on the CPU
cores. Streamed markers on 2 000 genes took 8.4 s.

## Other steps at a million cells

These are single runs on the 953 436-cell embryo embedding (50 principal components) on the
M3 Pro. "FFT-accelerated" t-SNE computes the long-range forces between cells on a grid with a
fast Fourier transform. This makes it fast on large data.

| step | Metalcyte | note |
|---|---:|---|
| t-SNE, FFT-accelerated (1 000 iterations) | 54 s | exact up to 20 000 cells, FFT above, the whole iteration on the GPU; 13 s on the 117 308-cell atlas |
| Wilcoxon markers on the variable genes kept by the head (`preprocess_backed(keep_hvg=True)`, 0.17 GB), 42 clusters | 0.3 s | the head's time is unchanged; this is the recommended route |
| Wilcoxon markers over the counts file, `tl.rank_genes_groups_backed`, 2 000 variable genes, 42 clusters | 4.5 s | the matrix never in memory (resident set unchanged at 2.3 GB); for a matrix whose variable genes do not fit either, or for every gene: all 45 676 in blocks of 4 096, 39 s |
| UMAP scatter of all cells, `pl.umap` (render, legend, PNG at 300 dpi) | 0.6 s | the Metal rasteriser draws the 953 436 points in 0.2 s; scanpy's matplotlib scatter of the same cells takes 4.1 s to save and holds a million path objects |
| Harmony, 7 experiment batches | 9 s | 4 outer iterations; harmonypy 2.1 (compiled) on the same input: 3 s; per-cell cosine between the two results 0.999 |

`regress_out` and `combat` read the sparse matrix in blocks of genes. They keep only their dense
result in memory. So the limit is the size of that result, which must fit in 60% of the
machine's memory.

## Energy use

We read the chip's own power counters every 100 ms while the 117 308-cell pipeline ran. Just
before each run we measured the power the idle machine draws, and we subtracted it.

| run | seconds | net energy | mean power |
|---|---:|---:|---:|
| scanpy (defaults) | 237 | 1 228 J | 6.1 W |
| Metalcyte, CPU only, parallel UMAP | 16 | 273 J | 15.8 W |
| Metalcyte, Metal, parallel UMAP | 13 | 168 J | 11.8 W |

Metalcyte on Metal does the same analysis with one seventh of the energy. It finishes 19 times
sooner at twice the power. scanpy keeps one core busy, while Metalcyte keeps eleven cores and
the GPU busy. The CPU-only run draws more power than the Metal run and takes longer. So the GPU
saves energy as well as time. On Metal, the run uses about 1.4 kJ per million cells. For
comparison, the published rapids-singlecell run on an NVIDIA L40S GPU takes 92 s for a million
cells. That card is rated at 300 W with a 200 W host computer. Even at half load, that run
would use 20 to 45 kJ.

On the full 4 062 980-cell survey (out-of-core pipeline, counts to clusters):

| run | seconds | gross energy | net energy | mean power |
|---|---:|---:|---:|---:|
| Metalcyte, Metal | 557 | 7 089 J | 6 244 J | 12.7 W |
| Metalcyte, CPU only | 568 | 7 028 J | not comparable | 12.3 W |

On Metal this is about 1.5 kJ per million cells, close to the 117 308-cell figure. The two runs
use the same energy within 1%. At this size the default path runs the approximate neighbour
search and UMAP on the CPU cores, so the GPU has little to do. We left out the net figure for
the CPU run. Its idle reading was taken just after the Metal run, while the machine was still
warm (3.9 W against 1.5 W). Measuring power slows the runs a little. They took 557 s and 568 s
under `powermetrics`, against 538 s without it.

```bash
sudo sh benches/run_energy.sh data/bone_marrow_117k_counts.h5ad
sudo sh benches/run_energy_4m.sh data/embryo_4m_counts.h5ad
```

## What each part of the chip contributes

We ran the 117 308-cell pipeline again and switched off one feature at a time
(`benches/ablation.py`). The table gives seconds for the whole pipeline and for the steps that
change. Times vary by about 2 s from run to run.

Some terms in the table:

- Accelerate is Apple's library of fast maths routines. Its matrix routines (BLAS) run on the
  AMX units, which are matrix-multiplication units built into the CPU.
- Zero-copy means Metalcyte reads numpy's arrays in place, without making a copy.
- The parallel UMAP optimiser lets all cores update the layout at the same time without waiting
  for each other (a lock-free optimiser). The sequential version uses one core.

| configuration | what is off | whole | PCA | neighbours | UMAP | markers |
|---|---|---:|---:|---:|---:|---:|
| everything on | nothing | 12.9 | 1.0 | 2.0 | 4.9 | 0.5 |
| no GPU | Metal | 16.3 | 2.3 | 4.6 | 4.8 | 0.5 |
| no Accelerate | Apple's BLAS on the AMX units | 13.0 | 1.0 | 2.1 | 4.6 | 0.5 |
| performance cores only | the 6 efficiency cores | 15.4 | 0.9 | 2.0 | 7.7 | 0.6 |
| one core | every core but one | 44.8 | 1.0 | 2.1 | 34.6 | 1.8 |
| no zero-copy | the in-place numpy borrows | 23.2 | 6.5 | 2.1 | 4.9 | 0.9 |
| sequential UMAP | the lock-free optimiser | 40.4 | 0.9 | 2.2 | 32.5 | 0.6 |

- The GPU makes the neighbour search 2.3 times faster and PCA 2.4 times faster. At this size it
  makes no difference to the other steps.
- Using all eleven cores makes the UMAP optimiser 7.0 times faster. The efficiency cores do a
  third of that work.
- Reading numpy's arrays without copying saves 10 s of a 13 s run, mostly in PCA.
- Accelerate makes no difference at 2 000 genes. The matrix products are too small for the AMX
  units to help.

## Limits

- The cost of the exact neighbour search grows with the square of the number of cells. The
  approximate search handles large datasets with a recall of 0.96 at a million cells. If you need
  the exact graph above 200 000 cells, you pay that higher cost.
- The approximate search does not give exactly the same result on every run. Its parallel
  updates make the graph vary slightly between runs, and so the number of clusters can vary.
- The Metal neighbour-search kernel (a program that runs on the GPU) reaches about 15% of the
  chip's peak arithmetic speed. A version that uses the GPU's matrix units is in the code. You
  must switch it on yourself, and it is not yet faster.
- t-SNE is exact up to 20 000 cells and FFT-accelerated above that. It takes 13 s for the
  117 308-cell atlas and 54 s for the 953 436 embryo cells (1 000 iterations, from the 50
  principal components).
- The exact t-SNE method (`method="exact"`) costs `O(n^2)` and is slower than scanpy's t-SNE once
  the data is not small. Against scanpy it is 1.42x faster at 499 cells, 0.25x at 2 638, and 0.06x
  at 10 000 cells (271.7 s against scanpy's 15.4 s). At 10 000 cells `tl.umap` is 4.84x faster
  than scanpy's UMAP.
- `regress_out` and `combat` produce a dense result that must fit within 60% of the machine's
  memory. They read the input in blocks of genes.
- Metalcyte is built and tested only on macOS on Apple silicon. The automatic test machines
  (continuous integration) have no usable GPU, so the GPU tests run on a local Mac.
