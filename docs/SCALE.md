# Scale: 100 000 cells, a real atlas, and a million cells on a laptop

This page is the scale measurement that `docs/BENCHMARKS.md` stops short of: the
per-operation sweep at 100 000 cells, the whole pipeline on a real 117 000-cell
atlas, a million cells pushed through the head of the pipeline without ever holding
the matrix, the energy each run draws, and how the numbers sit against the published
figures for the NVIDIA-GPU alternative. Everything here was measured on one machine:

| | |
|---|---|
| machine | Apple M3 Pro, 5 performance + 6 efficiency cores, 18 GB unified memory, 14-core GPU |
| scanpy | 1.12.4 (leidenalg Leiden, umap-learn UMAP) |
| scrust | this branch (`feat/scanpy-parity`), Accelerate BLAS on, Metal on |
| scripts | `benches/benchmark.py --sizes 100000`, `benches/pipeline.py`, `benches/pipeline_1m.py`, `benches/energy.py` |

Reproduce with the commands under each section. Every table below was produced by the
file named next to it in `benches/results/`.

## 1. Per-operation sweep at 100 000 cells

`benches/results/benchmark_100k_m3pro.txt`. PBMC 3k bootstrapped to 100 000 cells, each
stage prepared by scanpy, each operation timed on its own (best of up to 2 runs, repeats
stop after 30 s). Times in seconds; `+MB` is the physical footprint the call added.

| operation | genes | scanpy | scrust | speedup | +MB scanpy / scrust |
|---|---:|---:|---:|---:|---|
| `pp.filter_cells` | 32 738 | 0.525 | 0.533 | 0.99x | 1344 / 1263 |
| `pp.filter_genes` | 32 738 | 0.686 | 0.742 | 0.92x | 1246 / 1256 |
| `pp.normalize_total` | 16 579 | 0.084 | 0.097 | 0.86x | 633 / 635 |
| `pp.log1p` | 16 579 | 0.167 | 0.077 | 2.17x | 0 / 633 |
| `pp.highly_variable_genes` | 16 579 | 0.481 | 0.363 | 1.33x | 1267 / 951 |
| `pp.scale` | 2 000 | 0.441 | 0.029 | 15.4x | 2469 / 801 |
| `pp.pca` | 2 000 | 60.60 | 0.759 | 79.9x | 4884 / 1872 |
| `pp.neighbors` | 2 000 | 3.858 | 1.508 | 2.56x | 49 / 137 |
| `tl.umap` | 2 000 | 1690.8 | 26.91 | 62.8x | 248 / 89 |
| `tl.leiden` | 2 000 | 6.597 | 0.195 | 33.9x | 362 / 106 |
| `tl.rank_genes_groups` (wilcoxon) | 16 579 | 61.75 | 0.850 | 72.6x | 316 / 1590 |
| `tl.paga` | 2 000 | error | 0.005 | | |

What to read into it:

- **The graph and the decomposition are where the time goes, and they are where scrust is
  one to two orders of magnitude faster.** PCA, UMAP, Leiden and the rank-sum test are
  60x to 80x faster. The elementwise steps are at parity (they are memory-bound in both
  libraries, and scanpy's are already numpy).
- **The scanpy UMAP figure is an upper bound.** 1 690 s is far above the few minutes
  umap-learn usually needs at this size, and the probes that ran after it (`combat`,
  `regress_out`) pushed the process to 476 GB of virtual memory and filled the 40 GB of
  swap, so the scanpy worker was under memory pressure during the later rows and was
  eventually killed by hand. The real-atlas run in section 2 is the cleaner UMAP
  comparison.
- **`pp.neighbors` lost ground against the 50 000-cell figure** (4.5x there, 2.6x here).
  The brute-force Metal kernel is exact and quadratic; scanpy's pynndescent is
  approximate and near-linear. Above about 200 000 cells an approximate index will be
  needed to keep the gap.
- **The 1 590 MB `rank_genes_groups` adds** is the gene-major transpose the rank-sum test
  builds once; scanpy ranks in place. That copy is the price of being 73x faster and is
  released when the call returns.
- Two things refused the size and are reported as such rather than hidden: `tl.tsne`
  (exact, capped at 20 000 cells) and the dense `regress_out` / `combat` working set
  (capped at 8 GiB; 100 000 x 16 579 needs 12 and 31 GiB).

## 2. A real atlas end to end: 117 308 bone-marrow cells

Dataset: "Bone Marrow Reference" from CELLxGENE (dataset `b6037d0f-c415-45f7-8d01-0e69407fba9c`,
117 308 cells x 25 574 genes, 162 M stored counts). `benches/prepare_counts.py` copies
its `raw.X` into a counts-only file; `benches/pipeline.py` then runs one library through
QC, filters, `normalize_total`, `log1p`, 2 000 highly variable genes, `scale`, 50-component
PCA, 15-neighbour graph, UMAP, Leiden and a Wilcoxon marker test by author cell type,
each step seeing the previous step's output from the same library.

```bash
.venv/bin/python benches/prepare_counts.py data/bone_marrow_117k.h5ad data/bone_marrow_117k_counts.h5ad
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library scanpy --json benches/results/bm117k_scanpy.json
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library scrust --device cpu --json benches/results/bm117k_scrust_cpu.json
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data/bone_marrow_117k_counts.h5ad --library scrust --device auto --json benches/results/bm117k_scrust_metal.json
```

`benches/results/bm117k_*.json`. Seconds per step; 115 868 cells pass the filter, 2 000 variable genes, 36 or 37 Leiden clusters.

| step | scanpy | scanpy (tuned) | scrust CPU, parallel UMAP | scrust Metal | scrust Metal, `umap(parallel=True)` |
|---|---:|---:|---:|---:|---:|
| `pp.calculate_qc_metrics` | 1.46 | 1.40 | 1.37 | 0.97 | 0.86 |
| `pp.filter_cells` | 0.69 | 0.69 | 1.07 | 2.07 | 0.64 |
| `pp.filter_genes` | 1.10 | 1.12 | 1.23 | 1.09 | 1.01 |
| `pp.normalize_total` | 1.18 | 1.25 | 0.20 | 0.14 | 0.12 |
| `pp.log1p` | 0.34 | 0.35 | 0.19 | 0.15 | 0.12 |
| `pp.highly_variable_genes` | 1.34 | 1.37 | 0.35 | 0.30 | 0.25 |
| `subset` | 0.68 | 0.59 | 0.94 | 0.73 | 0.55 |
| `pp.scale` | 1.00 | 0.53 | 0.10 | 0.03 | 0.02 |
| `pp.pca` | 13.61 | 3.70 | 3.08 | 0.96 | 1.10 |
| `pp.neighbors` | 17.85 | 18.20 | 4.87 | 2.13 | 1.97 |
| `tl.umap` | 46.82 | 46.13 | 4.95 | 30.42 | 4.80 |
| `tl.leiden` | 135.40 | 2.49 | 0.73 | 0.64 | 0.66 |
| `tl.rank_genes_groups` | 11.83 | 8.29 | 10.09 | 7.33 | 7.61 |
| **whole pipeline** | **233.3** | **86.1** | **29.2** | **47.0** | **19.7** |

- **Against scanpy's defaults the pipeline is 5x faster on the same laptop, 11x with the
  parallel UMAP; against scanpy tuned (`covariance_eigh` PCA, `igraph` Leiden with two
  iterations, UMAP without a fixed seed) it is 1.8x and 4.0x.** The tuned column is the fair
  one and the one to quote. What remains after tuning is the neighbour graph (8.5x), UMAP
  (9.7x with the Hogwild optimiser; umap-learn's own parallel path did not engage through
  scanpy, 46 s either way), PCA (4x) and Leiden (3.9x against `igraph`, 210x against the
  default `leidenalg` run to convergence).
- **The GPU is the neighbour search, and the CPU column is now a fair one.** With
  `settings.device="cpu"` honoured (a bug fixed on this branch: sixteen functions used to ignore
  it) and the CPU search rewritten to use every core (query blocks across threads, Accelerate
  products per thread, fused top-k), the brute-force k-NN is 4.9 s on the CPU against 2.0 s on
  Metal, and PCA 3.1 s against 1.1 s; everything else is the same on either device. The first
  CPU measurement, before that rewrite, had the search at 79 s on effectively one core. Whole
  pipeline: 29 s CPU-only, 20 s with the GPU, against 233 s for scanpy's defaults and 86 s tuned.
- **The elementwise head is at parity or a little ahead**, as at 100 000 cells. `filter_cells`
  is slower in scrust (it copies the matrix once more than scanpy); it is 2 s of a 47 s run.
- **The rank-sum test is 1.3x to 1.6x here**, not the 73x of section 1: 2 000 genes by 37
  groups is a small test and both libraries spend their time elsewhere in the call.
- Scanpy's UMAP took 47 s on this real matrix, which is the figure to set against the 1 690 s
  of section 1: that number was a process under memory pressure, not umap-learn's speed.

## 3. A million cells on 18 GB: `pp.preprocess_backed`

Dataset: "Survey of human embryonic development (1 million cells subset)" from CELLxGENE
(dataset `f6271b1a-71f7-4c4e-a109-c9949ac42bc1`, 1 001 288 cells x 45 676 genes, 586 M
stored counts, 4.8 GB on disk as counts).

The dense scaled matrix scanpy's pipeline builds for PCA is 1 001 288 x 2 000 x 4 bytes =
8 GB, before the copies `scale` and the solver make. On an 18 GB machine that does not
fit next to the 4.7 GB count matrix. `scrust.pp.preprocess_backed` never builds it:

1. **pass 1** reads row blocks off the disk, drops cells under `min_genes`, normalises and
   log-transforms each block in place (the in-memory kernels, so the numbers match), and
   accumulates per-gene presence and the `sum` / `sum of squares` behind
   `highly_variable_genes` in `f64` across all cores;
2. the variable genes are ranked from those sums (`highly_variable_genes_from_sums`, the
   in-memory algorithm from the accumulators onward);
3. **pass 2** gathers the moments of the log data on the 2 000 variable columns;
4. **pass 3** scales each block against those moments straight into a dense buffer and
   accumulates the 2 000 x 2 000 scatter matrix on the GPU (`gram_dense`: one Metal
   matmul per block, summed in `f64`). The principal axes are the top eigenvectors of
   that scatter, found by subspace iteration on the device, which is scanpy's
   `svd_solver="covariance_eigh"` with the scatter built out of core;
5. **pass 4** scales each block again and multiplies it by the loadings, writing its rows
   of `X_pca`.

Peak memory is one block plus the `(n_cells, 50)` embedding. Apple's unified memory is
what makes pass 3 and 4 cheap: the buffer the CPU just scaled is the one the GPU
multiplies, with no copy to a device. The result is held to scanpy's exact solver in
`tests/test_streaming.py` and on real data: on 19 770 bone-marrow cells all 50 streamed
components span the same space as `covariance_eigh` (smallest canonical correlation
0.9994) and the variance ratios agree to five decimals.

```bash
.venv/bin/python benches/prepare_counts.py data/embryo_1m.h5ad data/embryo_1m_counts.h5ad
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline_1m.py data/embryo_1m_counts.h5ad --umap-parallel --json benches/results/embryo1m_scrust.json
PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline_1m.py data/embryo_1m_counts.h5ad --library scanpy --json benches/results/embryo1m_scanpy.json
```

`benches/results/embryo1m_scrust_{metal,cpu}.json`. Of the 1 001 288 cells, 953 436 pass `min_genes=200`;
9 022 genes fall under `min_cells=3`; 2 000 variable genes; 50 components; 42 Leiden clusters.
Seconds per step and the memory each step added (physical footprint):

| step | scrust Metal | scrust CPU | +MB (Metal) |
|---|---:|---:|---:|
| pass 1: QC, normalise, log1p, HVG sums | 3.6 | 4.7 | |
| pass 2: moments of the variable genes | 5.3 | 5.9 | |
| pass 3: scaled scatter on the device, eigenvectors | 15.4 | 23.1 | |
| pass 4: projection to 50 PCs | 7.5 | 8.2 | |
| `pp.preprocess_backed` (all four passes) | 32.3 | 42.6 | 1058 |
| `pp.neighbors` | 123.4 | 307.0 | 486 |
| `tl.umap` | 44.6 | 47.8 | 1163 |
| `tl.leiden` | 9.5 | 11.2 | 1245 |
| **whole run** | **210** | **409** | |

- **A million cells, start to Leiden, in 3.5 minutes on a laptop, never holding more than 1.4 GB
  beyond the baseline.** The head of the pipeline, four passes over 4.8 GB of counts, is 32 s;
  the counts stream off the SSD faster than the arithmetic can use them.
- **The exact neighbour search is the long pole and the GPU's whole case.** On Metal it is
  120 s, 57% of the run; on the CPU, with the search spread over every core, 307 s, and the
  whole run 6.8 minutes against 3.5. Brute force is quadratic in cells, and at 10⁶ cells
  that is 9 x 10¹¹ distance evaluations: the one step where the 14-core GPU is what keeps the
  laptop at minutes. (The first CPU measurement, before the search was parallelised, was
  5 294 s on effectively one core; it is kept in the git history as the lower bound it was.)
- **Everything else is level between the devices**: the streamed head is 32 s against 35 s
  (the scatter products 15 s against 20 s), UMAP and Leiden are on the CPU either way.
- **scanpy on the same file, same machine**: see `benches/results/embryo1m_scanpy.log`.
  `read_h5ad` of the 4.8 GB counts, the filters, `normalize_total`, `log1p`, `highly_variable_genes`
  and the subset ran (about 4 minutes); `pp.scale` then took 98 s and added **21.9 GB** of
  footprint on an 18 GB machine, and `pp.pca` (ARPACK over the 7.6 GB dense matrix) sat at
  10% CPU paging through 16 GB of swap. It was stopped at the 15-minute mark with PCA
  unfinished. scanpy's own answer at this size is `dask`-backed chunking, which this
  comparison did not configure: the point is what the default pipeline does on the
  machine on the desk.

## 4. Energy, not only time

`benches/energy.py` runs `benches/pipeline.py` under `powermetrics` (which needs `sudo` to
read the package power counters), samples the CPU, GPU and ANE rails every 100 ms,
integrates them over the run and subtracts the idle draw measured just before. The
figure that matters is the net joules per pipeline: what the analysis cost the battery.

```bash
sudo PYTHONPATH=$PWD/python .venv/bin/python benches/energy.py data/bone_marrow_117k_counts.h5ad --library scanpy --json benches/results/energy_bm117k_scanpy.json
sudo PYTHONPATH=$PWD/python .venv/bin/python benches/energy.py data/bone_marrow_117k_counts.h5ad --library scrust --device auto --json benches/results/energy_bm117k_scrust_metal.json
```

`benches/results/energy_bm117k_*.json`, raw samples next to them. Net joules (idle draw of the
package subtracted) for the whole 117k pipeline; the pipeline ran as the user, powermetrics as root.

| run | seconds | CPU rail J | GPU rail J | **net J** | mean W | kJ per million cells |
|---|---:|---:|---:|---:|---:|---:|
| scanpy (defaults) | 229 | 980 | 0 | **974** | 6.0 | 8 |
| scrust, Metal, parallel UMAP | 20 | 159 | 72 | **195** | 9.5 | 2 |
| scrust, run labelled "cpu" (was a Metal run, see note) | 19 | 161 | 72 | **197** | 9.6 | 2 |

- **scrust does the same analysis for one fifth of the energy**: 195 J against 974 J, because it is
  done 12x sooner while drawing only 1.6x the power (9.5 W against 6.0 W: scanpy keeps one core
  busy, scrust keeps eleven).
- **Per million cells that is about 1.7 kJ.** The published rapids-singlecell run on an L40S is
  92 s for a million cells; an L40S is rated at 300 W and the EPYC host at 200 W, so even at half
  load that is roughly 20 to 45 kJ for the same work: the laptop is one order of magnitude cheaper
  in energy and one order of magnitude slower in time.
- **The GPU rail is a quarter of scrust's energy** (72 J) and it was drawn in the "cpu" run as
  well, which is how the measurement caught a bug: `pp.neighbors`, `pp.pca`, `tl.umap`, `tl.leiden`,
  `tl.rank_genes_groups` and eleven other functions defaulted to `device="auto"` in their own
  signatures and ignored `settings.device`. Fixed on this branch (every function now resolves
  `device=None` through `settings`); the third row is therefore a second Metal run, and the true
  CPU-only pipeline is 99 s (section 2), which at the same 8 W would be about 800 J, close to
  scanpy's. The energy saving comes with the GPU; a re-measured CPU row needs one more `sudo` run.
- Idle draw was 1.8 W before the scanpy run and 0.6 W before the scrust runs (the machine had
  been busy); the net figures subtract each run's own idle.

Figure F8 (`docs/figures/F8_utilisation.png`) is the P-cluster, E-cluster and GPU active residency
over both runs from the raw samples. In the scrust trace the three phases are legible to the eye:
the GPU pinned at 100% for the three seconds of PCA and the neighbour search, both clusters at
100% through the parallel UMAP (the efficiency cores are saturated, not idle), and the P cluster
alone through the rank-sum test. In the scanpy trace the P cluster sits at 100% for 230 s with
one core busy, and the 20 to 50% GPU residency during its first 150 s is the display (Preview
windows were open on the machine), not the analysis, which is why its net GPU energy is zero.

## 5. Ablation: what each Apple-specific choice is worth

`benches/results/ablation_bm117k.json`, from `benches/ablation.py` on the 117 308-cell
atlas, scrust with the parallel UMAP. One knob off per row; seconds per step. Run-to-run
noise on this machine is about ±2 s on the whole pipeline (the rank-sum step alone moves
between 6.3 and 8.9 s with nothing changed), so differences under that are not differences.

| configuration | what is off | whole | PCA | neighbours | UMAP | Leiden | markers | scale | normalise |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | nothing | 23.3 | 1.37 | 2.08 | 5.04 | 0.94 | 8.88 | 0.05 | 0.15 |
| `no_metal` | the GPU (`device="cpu"`, after the fix and the parallel CPU search) | 29.2 | 3.08 | 4.87 | 4.95 | 0.73 | 10.09 | 0.10 | 0.20 |
| `no_accelerate` | Apple's BLAS, AMX (wheel built without the feature) | 20.4 | 0.92 | 2.13 | 4.76 | 0.63 | 7.84 | 0.03 | 0.14 |
| `p_cores_only` | the 6 efficiency cores (`RAYON_NUM_THREADS=5`) | 22.8 | 0.85 | 2.12 | 7.49 | 0.77 | 7.66 | 0.02 | 0.14 |
| `one_core` | every core but one (`RAYON_NUM_THREADS=1`) | 49.9 | 0.99 | 2.07 | 33.38 | 0.69 | 8.10 | 0.09 | 0.21 |
| `no_zero_copy` | the in-place numpy borrows (`SCRUST_FORCE_COPY=1`) | 25.1 | 4.83 | 1.99 | 4.73 | 0.63 | 6.32 | 1.59 | 0.85 |
| `sequential_umap` | the Hogwild optimiser | 46.6 | 0.89 | 2.08 | 31.33 | 0.65 | 7.71 | 0.03 | 0.12 |

What the table says, plainly:

- **The GPU is worth 2.5x on the one quadratic step, once the CPU is given a fair run.** The
  brute-force neighbour search is 4.9 s on all eleven cores through Accelerate and 2.0 s on
  Metal; PCA is 3.1 s against 1.4 s; nothing else moves. At a million cells the same two
  numbers are 307 s and 120 s (section 3). An earlier version of this row read 99 s, with the
  search at 79 s: that was the CPU search before it was parallelised, on effectively one
  core. The AMX units through Accelerate are within noise of the pure-Rust BLAS at this size.
- **The cores are what count, and all eleven of them.** UMAP's Hogwild sweep is 6.6x faster
  on 11 threads than on 1 and 1.5x faster than on the 5 performance cores alone: the
  efficiency cores are not idle ballast, they carry a third of the epochs. Everything else
  that scales with threads (`scale`, `normalise`, HVG) is already under half a second.
- **Zero-copy buys 4.5 s** on this run: PCA from the dense buffer scanpy's `scale` would
  have made (4.8 s) against PCA straight from the in-place result (1.4 s), and the
  elementwise steps 10x to 30x.
- **Two steps ignore the thread count and are the next targets.** `pp.neighbors` and
  `tl.rank_genes_groups` take the same 2 s and 8 s on one core as on eleven. The neighbour
  search runs through Accelerate or Metal whichever way rayon is set, so this knob never
  reached it; the rank-sum test is rayon-parallel in Rust, so its 8 s must be spent in the
  Python assembly of the result (structured arrays, DataFrames), which a profile will show.
- **The Apple-silicon case, in order of what the table shows:** (i) the GPU for the one
  quadratic step, 2.5x on the neighbour search against all eleven cores, which is the
  difference between 2 and 5 minutes at a million cells; (ii) all eleven cores for the optimiser and the
  elementwise passes; (iii) unified memory, which lets a million cells stream off the disk
  through buffers the CPU and GPU both read without a copy (section 3); (iv) a package that
  does all of this at under 10 W (section 4). The AMX units do not show at 2 000 genes; the
  plan's neighbour-search rewrite (`docs/PLAN_APPLE_SILICON.md`, 2.1) is where they would.

## 6. Scaling: seconds against cells

`benches/results/scaling_embryo.json`, from `benches/scaling.py`: random subsamples of the 1 M-cell
embryo atlas at 10k to 250k cells and the first 500k rows and the whole file above that, the same
pipeline as section 2, one process per point, a swap watchdog (a run that adds more than 10 GB of
swap is stopped) and a 40-minute cap. Up to 250k cells scrust runs the in-memory pipeline; from
500k it runs the streamed head and the graph steps (`pipeline_1m.py`), which is the recommended
use at that size. Whole-pipeline seconds (the in-memory points include the marker test, the
streamed points do not):

| cells | scanpy (defaults) | scanpy (tuned) | scrust CPU | scrust Metal |
|---:|---:|---:|---:|---:|
| 10,000 | 30 s | 27 s | 1 s | 1 s |
| 25,000 | 42 s | 30 s | 3 s | 3 s |
| 50,000 | 123 s | 45 s | 7 s | 8 s |
| 100,000 | 264 s | 73 s | 15 s | 16 s |
| 250,000 | 1368 s | 177 s | 68 s | 52 s |
| 500,000 | over 40 min | 377 s | 113 s | 75 s |
| 1,000,000 | not attempted (500k did not finish) | added more than 10 GB of swap | 366 s | 207 s |

- **The gap widens with size.** scrust Metal is 21x faster than scanpy's defaults at 10k cells and
  26x at 250k, 19x and 7x against scanpy tuned. scanpy's defaults stop at 250k (23 minutes) and did
  not finish 500k in 40 minutes; scanpy tuned reaches 500k in 6 minutes and does not fit a million.
- **The in-memory scrust pipeline also stops between 250k and 500k on 18 GB**, and the sweep showed
  why: `tl.rank_genes_groups` added 6 GB at 116k cells and 25 GB at 476k. The marker test's working
  set is the memory ceiling of the in-memory path and is the next engineering item
  (`docs/PLAN_APPLE_SILICON.md`, 2.3). Above 250k the streamed head is the path to use, and it
  carries a million cells at 207 s on Metal and 366 s on the CPU.
- **The neighbour panel of F5 is the honest one.** scanpy's pynndescent index is approximate and
  near-constant in time, about 17 s from 10k to 250k cells and 24 s at 500k, while scrust's exact
  search is quadratic: 0.05 s at 10k, 8 s at 250k, 30 s at 500k and 120 s at a million on Metal.
  The lines cross near 400k cells. Above that size an approximate index is what scrust needs
  (`docs/PLAN_APPLE_SILICON.md`, 2.2), and the GPU's 2.5x over the CPU search does not change
  where the crossing sits by much.
- Figure F5 draws every step on log-log axes; the streamed points above 250k have no separate PCA
  or marker step, which is why those two panels end at 250k for scrust.

## 7. Against the NVIDIA-GPU alternative

rapids-singlecell (Dicks et al., "GPU-accelerated single-cell analysis at scale with
rapids-singlecell", 2026, arXiv 2603.02402; NVIDIA developer blog, 12 June 2025) is the
other GPU implementation of the scanpy API. Its published numbers, with the hardware they
were measured on, next to this page's:

| pipeline | hardware | cells | total |
|---|---|---:|---:|
| scanpy 1.11.1, CPU | AMD EPYC 7413, 24 cores / 48 threads | 1 000 000 | 5 176 s |
| rapids-singlecell | NVIDIA L40S (48 GB) | 1 000 000 | 92 s |
| rapids-singlecell | NVIDIA RTX PRO 6000 (96 GB) | 1 000 000 | 28.4 s |
| rapids-singlecell | NVIDIA DGX B200 (8 x B200) | 1 000 000 | 24.6 s |
| rapids-singlecell (examples repo) | NVIDIA A100 40 GB | 1 300 000 | 686 s (UMAP 21 s, Leiden 1.7 s) |
| **scrust** | **Apple M3 Pro laptop, 18 GB** | **1 001 288** | **210 s** (Metal; 409 s CPU-only) |

Sources: NVIDIA developer blog "Driving toward billion-cell analysis and biological
breakthroughs with RAPIDS-singlecell" (12 June 2025) for the 1 M-cell rows;
`clara-parabricks/rapids-single-cell-examples` README for the 1.3 M-cell A100 row. The
pipelines are the same steps but not the same script, the datasets differ, and the
published runs were not re-executed here, so this is a placement, not a benchmark.

The placement is the point of the project:

- **rapids-singlecell needs a data-centre GPU and CUDA.** Its 1 M-cell runs were on a 48 GB
  L40S at the small end and an eight-GPU DGX at the large end; the 1.3 M-cell example
  wants an A100 and spends 475 s of its 686 s loading and preprocessing because the data
  does not fit a 40 GB card without `dask`. scrust runs the same steps on the GPU every
  recent Mac ships with, in the same memory the CPU uses, so a million cells stream off
  the SSD and no step waits on a PCIe copy.
- **The CPU baseline those speedups are quoted against is a 24-core server.** On that
  class of machine scanpy takes 86 minutes for a million cells. On this laptop scanpy
  cannot hold the dense matrix at all (section 3).
- **What this laptop does not match** is the raw throughput of a B200: 25 s for a million
  cells is out of reach of 14 GPU cores and an SSD. The claim here is a smaller one and is
  the one an analyst at a bench needs: the standard pipeline on an atlas-scale dataset, on
  the machine already on the desk, in minutes, for the energy in section 4.
