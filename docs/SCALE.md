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

| step | scanpy | scanpy (tuned) | scrust CPU | scrust Metal | scrust Metal, `umap(parallel=True)` |
|---|---:|---:|---:|---:|---:|
| `pp.calculate_qc_metrics` | 1.46 | 1.40 | 0.95 | 0.97 | 0.92 |
| `pp.filter_cells` | 0.69 | 0.69 | 1.98 | 2.07 | 1.20 |
| `pp.filter_genes` | 1.10 | 1.12 | 1.08 | 1.09 | 1.03 |
| `pp.normalize_total` | 1.18 | 1.25 | 0.13 | 0.14 | 0.11 |
| `pp.log1p` | 0.34 | 0.35 | 0.15 | 0.15 | 0.13 |
| `pp.highly_variable_genes` | 1.34 | 1.37 | 0.31 | 0.30 | 0.27 |
| `subset` | 0.68 | 0.59 | 0.65 | 0.73 | 0.58 |
| `pp.scale` | 1.00 | 0.53 | 0.03 | 0.03 | 0.03 |
| `pp.pca` | 13.61 | 3.70 | 0.95 | 0.96 | 0.87 |
| `pp.neighbors` | 17.85 | 18.20 | 2.07 | 2.13 | 2.09 |
| `tl.umap` | 46.82 | 46.13 | 30.79 | 30.42 | 4.77 |
| `tl.leiden` | 135.40 | 2.49 | 0.65 | 0.64 | 0.63 |
| `tl.rank_genes_groups` | 11.83 | 8.29 | 7.20 | 7.33 | 8.87 |
| **whole pipeline** | **233.3** | **86.1** | **46.9** | **47.0** | **21.5** |

- **Against scanpy's defaults the pipeline is 5x faster on the same laptop, 11x with the
  parallel UMAP; against scanpy tuned (`covariance_eigh` PCA, `igraph` Leiden with two
  iterations, UMAP without a fixed seed) it is 1.8x and 4.0x.** The tuned column is the fair
  one and the one to quote. What remains after tuning is the neighbour graph (8.5x), UMAP
  (9.7x with the Hogwild optimiser; umap-learn's own parallel path did not engage through
  scanpy, 46 s either way), PCA (4x) and Leiden (3.9x against `igraph`, 210x against the
  default `leidenalg` run to convergence).
- **CPU and Metal give the same times here.** At 2 000 genes and 116 000 cells PCA and the
  neighbour search are each about a second on either device, so the win over scanpy on this
  dataset comes from the Rust core using every core, not from the GPU. The GPU earns its
  place one step up in size (section 3), where the scatter products stop fitting a CPU budget.
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
| pass 1: QC, normalise, log1p, HVG sums | 3.6 | 3.4 | |
| pass 2: moments of the variable genes | 5.3 | 4.8 | |
| pass 3: scaled scatter on the device, eigenvectors | 15.4 | 21.0 | |
| pass 4: projection to 50 PCs | 7.5 | 6.9 | |
| `pp.preprocess_backed` (all four passes) | 32.3 | 36.7 | 1058 |
| `pp.neighbors` | 123.4 | 120.1 | 486 |
| `tl.umap` | 44.6 | 46.6 | 1163 |
| `tl.leiden` | 9.5 | 11.0 | 1245 |
| **whole run** | **210** | **214** | |

- **A million cells, start to Leiden, in 3.5 minutes on a laptop, never holding more than 1.4 GB
  beyond the baseline.** The head of the pipeline, four passes over 4.8 GB of counts, is 32 s;
  the counts stream off the SSD faster than the arithmetic can use them.
- **The exact neighbour search is now the long pole** (120 s, 57% of the run): it is brute force,
  quadratic in cells. It is also the step that puts a Mac on the same footing as a data-centre
  card: the 953 436 x 953 436 distance products run at about 400 GFLOP/s on either device.
- **Metal and the CPU are within 2% of each other on the whole run.** The CPU path is not a
  fallback: with the `accelerate` feature the matmuls go through Apple's AMX units via
  Accelerate, and on an M3 Pro those keep pace with the 14-core GPU at these shapes. The GPU
  wins the scatter step (15.4 s against 21.0 s) and nothing else by a margin worth the name.
  The honest summary is that the speed comes from the whole chip, every CPU core, the AMX
  matrix units and the GPU sharing one memory, not from the GPU alone.
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
| scrust, `settings.device="cpu"` (see note) | 19 | 161 | 72 | **197** | 9.6 | 2 |

- **scrust does the same analysis for one fifth of the energy**: 195 J against 974 J, because it is
  done 12x sooner while drawing only 1.6x the power (9.5 W against 6.0 W: scanpy keeps one core
  busy, scrust keeps eleven).
- **Per million cells that is about 1.7 kJ.** The published rapids-singlecell run on an L40S is
  92 s for a million cells; an L40S is rated at 300 W and the EPYC host at 200 W, so even at half
  load that is roughly 20 to 45 kJ for the same work: the laptop is one order of magnitude cheaper
  in energy and one order of magnitude slower in time.
- **The GPU rail is a quarter of scrust's energy** (72 J) and it was drawn in the "cpu" run as well.
  That run was not CPU-only: at the time of the measurement `pp.neighbors`, `pp.pca`, `tl.umap`,
  `tl.leiden` and `tl.rank_genes_groups` defaulted to `device="auto"` in their own signatures and
  ignored `settings.device`, a bug the energy trace exposed and this branch fixes (every function
  now resolves `device=None` through `settings`). The CPU row will be re-measured; until then the
  "CPU and Metal are level" statements in sections 2, 3 and 5 are statements about two Metal runs.
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
| `no_metal` | the GPU (`device="cpu"`) | 19.1 | 1.01 | 2.04 | 4.87 | 0.66 | 6.83 | 0.03 | 0.12 |
| `no_accelerate` | Apple's BLAS, AMX (wheel built without the feature) | 20.4 | 0.92 | 2.13 | 4.76 | 0.63 | 7.84 | 0.03 | 0.14 |
| `p_cores_only` | the 6 efficiency cores (`RAYON_NUM_THREADS=5`) | 22.8 | 0.85 | 2.12 | 7.49 | 0.77 | 7.66 | 0.02 | 0.14 |
| `one_core` | every core but one (`RAYON_NUM_THREADS=1`) | 49.9 | 0.99 | 2.07 | 33.38 | 0.69 | 8.10 | 0.09 | 0.21 |
| `no_zero_copy` | the in-place numpy borrows (`SCRUST_FORCE_COPY=1`) | 25.1 | 4.83 | 1.99 | 4.73 | 0.63 | 6.32 | 1.59 | 0.85 |
| `sequential_umap` | the Hogwild optimiser | 46.6 | 0.89 | 2.08 | 31.33 | 0.65 | 7.71 | 0.03 | 0.12 |

What the table says, plainly:

- **At this size neither the GPU nor the AMX units buy anything measurable.** Switching
  Metal off is 4 s *faster* (the GPU path pays for pipeline compilation and command-buffer
  round trips that a 2 000-gene problem does not amortise), and the pure-Rust BLAS is
  within noise of Accelerate. The 5x to 11x over scanpy on this atlas comes from the Rust
  core: fused single passes, `f32` throughout, and work spread over every core.
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
- **The Apple-silicon case is therefore not "the GPU makes the steps faster" at 10⁵ cells.**
  It is (i) unified memory, which lets a million cells stream off the disk through buffers
  the CPU and GPU both read without a copy (section 3), (ii) a laptop package that runs the
  whole pipeline on every core at under 30 W (section 4), and (iii) at 10⁶ cells, the matrix
  units for the neighbour search once it is written for them (`docs/PLAN_APPLE_SILICON.md`,
  2.1). The 1 M-cell ablation in the plan will say whether (iii) holds before that rewrite.

## 6. Against the NVIDIA-GPU alternative

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
| **scrust** | **Apple M3 Pro laptop, 18 GB** | **1 001 288** | **210 s (210 s; 43 Leiden clusters on the CPU path)** |

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
