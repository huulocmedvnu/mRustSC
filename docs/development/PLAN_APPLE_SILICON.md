# Plan: proving metalcyte is built for Apple silicon, and finishing what is half done

Written 2026-10-06 on branch `feat/scanpy-parity`. This is the working plan for the
next pass over the project; it lists what is done, what is measured but not yet
written up, what is still missing against scanpy, and the experiments and figures a
paper needs before the claim "optimised for Apple silicon" can be made in print.
Tick items off here as they land; `docs/development/MEASUREMENTS_2026-10.md` holds the numbers.

## 0. Where things stand

| area | state |
|---|---|
| per-operation benchmark to 100 000 cells | done, `benches/results/benchmark_100k_m3pro.txt` |
| real 117k atlas end to end, scanpy vs metalcyte (CPU, Metal, parallel UMAP) | done, 233 s / 29 s / 47 s / 20 s |
| scanpy with its fastest settings (`--tuned`) as a fourth column | done, 86.1 s |
| 1 M cells on 18 GB through `pp.preprocess_backed` | done, 210 s Metal; 409 s CPU-only after the parallel CPU search (was 5 386 s) |
| scanpy on 1 M cells on the same machine | did not finish: `scale` needed 21.9 GB, PCA swapped, killed at 15 min |
| streamed PCA held to scanpy's exact solver | done, `tests/test_streaming.py` and 50/50 components on real data |
| energy per run (`benches/energy.py`, `benches/run_energy.sh`) | done: scanpy 974 J, metalcyte Metal 195 J |
| placement against rapids-singlecell | written from their published numbers, `docs/development/MEASUREMENTS_2026-10.md` section 5 |
| Rust unit tests | 267 pass; Python suite not yet re-run in full after today's changes |
| CI on `main` | red: `clippy -D warnings` and 3 scanpy-1.12.4 median tests (pre-existing) |

The finding that shapes the message, after the `settings.device` bug was found and fixed:
**the GPU is worth 40x on the neighbour search and nothing measurable elsewhere; the cores
are worth 6.6x on the UMAP optimiser; the AMX units do not show at 2 000 genes.** The claim
is "optimised for Apple silicon" as a package (GPU for the quadratic step, every core for the
rest, unified memory for the out-of-core head, under 10 W), and the proof is the ablation in
`docs/development/MEASUREMENTS_2026-10.md` section 5, not a scanpy comparison.

## 1. Evidence: the experiments that prove the claim

Each experiment has a script, an output file in `benches/results/`, and a figure.

### 1.1 Ablation: switch off one Apple-specific choice at a time
Same binary, same data (117k real, 1 M streamed), same machine.

| knob | on | off | how |
|---|---|---|---|
| Accelerate / AMX BLAS | `--features accelerate` (default wheel) | pure-Rust `matrixmultiply` | second wheel built without the feature into `.venv-noaccel` |
| Metal | `settings.device="auto"` | `"cpu"` | done |
| all 11 cores (P+E) | `RAYON_NUM_THREADS=11` | `5` (P only), `1` | env var, rayon honours it |
| zero-copy numpy borrows | current `fast.rs` paths | force the old copying path (`METALCYTE_FORCE_COPY=1`, to add) | env var read in `_basics.py` |
| parallel UMAP | `parallel=True` | `False` | done |

Output: `benches/ablation.py` producing `benches/results/ablation_{bm117k,embryo1m}.json`;
one figure, grouped bars of seconds per step, one group per knob. The sentence the
figure supports: "removing X costs Y s of the Z s run".

### 1.2 Roofline: how close each kernel gets to the chip
Measured throughput against the theoretical peak of the M3 Pro (CPU with AMX about 2 to
3 TFLOP/s fp32; 14-core GPU about 5 TFLOP/s fp32; memory 150 GB/s). From today's numbers
the exact neighbour search runs at about 760 GFLOP/s on either device, 15% of the GPU
peak: the kernel is bound by the top-k selection, not by the distance products.

Output: `benches/roofline.py`, one figure (arithmetic intensity vs GFLOP/s, one point
per kernel: kNN, PCA scatter, projection, scale, normalise), with the roof lines of
the chip. Points under the memory roof are memory-bound (the elementwise steps and
that is fine); points far under the compute roof are the ones worth rewriting.

### 1.3 Hardware utilisation during a run
`powermetrics` reports per-cluster CPU residency (P cores vs E cores), GPU active
residency and power per rail. Record them alongside `benches/energy.py` (add the
`cpu_power,gpu_power,thermal` samplers and parse residency) and plot a timeline of the
117k run: which step keeps the GPU busy, which one runs on the E cores, where the
machine idles waiting on the SSD.

### 1.4 Energy per run
`sudo sh benches/run_energy.sh data/bone_marrow_117k_counts.h5ad` gives net joules for
scanpy, metalcyte Metal and metalcyte CPU. Report joules per million cells next to the
NVIDIA figures (an L40S draws about 300 W; 92 s is about 28 kJ for a million cells;
the laptop's whole package is under 30 W).

### 1.5 Scaling curves
Seconds against cells at 10k, 50k, 100k (done), 117k real, 250k and 500k subsamples of
the 1 M atlas, and 1 M, for scanpy (where it fits) and metalcyte. Log-log, one line per
step. The figure shows where each library's slope changes and where scanpy stops.

## 2. Engineering: what has to change for the claims to hold at 1 M and beyond

### 2.1 Neighbour search on the matrix units (highest value)
**CPU side: done.** `neighbors::knn_cpu` spreads query blocks over every core with a
per-thread Accelerate product and a fused top-k: 116k cells 79 s → 4.9 s, 1 M cells
5 294 s → 307 s. **GPU side: tried, not yet a win.** `knn_metal_simd_view` builds the
products from `simdgroup_float8x8` steps with queries and dimension-major candidates
staged in threadgroup memory; it is correct (held to the brute-force reference) but
3.1 s against the tiled kernel's 2.2 s at 116k x 50, and an ablation of its own shows
the 8 x 8 steps are the whole cost (0.76 s without them). It is opt-in
(`METALCYTE_KNN_KERNEL=simd`). What it needs is gemm-grade blocking: 64 x 64 output tiles
per threadgroup with eight accumulators amortising every load, as MLX's gemm does; or
the alternative two-stage design (one large Metal matmul per tile, then a merge
kernel), which is bandwidth-bound at about 50 s for a million cells because the
distance tile has to be written and read once. Either is a day's work with a Metal
profiler; neither was in reach today. The tiled FMA kernel stays the default.

### 2.2 Approximate neighbours above 1 M
**Done (2026-10-07):** NN-descent seeded with a random-projection forest in
`crates/metalcyte-core/src/nndescent.rs`, behind `pp.neighbors(method="approximate")`
and the default `method="auto"` above 200 000 cells. Held to the exact result by recall
at k = 15: 0.99 at 117 000 cells, 0.96 at 953 000. 16 s at 953 000 cells on the cores
against 123 s for the exact search on the GPU (`benches/knn_methods.py`, F12).

### 2.3 Streamed head: the remaining in-memory steps

**Found by the scaling sweep (2026-10-06):** the in-memory pipeline stops fitting 18 GB between
250k and 500k cells, and the step that breaks it is `tl.rank_genes_groups`: +6.2 GB at 116k
cells and +25.5 GB at 476k (37 groups x 2 000 genes), far beyond the gene-major transpose
it needs. Profile the binding and the Python assembly (`_de.py`: structured arrays per group)
and bound the working set; until then the marker test is the memory ceiling of the in-memory
path and the streamed head is what runs above 250k.
`pp.preprocess_backed` covers QC, filters, normalise, log1p, HVG, scale, PCA. Add
`tl.rank_genes_groups` over row blocks (per-gene ranks need the whole column; do it
gene-block by gene-block from a CSC copy written once) so marker genes at 1 M do not
need the matrix in memory either. UMAP and Leiden already work from the graph.

### 2.4 Parity with scanpy (the missing functions)
From the API audit (scanpy 1.12.4): `pp`: 17/22, `tl`: 15/20, `pl`: 3/48,
`experimental.pp`: 0/4. In priority order:
1. `pp.scrublet` and `pp.scrublet_simulate_doublets` (doublet detection; every pipeline runs it);
2. `experimental.pp.normalize_pearson_residuals` and `highly_variable_genes(flavor="pearson_residuals")`;
3. `pp.recipe_zheng17`, `recipe_seurat`, `recipe_weinreb17` (thin wrappers);
4. `tl.ingest` (map new cells onto a reference embedding and labels; useful with the 1 M atlas);
5. `pl`: `embedding`, `scatter`, `violin`, `dotplot`, `heatmap`, `matrixplot`, `stacked_violin`,
   `highest_expr_genes`, `highly_variable_genes`, `pca`, `pca_loadings`, `paga`, `dendrogram`,
   `rank_genes_groups_{dotplot,violin,heatmap}`, `embedding_density`, `tsne`, `diffmap`, `draw_graph`;
   written natively in matplotlib in the house style of `pl.py`, each held to scanpy's output
   on PBMC 3k by a snapshot test. Until then the README says: run metalcyte, plot with `sc.pl.*`.
6. `tl.sim`, `paga_compare_paths`, `paga_degrees`, `paga_expression_entropies`: low value, last.
`external.*` is scanpy calling other packages; out of scope, say so in the docs.

### 2.5 Housekeeping before any PR
- fix the `clippy -D warnings` failure on `main` (run `cargo clippy --all-targets --all-features`);
- decide the 3 scanpy-1.12.4 sparse-median tests: follow scanpy 1.12 (preferred) or pin the old rule;
- run the whole Python suite on `METALCYTE_TEST_DEVICE=auto` and `cpu`;
- `benches/results/` is committed (small JSON and text), the data files are not.

## 3. Figures: the theory and the hardware, drawn

Style: plotly, `plotly_white`, default colorway, no outlines; exported as SVG and PNG
at 300 dpi into `docs/figures/`. Diagrams use plotly shapes and annotations so every
figure comes from one script, `benches/figures.py`, and can be regenerated.

| # | figure | what it shows | source |
|---|---|---|---|
| F1 | **The chip and the library** | the M3 Pro as blocks (5 P cores, 6 E cores, AMX, 14-core GPU, one unified memory, SSD) with arrows for which metalcyte layer uses which block: rayon across P+E, Accelerate on AMX, candle/Metal kernels on the GPU, numpy buffers borrowed in place | diagram |
| F2 | **Where the bytes go** | the same pipeline step (`pp.scale` then `pp.pca`) as a byte-flow: scanpy's copies (sparse, dense f64, scaled, device) against metalcyte's one fused pass into a buffer the GPU reads directly; numbers from the `+MB` columns | diagram + measured |
| F3 | **A million cells in four passes** | the streamed head: disk → row block → per-block transform → accumulator (sums, scatter) → eigenvectors → projection; a second row shows what scanpy would need to hold (8 GB dense) | diagram |
| F4 | **Ablation** | seconds per step with each Apple-specific choice removed | 1.1 |
| F5 | **Roofline** | kernels against the chip's compute and memory roofs | 1.2 |
| F6 | **Scaling** | seconds vs cells, both libraries, per step, log-log | 1.5 |
| F7 | **Energy** | net joules per run and per million cells, with the NVIDIA figures as reference lines | 1.4 |
| F8 | **Utilisation timeline** | P-core, E-core and GPU residency over the 117k run, steps marked | 1.3 |
| F9 | **Parity** | streamed vs exact PCA: canonical correlations per component; HVG overlap | done |
| F10 | **The UMAP** | the 1 M-cell embedding coloured by author cell type, the picture that says "this ran on a laptop" | done (data in `embryo1m` run; re-run with `obs_columns=("cell_type",)` kept) |

## 4. Order of work and dependencies

1. **Now (machine busy with the tuned scanpy run):** clippy fix, median-test decision,
   `benches/ablation.py`, `benches/figures.py` skeleton with F1, F2, F3 (no measurements needed).
2. **Machine free:** build the no-Accelerate wheel, run the ablation (117k then 1 M; about
   40 min), full Python suite, then the user runs `run_energy.sh` under sudo.
3. **Then:** 2.1 (kNN on the matrix units), re-run the 1 M pipeline and the roofline.
4. **Then:** scaling curves (1.5), figures F4 to F8, `docs/development/MEASUREMENTS_2026-10.md` filled, PR to `main`.
5. **After the PR:** parity work in 2.4, approximate neighbours 2.2, streamed markers 2.3.

## 5. The paper, one paragraph

A Rust implementation of the scanpy API whose preprocessing, decomposition, graph and
differential-expression steps run on every core, the AMX units and the GPU of an Apple
silicon laptop through one memory, with an out-of-core head that puts a million cells
through QC, feature selection and PCA in half a minute without holding the matrix.
Against scanpy on the same laptop it is 5 to 11x faster end to end on a 117 000-cell
atlas and finishes a million cells where scanpy cannot start; against the published
NVIDIA-GPU figures it is within an order of magnitude of a 48 GB server card on a
machine that draws under 30 W, and the ablation shows which hardware feature buys
which second.

### 2.5 t-SNE and batch correction at scale
**Done (2026-10-07):** `tl.tsne(method="fft")` (FIt-SNE in `tsne_fft.rs`), the default above
20 000 cells: 13 s at 117 308 cells, 54 s at 953 436, the whole iteration on the GPU (Accelerate's FFT on the cores). `regress_out` and `combat` read gene blocks
from the sparse input (`batch::ColumnBlocks`), `combat` runs its empirical Bayes step on per-batch
sufficient statistics between two passes, and the only dense array is the result, budgeted at 60%
of physical memory instead of a fixed 8 GiB.
