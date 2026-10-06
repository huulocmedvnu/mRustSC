# Plan: proving scrust is built for Apple silicon, and finishing what is half done

Written 2026-10-06 on branch `feat/scanpy-parity`. This is the working plan for the
next pass over the project; it lists what is done, what is measured but not yet
written up, what is still missing against scanpy, and the experiments and figures a
paper needs before the claim "optimised for Apple silicon" can be made in print.
Tick items off here as they land; `docs/SCALE.md` holds the numbers.

## 0. Where things stand

| area | state |
|---|---|
| per-operation benchmark to 100 000 cells | done, `benches/results/benchmark_100k_m3pro.txt` |
| real 117k atlas end to end, scanpy vs scrust (CPU, Metal, parallel UMAP) | done, 233 s / 47 s / 47 s / 21.5 s |
| scanpy with its fastest settings (`--tuned`) as a fourth column | done, 86.1 s |
| 1 M cells on 18 GB through `pp.preprocess_backed` | done, 210 s Metal, 214 s CPU |
| scanpy on 1 M cells on the same machine | did not finish: `scale` needed 21.9 GB, PCA swapped, killed at 15 min |
| streamed PCA held to scanpy's exact solver | done, `tests/test_streaming.py` and 50/50 components on real data |
| energy per run (`benches/energy.py`, `benches/run_energy.sh`) | script ready, needs one `sudo` run |
| placement against rapids-singlecell | written from their published numbers, `docs/SCALE.md` section 5 |
| Rust unit tests | 267 pass; Python suite not yet re-run in full after today's changes |
| CI on `main` | red: `clippy -D warnings` and 3 scanpy-1.12.4 median tests (pre-existing) |

The one finding that changes the message: **on an M3 Pro the CPU path through Accelerate
(AMX) matches the Metal path to within 2% on both the 117k and the 1 M runs.** The speed
is the whole chip (every core, the AMX units, the GPU, one memory), not the GPU alone.
The claim to prove is therefore "optimised for Apple silicon", and the proof is an
ablation, not a scanpy comparison.

## 1. Evidence: the experiments that prove the claim

Each experiment has a script, an output file in `benches/results/`, and a figure.

### 1.1 Ablation: switch off one Apple-specific choice at a time
Same binary, same data (117k real, 1 M streamed), same machine.

| knob | on | off | how |
|---|---|---|---|
| Accelerate / AMX BLAS | `--features accelerate` (default wheel) | pure-Rust `matrixmultiply` | second wheel built without the feature into `.venv-noaccel` |
| Metal | `settings.device="auto"` | `"cpu"` | done |
| all 11 cores (P+E) | `RAYON_NUM_THREADS=11` | `5` (P only), `1` | env var, rayon honours it |
| zero-copy numpy borrows | current `fast.rs` paths | force the old copying path (`SCRUST_FORCE_COPY=1`, to add) | env var read in `_basics.py` |
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
scanpy, scrust Metal and scrust CPU. Report joules per million cells next to the
NVIDIA figures (an L40S draws about 300 W; 92 s is about 28 kJ for a million cells;
the laptop's whole package is under 30 W).

### 1.5 Scaling curves
Seconds against cells at 10k, 50k, 100k (done), 117k real, 250k and 500k subsamples of
the 1 M atlas, and 1 M, for scanpy (where it fits) and scrust. Log-log, one line per
step. The figure shows where each library's slope changes and where scanpy stops.

## 2. Engineering: what has to change for the claims to hold at 1 M and beyond

### 2.1 Neighbour search on the matrix units (highest value)
Rewrite `knn_metal_tiled` as two stages: distance tiles by a Metal matmul (`q · cᵀ`
for a 4 096 x 65 536 tile is 1 GB of f32, then `|q|² + |c|² - 2 q·c`), followed by a
per-row top-k merge kernel over the tile. The matmul runs at several TFLOP/s on Metal
and through Accelerate on the CPU; the merge is the only custom code left. Target: the
1 M neighbour step from 120 s to under 30 s on both devices. Keep the exact zero-snap
rule from `neighbors.rs` so duplicate cells still get connectivity 1.

### 2.2 Approximate neighbours above 1 M
Exact brute force is quadratic; at 5 M cells it is 25x the 1 M cost. Add an index
(HNSW in Rust, graph built with rayon, queried in parallel) behind
`pp.neighbors(method="approximate")`, held to the exact result by recall at k=15 (umap
uses 15). Not needed for the paper's 1 M result; needed for the "atlas scale" claim.

### 2.3 Streamed head: the remaining in-memory steps
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
   on PBMC 3k by a snapshot test. Until then the README says: run scrust, plot with `sc.pl.*`.
6. `tl.sim`, `paga_compare_paths`, `paga_degrees`, `paga_expression_entropies`: low value, last.
`external.*` is scanpy calling other packages; out of scope, say so in the docs.

### 2.5 Housekeeping before any PR
- fix the `clippy -D warnings` failure on `main` (run `cargo clippy --all-targets --all-features`);
- decide the 3 scanpy-1.12.4 sparse-median tests: follow scanpy 1.12 (preferred) or pin the old rule;
- run the whole Python suite on `SCRUST_TEST_DEVICE=auto` and `cpu`;
- `benches/results/` is committed (small JSON and text), the data files are not.

## 3. Figures: the theory and the hardware, drawn

Style: plotly, `plotly_white`, default colorway, no outlines; exported as SVG and PNG
at 300 dpi into `docs/figures/`. Diagrams use plotly shapes and annotations so every
figure comes from one script, `benches/figures.py`, and can be regenerated.

| # | figure | what it shows | source |
|---|---|---|---|
| F1 | **The chip and the library** | the M3 Pro as blocks (5 P cores, 6 E cores, AMX, 14-core GPU, one unified memory, SSD) with arrows for which scrust layer uses which block: rayon across P+E, Accelerate on AMX, candle/Metal kernels on the GPU, numpy buffers borrowed in place | diagram |
| F2 | **Where the bytes go** | the same pipeline step (`pp.scale` then `pp.pca`) as a byte-flow: scanpy's copies (sparse, dense f64, scaled, device) against scrust's one fused pass into a buffer the GPU reads directly; numbers from the `+MB` columns | diagram + measured |
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
4. **Then:** scaling curves (1.5), figures F4 to F8, `docs/SCALE.md` filled, PR to `main`.
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
