# Manuscript materials: what exists, what each section draws on, what is still missing

A checklist for writing the paper, kept next to the results so nothing is claimed that
was not measured. Target: an application note or methods paper (Bioinformatics,
Nature Methods brief communication style, or IEEE TCBB), one main figure set of six and
a supplement. Working title: *metalcyte: a Rust single-cell engine that puts
a million cells through the standard pipeline on an Apple silicon laptop.*

## 1. What the paper claims, and the evidence behind each claim

| claim | evidence | file | status |
|---|---|---|---|
| Same answers as scanpy | parity tests per algorithm (neighbours exact, UMAP ref. preservation, Leiden modularity = leidenalg quality, HVG, DE, PCA 50/50 components vs `covariance_eigh`) | `docs/VALIDATION.md`, `tests/*_audit.py`, `tests/test_streaming.py`, F9 | done |
| Same biology on a real atlas | HVG overlap, PCA subspace, k-NN overlap, Leiden ARI, marker-gene rank agreement, metalcyte vs scanpy on the 117k atlas | `benches/agreement.py` → `benches/results/agreement_bm117k.json`, figure F11 | **to do** |
| 5 to 11x faster than scanpy defaults, 3 to 4x than scanpy tuned, on a real atlas | 117k end-to-end, 5 configurations | `docs/SCALE.md` §2, F6 | done (single runs; repeat x3 for mean ± sd: **to do**) |
| A million cells on 18 GB, scanpy cannot | 1M streamed head + graph; scanpy killed at scale/PCA | `docs/SCALE.md` §3, F3 | done |
| One fifth of the energy | powermetrics, idle subtracted | `docs/SCALE.md` §4, F7, F8 | done (CPU-only row: optional) |
| Which hardware feature buys what | ablation: GPU, Accelerate, P/E cores, zero-copy, parallel UMAP | `docs/SCALE.md` §5, F4 | done |
| How it scales | seconds vs cells 10k → 1M, 4 configurations, real subsamples | `benches/scaling.py` → F5 | running |
| Where it sits against the NVIDIA alternative | published rapids-singlecell numbers with hardware | `docs/SCALE.md` §6 | done (placement, not a benchmark) |
| The embedding is sensible | 1M-cell UMAP coloured by author cell type | F10 | **to do** (save the embedding from `pipeline_1m.py --save`) |

## 2. Section by section

**Abstract / intro.** scanpy's defaults take 4 minutes on 117k cells and cannot start a
million on a laptop; rapids-singlecell needs a data-centre NVIDIA card. Apple silicon laptops
are what most analysts own. Numbers: `docs/SCALE.md` §2, §3, §6.

**Design (Methods 1).** Figure F1. Rust core, PyO3 zero-copy borrows, rayon over P and E cores,
Accelerate for BLAS, Metal kernels for the quadratic step, unified memory; `f32` with `f64`
reductions; seeded determinism except the Hogwild UMAP. Sources: `docs/ARCHITECTURE.md`,
`docs/API_CONTRACT.md`, `docs/HOW_IT_WORKS.md`.

**Algorithms (Methods 2).** Per step, what is computed and how it is held to scanpy:
`docs/VALIDATION.md`. The streamed head and the covariance PCA: `docs/SCALE.md` §3, F3.
The neighbour search on both devices: `neighbors.rs` (`knn_cpu`), `knn.rs` (tiled Metal).

**Benchmark methodology (Methods 3).** `docs/BENCHMARKS.md` "How it was measured";
`benches/pipeline.py` (end to end, one library per process, footprint by `task_vm_info`),
`benches/energy.py` (powermetrics, idle subtracted), `benches/ablation.py`,
`benches/scaling.py`. Machine, versions, seeds in each results JSON.

**Results 1: real atlas.** F6 + table. **Results 2: a million cells.** F3 + table + F10.
**Results 3: energy and utilisation.** F7, F8. **Results 4: ablation.** F4.
**Results 5: scaling.** F5. **Results 6: agreement with scanpy.** F9, F11.

**Discussion / limitations.** Exact neighbour search is quadratic (307 s CPU, 120 s GPU at
1M; approximate index needed beyond); the Metal kernel runs at about 15% of peak and the
simdgroup-matrix attempt is documented as not yet faster; Accelerate/AMX shows nothing at
2 000 genes; t-SNE is exact and capped at 20k; `regress_out`/`combat` cap at an 8 GiB
working set; 17/22 `pp`, 15/20 `tl`, 3/48 `pl` of scanpy's API; CI has no GPU; macOS only
for the GPU path (the CPU path builds anywhere). All stated in `docs/SCALE.md`,
`docs/BENCHMARKS.md`, `docs/PLAN_APPLE_SILICON.md`.

**Availability.** github.com/huulocmedvnu/metalcyte, MIT, `pip install` from source via maturin;
`benches/results/` holds every number in the paper and `benches/figures.py` draws every
figure.

## 3. Figure plan (main text: F1, F6, F3, F5, F4, F7; supplement: F2, F8, F9, F10, F11)

| # | figure | status |
|---|---|---|
| F1 | chip and library | done |
| F2 | where the bytes go | done |
| F3 | a million cells in four passes | done |
| F4 | ablation | done |
| F5 | scaling, seconds vs cells | running |
| F6 | 117k atlas step by step | done |
| F7 | energy | done |
| F8 | utilisation timeline | done |
| F9 | streamed PCA vs exact | done |
| F10 | 1M-cell UMAP by cell type | to do |
| F11 | agreement with scanpy on the 117k atlas (ARI, overlaps) | to do |

## 4. Still to gather before writing (in order)

1. `benches/scaling.py` sweep → F5 (running).
2. `benches/agreement.py`: both libraries on the 117k atlas, same seeds; HVG Jaccard,
   PCA canonical correlations, 15-NN overlap, Leiden ARI/NMI (and against author cell
   types), Spearman of marker scores per cell type → F11.
3. Three repeats of the 117k pipeline per configuration → mean ± sd in the table.
4. `pipeline_1m.py --save`: write `obs` + `X_umap` + `leiden`, draw F10 by `cell_type`.
5. Optional: energy for the CPU-only run (one more `sudo` pass), the 1M ablation.
