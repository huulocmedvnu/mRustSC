# Benchmarks

scrust is faster than scanpy at some things and slower at others, and the pattern is
consistent enough to plan around. This page is the measurement, wins and losses in
the same table.

**Read this first if you use t-SNE.** `tl.tsne` is the one operation that gets
*worse* as your data grows, and badly: 1.42x faster than scanpy at 499 cells,
0.25x at 2 638, and **0.06x at 10 000 cells — 271.73 s against scanpy's 15.41 s,
seventeen times slower.** It refuses more than 20 000 cells outright. This is
architectural, not a tuning problem; see [t-SNE does not scale](#tltsne-does-not-scale).

Reproduce it with:

```bash
PYTHONPATH=$PWD/python .venv/bin/python benches/benchmark.py
PYTHONPATH=$PWD/python .venv/bin/python benches/streaming.py
```

Both scripts take `settings.device` as it comes, which is `"auto"`. That resolves to
Metal where there is one and to the CPU otherwise — but only four of the timed
operations reach the device at all, so most of this page reproduces on either. See
[Which rows actually used the GPU](#which-rows-actually-used-the-gpu) before reading a
row as a GPU number, and [The two devices are not
interchangeable](#the-two-devices-are-not-interchangeable) before assuming the two
produce the same values.

## How it was measured

- Each library runs in its **own subprocess**, so a peak-memory reading belongs to
  one library only.
- Every timing is the **best of up to 2 runs**; an operation stops repeating once 30
  seconds have gone into it. Best-of rather than mean because the noise on a shared
  machine is one-sided.
- **Peak MB** is the highest resident size of the worker process while the call ran,
  sampled every 5 ms. It includes the interpreter, the imports and the input matrix
  the call was handed, so compare rows against each other, not against zero.
- Every stage is prepared **with scanpy**, so the only difference between the two
  runs of an operation is that operation.
- Sizes above PBMC 3k's 2 638 cells are bootstrapped from it and binomially thinned,
  which preserves sparsity and depth. Above 2 638 cells this is a cost model, not
  biology.
- Machine: Apple silicon, GPU path live (`gpu_available() == True`), scanpy 1.12.2,
  scrust 0.2.0. Every number on this page was measured on that one machine, on that
  configuration, and has **not** been re-measured since. Where a note below says a
  figure is stale, it is stale — it is kept with its provenance rather than deleted or
  guessed at.

### Which rows actually used the GPU

`settings.device` defaults to `"auto"`, and `DeviceKind::Auto` resolves to
`Device::new_metal(0).unwrap_or(Device::Cpu)` (`crates/scrust-core/src/device.rs`), so
the sweep above ran with Metal selected. **Selected is not the same as used.** Several
bindings take the device and then never touch it — the core signature is `_device` —
and those rows are CPU timings no matter what the machine offers:

| benchmarked operation | device reaches the arithmetic? | evidence |
| --- | --- | --- |
| `pp.filter_cells`, `pp.filter_genes` | no — no device parameter at all | `preprocess/filter.rs:68,84` |
| `pp.normalize_total` | **no** — `_device` | `preprocess/normalize.rs:35` |
| `pp.log1p` | no — no device parameter | `preprocess/normalize.rs:69` |
| `pp.highly_variable_genes` | **no** — `_device` | `preprocess/hvg.rs:44` |
| `pp.scale` | yes (moments in f64 on the CPU, the broadcast arithmetic on the device) | `preprocess/scale.rs:13` |
| `pp.pca` | yes | `pca.rs:79` |
| `pp.neighbors` | yes — hand-written `knn` Metal kernel on Metal, candle path on the CPU | `scrust-py/src/embedding.rs` (dispatch), `scrust-gpu/.../knn.rs` |
| `tl.umap` | **no** — `_device`; UMAP always runs on the CPU | `umap.rs:61` |
| `tl.tsne` | yes | `tsne.rs:77` |
| `tl.rank_genes_groups` (wilcoxon) | **no** — `_device` | `de/wilcoxon.rs:51` |
| `tl.paga` | no — no device parameter | `paga.rs:36` |
| `get.*` | no — plain Python and pandas | — |

`tl.leiden` and `tl.louvain` also take `_device` (`cluster.rs:93,129`) and always run on
the CPU; they are not in the timing tables. So do the parametric DE methods
(`de/parametric.rs:39,59`).

Read the table that way: on a machine without Metal, only the `pp.scale`, `pp.pca`,
`pp.neighbors` and `tl.tsne` rows can move. The rest are the same code either way.

### The two devices are not interchangeable

Choosing a device changes more than the clock. Both devices run the same candle source,
which means the same algorithm — it does **not** mean bit-identical output. `f32`
addition is not associative, and a GPU reduction lands a few ulps away from a sequential
CPU one.

That gap has already produced one real bug, fixed this session: `|a-b|^2` expanded as
`|a|^2 + |b|^2 - 2a.b` cancels to exactly 0 for two identical cells on the CPU but left
a sub-ulp positive on Metal, which the square root amplified. Duplicate cells then
carried a non-zero `rho` and their UMAP connectivities stopped being 1. `neighbors.rs`
now snaps squared distances below `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` to
zero.

So a timing on one device is not a claim about the other's *results*, only about its
speed. `tests/test_device_parity.py` is what holds the two against each other, and it
**skips entirely where there is no GPU** — including GitHub's hosted macOS runners. CI
being green is not evidence the GPU path was exercised.

## Results

`speedup` is scanpy seconds ÷ scrust seconds. **Above 1.00x scrust is faster; below
1.00x scanpy is faster.**

### 499 cells

| operation | genes | scanpy s | scrust s | speedup | scanpy MB | scrust MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pp.filter_cells` | 32738 | 0.0039 | 0.0043 | 0.89x | 696 | 690 |
| `pp.filter_genes` | 32738 | 0.0047 | 0.0045 | 1.05x | 697 | 697 |
| `pp.normalize_total` | 10339 | 0.0005 | 0.0061 | **0.09x** | 697 | 714 |
| `pp.log1p` | 10339 | 0.0009 | 0.0020 | **0.43x** | 697 | 721 |
| `pp.highly_variable_genes` | 10339 | 0.0053 | 0.0033 | 1.62x | 697 | 721 |
| `pp.scale` | 2000 | 0.0026 | 0.0156 | **0.17x** | 710 | 743 |
| `pp.pca` | 2000 | 0.0407 | 0.1133 | **0.36x** | 758 | 825 |
| `pp.neighbors` | 2000 | 0.0037 | 0.0058 | **0.64x** | 715 | 730 |
| `tl.umap` | 2000 | 0.4729 | 0.1388 | 3.41x | 778 | 791 |
| `tl.tsne` | 2000 | 0.8776 | 0.6189 | 1.42x | 245 | 799 |
| `tl.rank_genes_groups` | 10339 | 0.1329 | 0.0131 | 10.15x | 355 | 797 |
| `tl.paga` | 2000 | 0.0044 | 0.0002 | 23.00x | 378 | 797 |
| `get.obs_df` | 10339 | 0.0007 | 0.0007 | 1.11x | 382 | 797 |
| `get.var_df` | 10339 | 0.0004 | 0.0005 | 0.84x | 383 | 797 |
| `get.rank_genes_groups_df` | 10339 | 0.0098 | 0.0108 | 0.90x | 395 | 798 |
| `get.aggregate` | 10339 | 0.0015 | 0.0023 | **0.65x** | 421 | 798 |

### 2 638 cells (real PBMC 3k)

| operation | genes | scanpy s | scrust s | speedup | scanpy MB | scrust MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pp.filter_cells` | 32738 | 0.0131 | 0.0126 | 1.04x | 1033 | 1055 |
| `pp.filter_genes` | 32738 | 0.0207 | 0.0196 | 1.06x | 1042 | 1064 |
| `pp.normalize_total` | 13656 | 0.0019 | 0.0113 | **0.17x** | 1042 | 1119 |
| `pp.log1p` | 13656 | 0.0047 | 0.0107 | **0.44x** | 1042 | 1119 |
| `pp.highly_variable_genes` | 13656 | 0.0129 | 0.0120 | 1.07x | 1060 | 1119 |
| `pp.scale` | 2000 | 0.0106 | 0.0299 | **0.35x** | 1060 | 1148 |
| `pp.pca` | 2000 | 0.4699 | 0.3460 | 1.36x | 1186 | 1478 |
| `pp.neighbors` | 2000 | 0.0188 | 0.0163 | 1.15x | 1012 | 1023 |
| `tl.umap` | 2000 | 2.2144 | 0.7482 | 2.96x | 1197 | 1190 |
| `tl.tsne` | 2000 | 4.1346 | 16.6965 | **0.25x** | 1187 | 1258 |
| `tl.rank_genes_groups` | 13656 | 0.9081 | 0.0316 | 28.72x | 1302 | 343 |
| `tl.paga` | 2000 | 0.0151 | 0.0003 | 55.67x | 1249 | 421 |
| `get.obs_df` | 13656 | 0.0018 | 0.0018 | 1.03x | 1249 | 428 |
| `get.var_df` | 13656 | 0.0005 | 0.0003 | 1.44x | 1249 | 428 |
| `get.rank_genes_groups_df` | 13656 | 0.0150 | 0.0133 | 1.12x | 1249 | 476 |
| `get.aggregate` | 13656 | 0.0025 | 0.0057 | **0.44x** | 1255 | 494 |

### 10 000 cells (bootstrapped)

| operation | genes | scanpy s | scrust s | speedup | scanpy MB | scrust MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pp.filter_cells` | 32738 | 0.0394 | 0.0430 | 0.92x | 2170 | 2168 |
| `pp.filter_genes` | 32738 | 0.0642 | 0.0603 | 1.07x | 2178 | 2200 |
| `pp.normalize_total` | 16004 | 0.0087 | 0.0286 | **0.30x** | 2178 | 2330 |
| `pp.log1p` | 16004 | 0.0164 | 0.0361 | **0.45x** | 2178 | 2256 |
| `pp.highly_variable_genes` | 16004 | 0.0360 | 0.0360 | 1.00x | 2242 | 2257 |
| `pp.scale` | 2000 | 0.0409 | 0.0981 | **0.42x** | 2403 | 2522 |
| `pp.pca` | 2000 | 1.4019 | 0.8592 | 1.63x | 3028 | 3627 |
| `pp.neighbors` | 2000 | 0.7251 | 0.1251 | 5.80x | 2221 | 2230 |
| `tl.umap` | 2000 | 9.8716 | 2.0403 | 4.84x | 2402 | 2413 |
| `tl.tsne` | 2000 | 15.4091 | 271.7281 | **0.06x** | 2489 | 2920 |
| `tl.rank_genes_groups` | 16004 | 3.8273 | 0.0912 | 41.95x | 2422 | 365 |
| `tl.paga` | 2000 | 0.0584 | 0.0006 | 90.40x | 2496 | 702 |
| `get.obs_df` | 16004 | 0.0054 | 0.0068 | **0.79x** | 2206 | 545 |
| `get.var_df` | 16004 | 0.0005 | 0.0004 | 1.16x | 2042 | 2041 |
| `get.rank_genes_groups_df` | 16004 | 0.0152 | 0.0198 | **0.77x** | 2209 | 640 |
| `get.aggregate` | 16004 | 0.0057 | 0.0227 | **0.25x** | 2213 | 708 |

The `get.var_df` row comes from a second, single-operation run
(`--ops get.var_df`). In the first sweep **both** libraries raised
`ValueError: adata.obs_names contains duplicated items` — a defect in the benchmark's
own fixture, not in either library. Growing PBMC 3k to 10 000 cells resamples cells
with replacement, so a cell can appear twice, and `get.var_df` selects cells by name
and refuses a duplicated index. scanpy refuses it with the identical message, which
is why the failure appeared in both columns. `resize()` now calls
`obs_names_make_unique()`; that changes labels only, so the rest of this table is
unaffected and was not re-run.

## Reading the table

### Where scrust wins, and why

`rank_genes_groups` and `paga` are the clearest wins, and both improve with size:

| operation | 499 | 2 638 | 10 000 |
| --- | ---: | ---: | ---: |
| `tl.paga` | 23.00x | 55.67x | **90.40x** |
| `tl.rank_genes_groups` | 10.15x | 28.72x | **41.95x** |
| `tl.umap` | 3.41x | 2.96x | 4.84x |
| `pp.pca` | 0.36x | 1.36x | 1.63x |
| `pp.neighbors` | 0.64x | 1.15x | 5.80x |

`rank_genes_groups` and `paga` are the same shape of problem: a large amount of
independent per-gene or per-edge work that scanpy does through numpy in several full
passes over the matrix, and that Rust does in one pass with no interpreter in the
loop. Neither touches the GPU — `wilcoxon.rs` takes `_device` and `paga.rs` takes no
device at all — so these are CPU-against-CPU wins.

`pca` and `neighbors` are the tensor-algebra wins: dense batched work that does reach
the device, on hardware with unified memory. `neighbors` is now the strongest of the
two, because it is the one row backed by a **hand-written Metal kernel**: a Metal caller
runs `scrust-gpu`'s `knn` kernel rather than candle's backend, which does the distance
matrix and the k-selection in one GPU pass. That is what re-measured the row from
`0.43x / 0.59x / 2.33x` (candle) to `0.64x / 1.15x / 5.80x` (kernel) — the scanpy
baseline reproduced within noise, so the change is the kernel, not the machine. The
kernel reproduces the CPU path bit-for-bit on degenerate input (`tests/test_device_parity.py`,
4 of 4). `umap` is **not** a GPU win despite being large in the top block — `umap.rs:61`
takes `_device` and never uses it. Its 3-5x is single-threaded-interpreter-versus-Rust,
not CPU-versus-GPU, and it will hold on a machine with no GPU at all.

Note that `pca` and `neighbors` are *slower* than scanpy at 499 cells and only pull
ahead as the matrix grows, for the same boundary-cost reason as the elementwise
steps below; `neighbors` now crosses into a win by 2 638 cells where before it took
until 10 000.

### Where scrust loses, and why

**The cheap elementwise steps are slower at every size measured, and much slower at
small sizes.**

| operation | 499 | 2 638 | 10 000 |
| --- | ---: | ---: | ---: |
| `pp.normalize_total` | **0.09x** | 0.17x | 0.30x |
| `pp.scale` | **0.17x** | 0.35x | 0.42x |
| `pp.log1p` | 0.43x | 0.44x | 0.45x |
| `get.aggregate` | 0.65x | 0.44x | 0.25x |

These are bandwidth-bound passes over a sparse matrix that numpy already does in
optimised C, and scrust adds a fixed per-call cost that numpy does not pay:

1. the CSR arrays are cast to `uint32`/`float32` if they are not already,
2. they cross the FFI boundary,
3. the result is rebuilt into a new scipy CSR matrix on the way back.

At 499 cells that boundary cost *is* the runtime — the arithmetic itself is a
fraction of a millisecond either way, so the ratio measures the boundary and nothing
else. The trend confirms it: `normalize_total` climbs 0.09x → 0.17x → 0.30x and
`scale` 0.17x → 0.35x → 0.42x as the fixed cost is amortised over more data. But
neither has reached parity by 10 000 cells, and `log1p` sits flat at ~0.44x
throughout. **If your pipeline is dominated by normalising small matrices, scrust is
the wrong tool**; these steps are cheap in absolute terms (tens of milliseconds at
10 000 cells) and the pipeline-level win has to come from `pca`, `umap`,
`rank_genes_groups` and `paga`.

### `tl.tsne` does not scale

This is the one large, structural loss, and it is worth stating without softening:

| cells | scanpy s | scrust s | speedup |
| ---: | ---: | ---: | ---: |
| 499 | 0.8776 | 0.6189 | 1.42x — scrust faster |
| 2 638 | 4.1346 | 16.6965 | 0.25x — 4x slower |
| 10 000 | 15.4091 | **271.7281** | **0.06x — 17x slower** |
| above 20 000 | — | refuses | `ValueError` |

The trend is the finding: scrust wins at 499 cells, loses by 4x at 2 638, and loses
by 17x at 10 000. It gets worse, not better, with size.

The cause is a deliberate design choice going the wrong way at scale. scrust's t-SNE
is **exact**: it materialises the full `(n, n)` affinity matrix, because a dense
quadratic form is a matmul the GPU eats, where a Barnes-Hut tree walk is exactly the
irregular pointer-chasing a GPU is worst at. scanpy uses scikit-learn's Barnes-Hut,
which is `O(n log n)`. Exact beats `n log n` while `n` is small enough that constant
factors dominate — that is the 499-cell row — and loses increasingly fast after that.

The exact formulation is also why there is a hard ceiling. `MAX_CELLS = 20 000` in
`crates/scrust-core/src/tsne.rs`: above it the call returns
`Error::InvalidParameter`, surfacing in Python as a `ValueError`, rather than
attempting the allocation. At 20 000 cells the affinity matrix alone is 1.6 GB and
the gradient step holds three more buffers of that shape, for a peak near 6.5 GB.
Refusing with a documented limit is the designed behaviour; exhausting unified
memory is the alternative.

What it buys is a marginally better optimum of the objective:
[VALIDATION.md](VALIDATION.md) records scrust reaching KL 2.028 against scanpy's
2.076 on PBMC 3k. That is a real but small gain for 4x the time at 2 638 cells and
17x at 10 000.

**What to do instead.** Above roughly 2 000 cells, use `sc.tl.tsne` — it writes the
same `obsm["X_tsne"]` slot, so nothing else in a scrust pipeline changes:

```python
sr.pp.neighbors(adata, use_rep="X_pca")
sc.tl.tsne(adata)          # Barnes-Hut; scrust's exact version is for small n
sr.tl.umap(adata)          # UMAP, by contrast, is 4.84x faster at 10 000 cells
```

If you want a 2-D embedding of a large dataset, prefer `tl.umap`, which goes the
other way with size: 3.41x faster at 499 cells, 2.96x at 2 638, 4.84x at 10 000.

**`get.*` are a wash, with one loss.** They are plain Python and pandas in both
libraries — there is no Rust behind them — so most differences are incidental.
`get.aggregate` is the exception and the only `get` row that moves with size:
0.65x → 0.44x → 0.25x. It is scrust doing per-group reductions in scipy where scanpy
has a more specialised path, and it gets relatively worse as groups grow.

### The 50 000-cell case

**Not measured.** The run was started and deliberately cut short: scanpy's UMAP and
t-SNE at 50 000 cells were on course to take on the order of an hour, and the
benchmark was stopped at 10 000. No 50 000-cell timings are reported here — no
estimate, no extrapolation.

Two things are known about that size without measuring it:

- **`tl.tsne` would not have produced a timing at all.** It refuses more than
  20 000 cells with a `ValueError`, because the exact formulation materialises an
  `(n, n)` affinity matrix — 1.6 GB at 20 000 cells, with three more buffers of that
  shape live during the gradient step, for a peak near 6.5 GB. Refusing with a
  documented limit is the designed behaviour; the alternative is an out-of-memory
  kill. `benchmark.py` encodes the same limit and does not time scanpy there either,
  because a one-sided row compares nothing.
- **`pp.neighbors` would be slow.** The k-NN search is exact, which is why its
  results match scanpy's neighbour sets completely
  ([VALIDATION.md](VALIDATION.md)), and it is `O(n^2)` in cells where scanpy's
  pynndescent is approximate. It is already only 2.33x ahead at 10 000 cells rather
  than pulling away.

Run it yourself with `benches/benchmark.py --sizes 50000 --repeats 1` if you have the
hour.

## Memory: streaming row blocks

`benches/streaming.py` measures the other axis, where a large matrix *is* cheap to
test: peak resident memory computing the same per-gene sums over a synthetic
50 000 x 20 000 matrix at 2% density, two ways.

| mode | seconds | peak GB |
| --- | ---: | ---: |
| read whole file, densify, reduce | 0.86 | 4.34 |
| stream row blocks (`scrust._backed`) | 0.51 | **1.41** |

Both produce identical sums. The file is 0.15 GB on disk; dense `float32` would be
3.73 GB, which is what the first mode pays for and the second does not. Streaming
used blocks of 12 904 rows, sized from `settings.max_memory_gb = 1.0`.

**This is not a benchmark against scanpy and it does not fill the gap left by the
abandoned 50 000-cell sweep.** It compares two ways of reading a file inside scrust,
it measures bytes rather than seconds against a reference, and it completed in under
a second. No operation timing at 50 000 cells is reported anywhere on this page.

One caveat. The win is bounded by what you do with each block: densifying a block is
still `rows x n_vars x 4` bytes, which is what the block sizing is derived from.

### Out-of-core `pp.normalize_total` and `pp.log1p` (v0.2.0)

As of v0.2.0 `scrust._backed` is wired into `pp.normalize_total` and `pp.log1p`: when
`adata.isbacked`, they stream `X` in row blocks and rewrite it on disk in place, so peak
memory is one block rather than the whole matrix. `benches/backed_transform.py` measures
the peak resident memory of the two paths — read the file into memory and transform it
there, versus stream it — applying counts-per-10k normalisation then `log1p`.

| mode | peak MB |
| --- | ---: |
| whole file in memory, then transform | 1141 |
| streamed on disk (`isbacked=True`) | **737** |

Measured on a synthetic 40 000 x 6 000 matrix at 8% density: **404 MB less peak RAM
(0.65× of the in-memory peak)**, with **bit-for-bit identical** output (the two paths
produce the same matrix; the checksums agree). Each block is transformed by the same
`scrust-core` routine the in-memory path uses, so normalisation stays per-row exact and
`log1p` element-wise exact.

## Harmony batch integration (v0.2.0)

`sr.pp.harmony_integrate` is a native Rust/candle implementation of Harmony (Korsunsky et
al. 2019). It is iterative and k-means seeded, so it is not bit-for-bit with `harmonypy` (a
compiled C++ backend); correctness is batch mixing, measured by iLISI, and the timing is
reported against `harmonypy` as a reference. Measured on PBMC 3k (2 638 cells, 50 PCs, two
batches with a shift injected into the embedding):

| | time | iLISI |
| --- | ---: | ---: |
| before correction | — | 1.00 |
| scrust harmony (CPU) | **0.182 s** | **1.90** |
| `harmonypy` 2.0.0 (C++) | 0.064 s | — |

scrust raises iLISI from 1.00 (batches separated) to 1.90, near the maximum of 2 for two
batches, and its objective converges. The CPU path is **0.182 s, a 3.2× improvement** over
the 0.590 s of the first working version — from an adaptive ndarray/candle matmul,
rayon-parallel M-step and E-step reductions, and harmonypy's convergence tolerances.

It does **not** reach harmonypy's 0.064 s, and this is stated rather than glossed: harmonypy
uses a hand-tuned Accelerate BLAS, and the remaining gap is small-matrix matmul that a
pure-Rust build without a BLAS backend cannot close. A profiled attempt at an `(N, K)`
contiguous layout and a batch-parallel E-step was measured, found either neutral or
correctness-breaking (iLISI collapsed to 1.00), and reverted; 0.182 s is the honest floor
that keeps iLISI ≥ 1.85. An optional Accelerate BLAS backend (below) narrows the CPU-side
margin but does not close the harmonypy gap.

## Optional Apple Accelerate BLAS (post-v0.2.0)

The Harmony gap above motivated an opt-in `accelerate` cargo feature (`maturin develop
--release --features accelerate`) that links Apple's Accelerate (vecLib) BLAS/LAPACK behind
the dense CPU paths — ndarray's `.dot()` and candle's CPU backend — for PCA, Harmony,
neighbour distances and diffusion. It is **off by default**: the shipped build stays
pure-Rust and Linux/CI-portable, and the sparse CSR paths (`normalize_total`, `log1p`)
never touch BLAS either way, so enabling it cannot change their memory profile.

The gain is real but modest, and CPU-path only. Measured on an M3 Pro, each config rebuilt
and the extension reinstalled, best (min) of 5 runs, `device="cpu"`:

| workload | pure-Rust | `--features accelerate` | gain |
| --- | ---: | ---: | ---: |
| Harmony, 2 638 cells | 0.1163 s | 0.1083 s | ~7% |
| Harmony, 10 000 cells | 0.6369 s | 0.5872 s | ~8% |
| Harmony + PCA | 0.513 s | 0.467 s | ~9% |
| pipeline PCA→Neighbors→UMAP→Harmony | 1.321 s | 1.274 s | ~4% |

Two honest caveats:

- **These absolutes come from a separate A/B harness** (`device="cpu"`, min of 5), not the
  sweep at the top of this page — so read them as a with/without *ratio* on one machine,
  not against the 0.182 s Harmony row. A different injected shift and repeat policy move the
  absolute Harmony time, which is why the pure-Rust column here reads faster than 0.182 s;
  only the two columns beside each other are comparable.
- **The gain is small because these routines are not purely matmul-bound at single-cell
  sizes**, and the default `device="auto"` runs the dense ops on Metal, not CPU BLAS.
  Accelerate only touches the explicit CPU path; on the GPU path it changes nothing, and it
  narrows rather than closes the harmonypy margin.

## What could not be benchmarked

**The 24 probed functions.** `benchmark.py` keeps a second list, `PROBES`, of every
remaining public name — `pp.calculate_qc_metrics`, `pp.regress_out`, `pp.combat`,
`pp.subsample`, `tl.leiden`, `tl.louvain`, `tl.diffmap`, `tl.dpt`, `tl.score_genes`,
`tl.dendrogram`, `tl.draw_graph`, all four `metrics`, and the rest — and calls each
once on a small input so that its outcome is measured rather than asserted.

**These are no longer stubs.** An earlier version of this page said all 24 raise
`NotImplementedError` naming the branch that owed them; that was true while the feature
branches were in flight and is not true now. Every name in `PROBES` is exported and
implemented, and as of v0.2.0 there is **no `NotImplementedError` left anywhere in
`python/scrust/`** — `tl.dpt(n_branchings > 0)`, the last one, now runs native branch
detection. What is still missing is *timings*: none of these 24 appears in the tables
above, because `PROBES` calls them once and does not time them. Benchmarking them means
adding them to `OPS` and re-running the sweep. See [API.md](API.md) for the current state
of each.

**The hand-written Metal kernels.** `crates/scrust-gpu` contains four kernel modules —
`spmm` (CSR SpMM, transposed SpMM, column moments, row scaling), `knn`, `umap_sgd` and
`tsne_gradient` — tested in Rust against their `scrust-core` counterparts.

**One of them, `knn`, is now on the call path.** `crates/scrust-py` depends on
`crates/scrust-gpu` and its `embedding` binding dispatches a Metal caller's k-NN to the
`knn` kernel (`scrust-py/src/embedding.rs`), falling back to the candle path on the CPU
or where no Metal context builds. The `pp.neighbors` row above is measured with that
kernel live, which is why it improved: the figure is no longer a Rust-against-Rust
microbenchmark of unreachable code but the shipped path a Metal caller takes. The kernel
matches the CPU oracle bit-for-bit on the degenerate inputs that matter
(`tests/test_device_parity.py`, 4 of 4).

**The other three are still not reachable.** `spmm` has no Python-reachable consumer
(`core::pca` does a *centred* product, not the plain sparse×dense it offers),
`tsne_gradient` is unwired, and `umap_sgd` is left unwired on purpose — it is Hogwild, so
wiring it would make a UMAP layout depend on whether the caller has a GPU. Their speedup
figures elsewhere remain Rust-against-Rust microbenchmarks. For every row other than
`pp.neighbors`, what the GPU contributes is candle's Metal backend, on the operations
marked "yes" in [Which rows actually used the GPU](#which-rows-actually-used-the-gpu).
