# Validation

This page shows how we check that Metalcyte gives correct results. We compare it against scanpy,
the standard Python library for single-cell analysis, and against the other reference packages
that scanpy relies on. Three kinds of test do this, and each answers a different question.

* **Unit tests** (`cargo test --workspace`) check each Rust function on its own, on inputs the
  author chose.
* **Reference tests** (`tests/test_reference.py`, 26 collected) run the whole pipeline in both
  Metalcyte and scanpy and check that the results agree. They run twice, on a 240-cell synthetic
  matrix and on the real PBMC 3k dataset (2 638 cells), under the `reference` marker. The
  required kind of agreement is chosen for each algorithm. It is fixed in
  [development/API_CONTRACT.md](development/API_CONTRACT.md#scanpy-is-the-reference).
* **Audits** (`tests/test_*_audit.py`, 21 files) test one module at a time. They target errors
  that the reference tests could miss. They check formulas term by term against the reference
  package's own code. They also test edge cases, unusual inputs, and the places where Metalcyte
  differs from scanpy on purpose, with the size of each difference.

To reproduce the reference numbers, run:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_reference.py \
    -o junit_family=xunit1 --junitxml=ref.xml
```

The figures on this page come from one such run on Apple silicon with the GPU in use
(`gpu_available() == True`). All 26 of 26 reference tests passed. Steps that use random numbers
can change in the last digit or two between runs. Steps without randomness give the same result
every time.

### The Accelerate build passes the same tests

Accelerate is Apple's library of fast maths routines. The `accelerate` build option makes the
dense matrix maths on the CPU use Apple's vecLib BLAS (see [INSTALL.md](INSTALL.md)). It does
not change the algorithms. It sends ndarray's `.dot()` and candle's CPU backend through a
different BLAS library. The sparse-matrix (CSR) code never uses it. We hold this build to the
same standard as the pure-Rust build, and it passes:

| build | command | result |
| --- | --- | --- |
| pure-Rust | `cargo test --workspace` | pass |
| Accelerate | `cargo test --workspace --features accelerate` | **300 / 300 pass** |

`cargo clippy --workspace` reports no warnings with either build option. There is one caveat.
The pure-Rust build has a unit test, `umap_sgd::is_reproducible_only_in_structure...`, that can
fail at random on its own. It checks the parallel UMAP optimiser, where threads write to shared
memory without waiting for each other ("Hogwild"), so its result varies a little by design. It
passes with `--features accelerate`. Both builds compute the same algorithm. They differ only in
the order of floating-point additions, which changes the last digits. So the reference and audit
numbers below are not repeated for each build. They hold for either build.

## Deterministic steps: compared value by value

These steps involve no randomness. We compare every value with `numpy.testing.assert_allclose`
at `rtol=1e-5, atol=1e-6`. The table gives the largest absolute difference on PBMC 3k.

| step | max abs difference | note |
| --- | --- | --- |
| `pp.normalize_total` | 1.2e-4 (rel 7.0e-8) | f32 rounding of the scale factor |
| `pp.log1p` | 0.0 | bit-identical |
| `pp.scale` | 1.8e-6 (rel 1.8e-7) | f32 rounding of mean and variance |
| `pp.filter_cells` / `pp.filter_genes` | 0.0 | same cells, same genes kept |

"Bit-identical" means the two results are exactly the same numbers, down to the last bit.

## Gene and neighbour selection: overlap of the chosen sets

| step | criterion | measured (PBMC 3k) |
| --- | --- | --- |
| `pp.highly_variable_genes` | ≥ 0.95 of scanpy's 2 000 genes | **1.00** (identical set) |
| `pp.neighbors` (exact) | mean per-cell neighbour overlap ≥ 0.90 | **1.00**, worst cell 1.00 |
| `pp.neighbors(method="approximate")` | mean recall against the exact lists ≥ 0.95 | **0.99** |

The exact search finds the same k nearest cells as scanpy, so the overlap is complete. The
approximate search uses NN-descent. NN-descent starts from rough neighbour lists from a
random-projection forest and improves them by checking each cell's neighbours' neighbours. We
measure it by recall at k = 15. Recall is the share of the exact neighbours that the approximate
lists contain (`tests/test_neighbors_approximate` in `tests/test_reference.py`). On the embryo
embedding, `benches/knn_methods.py` measures a recall of 0.98 at 100 000 cells and 0.96 at
953 000.

## t-SNE above 20 000 cells

Above 20 000 cells, t-SNE uses an FFT method. It computes the long-range forces between cells on
a grid with a fast Fourier transform. We compare it with the exact method on the same input
(tests in `crates/metalcyte-core/src/tsne_fft.rs`):

- The grid forces agree with the exact forces to within 2% of the root-mean-square (RMS) force at
  the default grid.
- Ten planted clusters stay separated.
- The t-SNE objective of the FFT layout is within 15% of the exact method's (measured: 1.37
  against 1.26 on 3 000 cells).

On PBMC 3k, the two layouts share as many of each cell's 15 neighbours as two t-SNE runs
normally do (`tests/test_reference.py::test_tsne_fft_keeps_cell_types_together`).

We also compare the GPU version with the CPU version (tests in
`crates/metalcyte-gpu/src/kernels/tsne_fft_gpu.rs`). The GPU convolution is within 1e-4 relative
RMS of Accelerate's. The forces are within 1e-3 of their RMS. The objective is within 1e-3 at
every step and within 0.5% after 500 iterations. The same seed gives exactly the same bytes.

## Embedding plots

Metalcyte can draw scatter plots on the GPU. We compare the GPU drawing with the CPU drawing
(tests in `crates/metalcyte-gpu/src/kernels/raster.rs` and `tests/test_plotting_gpu.py`). On
5 000 overlapping semi-transparent discs, the two images differ by 0.03 of 255 per colour channel
on average. Each point lands on the same pixel in both images.

## PCA: stable components and explained variance

scanpy's default PCA solver, `arpack`, gives the same answer every time. Metalcyte uses a
randomised SVD, the same kind of method as scanpy's randomised solver. Some components are
well determined by the data. We call a component "determined" when scanpy's own randomised
solver reproduces scanpy's arpack result with a correlation of at least 0.99. The later
components are not well determined. Small changes in the method can rotate them, so comparing
them one by one tells us nothing. For those we only check the variance each component explains.

| dataset | components | determined | Metalcyte matches (corr ≥ 0.99) | worst variance-ratio gap |
| --- | --- | --- | --- | --- |
| synthetic | 50 | 31 | 48 | 0.003 |
| PBMC 3k | 50 | 7 | 8 | 0.042 |

On PBMC 3k only 7 of 50 components are determined. This comes from the data. After the 7th
component, scanpy cannot reproduce its own result either. Metalcyte matches every determined
component. Its variance ratios stay within the range that a randomised SVD varies by on its own.

## UMAP: how well neighbourhoods are kept

UMAP uses random numbers, and two runs with different seeds give different layouts. So we use a
relative standard. We measure how many of each cell's neighbours in scanpy's layout are also
near it in Metalcyte's layout (K_REF=15 in the reference layout, K_CAND=30 in the candidate). We
call this neighbourhood preservation. We also measure it for scanpy against scanpy with a new
seed. This gives the best value we can expect (the ceiling). Metalcyte must reach at least 85%
of that ceiling.

| dataset | Metalcyte vs scanpy | scanpy vs itself (ceiling) | floor (0.85 × ceiling) | pass |
| --- | --- | --- | --- | --- |
| synthetic | 0.564 | 0.623 | 0.530 | yes |
| PBMC 3k | 0.456 | 0.511 | 0.434 | yes |

The ceiling of about 0.51 on PBMC 3k is the main point. When umap-learn runs again with a new
seed, it keeps only about half of each cell's neighbourhood. Metalcyte comes just below that
ceiling. A different program cannot get much closer to a target that changes this much between
runs.

The `blobs` test data has six clusters, each smaller than K_REF. So the neighbour sets do not
depend on the seed. Here a fixed threshold of 0.80 is possible, and Metalcyte scores **1.00**.

This kind of test is weak on its own. A roughly similar optimiser could pass it even with a
mistake. So the UMAP audit also compares Metalcyte term by term with a copy of umap-learn's
`layouts.py`.

## t-SNE: the quality score

t-SNE minimises a defined score, the KL divergence. Lower is better. So the test asks whether
Metalcyte reaches a KL divergence no worse than scanpy's. We allow 5%, because Metalcyte uses
32-bit numbers (f32) and a different random start. Both libraries use scikit-learn's `auto`
learning rate. scanpy's older default of 1000 makes scanpy's own KL about ten times worse at
these sizes, which would make Metalcyte look better than it is.

| dataset | Metalcyte KL | scanpy KL | ratio | pass (≤ 1.05) |
| --- | --- | --- | --- | --- |
| synthetic | 0.985 | 0.971 | 1.014 | yes |
| PBMC 3k | 2.028 | 2.076 | 0.977 | yes |
| blobs | 0.170 | 0.180 | 0.943 | yes |

On PBMC 3k and blobs, Metalcyte reaches a lower KL than scanpy. It finds a better solution of the
same problem. On the synthetic set its KL is 1.4% higher, within the allowed range.

## Differential expression: top genes compared value by value

For each group, we compare the top 100 genes field by field with scanpy's Wilcoxon test. The
table gives the largest relative difference over all groups and both datasets.

| field | worst deviation |
| --- | --- |
| `scores` | 0.0 |
| `logfoldchanges` | 0.0 |
| `pvals` | ~2e-13 |
| `pvals_adj` | ~2e-13 |

Scores and fold changes are bit-identical. The p-values differ only in the last digits of a
`float64` number. This comes from adding a long list of numbers in a different order. It holds
for every cell type in PBMC 3k, including the Megakaryocytes, which have only 8 cells.

## PAGA: connection strengths compared value by value

PAGA builds a simplified graph of how clusters connect. `tests/test_paga.py` compares the
connection strengths with scanpy's v1.2 model. It also requires the same spanning tree.

| dataset | max relative deviation | tree |
| --- | --- | --- |
| synthetic | 2.3e-8 | identical edges |
| PBMC 3k | 5.5e-8 | identical edges |

## The audits: what each one checks, and against what

There is one audit file per module. Each names the reference code it compares the Rust code
with. The test counts come from `pytest --collect-only -q` on each file, so each parameter case
counts as one test. `test_de_audit.py` is an exception. It loads the compiled library when it is
imported and skips the whole file if the library is missing. So its 56 tests are counted from
its parameter lists, without collection.

| module | file | reference | tests |
| --- | --- | --- | --- |
| `pca` | `test_pca_audit.py` | scikit-learn `randomized_svd`, Halko et al., scanpy `_pca` | 11 |
| `neighbors` | `test_neighbors_audit.py` | umap-learn `umap.umap_.fuzzy_simplicial_set`, called directly | 17 |
| `umap` | `test_umap_audit.py` | umap-learn `layouts.py`, transcribed and re-checked against the package | 27 |
| `tsne` | `test_tsne_audit.py` | scikit-learn `manifold._t_sne` and `_utils._binary_search_perplexity` | 17 |
| `cluster` | `test_cluster_audit.py` | leidenalg 0.12.0 / libleidenalg, Traag et al. 2019, scanpy `_leiden.py` | 11 |
| `preprocess` | `test_preprocess_audit.py` | scanpy `normalize_total`, `log1p`, `highly_variable_genes` | 20 |
| `de` + `scale` + `hvg` boundaries | `test_de_audit.py` | scanpy, `scipy.stats` | 56 |
| `de/wilcoxon` | `test_wilcoxon_audit.py` | scanpy `rank_genes_groups(method="wilcoxon")`, scipy in the far tail | 21 |
| `de/parametric` | `test_parametric_audit.py` | scanpy t-test, t-test_overestim_var, logreg | 12 |
| `de/hypothesis`, `de/glm`, `de/dispersion` | `test_destats_audit.py` | `scipy.special`, statsmodels 0.14.6 | 12 |
| `qc` + `filter` | `test_qc_audit.py` | scanpy `calculate_qc_metrics`, `filter_cells`, `filter_genes` | 24 |
| `batch` + autocorrelation | `test_batch_audit.py` | scanpy `regress_out`, `combat`, `sc.metrics.morans_i` / `gearys_c` | 14 |
| `scoring` + sampling | `test_scoring_audit.py` | scanpy `score_genes` and its pandas bin/sample chain | 25 |
| `diffusion` | `test_diffusion_audit.py` | scanpy `tl.diffmap`, `tl.dpt` | 12 |
| `paga` | `test_paga_audit.py` | scanpy `tl.paga(model="v1.2")` | 17 |
| `layout` | `test_layout_audit.py` | scanpy `tl.dendrogram`, `scipy.cluster.hierarchy`, `scipy.stats.gaussian_kde` | 18 |
| cell cycle | `test_cell_cycle_audit.py` | scanpy `tl.score_genes_cell_cycle` (scores + phase rule) | 5 |
| `dpt` branching | `test_dpt_branching_audit.py` | scanpy `tl.dpt(n_branchings>0)`, adjusted Rand index | 4 |
| `harmony` | `test_harmony_audit.py` | `harmonypy 2.0.0` (iLISI, objective convergence) | 3 |

`test_backed_streaming.py` checks that the out-of-core path gives the same result as the
in-memory path. (The out-of-core path reads the data from disk in blocks.) Where we can run the
reference package on the same input, we do. In three places we cannot call the inner code
directly: umap-learn's SGD inner loop, scikit-learn's t-SNE gradient and scanpy's
`transitions_sym` spectrum. There we copy the reference code into the test. A separate test then
checks the copy against the installed package.

Two known differences are marked with `xfail(strict=True)`. This means the test reports an error
on the day a difference disappears, so we will notice. The first is in PCA. The Gram
eigendecomposition loses about half the digits of the smallest singular values (3.6e-4 against
an exact f64 reference, where scikit-learn gives 6.3e-7). The second is in UMAP. A random
starting layout does not recover the overall arrangement of the clusters, which a spectral
starting layout would.

`chunked` and `sparse` are internal modules. No scanpy or scipy function does the same thing, so
there is nothing to compare them against. Their audits (`test_chunked_audit.py`,
`test_sparse_audit.py`) check fixed properties instead. For `chunked`, the result must not
depend on the block size. For `sparse`, data must survive a round trip, and input checks must
work. Where the two overlap, `sparse` is compared with `scipy.sparse.csr_matrix`.

## CPU or GPU

`settings.device` defaults to `"auto"`. With `"auto"`, Metalcyte uses the GPU (Metal) if a Metal
device starts up, and the CPU otherwise (`crates/metalcyte-core/src/device.rs`). So on any Mac
with Metal, **code that does not name a device runs on the GPU.** Which device produced a result
depends on the machine, and the code does not decide it.

The audits compare with scanpy on one device at a time. `METALCYTE_TEST_DEVICE`
(`tests/metalcyte_call.py`, default `"cpu"`) chooses the device. Set it to `"auto"` to run the
same tests on the GPU. Both settings pass on Apple silicon.

The CPU and GPU run the same candle code, so they run the same algorithm. They can still give
slightly different numbers. With 32-bit numbers (f32), the order of additions changes the last
digits. A GPU adds numbers in a different order from a single CPU thread, so its result can
differ by a few units in the last place (ulps). So `tests/test_device_parity.py` (4 tests)
compares the two devices with each other. This is a separate check from comparing either device
with scanpy.

Not every module uses the device. `pca`, `neighbors` and `tsne` use the chosen `Device`. `umap`
and `cluster` accept it but ignore it (`_device` in `crates/metalcyte-core/src/umap.rs` and
`cluster.rs`). They always run on the CPU.

## What the audits found

The audits found each behaviour below in code that already passed its tests. Each is now fixed
in place by a test that fails if the behaviour changes.

* **The neighbour graph sets tiny distances to zero.** The squared distance is computed as
  `|a-b|² = |a|² + |b|² - 2a·b`. For two identical cells this gives exactly 0 on the CPU. On
  Metal it leaves a tiny positive value (9.5e-7 at norm scale 12). The square root enlarges this
  to 9.8e-4. Then `rho` would not be zero. UMAP subtracts `rho` when it builds its fuzzy
  neighbour set, so duplicate cells would no longer get a connection strength of 1. Metalcyte
  now sets squared distances below `(n_dims + 2) * f32::EPSILON * (|a|² + |b|²)` to zero. On
  PBMC 3k's 50 PCs that cut-off is 0.049. The smallest nearest-neighbour distance is 6.40. So it
  changes 0 of 39 570 neighbours. The problem appears only with `METALCYTE_TEST_DEVICE=auto`.
  Every test that names `"cpu"` passes either way.
* **PAGA counts stored zeros as edges.** scanpy first sets every stored entry to one
  (`ones.data = np.ones(len(ones.data))`, `_paga.py:182-183`). So every stored entry counts as an
  edge. Take 120 cells of which 60 are duplicates. Here `sc.pp.neighbors(n_neighbors=10)` stores
  540 zeros out of 1080 entries. Dropping those zeros would overstate connection strengths by up
  to 0.096. Metalcyte agrees with scanpy to ~2e-8.
* **The diffusion map counts only edges with a non-zero weight** when it checks that the graph is
  connected. If it also counted stored zero entries, it could join separate parts of the graph.
  It would then return a useless map (spectrum `[1.0000001, 1.0, ...]`) and no error. This check
  is stricter than in scipy and scanpy on purpose. They count every stored entry. The code
  explains the reason.
* **The dendrogram uses `complete` linkage**, as `sc.tl.dendrogram` does. It records
  `linkage_method="complete"` in `uns`. On cluster centres where `average` and `complete` linkage
  disagree, the leaf order differs completely and merge heights differ by up to 0.52.
* **The force-directed layout treats the graph as undirected** and sorts the list of edges. So
  the layout depends only on the graph. If it kept only `column > row`, a graph stored as a lower
  triangle would have no attraction at all. The layout would then use only repulsion, with no
  warning. The masses taken from row counts would have the same error.
* **Scaling computes each gene's mean and variance in 64-bit numbers (f64).** In f32, the mean of
  a constant gene is off by one unit in the last place. The whole column then comes back as a
  constant `-sqrt((n-1)/n)` in place of 0. Without zero-centering this value is of order 1e7.
* **The neighbour graph and t-SNE centre the coordinates before expanding `|a-b|²`.** Without
  centring, the distances in an embedding far from the origin round to zero, and the graph
  breaks down.
* **The UMAP GPU kernel's learning-rate (alpha) schedule matches umap-learn's**
  `layouts.py:431` at every epoch.
* **The t-SNE perplexity check is the same as scikit-learn's.** It requires only
  `perplexity < n_samples`. scanpy has no stricter rule.

The audits also fix about a dozen intended differences from scanpy. We consider these correct
and keep them on purpose. Examples are `score_genes` ignoring `random_state` below ~1200 genes,
the median in `normalize_total` that depends on how the matrix is stored, Wilcoxon on a group of
one cell, and the `cell_ranger` HVG method at two genes per bin. The full list is in
[development/API_CONTRACT.md](development/API_CONTRACT.md). Each one has a test that fails if
the behaviour changes.

## What is not validated

**The GPU code is validated only on Apple hardware.** `tests/test_device_parity.py` skips all its
tests where `gpu_available()` is false. The `cargo test` unit tests in `crates/metalcyte-gpu`
also return early when `MetalContext::new()` fails. The audits run on the device set by
`METALCYTE_TEST_DEVICE`, which defaults to `cpu`. A passing run on a machine without Metal only
tells you about the CPU code. Most users run on the GPU (see "CPU or GPU" above), and that path
is checked only on a machine with a Metal device.

**Python can reach one Metal kernel, `knn`. It cannot reach the other three.** A kernel is a
small program that runs on the GPU. `crates/metalcyte-py` depends on `metalcyte-gpu`. It sends a
Metal user's k-nearest-neighbour search to `knn_metal` (`metalcyte-py/src/embedding.rs`). We
validate this kernel in two ways on Apple silicon:

- As a kernel, `cargo test -p metalcyte-gpu` compares it with a brute-force CPU search. 35 of 35
  tests pass.
- From end to end, `tests/test_device_parity.py` checks that its neighbour lists equal those of
  the candle CPU code (4 of 4, `METALCYTE_TEST_DEVICE=auto`). One test uses points packed closer
  than `f32` can tell apart. Both devices merge them the same way, because the kernel centres the
  data and sets tiny squared distances to zero, as the CPU code does.

`spmm`, `tsne_gradient` and `umap_sgd` cannot be reached from Python. `spmm` has no caller that
multiplies a plain sparse matrix by a dense one. `umap_sgd` uses the Hogwild method and is left
unconnected on purpose. These three are checked only against CPU code in the same module. On a
machine without Metal, that check tests nothing. All other GPU work goes through candle.

**Python cannot reach `de/glm` and `de/dispersion`.** No Python-facing function in
`crates/metalcyte-py/src/` mentions `fit_negative_binomial`, `size_factors_median_of_ratios`,
`dispersions_method_of_moments` or `shrink_towards_trend`. `metalcyte._metalcyte` exports no
entry point to them. (The `dispersions` that `_metalcyte.highly_variable_genes` reports come from
`preprocess.rs`, and not from `de/dispersion.rs`.) So `test_destats_audit.py` does not validate
that code. It validates the reference data that the unit tests of `glm.rs` use. It reads the
counts and design from the Rust source and recomputes the two fixed tables of coefficients with
statsmodels. `dispersion.rs` contains no copied reference numbers, so nothing about it can be
checked from Python. `de/hypothesis.rs` is partly reachable. Its `erfc` runs on every Wilcoxon
call. It agrees with `scipy.special` to 6.6e-15 relative. The smallest p-value returned is
2.5e-34, correct to 14 digits. Its `wald_test` is not called from outside the module and is not
tested from Python.

Rust tests cover the parts of `glm.rs` that the Python audit cannot reach: a gene with all zero
counts, a covariate that separates the samples perfectly, a single sample, an extremely large
count, reporting of convergence for each gene, and a design matrix with dependent columns.
`glm.rs` has four safeguard constants for these cases. We checked each one by switching it off
and running the tests again:

| constant | caught by |
| --- | --- |
| `RIDGE` | `a_rank_deficient_design_is_solved_rather_than_returning_nonsense` |
| `MAXIMUM_LINEAR_PREDICTOR` | `an_extreme_count_does_not_poison_the_first_iteration` |
| `MAXIMUM_WORKING_RESIDUAL` | **nothing** |
| `MINIMUM_MEAN` | **nothing** |

The last two can be set to absurd values and all tests still pass. So no test depends on them,
and no test would notice if they were wrong. Both have a comment in `glm.rs` that says so.

No function in `python/metalcyte/` raises `NotImplementedError`. See [API.md](API.md) for what
each function accepts and what it leaves out.

## DPT branch detection: agreement with scanpy

`tl.dpt(n_branchings > 0)` finds branches in a developmental trajectory. It is a direct port of
scanpy's method (Haghverdi 2016) and writes the branches to `obs["dpt_groups"]`. Branch labels
are arbitrary. So we measure agreement with the adjusted Rand index (ARI) of the two groupings,
following the clustering rule in [development/API_CONTRACT.md](development/API_CONTRACT.md).
ARI is 1 for identical groupings and about 0 for chance agreement. To test only the branching
step, `tests/test_dpt_branching_audit.py` gives Metalcyte and scanpy's `dpt` the **same**
diffusion map. So any difference must come from the branching code and not from the diffusion
map.

| dataset | `n_branchings` | ARI vs scanpy | group sizes |
| --- | --- | --- | --- |
| PBMC 3k | 1 | **1.0000** | identical (e.g. 153 / 2275 / 266 / 6) |
| PBMC 3k | 2 | **1.0000** | identical |

The groupings are identical, and the group sizes match. When Metalcyte also computes its own
diffusion map, the ARI is still 1.0000. Its diffusion map agrees with scanpy's closely enough
that the branches fall in the same place. The test passes on the CPU and on Metal.

## Harmony batch correction: checked by how well batches mix

`pp.harmony_integrate` removes batch effects. It works by repeated steps and starts from a
random k-means clustering. So it does not reproduce `harmonypy` (a compiled C++ program) exactly.
For batch correction, the right check is whether the batches mix and whether the method
converges. `tests/test_harmony_audit.py` measures this on PBMC 3k, after adding an artificial
batch shift to the PCA embedding:

* **iLISI** (integration Local Inverse Simpson's Index) is the effective number of batches among
  each cell's neighbours. For two batches it runs from 1 (fully separated) to 2 (fully mixed).
  Metalcyte raises it from **1.00 before correction to 1.90 after**. The test requires
  iLISI ≥ 1.85.
* **Convergence**: the harmony objective decreases, and its last relative step is within the
  harmony tolerance (~1%). The test checks this.
* **Reference**: we run `harmonypy 2.0.0` on the same data. Metalcyte must mix batches at least
  as well (iLISI). We record the cosine correlation with harmonypy (0.88) but do not test it,
  because the two programs start and converge differently.

The test passes on the CPU and on Metal. `tests/test_cell_cycle_audit.py` validates cell-cycle
scoring separately. It compares the S and G2M scores value by value with scanpy, and checks the
exact three-way phase rule (4/4 on the CPU and on Metal). `tests/test_backed_streaming.py`
checks that the out-of-core path gives exactly the same result as the in-memory path.
