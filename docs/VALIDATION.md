# Validation

Three kinds of test run against Metalcyte, and they answer different questions.

* **Unit tests** (`cargo test --workspace`) hold each Rust function to its own
  contract on inputs the author chose.
* **Reference tests** (`tests/test_reference.py`, 26 collected) run the whole pipeline
  against scanpy twice, on a 240-cell synthetic matrix and on real PBMC 3k
  (2 638 cells) under the `reference` marker, and ask whether the results agree.
  The form of agreement is chosen per algorithm and fixed in
  [development/API_CONTRACT.md](development/API_CONTRACT.md#scanpy-is-the-reference).
* **Audits** (`tests/test_*_audit.py`, 21 files) take one module at a time and go
  after the places the reference tests cannot fail: term-by-term identities against
  the reference implementation's own code, boundaries, degenerate inputs, and
  deliberate divergences pinned with the size of the gap.

Reproduce the reference numbers with:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_reference.py \
    -o junit_family=xunit1 --junitxml=ref.xml
```

The figures here are from one such run on Apple silicon with the GPU path live
(`gpu_available() == True`), 26 of 26 reference tests passing. A stochastic step
moves in the last digit or two between runs. A deterministic one does not.

### The Accelerate build passes the same suite

The `accelerate` feature (Apple vecLib BLAS behind the dense CPU paths, see
[INSTALL.md](INSTALL.md)) is a build switch and does not change the numerics: it
routes ndarray's `.dot()` and candle's CPU backend through a different BLAS, and the
sparse CSR paths never touch it. It is held to the same bar as the pure-Rust build,
and passes it:

| build | command | result |
| --- | --- | --- |
| pure-Rust | `cargo test --workspace` | pass |
| Accelerate | `cargo test --workspace --features accelerate` | **300 / 300 pass** |

`cargo clippy --workspace` is clean under both feature sets. The one caveat is the
pure-Rust build's `umap_sgd::is_reproducible_only_in_structure...` unit test, a
Hogwild race check that accepts racing writes by design and can flake on its own.
It passes under `--features accelerate`. Because both builds compute the same
algorithm to within floating-point reassociation, the reference and audit numbers
below are not re-tabulated per feature. They hold for either build.

## Deterministic transforms: element-wise

Compared with `numpy.testing.assert_allclose` at `rtol=1e-5, atol=1e-6`. Measured
worst absolute deviation on PBMC 3k:

| step | max abs difference | note |
| --- | --- | --- |
| `pp.normalize_total` | 1.2e-4 (rel 7.0e-8) | f32 rounding of the scale factor |
| `pp.log1p` | 0.0 | bit-identical |
| `pp.scale` | 1.8e-6 (rel 1.8e-7) | f32 rounding of mean and variance |
| `pp.filter_cells` / `pp.filter_genes` | 0.0 | same cells, same genes kept |

## Selections: set overlap

| step | criterion | measured (PBMC 3k) |
| --- | --- | --- |
| `pp.highly_variable_genes` | ≥ 0.95 of scanpy's 2 000 genes | **1.00** (identical set) |
| `pp.neighbors` (exact) | mean per-cell neighbour overlap ≥ 0.90 | **1.00**, worst cell 1.00 |
| `pp.neighbors(method="approximate")` | mean recall against the exact lists ≥ 0.95 | **0.99** |

The exact search is why the overlap is total: the same k nearest points, in the same
graph. The approximate search (NN-descent seeded with a random-projection forest) is
held to the exact lists by recall at k = 15 (`tests/test_neighbors_approximate` in
`tests/test_reference.py`); on the embryo embedding `benches/knn_methods.py` measures
0.98 at 100 000 cells and 0.96 at 953 000.

## PCA: determined components and spectrum

scanpy's default solver is deterministic `arpack`. Metalcyte does a randomised SVD, the
same algorithm class as scanpy's randomised solver. A component is "determined"
when scanpy's own randomised solver reproduces its arpack result to correlation
≥ 0.99. Beyond those, the eigenvectors are free to rotate and per-component
correlation measures nothing, so only the spectrum is asserted there.

| dataset | components | determined | Metalcyte matches (corr ≥ 0.99) | worst variance-ratio gap |
| --- | --- | --- | --- | --- |
| synthetic | 50 | 31 | 48 | 0.003 |
| PBMC 3k | 50 | 7 | 8 | 0.042 |

The `7 of 50` on PBMC 3k is a property of the data's spectrum: past the 7th
component the reference implementation cannot reproduce itself either. Metalcyte
matches every determined component and holds the variance ratios within the
tolerance a randomised SVD drifts on its own.

## UMAP: preservation band

UMAP is stochastic and does not reproduce itself across seeds, so the bar is
relative: Metalcyte's neighbourhood preservation against scanpy must reach at least 85%
of the preservation scanpy reaches against itself reseeded (K_REF=15 in the
reference layout, K_CAND=30 in the candidate).

| dataset | Metalcyte vs scanpy | scanpy vs itself (ceiling) | floor (0.85 × ceiling) | pass |
| --- | --- | --- | --- | --- |
| synthetic | 0.564 | 0.623 | 0.530 | yes |
| PBMC 3k | 0.456 | 0.511 | 0.434 | yes |

The ceiling of ~0.51 on PBMC 3k is the headline: umap-learn agrees with itself, on
its own output, on only about half of each cell's neighbourhood across a change of
seed. Metalcyte sits just under that ceiling, which is as close as a different
implementation can come to a target that unstable.

On the `blobs` fixture (six clusters smaller than K_REF, so neighbour sets are
seed-independent) the absolute 0.80 threshold is reachable, and Metalcyte records
**1.00**.

This is the shape of test the audits exist to supplement: a bar that a
roughly-similar optimiser clears whether or not it is right. The UMAP audit
compares term by term against a transcription of umap-learn's `layouts.py` instead.

## t-SNE: the objective

t-SNE has an explicit objective, so the test asks the direct question: is the KL
divergence Metalcyte reaches no worse than scanpy's (within 5% for f32 and a different
random start)? Both libraries are given scikit-learn's `auto` learning rate, because
scanpy's legacy default of 1000 costs scanpy itself an order of magnitude in KL at
these sizes and would flatter Metalcyte.

| dataset | Metalcyte KL | scanpy KL | ratio | pass (≤ 1.05) |
| --- | --- | --- | --- | --- |
| synthetic | 0.985 | 0.971 | 1.014 | yes |
| PBMC 3k | 2.028 | 2.076 | 0.977 | yes |
| blobs | 0.170 | 0.180 | 0.943 | yes |

On PBMC 3k and blobs Metalcyte reaches a lower KL than scanpy, a better local
optimum of the same objective. On the synthetic set it is 1.4% higher, inside
tolerance.

## Differential expression: element-wise on the top genes

Per group, the top 100 genes are compared field by field against scanpy's Wilcoxon.
Worst relative deviation across all groups and both datasets:

| field | worst deviation |
| --- | --- |
| `scores` | 0.0 |
| `logfoldchanges` | 0.0 |
| `pvals` | ~2e-13 |
| `pvals_adj` | ~2e-13 |

Scores and fold changes are bit-identical. The p-values differ only in the last
digits of `float64`, from the order of a long sum. This holds for every cell type in
PBMC 3k, including the 8-cell Megakaryocytes.

## PAGA: element-wise on the connectivities

`tests/test_paga.py` compares the abstracted-graph connectivities against scanpy's
v1.2 model and requires the same spanning tree.

| dataset | max relative deviation | tree |
| --- | --- | --- |
| synthetic | 2.3e-8 | identical edges |
| PBMC 3k | 5.5e-8 | identical edges |

## The audits: what is cross-checked, and against what

One file per module, each naming the reference line it holds the Rust to. Test counts
are `pytest --collect-only -q` on that file, so parametrised cases are counted
individually. (`test_de_audit.py` loads the compiled cdylib at import time and skips at
module level when it is absent, so its 56 are counted from the parametrisations and
not from a collection.)

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

The suite also holds backed streaming to the in-memory path in
`test_backed_streaming.py`. Wherever the reference can be driven on the same input it
is driven. The exceptions (umap-learn's SGD inner loop, scikit-learn's t-SNE gradient,
scanpy's `transitions_sym` spectrum) transcribe the reference and then check the
transcription against the installed package in a test of its own.

Two divergences are pinned with `xfail(strict=True)`, so they will announce themselves
the day they are closed: the PCA Gram eigendecomposition losing roughly half the digits
of the trailing singular values (3.6e-4 against an exact f64 reference where
scikit-learn is 6.3e-7), and UMAP's random initialisation failing to recover the global
arrangement of clusters that a spectral initialisation would.

`chunked` and `sparse` are internal and have no scanpy or scipy equivalent to compare
against, so their audits (`test_chunked_audit.py`, `test_sparse_audit.py`) pin
invariants instead: block-size invariance for `chunked`, and round-trip and validation
behaviour for `sparse`, against `scipy.sparse.csr_matrix` where the two overlap.

## The device dimension

`settings.device` defaults to `"auto"`, which resolves to Metal when a Metal device
initialises and to the CPU otherwise (`crates/metalcyte-core/src/device.rs`). On any
Mac with Metal, **a caller who names no device is on the GPU.** Which device a result
came from is therefore a property of the machine, not of the code.

The audits pin behaviour against scanpy on one device at a time. `METALCYTE_TEST_DEVICE`
(`tests/metalcyte_call.py`, default `"cpu"`) selects which. Set it to `"auto"` to run the
same suite the other way. Both legs pass on Apple silicon.

Same candle source means the same algorithm, and does not mean bit-identical results:
f32 addition is not associative, and a GPU reduction lands a few ulps from a sequential
one. So `tests/test_device_parity.py` (4 tests) holds the two devices against each
other, which is a different question from either device against scanpy.

Not every module has a device to differ on. `pca`, `neighbors` and `tsne` use the
resolved `Device`. `umap` and `cluster` take it and ignore it (`_device` in
`crates/metalcyte-core/src/umap.rs` and `cluster.rs`) and always run on the CPU.

## What the audits found

Each behaviour below was found by an audit in code that already had passing tests,
and each is pinned by a test that fails if it drifts.

* **The neighbour graph snaps sub-resolution distances to zero.**
  `|a-b|² = |a|² + |b|² - 2a·b` cancels to exactly 0 for two identical cells on the CPU
  but leaves a sub-ulp positive on Metal (9.5e-7 at norm scale 12), which the square
  root amplifies to 9.8e-4. `rho` would then be non-zero, and it is subtracted when the
  UMAP fuzzy set is built, so duplicate cells' connectivities would stop being 1.
  Squared distances below `(n_dims + 2) * f32::EPSILON * (|a|² + |b|²)` snap to zero.
  On PBMC 3k's 50 PCs that floor is 0.049 against a smallest nearest-neighbour
  distance of 6.40: it snaps 0 of 39 570 neighbours. This shows only with
  `METALCYTE_TEST_DEVICE=auto`. Every test naming `"cpu"` passes either way.
* **PAGA counts stored zeros as edges.** scanpy binarises first
  (`ones.data = np.ones(len(ones.data))`, `_paga.py:182-183`), so every stored entry
  is an edge. On 120 cells of which 60 are duplicates, `sc.pp.neighbors(n_neighbors=10)`
  stores 540 zeros out of 1080 entries. Dropping them would overstate connectivities by
  up to 0.096. Metalcyte agrees with scanpy to ~2e-8.
* **The diffusion map counts only edges that carry weight** in its connected-graph
  guard. A stored-but-zero entry walked as part of the sparsity pattern would join
  separate components and return a degenerate map (spectrum `[1.0000001, 1.0, ...]`)
  in place of an error. This is deliberately stricter than scipy and scanpy, which
  read the pattern, and the reasoning is in the code.
* **The dendrogram uses `complete` linkage**, as `sc.tl.dendrogram` does, and records
  `linkage_method="complete"` in `uns`. On centroids where `average` and `complete`
  disagree, leaf order differs outright and merge heights by up to 0.52.
* **The force-directed layout reads the graph as undirected** and sorts the edge
  list, so the layout is a function of the graph alone. Keeping only `column > row`
  would give a lower-triangular graph no attraction at all and a silently
  pure-repulsive layout, and the masses taken from row counts would carry the same
  error.
* **Scaling reduces per-gene moments in f64.** Reduced in f32, a constant gene's mean
  lands one ulp out, and the whole column comes back as a constant `-sqrt((n-1)/n)`
  in place of 0, which is order 1e7 without zero-centering.
* **The neighbour graph and t-SNE centre coordinates before expanding `|a-b|²`.**
  Without centring, an embedding far from the origin cancels to zero and the graph
  degenerates.
* **The UMAP GPU kernel's alpha schedule matches umap-learn's** `layouts.py:431`
  epoch for epoch.
* **The t-SNE perplexity guard is scikit-learn's.** It requires only
  `perplexity < n_samples`. scanpy has no stricter rule.

The audits also pin a dozen divergences from scanpy that are correct and are kept
deliberately: `score_genes` ignoring `random_state` below ~1200 genes,
`normalize_total`'s storage-dependent median, Wilcoxon on a one-cell group, the
`cell_ranger` HVG flavour at two genes per bin, and others. Those live in
[development/API_CONTRACT.md](development/API_CONTRACT.md). Each is a test that fails
if the behaviour drifts.

## What is not validated

**The GPU path is validated only on Apple hardware.** `tests/test_device_parity.py`
skips in its entirety where `gpu_available()` is false, and the `cargo test` unit
tests inside `crates/metalcyte-gpu` likewise return early when `MetalContext::new()`
fails. The audits themselves run against `METALCYTE_TEST_DEVICE`, which defaults to
`cpu`. A passing run on a machine without Metal is evidence about the CPU path and
nothing else, and the device most callers get (see "The device dimension" above) is
checked only on a machine with a Metal device.

**One Metal kernel, `knn`, is reachable from Python. The other three are not.**
`crates/metalcyte-py` depends on `metalcyte-gpu` and routes a Metal caller's k-NN to
`knn_metal` (`metalcyte-py/src/embedding.rs`). It is validated two ways on Apple
silicon: as a kernel, `cargo test -p metalcyte-gpu` holds it against a brute-force CPU
reference (35 of 35 pass), and end to end, `tests/test_device_parity.py` holds its
neighbour lists equal to the candle CPU path (4 of 4, `METALCYTE_TEST_DEVICE=auto`),
including a knot tighter than `f32` can resolve, which both devices collapse the same
way because the kernel reproduces the CPU path's mean-centering and squared-distance
snapping. `spmm`, `tsne_gradient` and `umap_sgd` are unreachable (`spmm` has no plain
sparse times dense caller, and `umap_sgd` is Hogwild and left unwired on purpose) and
are checked only against their in-module CPU references, which is vacuous on a machine
without Metal. All other GPU work goes through candle.

**`de/glm` and `de/dispersion` are not reachable from Python.** No pyfunction in
`crates/metalcyte-py/src/` mentions `fit_negative_binomial`,
`size_factors_median_of_ratios`, `dispersions_method_of_moments` or
`shrink_towards_trend`, and `metalcyte._metalcyte` exports no entry point to them. (The
`dispersions` that `_metalcyte.highly_variable_genes` reports come from `preprocess.rs`,
not from `de/dispersion.rs`.) `test_destats_audit.py` therefore does not validate that
code. What it validates is the reference data `glm.rs`'s own unit tests are judged
against, re-deriving the two hard-coded coefficient tables with statsmodels straight
from the counts and design parsed out of the Rust source. `dispersion.rs` carries no
transcribed numeric reference, so nothing about it can be checked from Python at all.
`de/hypothesis.rs` is partly reachable: its `erfc` runs on every Wilcoxon call and
agrees with `scipy.special` to 6.6e-15 relative (the smallest p-value returned is
2.5e-34, correct to 14 digits). Its `wald_test` is called by nothing outside the module
and is untested from Python.

What the Python audit cannot reach in `glm.rs` is covered by Rust tests: an all-zero
gene, a perfectly separating covariate, a single sample, an out-of-scale count,
per-gene convergence reporting, and a rank-deficient design. `glm.rs` has four guard
constants for those cases, and each was checked by disabling it and re-running the
suite:

| constant | caught by |
| --- | --- |
| `RIDGE` | `a_rank_deficient_design_is_solved_rather_than_returning_nonsense` |
| `MAXIMUM_LINEAR_PREDICTOR` | `an_extreme_count_does_not_poison_the_first_iteration` |
| `MAXIMUM_WORKING_RESIDUAL` | **nothing** |
| `MINIMUM_MEAN` | **nothing** |

The last two can be set to absurd values with the whole suite still green, so nothing
depends on them and nothing would notice if they were wrong. Both carry a comment in
`glm.rs` saying so.

No entry point in `python/metalcyte/` raises `NotImplementedError`. See
[API.md](API.md) for what each function accepts and what it leaves out.

## DPT branch detection: adjusted Rand index against scanpy

`tl.dpt(n_branchings > 0)` runs a native port of scanpy's Haghverdi 2016 branch
detection and writes `obs["dpt_groups"]`. Branch labels are arbitrary, so parity is the
adjusted Rand index of the two partitions, per the clustering rule in
[development/API_CONTRACT.md](development/API_CONTRACT.md). To test the branching
alone, `tests/test_dpt_branching_audit.py` feeds Metalcyte's port and scanpy's `dpt`
the **same** diffusion map, so any difference is the branching logic and not the
diffmap:

| dataset | `n_branchings` | ARI vs scanpy | group sizes |
| --- | --- | --- | --- |
| PBMC 3k | 1 | **1.0000** | identical (e.g. 153 / 2275 / 266 / 6) |
| PBMC 3k | 2 | **1.0000** | identical |

The partition is identical: the group sizes match as multisets. End-to-end, with
Metalcyte computing its own diffmap, the ARI is also 1.0000, because Metalcyte's
diffmap agrees with scanpy's closely enough that the branch cut lands in the same
place. Passes on cpu and Metal.

## Harmony batch integration: integration metrics, not bit-for-bit

`pp.harmony_integrate` is iterative and k-means seeded, so it does not reproduce
`harmonypy` (a compiled C++ backend) to the bit. Correctness for a batch-integration
method is instead batch mixing and convergence, measured in `tests/test_harmony_audit.py`
on PBMC 3k with a batch shift injected into the PCA embedding:

* **iLISI** (integration Local Inverse Simpson's Index): the effective number of batches
  in each cell's neighbourhood, 1 (separated) to 2 (mixed for two batches). Metalcyte
  raises it from **1.00 before correction to 1.90 after**. The test asserts iLISI ≥ 1.85.
* **Objective convergence**: the harmony objective decreases and its final relative step
  is within the harmony tolerance (~1%). The test asserts this.
* **Reference**: `harmonypy 2.0.0` is run on the same data as a black box. Metalcyte is
  asserted to mix batches at least as well (iLISI), and the cosine correlation with
  harmonypy is recorded (0.88) and not asserted, since the two seed and converge
  differently.

Passes on cpu and Metal. Cell-cycle scoring is validated separately by
`tests/test_cell_cycle_audit.py` (S/G2M scores element-wise against scanpy, its exact
three-way phase rule, 4/4 on cpu and Metal), and backed streaming by
`tests/test_backed_streaming.py` (bit-for-bit against the in-memory path).
