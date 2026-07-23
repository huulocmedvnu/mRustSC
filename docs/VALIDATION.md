# Validation

Three kinds of test run against this crate, and they answer different questions.

* **Unit tests** (`cargo test --workspace`) hold each Rust function to its own
  contract on inputs the author chose.
* **Reference tests** (`tests/test_reference.py`, 26 collected) run the whole pipeline
  against scanpy twice — on a 240-cell synthetic matrix, and on real PBMC 3k
  (2 638 cells) under the `reference` marker — and ask whether the *results* agree.
  The form of agreement is chosen per algorithm and fixed in
  [API_CONTRACT.md](API_CONTRACT.md#scanpy-is-the-reference).
* **Audits** (`tests/test_*_audit.py`, 16 files, 314 collected tests) take one module
  at a time and go after the places the reference tests cannot fail: term-by-term
  identities against the reference implementation's own code, boundaries, degenerate
  inputs, and deliberate divergences pinned with the size of the gap. These are new,
  and they are where the defects in §"What the audits found" came from.

Reproduce the reference numbers with:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_reference.py \
    -o junit_family=xunit1 --junitxml=ref.xml
```

The figures here are from one such run on Apple silicon with the GPU path live
(`gpu_available() == True`), 26 of 26 reference tests passing. A stochastic step
moves in the last digit or two between runs; a deterministic one does not.

## Deterministic transforms — element-wise

Compared with `numpy.testing.assert_allclose` at `rtol=1e-5, atol=1e-6`. Measured
worst absolute deviation on PBMC 3k:

| step | max abs difference | note |
| --- | --- | --- |
| `pp.normalize_total` | 1.2e-4 (rel 7.0e-8) | f32 rounding of the scale factor |
| `pp.log1p` | 0.0 | bit-identical |
| `pp.scale` | 1.8e-6 (rel 1.8e-7) | f32 rounding of mean and variance |
| `pp.filter_cells` / `pp.filter_genes` | 0.0 | same cells, same genes kept |

## Selections — set overlap

| step | criterion | measured (PBMC 3k) |
| --- | --- | --- |
| `pp.highly_variable_genes` | ≥ 0.95 of scanpy's 2 000 genes | **1.00** (identical set) |
| `pp.neighbors` | mean per-cell neighbour overlap ≥ 0.90 | **1.00**, worst cell 1.00 |

The neighbour search is exact, which is why the overlap is not merely high but
total: the same k nearest points, in the same graph.

## PCA — determined components and spectrum

scanpy's default solver is deterministic `arpack`; scrust does a randomised SVD, the
same algorithm class as scanpy's *randomised* solver. A component is "determined"
when scanpy's own randomised solver reproduces its arpack result to correlation
≥ 0.99 — beyond those, the eigenvectors are free to rotate and per-component
correlation measures nothing, so only the spectrum is asserted there.

| dataset | components | determined | scrust matches (corr ≥ 0.99) | worst variance-ratio gap |
| --- | --- | --- | --- | --- |
| synthetic | 50 | 31 | 48 | 0.003 |
| PBMC 3k | 50 | 7 | 8 | 0.042 |

The `7 of 50` on PBMC 3k is a property of the data's spectrum, not a scrust
weakness: past the 7th component the reference implementation cannot reproduce
itself either. scrust matches every determined component and holds the variance
ratios within the tolerance a randomised SVD drifts on its own.

## UMAP — preservation band

UMAP is stochastic and does not reproduce itself across seeds, so the bar is
relative: scrust's neighbourhood preservation against scanpy must reach at least 85%
of the preservation scanpy reaches against itself reseeded (K_REF=15 in the
reference layout, K_CAND=30 in the candidate).

| dataset | scrust vs scanpy | scanpy vs itself (ceiling) | floor (0.85 × ceiling) | pass |
| --- | --- | --- | --- | --- |
| synthetic | 0.564 | 0.623 | 0.530 | yes |
| PBMC 3k | 0.456 | 0.511 | 0.434 | yes |

The ceiling of ~0.51 on PBMC 3k is the headline: umap-learn agrees with *itself*, on
its own output, on only about half of each cell's neighbourhood across a change of
seed. scrust sits just under that ceiling, which is as close as a different
implementation can come to a target that unstable.

On the `blobs` fixture — six clusters smaller than K_REF, so neighbour sets are
seed-independent — the absolute 0.80 threshold *is* reachable, and scrust records
**1.00**.

This is exactly the shape of test the audits exist to supplement: a bar that a
roughly-similar optimiser clears whether or not it is right. `test_umap_audit.py`
compares term by term against a transcription of umap-learn's `layouts.py` instead.

## t-SNE — the objective

t-SNE has an explicit objective, so the test asks the direct question: is the KL
divergence scrust reaches no worse than scanpy's (within 5% for f32 and a different
random start)? Both libraries are given scikit-learn's `auto` learning rate, because
scanpy's legacy default of 1000 costs scanpy itself an order of magnitude in KL at
these sizes and would flatter scrust.

| dataset | scrust KL | scanpy KL | ratio | pass (≤ 1.05) |
| --- | --- | --- | --- | --- |
| synthetic | 0.985 | 0.971 | 1.014 | yes |
| PBMC 3k | 2.028 | 2.076 | 0.977 | yes |
| blobs | 0.170 | 0.180 | 0.943 | yes |

On PBMC 3k and blobs scrust reaches a *lower* KL than scanpy — a better local
optimum of the same objective — and on the synthetic set it is 1.4% higher, inside
tolerance.

## Differential expression — element-wise on the top genes

Per group, the top 100 genes are compared field by field against scanpy's Wilcoxon.
Worst relative deviation across all groups and both datasets:

| field | worst deviation |
| --- | --- |
| `scores` | 0.0 |
| `logfoldchanges` | 0.0 |
| `pvals` | ~2e-13 |
| `pvals_adj` | ~2e-13 |

Scores and fold changes are bit-identical; the p-values differ only in the last
digits of `float64`, from the order of a long sum. This holds for every cell type in
PBMC 3k, including the 8-cell Megakaryocytes.

## PAGA — element-wise on the connectivities

`tests/test_paga.py` compares the abstracted-graph connectivities against scanpy's
v1.2 model and requires the same spanning tree.

| dataset | max relative deviation | tree |
| --- | --- | --- |
| synthetic | 2.3e-8 | identical edges |
| PBMC 3k | 5.5e-8 | identical edges |

## The audits — what is cross-checked, and against what

One file per module, each naming the reference line it holds the Rust to. Test counts
are `pytest --collect-only -q` on that file, so parametrised cases are counted
individually. (`test_de_audit.py` loads the compiled cdylib at import time and skips at
module level when it is absent, so its 56 are counted from the parametrisations rather
than collected.)

| module | file | reference | tests |
| --- | --- | --- | --- |
| `pca` | `test_pca_audit.py` | scikit-learn `randomized_svd`, Halko et al., scanpy `_pca` | 11 |
| `neighbors` | `test_neighbors_audit.py` | umap-learn `umap.umap_.fuzzy_simplicial_set`, called directly | 17 |
| `umap` | `test_umap_audit.py` | umap-learn `layouts.py`, transcribed and re-checked against the package | 27 |
| `tsne` | `test_tsne_audit.py` | scikit-learn `manifold._t_sne` and `_utils._binary_search_perplexity` | 17 |
| `cluster` | `test_cluster_audit.py` | leidenalg 0.12.0 / libleidenalg, Traag et al. 2019, scanpy `_leiden.py` | 11 |
| `preprocess` | `test_preprocess_audit.py` | scanpy `normalize_total`, `log1p`, `highly_variable_genes` | 20 |
| `de` + `scale` + `hvg` boundaries | `test_de_audit.py` | scanpy, `scipy.stats` | 56 |
| `de/wilcoxon` | `test_wilcoxon_audit.py` | scanpy `rank_genes_groups(method="wilcoxon")`; scipy in the far tail | 21 |
| `de/parametric` | `test_parametric_audit.py` | scanpy t-test, t-test_overestim_var, logreg | 12 |
| `de/hypothesis`, `de/glm`, `de/dispersion` | `test_destats_audit.py` | `scipy.special`, statsmodels 0.14.6 | 12 |
| `qc` + `filter` | `test_qc_audit.py` | scanpy `calculate_qc_metrics`, `filter_cells`, `filter_genes` | 24 |
| `batch` + autocorrelation | `test_batch_audit.py` | scanpy `regress_out`, `combat`, `sc.metrics.morans_i` / `gearys_c` | 14 |
| `scoring` + sampling | `test_scoring_audit.py` | scanpy `score_genes` and its pandas bin/sample chain | 25 |
| `diffusion` | `test_diffusion_audit.py` | scanpy `tl.diffmap`, `tl.dpt` | 12 |
| `paga` | `test_paga_audit.py` | scanpy `tl.paga(model="v1.2")` | 17 |
| `layout` | `test_layout_audit.py` | scanpy `tl.dendrogram`, `scipy.cluster.hierarchy`, `scipy.stats.gaussian_kde` | 18 |

314 tests over 16 files. Wherever the reference can be *driven* on the same input it
is driven rather than transcribed; the exceptions (umap-learn's SGD inner loop,
scikit-learn's t-SNE gradient, scanpy's `transitions_sym` spectrum) transcribe the
reference and then check the transcription against the installed package in a test of
its own.

Two divergences are pinned with `xfail(strict=True)`, so they will announce themselves
the day they are closed: the PCA Gram eigendecomposition losing roughly half the digits
of the trailing singular values (3.6e-4 against an exact f64 reference where
scikit-learn is 6.3e-7), and UMAP's random initialisation failing to recover the global
arrangement of clusters that a spectral initialisation would.

`chunked` and `sparse` have no audit: they are internal and have no scanpy equivalent
to check against.

## The device dimension

`settings.device` defaults to `"auto"`, and `DeviceKind::Auto` resolves to
`Device::new_metal(0).unwrap_or(Device::Cpu)` (`crates/scrust-core/src/device.rs`). On
any Mac with Metal, **a caller who names no device is on the GPU.** Which device a
result came from is therefore a property of the machine, not of the code.

The audits pin behaviour against scanpy on one device at a time. `SCRUST_TEST_DEVICE`
(`tests/scrust_call.py`, default `"cpu"`) selects which; set it to `"auto"` to run the
same suite the other way. Both legs pass on Apple silicon.

Same candle source means the same algorithm, not bit-identical results: f32 addition is
not associative, and a GPU reduction lands a few ulps from a sequential one. So
`tests/test_device_parity.py` (4 tests) holds the two devices against *each other*,
which is a different question from either device against scanpy.

Not every module has a device to differ on. `pca`, `neighbors` and `tsne` use the
resolved `Device`; `umap` and `cluster` take it and ignore it (`_device` in
`crates/scrust-core/src/umap.rs` and `cluster.rs`) and always run on the CPU.

## What the audits found

Every defect below was in code that already had tests and was already green. That is
the argument for this kind of testing, and it is worth being specific about.

* **`neighbors` — duplicate cells stopped being duplicates on the GPU.**
  `|a-b|² = |a|² + |b|² - 2a·b` cancels to exactly 0 for two identical cells on the CPU
  but left a sub-ulp positive on Metal (9.5e-7 at norm scale 12), which the square root
  amplified to 9.8e-4. `rho` was then non-zero and is subtracted when the UMAP fuzzy set
  is built, so duplicate cells' connectivities stopped being 1. Squared distances below
  `(n_dims + 2) * f32::EPSILON * (|a|² + |b|²)` now snap to zero. On PBMC 3k's 50 PCs
  that floor is 0.049 against a *smallest* nearest-neighbour distance of 6.40: it snaps
  0 of 39 570 neighbours. Found only because the audit was re-run with
  `SCRUST_TEST_DEVICE=auto`; every test naming `"cpu"` had passed.
* **`paga` — stored zeros were dropped from the edge count.** `count_edges` skipped
  entries whose value was 0.0, citing a `nonzero()` in scanpy; scanpy binarises *first*
  (`ones.data = np.ones(len(ones.data))`, `_paga.py:182-183`), so every stored entry is
  an edge. On 120 cells of which 60 are duplicates, `sc.pp.neighbors(n_neighbors=10)`
  stores 540 zeros out of 1080 entries and connectivities were overstated by up to
  0.096. Now agrees with scanpy to ~2e-8.
* **`diffusion` — a stored zero bridged two components.** The connected-graph guard
  walked the sparsity pattern, so a stored-but-zero entry joined separate components and
  a degenerate map (spectrum `[1.0000001, 1.0, ...]`) came back instead of an error. It
  now counts only edges carrying weight — deliberately stricter than scipy and scanpy,
  which read the pattern too, and argued in the code.
* **`layout` — three defects, one of them user-visible.** `dendrogram` computed
  `average` linkage where `sc.tl.dendrogram` defaults to `complete`; on centroids where
  the two disagree, leaf order differs outright and merge heights by up to 0.52. The
  wrapper now records `linkage_method="complete"` in `uns`. Separately,
  `undirected_edges` kept only `column > row`, so a lower-triangular graph produced no
  attraction at all and a silently pure-repulsive layout, and masses were taken from row
  counts with the same bug one level up. The edge list is now sorted, so the layout is a
  function of the graph alone.
* **`preprocess/scale` — a constant gene came back at order 1e7.** Per-gene moments
  reduced in f32 left a constant gene's mean one ulp out, so the whole column returned a
  constant `-sqrt((n-1)/n)` instead of 0 — without zero-centering, order 1e7. Reduced in
  f64 now.
* **`neighbors` and `tsne` — expansion on uncentred coordinates.** Both expanded
  `|a-b|²` without centring first, so an embedding far from the origin cancelled to zero
  and the graph degenerated. Both centre now.
* **`umap` — the GPU kernel's alpha schedule ran one epoch ahead** of umap-learn's
  `layouts.py:431`.
* **`tsne` — a guard invented from nothing.** The perplexity check refused
  `n_cells < 3 * perplexity` and credited the rule to scanpy. scanpy has no such rule;
  scikit-learn requires only `perplexity < n_samples`, which is the guard now.

The audits also *pinned* a dozen divergences from scanpy that turned out to be correct
and are kept deliberately — `score_genes` ignoring `random_state` below ~1200 genes,
`normalize_total`'s storage-dependent median, Wilcoxon on a one-cell group, the
`cell_ranger` HVG flavour at two genes per bin, and others. Those live in
[API_CONTRACT.md](API_CONTRACT.md); the point here is that each is now a test that fails
if the behaviour drifts, rather than an undocumented difference.

## What is not validated

**CI does not test the GPU path.** `tests/test_device_parity.py` skips in its entirety
where `gpu_available()` is false, which includes GitHub's hosted macOS runners; the
`cargo test` unit tests inside `crates/scrust-gpu` likewise return early when
`MetalContext::new()` fails. The audits themselves run against `SCRUST_TEST_DEVICE`,
which defaults to `cpu`. So the device most callers get — see "The device dimension"
above — is the device CI never exercises. A green tick on `ci.yml` is evidence about the
CPU path and nothing else; the GPU legs run on developer hardware and on a self-hosted
Apple-silicon runner.

**The Metal kernels are not reachable from Python.** `crates/scrust-gpu` holds four
hand-written kernels — `knn`, `spmm`, `tsne_gradient`, `umap_sgd` — and the dependency
was removed from `crates/scrust-py/Cargo.toml` because nothing there called it. No path
a Python caller can take reaches them; GPU work that does happen goes through candle.
They are checked in Rust against brute-force CPU references written in the same files,
which as above is vacuous on a machine without Metal.

**`de/glm` and `de/dispersion` are not reachable from Python.** No pyfunction in
`crates/scrust-py/src/` mentions `fit_negative_binomial`,
`size_factors_median_of_ratios`, `dispersions_method_of_moments` or
`shrink_towards_trend`, and `scrust._scrust` exports no entry point to them. (The
`dispersions` that `_scrust.highly_variable_genes` reports come from `preprocess.rs`,
not from `de/dispersion.rs`.) `test_destats_audit.py` therefore does not validate that
code: what it validates is the *reference data* `glm.rs`'s own unit tests are judged
against, re-deriving the two hard-coded coefficient tables with statsmodels straight
from the counts and design parsed out of the Rust source. `dispersion.rs` carries no
transcribed numeric reference, so nothing about it can be checked from Python at all.
`de/hypothesis.rs` is partly reachable: its `erfc` runs on every Wilcoxon call and
agrees with `scipy.special` to 6.6e-15 relative (the smallest p-value returned is
2.5e-34, correct to 14 digits); its `wald_test` is called by nothing outside the module
and is untested from Python.

**Not audited at all:** `chunked` and `sparse`, both internal with no external reference
to check against.

Of the Python API itself, the only entry point that still raises `NotImplementedError`
is `tl.dpt(n_branchings > 0)` — branch detection. See [API.md](API.md) for the current
implemented/not-implemented split.
