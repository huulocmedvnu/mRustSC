# API contract

What this crate promises about its numbers: the dtypes and shapes that cross the
Python boundary, what "agrees with scanpy" means per algorithm and to what
tolerance, what the two devices guarantee about each other, and where scrust
deliberately does something scanpy does not.

The parallel-branch round this document was written to coordinate is over. Every
branch in it is merged: `grep -rn 'todo!' crates/` returns nothing, and the only
`NotImplementedError` left in `python/` is `tl.dpt(n_branchings=)` above 0
(`python/scrust/tl/_trajectory.py:46`) — branch detection, which is genuinely not
implemented, not a placeholder. The ownership table that assigned files to
branches has been deleted; `docs/ARCHITECTURE.md` describes the layout as it now
stands.

Everything below is binding on new work. Every tolerance is one an existing test
asserts, and the test is named. Do not state a tolerance here that nothing
measures.

## Conventions

- Matrices are **cells by genes**. `CsrMatrix` crosses the Python boundary as
  `(indptr, indices, values, n_cols)` with `uint32` indices and `float32` values
  (`crates/scrust-py/src/convert.rs`, `python/scrust/_shared.py`).
- `f32` throughout, except **p-values, which are `f64`** — a rank-sum p-value
  underflows `f32` to exactly zero. The wrapper writes `pvals` and `pvals_adj`
  as `float64` (`python/scrust/tl/_de.py:23-24`).
- Reductions whose result is a small number obtained from large ones accumulate
  in `f64` even when the inputs and outputs are `f32`. Per-gene moments in
  `preprocess/scale.rs` and the column means in `neighbors.rs` do this because
  an `f32` reduction left a constant gene one ulp from its own mean and the
  scaled column came back constant instead of zero.
- Every algorithm takes a `&candle_core::Device`. The CPU path is the same code
  as the GPU path and is the oracle the GPU path is tested against.
- Randomness takes an explicit seed. Same seed, same bytes.
- Errors are `scrust_core::Error`; nothing panics on user input.
- The Python layer is AnnData plumbing and defaults only. No arithmetic.
- Bindings convert and nothing else: no defaults, no AnnData knowledge, no
  algorithm. Release the GIL around the core call.

## Devices

**The device a caller gets is a property of their machine, not of their code.**
`settings.device` defaults to `"auto"`, and `DeviceKind::Auto` resolves to
`Device::new_metal(0).unwrap_or(Device::Cpu)`
(`crates/scrust-core/src/device.rs:33`). On any Mac with Metal, a caller who
names no device is on the GPU.

- **The two devices agree to `f32`, not exactly.** Same candle source means the
  same algorithm, not bit-identical results: `f32` addition is not associative
  and a GPU reduction lands a few ulps from a sequential one. Quantities that
  are *choices* rather than measurements must still match exactly.
  `tests/test_device_parity.py` holds k-NN to precisely this split — neighbour
  *lists* by `assert_array_equal`, distances at `rtol=1e-5, atol=1e-6`.
- **An expansion's noise floor is part of the algorithm, not an accident.**
  `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b` cancels to exactly zero for two identical
  cells on the CPU but leaves a sub-ulp positive on Metal (9.5e-7 at a norm
  scale of 12), and the square root amplifies that to 9.8e-4. Duplicate cells
  then had non-zero `rho`, which is subtracted when the UMAP fuzzy set is built,
  so their connectivities stopped being 1. `neighbors.rs` now snaps squared
  distances below `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` to zero. On
  PBMC 3k's 50 PCs the floor is 0.049 as a distance against a *smallest*
  nearest-neighbour distance of 6.40; it snaps 0 of 39570 neighbours. Any new
  code that expands a squared distance owes the same treatment, and owes
  centring first — un-centred coordinates make the resolution a function of the
  distance to the origin rather than the radius of the cloud.
- **CI does not cover the GPU.** GitHub's hosted macOS runners have no usable
  GPU, so `gpu_available()` is false and `tests/test_device_parity.py` skips in
  its entirety there. The audits run against `SCRUST_TEST_DEVICE`, which
  defaults to `"cpu"`; set it to `"auto"` on hardware with a GPU to run the same
  suite the other way. Both legs pass locally. Do not present a green CI as GPU
  coverage — this is stated in `.github/workflows/ci.yml` as well.
- **Not everything that takes a `device` uses one.** `umap.rs`, `cluster.rs`
  (leiden and louvain), `preprocess/normalize.rs`, `preprocess/hvg.rs`,
  `de/wilcoxon.rs` and `de/parametric.rs` bind it as `_device` and run on the
  CPU. `pca`, `neighbors`, `tsne`, `diffusion`, `layout`, `batch`, `scoring` and
  `autocorrelation` use it. Accepting `device` and ignoring it is allowed, but
  the parameter must be spelled `_device` so the fact is visible at the
  signature, and the reason belongs in the doc comment.
- **`scrust-gpu` is partly reachable: `knn` is wired, the rest is not.**
  `crates/scrust-py` depends on `scrust-gpu` and its `embedding` binding routes a
  Metal caller's k-NN to `kernels::knn::knn_metal`, falling back to the candle path
  on the CPU or where no Metal context builds (`scrust-py/src/embedding.rs`). A wired
  kernel is an **optimisation, not a separate algorithm**: it must return what its
  `scrust-core` counterpart returns, and its tests must assert that. `knn` meets this —
  it reproduces `neighbors::knn`'s f64 mean-centering and its
  `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` snapping inside the MSL, so
  `tests/test_device_parity.py` holds the two devices' neighbour lists equal (4 of 4).
  The other three kernels (`spmm`, `tsne_gradient`, `umap_sgd`) sit on no path a Python
  caller can take: `spmm` has no plain sparse×dense consumer, and `umap_sgd` is Hogwild
  and left unwired on purpose. All other GPU work still goes through candle.

## scanpy is the reference

Correctness is agreement with scanpy on the same input. Tests live in
`tests/test_<area>.py` and `tests/test_<area>_audit.py`; use the fixtures in
`tests/conftest.py` and the helpers in `tests/reference_metrics.py`.

**What agreement means differs per algorithm, and picking the wrong form is the
mistake that has cost this project the most time.**

- **Deterministic transforms** are compared element-wise. The working tolerance
  is `rtol=1e-3` relative, tightened wherever the two implementations should
  agree to machine precision: `tests/test_reference.py` uses
  `rtol=1e-5, atol=1e-6` for `scale`, `log1p`, `normalize_total` and the
  filters; `tests/reference_metrics.py:67` holds DE scores and log fold changes
  to `1e-3` and p-values, adjusted and raw, to `1e-6`; `tests/test_paga_audit.py`
  holds PAGA connectivities to `rtol=1e-6`; `tests/test_parametric_audit.py`
  holds t-test scores to `rtol=1e-4, atol=1e-5` and its p-values to
  `rtol=1e-4, atol=1e-9`; `tests/test_diffusion_audit.py` holds dpt pseudotime
  to one `f32` epsilon absolute and diffmap eigenvalues to `rtol=1e-4`.
- **Set selections** are compared as sets. Highly variable genes must overlap
  scanpy's selection by at least 0.95 (`tests/test_reference.py:128`); the mean
  per-cell neighbour overlap must be at least 0.90
  (`tests/test_reference.py:156`).
- **Stochastic embeddings** (`umap`, `draw_graph`) are compared against the band
  the reference reaches against *itself* reseeded, never an absolute number.
  The bar is `CEILING_FRACTION = 0.85` of that measured ceiling
  (`tests/conftest.py:46`), with the ceiling recorded whether the test passes or
  not. Measured: umap-learn agrees with itself only ~44% on PBMC 3k. The
  absolute `STRICT_PRESERVATION = 0.80` is kept only on the `blobs` fixture,
  where scanpy does reproduce itself.
- **An explicit objective beats a proxy.** t-SNE is judged on the KL divergence
  it minimises — no worse than `1.05 x` the reference's
  (`tests/test_reference.py:47`) — not on landing in scanpy's local optimum.
  If your algorithm minimises something, assert on that.
- **Degenerate directions.** Where the reference cannot reproduce itself — PCA's
  trailing components — assert the quantity that *is* determined (the spectrum,
  to `VARIANCE_RATIO_TOLERANCE = 1e-3` or the randomised SVD's own drift,
  whichever is looser), not the one that is not (the eigenvectors).
- **Clustering labels are arbitrary.** Compare with the adjusted Rand index or
  normalised mutual information against scanpy's labels, never label by label,
  and against the same 0.85-of-ceiling band (`tests/test_cluster.py:387-390`).
  Modularity is the objective both algorithms maximise and is asserted directly.

If the criterion in this document is itself unreachable, **measure the ceiling
and say so** rather than lowering a threshold until it passes. That has happened
three times and each time the measurement was the real result.

## Deliberate divergences from scanpy

Each of these is a decision, not a defect, and each is pinned by a test that
fails if parity is reached — at which point the divergence and its test should
both go. `docs/API.md` carries the caller-facing version.

- **`tl.dendrogram` uses `complete` linkage**, matching `sc.tl.dendrogram`'s
  default; an earlier `average` was the bug. The wrapper records
  `linkage_method="complete"` in `uns` so the choice is readable from the
  result. Where the two methods differ, leaf order differs outright.
- **`tl.diffmap` raises for `n_comps >= n_cells`** where scanpy clamps to
  `n_cells - 1` (`crates/scrust-core/src/diffusion.rs:85`): the last eigenvector
  of an `n_cells` operator is not determined by a subspace of the same size.
- **The connected-graph guard counts only edges carrying weight**, not stored
  entries. This is stricter than scipy and scanpy, deliberately: walking the
  sparsity pattern let stored zeros bridge separate components and returned a
  degenerate map instead of an error.
- **`tl.dpt` gives unreachable cells finite pseudotime** where scanpy writes
  `inf`, and returns 0 rather than NaN when every cell coincides with the root.
- **`tl.paga` does not write `uns["<groups>_sizes"]`**, which `sc.pl.paga` reads
  to size nodes (`tests/test_paga_audit.py:523`). Its spanning tree may differ
  from scanpy's on ties, because of the `min(..., 1)` cap; both are valid maximum
  spanning trees and total weight agrees to 1e-6.
- **`rank_genes_groups(method="logreg")` reports scores only** — no p-values, no
  fold changes — because scanpy reports none for it. The core returns NaN and
  the wrapper drops the fields.
- **Wilcoxon accepts a one-cell group**, which scanpy refuses. The rank test is
  well defined at `n_active = 1`; scanpy's guard protects its own per-group
  variance.
- **Fold changes always use the natural log base.** scanpy uses
  `expm1(x * log(base))` when `uns["log1p"]["base"]` is set; the binding takes a
  matrix, not an AnnData. `sc.pp.log1p` leaves the base unset by default.
- **`scores` is reported as `f32` while p-values come from the unrounded `f64`
  score**, so `2 * norm.sf(|reported score|)` is not the reported p-value.
  scanpy does the same.
- **`normalize_total(target_sum=None)` always takes the CSR median.** scanpy
  picks the median over *all* cells for CSR and over cells with non-zero counts
  otherwise (`_normalization.py:93-117`); the core always holds CSR.
- **`highly_variable_genes(flavor="cell_ranger")` at two genes per bin** is
  degenerate: every gene normalises to exactly `±0.6744897501960817` and the
  top-N selection is a tie. The core is arithmetically right; scanpy's values
  land a few ulps apart and its selection stops earlier. Only at 2 genes/bin —
  from 3 upward they agree exactly.
- **`score_genes` ignores `random_state` below ~1200 genes**, because bins hold
  about `n_genes / (n_bins - 1)` genes and `ctrl_size` are drawn only
  `if ctrl_size < len(bin)`. scanpy behaves identically; this is documented
  rather than fixed.
- **`filter_cells(min_genes=)` counts entries `> 0` while
  `calculate_qc_metrics` counts entries `!= 0`.** Identical on counts, different
  on centred data. Both follow scanpy.
- **`de/glm` and `de/dispersion` are not reachable from Python at all.** They
  exist in `crates/scrust-core/src/de/` and are tested there; no binding exposes
  them.

## Big data

- Never densify a whole matrix. Densify a row block, or use the sparse kernels.
  If an algorithm cannot avoid a dense `(n_cells, n_genes)` intermediate, say so
  with the size it implies.
- State the peak memory an implementation needs as a function of its input, in
  the doc comment, and return `Error::InvalidParameter` above a documented limit
  rather than exhausting memory. `diffusion.rs:66-68` is the pattern.
- `settings.max_memory_gb` (default 4.0) is the budget the chunked paths size
  blocks against.

## Coverage

Cross-checked against scanpy, scikit-learn, umap-learn, scipy or statsmodels,
each in its own `tests/test_*_audit.py`: pca, umap, tsne, neighbors, cluster
(leiden/louvain), preprocess (normalize/log1p/hvg), scale, multiple_testing,
wilcoxon, qc, filter, batch (regress_out/combat), autocorrelation, scoring,
sampling, parametric (t-test/logreg), diffusion, paga, layout, hypothesis
(erfc). `erfc` agrees with `scipy.special` to 6.6e-15 relative; the smallest
p-value returned is 2.5e-34 and correct to 14 digits.

Not cross-checked: `chunked` and `sparse`, which are internal and have no scanpy
equivalent.
