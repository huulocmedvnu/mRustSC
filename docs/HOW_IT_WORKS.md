# How scrust works

A user-facing tour: what happens between your `sr.pp.pca(adata)` and the array that
lands in `obsm["X_pca"]`, and which of the project's design choices you can actually
feel from Python. The internal contract that branches are written against is
[ARCHITECTURE.md](ARCHITECTURE.md); this page is the version you need in order to
use the library well.

## The shape of a call

```
your script                 sr.pp.pca(adata, n_comps=50)
python/scrust/pp/_basics.py pulls X apart into (indptr, indices, values, n_cols)
crates/scrust-py            converts those numpy arrays into Rust types
crates/scrust-core          runs the algorithm against candle tensors
                            on Device::Metal or Device::Cpu
python/scrust/pp/_basics.py writes obsm["X_pca"], varm["PCs"], uns["pca"]
```

Three consequences you can observe:

- **No AnnData knowledge below Python.** Rust sees three flat arrays and a column
  count. Layers, `raw`, views and masks are handled — or not handled — in Python.
- **No defaults below Python.** Every value the core needs is passed explicitly.
  When a scrust default differs from scanpy's, the difference is one line in
  `python/scrust/`, and [API.md](API.md) says where.
- **Matrices are cells by genes and `float32`**, with `uint32` indices. A
  `float64` matrix is converted on the way in, so results come back `float32` even
  if you handed in doubles. p-values are the exception: a rank-sum p-value underflows
  `float32` to exactly zero, so those are `float64`.

## Where the GPU is, and where it is not

Algorithms are written once against `candle_core::Tensor` and take a device. Asking
for `"auto"` resolves to Metal when a Metal device initialises, and to the CPU
otherwise. The same source runs both ways, so the CPU path is not a second
implementation that can drift — it is the oracle the GPU path is tested against.
`scrust.gpu_available()` tells you which one you will get.

Apple's unified memory is what makes this worth doing at single-cell sizes: a Metal
buffer over a Rust slice is a view, not a copy across a PCIe bus, so an operation
does not have to be large enough to amortise a transfer before it pays.

Two honest qualifications:

- **A GPU does not make everything faster.** Elementwise work over a sparse matrix
  — `log1p`, `normalize_total`, `scale` — is bandwidth-bound and small, and scrust
  is *slower* than scanpy on several of these. The wins are in the dense batched
  work: PCA, the layouts, and the rank-sum test. [BENCHMARKS.md](BENCHMARKS.md) has
  both columns.
- **The hand-written Metal kernels are not on your call path.** `crates/scrust-gpu`
  holds CSR SpMM, column moments, row scaling, k-NN, UMAP SGD and t-SNE gradient
  kernels, and they are tested against their core counterparts. But
  `crates/scrust-core` does not depend on `scrust-gpu`, and
  `crates/scrust-py/src/lib.rs` registers only `preprocess`, `embedding`, `de` and
  `paga` — so nothing you can call from Python reaches them. Today the GPU you get
  is candle's Metal backend. The kernels are the next step, not the current state.

## Memory

The count matrix stays sparse across the FFI boundary: the three CSR arrays are
passed as they are, so a 95%-zero matrix is never densified to be handed to Rust.

Two places where something does densify, both worth planning around:

- `pp.scale(zero_center=True)` returns a dense `(cells, genes)` array and assigns it
  to `adata.X`. That is 400 MB at 50 000 x 2 000 and 4 GB at 50 000 x 20 000. scanpy
  does the same thing, and both are why you subset to highly variable genes first.
- `tl.tsne` materialises an `(n, n)` affinity matrix — that is what makes it exact,
  and it is why the call refuses more than 20 000 cells with a `ValueError` rather
  than exhausting memory. It is also why t-SNE is the one operation that gets
  *slower* relative to scanpy as your data grows: 17x slower at 10 000 cells. Use
  `sc.tl.tsne` above a couple of thousand cells.

For matrices that do not fit at all there is `scrust._backed.open_backed`, which
iterates row blocks straight out of an `.h5ad` and sizes them against
`settings.max_memory_gb`. It is private and no `pp` function consumes it yet, so
today it is a tool you drive yourself. Measured on a 50 000 x 20 000 file: 1.41 GB
peak streaming against 4.34 GB for read-then-densify, computing the same per-gene
sums.

## What "agrees with scanpy" means

scanpy is the reference, and the form of agreement is chosen per algorithm, because
using the wrong form is how a correctness claim becomes meaningless:

- **Deterministic transforms** — `normalize_total`, `log1p`, `scale` — are compared
  element by element.
- **Selections** — highly variable genes, nearest neighbours — are compared as sets.
- **Stochastic layouts** — UMAP — are compared against the band the reference
  reaches against *itself* reseeded. umap-learn agrees with itself on about half of
  each cell's neighbourhood on PBMC 3k, so an absolute threshold above that would
  be a claim about nothing.
- **Anything with an explicit objective** — t-SNE — is judged on the objective. The
  question is whether the KL divergence is as good, not whether it found the same
  local optimum.
- **PCA** is asserted on the components a randomised SVD can determine at all, and
  on the spectrum everywhere. On PBMC 3k that is the first 7 of 50 components; past
  those, scanpy's own randomised solver does not reproduce scanpy's arpack either.

Every number behind those statements is in [VALIDATION.md](VALIDATION.md), measured
by `tests/test_reference.py` rather than asserted here.

## Reproducibility

Randomness takes an explicit seed and the same seed gives the same bytes. Note that
this holds *within* scrust: a scrust UMAP with `random_state=0` will not match a
scanpy UMAP with `random_state=0`, because they are different implementations of a
stochastic method. That is what the preservation band exists to measure.
