# How Metalcyte works

A user-facing tour: what happens between your `mc.pp.pca(adata)` and the array that
lands in `obsm["X_pca"]`, and which of the design choices you can feel from Python.
The internal contract is [ARCHITECTURE.md](ARCHITECTURE.md). This page is the version
you need in order to use the library well.

## The shape of a call

```
your script                 mc.pp.pca(adata, n_comps=50)
python/metalcyte/pp/_basics.py pulls X apart into (indptr, indices, values, n_cols)
crates/metalcyte-py            converts those numpy arrays into Rust types
crates/metalcyte-core          runs the algorithm against candle tensors
                            on Device::Metal or Device::Cpu
python/metalcyte/pp/_basics.py writes obsm["X_pca"], varm["PCs"], uns["pca"]
```

Three consequences you can observe:

- **No AnnData knowledge below Python.** Rust sees three flat arrays and a column
  count. Layers, `raw`, views and masks are handled, or not handled, in Python.
- **No defaults below Python.** Every value the core needs is passed explicitly.
  When a Metalcyte default differs from scanpy's, the difference is one line in
  `python/metalcyte/`, and [API.md](API.md) says where.
- **Matrices are cells by genes and `float32`**, with `uint32` indices. A
  `float64` matrix is converted on the way in, so results come back `float32` even
  if you handed in doubles. p-values are the exception: a rank-sum p-value underflows
  `float32` to exactly zero, so those are `float64`.

## Where the GPU is, and where it is not

Algorithms are written once against `candle_core::Tensor` and take a device. Asking
for `"auto"` resolves to Metal when a Metal device initialises, and to the CPU
otherwise (`crates/metalcyte-core/src/device.rs`). `settings.device` is `"auto"`, so
on a Mac with Metal a caller who names no device is on the GPU without having chosen
to be. The same source runs both ways, so the CPU path is the same implementation
and serves as the oracle the GPU path is tested against. `metalcyte.gpu_available()`
tells you which one you will get, and `METALCYTE_DEVICE=cpu` in the environment
keeps a whole session on the CPU.

Apple's unified memory is what makes this worth doing at single-cell sizes: a Metal
buffer over a Rust slice is a view, with no copy across a PCIe bus, so an operation
does not have to be large enough to amortise a transfer before it pays.

Three qualifications:

- **A GPU does not make everything faster**, which is why some paths do not use one.
  Elementwise work over a sparse matrix (log transform, total-count normalisation,
  scaling) is bandwidth-bound and small, and Metalcyte is slower than scanpy on
  several of these. The tensor-algebra wins that do reach the device are PCA and the
  neighbour graph. The largest speedups over scanpy, the marker test and PAGA, are
  plain Rust that never touches it. [PERFORMANCE.md](PERFORMANCE.md) has both columns.
- **One hand-written Metal kernel is on your call path: `knn`.** `crates/metalcyte-gpu`
  holds CSR SpMM, column moments, row scaling, k-NN, UMAP SGD and t-SNE gradient
  kernels, all tested against their core counterparts. The bindings route a Metal
  caller's k-NN, the search behind the neighbour graph, to the `knn` kernel, which is
  ~2-2.5x faster than the candle path and agrees with the CPU result bit-for-bit
  (`tests/test_device_parity.py`). The other kernels (SpMM, UMAP SGD, t-SNE gradient)
  are not reachable: SpMM has no plain sparse times dense caller, and UMAP SGD is left
  unwired on purpose because it is non-deterministic. For everything except k-NN,
  the GPU you get is candle's Metal backend.
- **Not every call that takes a `device` uses it.** PCA, scaling, the neighbour
  graph, t-SNE, the diffusion map, the graph layouts, batch correction, gene scoring
  and spatial autocorrelation build tensors on it. Clustering, UMAP, the Wilcoxon and
  t-test markers, total-count normalisation and highly variable gene selection bind
  it as `_device` and never touch it: they are CPU code whatever you ask for (the log
  transform takes no `device` at all). The binding still resolves the name, so
  `device="gpu"` on a machine without Metal raises. It never changes where that work
  runs.

One build switch touches the CPU side of all this. The `accelerate` cargo feature
links Apple's Accelerate (vecLib) BLAS behind the dense CPU paths, the ndarray
`.dot()` and candle-CPU work in PCA, Harmony, the neighbour graph and the diffusion
map. It never touches the sparse CSR paths, and it buys a modest CPU-only margin
(~7-8% on Harmony, ~4-9% on the full pipeline). On the default `"auto"`/Metal path it
changes nothing, because that work is already on the GPU. [INSTALL.md](INSTALL.md)
says how it is switched on and off.

## Why the two devices do not agree bit for bit

One source is not one answer. `f32` addition is not associative, and a GPU reduction
adds its terms in a different order from a sequential one, so the same tensor
expression lands a few ulps apart on Metal and on the CPU. Usually that is invisible.
One case where it is not is worth understanding, because it is the shape the problem
takes generally: a cancellation whose exact result carries meaning downstream.

The neighbour search gets the whole distance matrix out of one matmul by expanding
`|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`. For two identical cells the three terms cancel to
exactly zero on the CPU. On Metal they leave a sub-ulp positive, 9.5e-7 against a norm
scale of 12, and the square root amplifies it a thousandfold, to 9.8e-4. A duplicated
cell would then have a non-zero `rho`, `rho` is subtracted when the UMAP fuzzy
simplicial set is built, and its connectivities would stop being 1. A distance of
1e-3 would become a visible change in the graph.

The guard is the expansion's own resolution: the expansion accumulates one rounding
per term of the dot product and one for each addition, so anything below
`(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` is noise it cannot tell from zero, and
is snapped to zero before the square root. The bound scales with the norms, so it
cannot swallow a real neighbour: on PBMC 3k's first 50 principal components the floor
is 0.049 as a distance, against a smallest nearest-neighbour distance of 6.40 and a
first percentile of 7.21. 0 of 39 570 neighbours are snapped, a margin of 130x. What
it reaches is cells that are identical, or closer together than `f32` can say.

Two things follow for you as a caller:

- **Hold cross-device results to `f32`, not to equality.** The device parity test
  requires the neighbour lists to match exactly, because a different neighbour is a
  different graph, and the distances only to `rtol=1e-5, atol=1e-6`.
- **That test skips entirely where there is no Metal device.** A run on a machine
  without Metal says nothing about the GPU path. `METALCYTE_TEST_DEVICE` (default
  `"cpu"`) chooses the device the rest of the audits run against.

## A stored zero means different things in different modules

A CSR matrix distinguishes "no entry" from "an entry whose value is 0.0", and the
neighbour graph is full of the second kind: two identical cells are at distance zero,
and that zero is stored. On 120 cells of which 60 are exact duplicates,
`sc.pp.neighbors(n_neighbors=10)` stores 540 zeros out of 1080 entries, half the
graph. Whether those entries count is a question each consumer has to answer for
itself, and the two in this codebase answer it opposite ways.

- **PAGA counts every stored entry, whatever its value.** scanpy binarises the graph
  before building it (`ones.data = np.ones(len(ones.data))`, `_paga.py:182-183`), so
  the `nonzero()` inside `get_igraph_from_adjacency` sees nothing but ones and drops
  none of them. Skipping zero-valued entries would overstate a connectivity by 0.096
  on the duplicate-heavy graph above.
- **The diffusion map counts only entries that carry weight.** It propagates along
  the weights, so an edge of weight zero transports nothing and cannot join two
  components. Its connected-graph guard therefore ignores stored zeros, and a graph
  they alone hold together is reported as disconnected, with an error, in place of a
  degenerate map (spectrum `[1.0000001, 1.0, ...]`). This is deliberately stricter
  than `scipy.sparse.csgraph.connected_components`, which walks the pattern and
  calls such a graph connected. scanpy inherits that.

Both readings are argued in the source, at `paga.rs::count_edges` and
`diffusion.rs::component_count`. If you write a new consumer of `obsp`, decide which
of the two it is before you index.

## Memory

The count matrix stays sparse across the FFI boundary: the three CSR arrays are
passed as they are, so a 95%-zero matrix is never densified to be handed to Rust.

Two places where something does densify, both worth planning around:

- Scaling with `zero_center=True` returns a dense `(cells, genes)` array and assigns
  it to `adata.X`. That is 400 MB at 50 000 x 2 000 and 4 GB at 50 000 x 20 000.
  scanpy does the same thing, and both are why you subset to highly variable genes
  first.
- t-SNE materialises an `(n, n)` affinity matrix. That is what makes it exact, and
  it is why the call refuses more than 20 000 cells with a `ValueError` instead of
  exhausting memory. It is also why t-SNE is the one operation that gets slower
  relative to scanpy as your data grows: 17x slower at 10 000 cells. Use
  `sc.tl.tsne` above a couple of thousand cells.

For matrices that do not fit at all there is the out-of-core head, which iterates row
blocks straight out of an `.h5ad` and sizes them against `settings.max_memory_gb`.
Total-count normalisation and the log transform consume it automatically when
`adata.isbacked`: they stream `X` in row blocks and rewrite it on disk in place, so
peak memory is one block and not the whole matrix, and the output is bit-for-bit
the in-memory result (`benches/backed_transform.py`: 737 MB peak against 1141 MB in
memory, 0.65x). The lower-level `open_backed` iterator is there to drive yourself for
other reductions. Measured on a 50 000 x 20 000 file computing per-gene sums, it
peaks at 1.41 GB streaming against 4.34 GB for read-then-densify. [API.md](API.md)
documents both entry points.

## What "agrees with scanpy" means

scanpy is the reference, and the form of agreement is chosen per algorithm, because
using the wrong form is how a correctness claim becomes meaningless:

- **Deterministic transforms** (total-count normalisation, the log transform,
  scaling) are compared element by element.
- **Selections** (highly variable genes, nearest neighbours) are compared as sets.
- **Stochastic layouts** (UMAP) are compared against the band the reference
  reaches against itself reseeded. umap-learn agrees with itself on about half of
  each cell's neighbourhood on PBMC 3k, so an absolute threshold above that would
  be a claim about nothing.
- **Anything with an explicit objective** (t-SNE) is judged on the objective. The
  question is whether the KL divergence is as good, and whether it found the same
  local optimum is left aside.
- **PCA** is asserted on the components a randomised SVD can determine at all, and
  on the spectrum everywhere. On PBMC 3k that is the first 7 of 50 components. Past
  those, scanpy's own randomised solver does not reproduce scanpy's arpack either.

Every number behind those statements is in [VALIDATION.md](VALIDATION.md), measured
by the reference test suite.

## Reproducibility

Randomness takes an explicit seed and the same seed gives the same bytes, on one
device. Across devices the seed fixes the algorithm and not the last few bits of the
arithmetic, for the reason above. This holds within Metalcyte only: a Metalcyte UMAP
with `random_state=0` will not match a scanpy UMAP with `random_state=0`, because
they are different implementations of a stochastic method. That is what the
preservation band exists to measure.
