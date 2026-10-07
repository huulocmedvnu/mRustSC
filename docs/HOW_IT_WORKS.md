# How Metalcyte works

This page follows one call, `mc.pp.pca(adata)`, from your script to the array that appears in
`obsm["X_pca"]`. It also explains the design choices that you can notice from Python. The internal
design is described in [ARCHITECTURE.md](ARCHITECTURE.md). This page covers what you need to use
the library well.

## The path of a call

```
your script                 mc.pp.pca(adata, n_comps=50)
python/metalcyte/pp/_basics.py pulls X apart into (indptr, indices, values, n_cols)
crates/metalcyte-py            converts those numpy arrays into Rust types
crates/metalcyte-core          runs the algorithm against candle tensors
                            on Device::Metal or Device::Cpu
python/metalcyte/pp/_basics.py writes obsm["X_pca"], varm["PCs"], uns["pca"]
```

The Python layer takes the matrix out of the AnnData object and passes it to Rust as three plain
arrays. Rust runs the algorithm with candle, a Rust library for array calculations that can run on
the CPU or the GPU. Python then writes the results back into the AnnData object.

You can notice three effects of this design:

- **Only Python knows about AnnData.** Rust receives three flat arrays and a column count. Python
  handles layers, `raw`, views and masks, or declines them.
- **Only Python sets default values.** Python passes every value that the Rust code needs. When a
  Metalcyte default differs from scanpy's, the difference is one line in `python/metalcyte/`.
  [API.md](API.md) says where.
- **Matrices are cells by genes, stored as `float32` with `uint32` indices.** Metalcyte converts a
  `float64` matrix on the way in, so results come back as `float32` even if you passed 64-bit
  numbers. p-values are the exception and stay `float64`. A small rank-sum p-value would become
  exactly zero in `float32`.

## What runs on the GPU

Each algorithm is written once with `candle_core::Tensor` and receives a device. The device
`"auto"` means Metal, Apple's interface to the GPU, when a Metal device starts. Otherwise it means
the CPU (`crates/metalcyte-core/src/device.rs`). `settings.device` is `"auto"`. So on a Mac with
Metal, a call that names no device runs on the GPU. The same source code runs on both devices, and
the tests use the CPU version as the reference for the GPU version. `metalcyte.gpu_available()`
tells you which device you will get. `METALCYTE_DEVICE=cpu` in the environment keeps a whole session
on the CPU.

On Apple silicon the CPU and the GPU share the same memory (unified memory). The GPU reads data
where the CPU left it, with no copy. On a computer with a separate graphics card, data must first
travel over the PCIe connection to the card. That copy only pays off for large operations. Without
it, the GPU helps even at single-cell sizes.

Three points limit this:

- **The GPU does not speed up every step, so some steps do not use it.** Element-by-element work on
  a sparse matrix (log transform, total-count normalisation, scaling) is small and limited by memory
  speed. Metalcyte is slower than scanpy on several of these steps. The GPU helps most with PCA and
  the neighbour graph. The largest speedups over scanpy, in the marker test and PAGA, come from
  plain Rust code on the CPU. [PERFORMANCE.md](PERFORMANCE.md) shows both.
- **Only one hand-written GPU program (kernel) runs in a normal call: `knn`.**
  `crates/metalcyte-gpu` holds kernels for sparse matrix products (CSR SpMM), column moments, row
  scaling, k-NN, the UMAP optimisation (SGD) and the t-SNE gradient. Each is tested against the
  matching Rust code. For a call on Metal, the k-NN search behind the neighbour graph goes to the
  `knn` kernel. It is ~2-2.5x faster than the candle version and gives exactly the same result as
  the CPU (`tests/test_device_parity.py`). The other kernels (SpMM, UMAP SGD, t-SNE gradient) are
  not connected to any function. No function needs the plain sparse times dense product that SpMM
  computes. The UMAP SGD kernel stays disconnected on purpose, because its results change from run
  to run. For everything except k-NN, GPU work goes through candle's own Metal code.
- **Some functions accept a `device` argument and ignore it.** PCA, scaling, the neighbour graph,
  t-SNE, the diffusion map, the graph layouts, batch correction, gene scoring and spatial
  autocorrelation run on the chosen device. Clustering, UMAP, the Wilcoxon and t-test markers,
  total-count normalisation and highly variable gene selection always run on the CPU. Their code
  receives the device as `_device` and does not use it. The log transform takes no `device` at all.
  Metalcyte still checks the device name, so `device="gpu"` on a machine without Metal raises an
  error. The name never changes where that work runs.

One build option affects the CPU side. The `accelerate` cargo feature links Apple's Accelerate maths
library (vecLib BLAS), which uses the chip's matrix unit. Metalcyte then uses it for dense matrix
work on the CPU: the ndarray `.dot()` and candle-CPU calculations in PCA, Harmony, the neighbour
graph and the diffusion map. It does not affect the sparse matrix code. It makes CPU runs a little
faster (~7-8% on Harmony, ~4-9% on the full pipeline). With the default `"auto"` device it changes
nothing, because that work already runs on the GPU. [INSTALL.md](INSTALL.md) explains how to turn it
on and off.

## Why the CPU and GPU results differ in the last digits

The same code can give slightly different numbers on the two devices. With 32-bit floating point
numbers (`f32`), the order of additions changes the last bits of a sum. The GPU adds terms in a
different order than the CPU, so the same formula ends a few units in the last place (ulps) apart.
Usually nobody notices. One case matters, and it shows the general risk. It happens when a
subtraction should give exactly zero and later steps depend on that zero.

The neighbour search computes all distances with one matrix product, using
`|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`. For two identical cells the three terms cancel to exactly
zero on the CPU. On Metal they leave a tiny positive value, 9.5e-7 against a norm scale of 12. The
square root makes it a thousand times larger, 9.8e-4. A duplicated cell would then get a non-zero
`rho`, the distance to its nearest neighbour. UMAP subtracts `rho` when it builds its graph of
neighbour weights (the fuzzy simplicial set), so the cell's connectivities would no longer be 1. A
distance of 1e-3 would visibly change the graph.

Metalcyte fixes this with the precision limit of the formula. Each multiplication and addition adds
one rounding error. Any value below `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` is rounding
noise that cannot be told apart from zero. Metalcyte sets such values to zero before the square root.
The limit grows with the size of the vectors, so it cannot remove a real neighbour. On the first 50
principal components of PBMC 3k, the limit is 0.049 as a distance. The smallest nearest-neighbour
distance is 6.40 and the first percentile is 7.21. 0 of 39 570 neighbours are set to zero, a margin
of 130x. The rule only affects cells that are identical, or closer than `f32` can represent.

Two practical points follow:

- **Compare results across devices within `f32` precision, and do not expect exact equality.** The
  device comparison test requires the neighbour lists to match exactly, because a different
  neighbour gives a different graph. It requires the distances to match only within
  `rtol=1e-5, atol=1e-6`.
- **That test skips itself on a machine without Metal.** A test run on such a machine does not
  check the GPU code. `METALCYTE_TEST_DEVICE` (default `"cpu"`) chooses the device for the other
  tests.

## A stored zero means different things in different modules

A sparse matrix in CSR format can hold "no entry" or "an entry with value 0.0". The neighbour graph
contains many entries of the second kind. Two identical cells are at distance zero, and the graph
stores that zero. On 120 cells, of which 60 are exact duplicates, `sc.pp.neighbors(n_neighbors=10)`
stores 540 zeros out of 1080 entries, half the graph. Each function that reads the graph must decide
whether these entries count. The two such functions in Metalcyte decide in opposite ways.

- **PAGA counts every stored entry, whatever its value.** scanpy sets all graph values to one before
  building PAGA (`ones.data = np.ones(len(ones.data))`, `_paga.py:182-183`). So the `nonzero()` call
  inside `get_igraph_from_adjacency` sees only ones and drops none of them. If Metalcyte skipped
  zero-valued entries, one connectivity in the duplicate-heavy graph above would be 0.096 too high.
- **The diffusion map counts only entries with a non-zero weight.** It spreads values along the
  edge weights, so an edge of weight zero carries nothing and cannot join two parts of the graph.
  So its check for a connected graph ignores stored zeros. If only stored zeros hold a graph
  together, Metalcyte reports it as disconnected and raises an error. Otherwise it would return a
  useless map (spectrum `[1.0000001, 1.0, ...]`). This check is stricter on purpose than
  `scipy.sparse.csgraph.connected_components`, which counts every stored entry and calls such a
  graph connected. scanpy uses the scipy function.

The source code explains both choices, at `paga.rs::count_edges` and
`diffusion.rs::component_count`. If you write new code that reads `obsp`, decide which of the two
rules it follows before you start.

## Memory

The count matrix stays sparse when it passes from Python to Rust. Python hands over the three CSR
arrays as they are. A matrix that is 95% zeros is never expanded into a full (dense) matrix.

Two steps do create a dense matrix, and you should plan for them:

- Scaling with `zero_center=True` returns a dense `(cells, genes)` array and stores it in `adata.X`.
  That takes 400 MB at 50 000 x 2 000 and 4 GB at 50 000 x 20 000. scanpy does the same. This is
  why you keep only the highly variable genes before scaling.
- The exact t-SNE builds an `(n, n)` matrix of cell-to-cell similarities, so it stops at 20 000
  cells. Above that, `tl.tsne` switches to FIt-SNE, a fast approximation. FIt-SNE keeps
  similarities only between nearest neighbours. It estimates the pushing-apart force between all
  cells on a grid with the fast Fourier transform (FFT). The time per iteration then grows in
  proportion to the number of cells.

For matrices that do not fit in memory at all, Metalcyte reads the `.h5ad` file from disk in
blocks of rows. It sizes the blocks to stay within `settings.max_memory_gb`. Total-count
normalisation and the log transform do this automatically when `adata.isbacked` is true. They read
`X` one block at a time and write the result back to the same file. Peak memory is one block, and
the output is exactly the same, bit for bit, as the in-memory result (`benches/backed_transform.py`:
737 MB peak against 1141 MB in memory, 0.65x). For other calculations you can use the lower-level
`open_backed` iterator yourself. To sum each gene on a 50 000 x 20 000 file, it peaks at 1.41 GB.
Reading the file and making the matrix dense peaks at 4.34 GB. [API.md](API.md) documents both
functions.

## What "agrees with scanpy" means

scanpy is the reference. The kind of agreement we test depends on the algorithm. A test of the wrong
kind would prove nothing.

- **Calculations with one correct answer** (total-count normalisation, the log transform, scaling)
  are compared value by value.
- **Selections** (highly variable genes, nearest neighbours) are compared as sets.
- **Layouts that depend on random numbers** (UMAP) are compared with the range that the reference
  reaches when it runs again with a different seed. On PBMC 3k, umap-learn agrees with itself on
  about half of each cell's neighbours. A required agreement above that level would test nothing.
- **Methods that minimise a stated quantity** (t-SNE) are judged on that quantity. The test asks
  whether the final KL divergence is as low as scanpy's. It does not ask whether both runs ended at
  the same layout.
- **PCA** is compared on the components that a randomised solver can determine reliably, and on the
  variances of all components (the spectrum). On PBMC 3k that is the first 7 of 50 components.
  Beyond those, scanpy's own randomised solver does not match scanpy's arpack solver either.

[VALIDATION.md](VALIDATION.md) gives every number behind these statements, measured by the
reference test suite.

## Reproducibility

Every random step takes an explicit seed. On one device, the same seed gives exactly the same
bytes. Across devices, the seed fixes the steps of the algorithm, but the last few bits of the
arithmetic can differ, as explained above. This holds within Metalcyte only. A Metalcyte UMAP with
`random_state=0` will not match a scanpy UMAP with `random_state=0`. They are different programs of a
method that uses random numbers. The comparison with the reference range, described above, measures
how close they are.
