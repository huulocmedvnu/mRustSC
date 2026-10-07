# Architecture

## Layers

Metalcyte has four parts. Python code sits on top and Rust code below it.

```
python/metalcyte/{pp,tl,get,metrics}   AnnData plumbing and defaults
        │
crates/metalcyte-py                    PyO3: conversion only, no logic
        │
crates/metalcyte-core                  data types and every algorithm, written against candle
        │
        └── crates/metalcyte-gpu       Metal context and hand written kernels
```

Each layer uses only the layer below it, and each layer has one job:

- **Python** sets default values and argument names, and decides where in the AnnData object each
  result goes. It does no calculations.
- **Bindings** (PyO3, the library that connects Python and Rust) convert numpy and scipy arrays
  into Rust types and back. They turn `metalcyte_core::Error` into Python exceptions. They hold no
  default values, so every binding argument is required.
- **Core** holds the algorithms and the data types. It knows nothing about Python or AnnData.
- **GPU** holds the Metal code: the device, a cache of compiled GPU programs, and the kernels
  (hand-written GPU programs). It sits beside the core and depends on `metalcyte-core`. The bindings
  also depend on it, for the one kernel (`knn`) that is connected. See below.

## The GPU code is candle

candle is a Rust library for array (tensor) calculations that runs on the CPU or on the Apple GPU.
Every calculation that candle can express is written once with `candle_core::Tensor` and receives a
`Device`. The same source code runs on both devices. The CPU version serves as the reference that
the GPU version is tested against. This is how a Python call reaches the GPU. `DeviceKind` becomes a
`candle_core::Device`, the binding passes it to the core, and candle's Metal code does the
calculation.

Some algorithms ignore the device. `pca`, `neighbors`, `tsne`, `diffusion`, `batch`, `layout`,
`autocorrelation`, `scoring` and `de/glm` run on it. `umap`, `cluster`, `normalize`, `hvg`,
`de/wilcoxon` and `de/parametric` receive it as `_device` and always run on the CPU. Their inner
loops walk graphs or rank values, and contain no matrix calculations to move to the GPU. So
`device="gpu"` is a request. The documentation of each of those modules says why it runs on the CPU.

## `metalcyte-gpu` sits beside the core

`metalcyte-gpu` holds four hand-written Metal kernels: `knn`, `spmm`, `tsne_gradient` and
`umap_sgd`. They do work that candle cannot express directly. That work is choosing the nearest
neighbours, multiplying sparse matrices without making them dense first, and the combined attract
and repel steps of t-SNE and UMAP. With candle, each of these would need a full `(n, n)` matrix
that is built only to be discarded.

**Python can reach one of them, `knn`.** `crates/metalcyte-py/Cargo.toml` depends on
`metalcyte-gpu`. For a call on Metal, the `embedding` binding sends the k-NN search to `knn_metal`.
On the CPU, or when Metal fails to start, it uses the candle version
(`metalcyte-py/src/embedding.rs`). The kernel must give the same answer as the CPU reference. So
its MSL code (Metal Shading Language) repeats two steps of the core k-NN: it centres the data on the
mean, and it sets tiny squared distances to zero. The device comparison test requires both devices
to return the same neighbour lists.

**Python cannot reach the other three.** `spmm` has no caller. The core PCA multiplies a centred
sparse matrix, which needs an extra rank-one correction, and the kernel computes only a plain sparse
times dense product. `tsne_gradient` is not connected. `umap_sgd` is left unconnected on purpose. It
uses Hogwild updates, where parallel threads change the layout at the same time without locks, so
results vary from run to run. Connecting it would make a UMAP layout depend on whether the computer
has a GPU. Each kernel is still tested against a simple CPU version in its own module. Outside of
k-NN, when this repository mentions "the GPU path", it means candle, unless it names a kernel.

If a kernel and its reference ever disagree, the core version is right.

## Data flow

`AnnData.X` is a sparse matrix in CSR format. Python passes its three CSR arrays straight to Rust,
so a matrix that is 90-95% zeros is never expanded into a full (dense) matrix. When an algorithm in
the core needs a dense array, it expands one block of rows at a time. Peak memory then depends on
the block size and does not grow with the matrix.

On Apple silicon the CPU and the GPU share the same memory (unified memory). The GPU reads the data
where Rust stored it, and nothing is copied. This is why the GPU helps even with single-cell matrix
sizes. A separate graphics card would spend more time copying data than calculating.

## Conventions

- Matrices are cells by genes, as in AnnData.
- Expression data and all tensors use `f32` (32-bit floating point). The Apple GPU has no `f64`,
  and scanpy's own results are `f32` after normalisation. There are two exceptions, both where
  `f32` would lose the answer completely. CPU sums are computed in `f64` and rounded once at the end
  (per-gene moments in `scale`, the rank sums in `wilcoxon`). p-values stay `f64` everywhere,
  because a rank-sum p-value is often too small for `f32` and would become exactly zero.
- Every random step takes an explicit seed. The same seed gives the same bytes. The exception is a
  kernel whose threads race on purpose, and its module documentation must say so.
- Functions and modules use `snake_case`, types use `PascalCase`, and names are long enough to
  explain themselves.
- Errors are `metalcyte_core::Error`. Bad user input never makes the program crash (panic).

## Correctness

scanpy defines the correct answer. The kind of agreement depends on the algorithm. Calculations with
one correct answer are compared value by value. Selections are compared by the overlap of the
selected sets. Embeddings that depend on random numbers are compared by how well they keep each
cell's neighbours. These rules are fixed in
[development/API_CONTRACT.md](development/API_CONTRACT.md), so that no change can quietly lower its
own standard. The measured results are in [VALIDATION.md](VALIDATION.md).
