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
        └── crates/metalcyte-gpu       Metal context, hand-written kernels (src/shaders/*.metal)
```

Each layer uses only the layer below it, and each layer has one job:

- **Python** sets default values and argument names, and decides where in the AnnData object each
  result goes. It does no calculations.
- **Bindings** (PyO3) convert numpy and scipy arrays
  into Rust types and back. They turn `metalcyte_core::Error` into Python exceptions. They hold no
  default values, so every binding argument is required.
- **Core** holds the algorithms and the data types. It knows nothing about Python or AnnData.
- **GPU** holds the Metal code: the device, a cache of compiled pipelines, and the
  hand-written kernels. It sits beside the core and depends on `metalcyte-core`. The bindings
  also depend on it, for the one kernel (`knn`) that is connected. See below.

## The GPU code is candle

candle is a Rust tensor library that runs on the CPU or on the Apple GPU.
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

`metalcyte-gpu` holds the hand-written Metal kernels. Their Metal Shading Language source lives in
`crates/metalcyte-gpu/src/shaders/*.metal`, one file per kernel family. The Rust side embeds each file
with `include_str!` and the system Metal framework compiles it the first time it is used, so neither
Xcode nor a prebuilt `.metallib` is needed. Compiled pipelines are cached for the rest of the process.

| shader file | kernels | connected to Python |
|---|---|---|
| `knn.metal`, `knn_tiled.metal`, `knn_simd.metal` | exact k-nearest-neighbour search | yes, `pp.neighbors` on Metal |
| `tsne_fft.metal` | the FFT-accelerated t-SNE iteration: attraction, placement, spreading, FFT, gather, update | yes, `tl.tsne` above 20 000 cells on Metal |
| `raster.metal` | point rasteriser (vertex and fragment shader) | yes, `pl.embedding` and the plots built on it |
| `spmm.metal`, `column_moments.metal`, `scale_rows.metal` | sparse times dense product, column moments, row scaling | no |
| `tsne_gradient.metal` | exact t-SNE gradient | no |
| `umap_sgd.metal` | one UMAP epoch | no, on purpose: its lock-free updates would make a layout depend on the device |
| `trivial.metal` | a one-line kernel that probes whether the GPU is usable | internal |

Every kernel must give the answer of its CPU reference in the core. The exact neighbour search repeats
two steps of the core search: it centres the data on the mean and sets tiny squared distances to zero,
and the device test requires identical neighbour lists. The unconnected kernels are still tested
against a CPU version in their own modules. Elsewhere, "the GPU path" means candle unless a kernel is
named.

If a kernel and its reference ever disagree, the core version is right.

## Data flow

`AnnData.X` is a CSR matrix. Python passes its three CSR arrays straight to Rust, so a matrix that
is 90-95% zeros is never densified. When an algorithm in
the core needs a dense array, it expands one block of rows at a time. Peak memory then depends on
the block size and does not grow with the matrix.

On Apple silicon the CPU and the GPU share the same memory (unified memory). The GPU reads the data
where Rust stored it, and nothing is copied. This is why the GPU helps even with single-cell matrix
sizes. A discrete GPU would spend more time on PCIe transfers than on calculation.

## Conventions

- Matrices are cells by genes, as in AnnData.
- Expression data and all tensors use `f32`, because the Apple GPU has no
  `f64`. There are two exceptions, both where `f32` would lose the answer completely. CPU sums
  are computed in `f64` and rounded once at the end (per-gene moments in `scale`, the rank sums
  in `wilcoxon`). p-values stay `f64` everywhere, because a rank-sum p-value is often too small for `f32` and would become exactly zero.
- Every random step takes an explicit seed. The same seed gives the same bytes. The exception is a
  kernel whose threads race on purpose, and its module documentation must say so.
- Functions and modules use `snake_case`, types use `PascalCase`, and names are long enough to
  explain themselves.
- Errors are `metalcyte_core::Error`. Bad user input never causes a panic.

## Correctness

Each function is tested against a reference implementation of the same method. The kind of
agreement depends on the algorithm. Calculations with
one correct answer are compared value by value. Selections are compared by the overlap of the
selected sets. Stochastic embeddings are compared by how well they keep each
cell's neighbours. These rules are fixed in
[development/API_CONTRACT.md](development/API_CONTRACT.md), so that no change can quietly lower its
own standard. The measured results are in [VALIDATION.md](VALIDATION.md).
