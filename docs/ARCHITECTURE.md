# Architecture

## Layers

```
python/scrust/{pp,tl,get,metrics}   AnnData plumbing and defaults
        │
crates/scrust-py                    PyO3: conversion only, no logic
        │
crates/scrust-core                  data types and every algorithm, written against candle
        │
        └── crates/scrust-gpu       Metal context and hand written kernels
```

Dependencies point downwards. Each layer is allowed to know only about the one
below it, and each has exactly one job:

- **Python** owns defaults, argument names and where a result lands in an
  AnnData. It performs no arithmetic.
- **Bindings** own conversion between numpy/scipy and Rust types, and the map
  from `scrust_core::Error` to Python exceptions. They own no defaults, which is
  why every binding argument is required.
- **Core** owns the algorithms and the data types. It knows nothing about Python
  or AnnData.
- **GPU** owns Metal: the device, the pipeline cache, and the kernels. It sits
  *beside* the pipeline rather than under it — it depends on `scrust-core`, and
  nothing above it depends on it.

## The GPU path is candle

Everything expressible as tensor algebra is written once against
`candle_core::Tensor` and takes a `Device`. The same source runs on the CPU and
on the Apple GPU, so the CPU path is not a second implementation to keep in sync
— it is the correctness oracle the GPU path is tested against. This is the only
way a Python caller reaches the GPU: `DeviceKind` resolves to a
`candle_core::Device`, the binding hands it to the core, and candle's Metal
backend does the arithmetic.

Not every algorithm takes that device to heart. `pca`, `neighbors`, `tsne`,
`diffusion`, `batch`, `layout`, `autocorrelation`, `scoring` and `de/glm` build
tensors on it; `umap`, `cluster`, `normalize`, `hvg`, `de/wilcoxon` and
`de/parametric` take the argument as `_device` and run on the CPU regardless,
because their inner loops are graph or rank work rather than tensor algebra.
Passing `device="gpu"` is a request, not a guarantee, and each of those modules
says in its own docs why it declines.

## `scrust-gpu` is a sidecar, not a layer

`scrust-gpu` holds four hand written Metal kernels — `knn`, `spmm`,
`tsne_gradient`, `umap_sgd` — for the loops candle cannot express: nearest
neighbour *selection*, sparse products that would have to be densified first,
and the fused attract/repel passes of t-SNE and UMAP. Expressing those with
tensor ops would mean materialising an `(n, n)` matrix that only exists to be
thrown away.

**None of them is reachable from Python.** `crates/scrust-py/Cargo.toml` does
not depend on `scrust-gpu` and no binding names it; the dependency was removed
once it became clear nothing there called it. The crate is still built and
tested in the workspace — each kernel's tests hold it against a brute force CPU
reference written in the same module — but it is exploratory work, not the path
a `scrust.pp` or `scrust.tl` call takes. Read a claim about "the GPU path" in
this repository as candle unless it names a kernel.

If a kernel and its reference ever disagree, the core version is right.

## Data flow

`AnnData.X` is CSR. The three CSR arrays cross the FFI boundary directly, which
avoids densifying a matrix that is 90-95% zeros. Inside the core, algorithms
densify a *row block* at a time when they need a tensor, so peak memory stays
bounded by the tile size rather than the matrix.

Apple silicon has unified memory, so a Metal buffer over a Rust slice is a view,
not a copy across a bus. That is the property that makes GPU acceleration worth
it at single-cell matrix sizes, where a discrete GPU would spend more time on
transfers than on arithmetic.

## Conventions

- Matrices are cells by genes, matching AnnData.
- `f32` for expression data and for every tensor: the Apple GPU has no `f64`,
  and scanpy's own results are `f32` after normalisation. Two exceptions, both
  because `f32` loses the answer outright: CPU reductions accumulate in `f64`
  and round once at the end (per gene moments in `scale`, the rank sums in
  `wilcoxon`), and p-values stay `f64` throughout, since a rank sum p-value
  routinely underflows `f32` to exactly zero.
- Randomness takes an explicit seed. Same seed, same bytes — except where a
  kernel documents a deliberate race, which must be stated in its module docs.
- `snake_case` for functions and modules, `PascalCase` for types, names long
  enough to explain themselves.
- Errors are `scrust_core::Error`. Nothing panics on user input.

## Correctness

scanpy defines correct. The form of agreement differs per algorithm — element
wise for deterministic transforms, set overlap for selections, neighbourhood
preservation for stochastic embeddings — and is fixed in `docs/API_CONTRACT.md`
so that no branch can quietly weaken its own bar. Measured results live in
`docs/VALIDATION.md`.
