# Architecture

## Layers

```
python/scrust/{pp,tl}.py      AnnData plumbing and defaults
        │
crates/scrust-py              PyO3: conversion only, no logic
        │
crates/scrust-core            data types and every algorithm, written against candle
        │
crates/scrust-gpu             Metal context and hand written kernels
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
- **GPU** owns Metal: the device, the pipeline cache, and the kernels.

## Why candle *and* hand written kernels

Everything expressible as tensor algebra is written once against
`candle_core::Tensor` and takes a `Device`. The same source runs on the CPU and
on the Apple GPU, so the CPU path is not a second implementation to keep in sync
— it is the correctness oracle the GPU path is tested against.

Three inner loops are not tensor algebra: nearest-neighbour *selection*, UMAP's
negative sampling, and t-SNE's fused attract/repel pass. Expressing them with
tensor ops would mean materialising an `(n, n)` matrix that only exists to be
thrown away. Those get hand written Metal kernels in `scrust-gpu`, and each
kernel's test asserts it returns what its core counterpart returns.

A kernel is therefore always an *optimisation*, never a separate algorithm. If
the two ever disagree, the core version is right.

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
- `f32` everywhere: the Apple GPU has no `f64`, and scanpy's own results are
  `f32` after normalisation.
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
