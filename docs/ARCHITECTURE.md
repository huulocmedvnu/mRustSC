# Architecture

## Layers

```
cli ──► factory ──► pipeline ──► contracts ◄── stats / preprocess / io / report
                                    ▲
                                    └── backend (mlx, numpy)
```

Dependencies point inwards, towards `contracts.py`. The pipeline never imports a
concrete estimator, a backend, or a file format — it receives them through its
constructor. Only `factory.py` knows which implementation is the default, so
swapping the GPU backend for the CPU one, or the Wald test for a likelihood
ratio test, changes one line in one file.

## Rules

1. **One module, one responsibility.** Normalisation does not filter genes; the
   GLM does not compute p-values; the writer does not format plots.
2. **Depend on protocols.** Anything typed as `SizeFactorEstimator`,
   `GLMFitter`, `ComputeBackend`, … accepts every implementation of that
   protocol. New behaviour is added as a new class, not as a branch inside an
   existing one.
3. **Small interfaces.** `ComputeBackend` is composed from `ArrayOps` and
   `LinearAlgebraOps`; a caller that only reduces arrays depends on the smaller
   protocol.
4. **NumPy at the boundaries, backend arrays inside.** Public signatures take and
   return `np.ndarray`. Conversion to device arrays happens inside the component
   that does the arithmetic, so nothing else has to know a GPU exists.
5. **No premature generality.** Formats, models, and options are added when a
   caller needs them.

## Why the GPU helps

A differential expression run fits one GLM per gene — tens of thousands of tiny,
identical, independent problems. Fitting them one at a time is latency-bound; the
implementation instead stacks every gene into one batched IRLS iteration, so each
step is a handful of large kernel launches over `(n_genes, n_samples)` and
`(n_genes, p, p)` tensors. Apple silicon's unified memory means the count matrix
is not copied across a PCIe bus, which is the usual reason small-batch GPU
statistics are not worth it.

## Naming

`snake_case` for functions, variables, and modules; `PascalCase` for classes;
descriptive names over abbreviations (`size_factors`, not `sf`). Names carry the
meaning, so comments explain *why*, not *what*.
