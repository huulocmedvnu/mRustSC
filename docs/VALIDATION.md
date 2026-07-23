# Validation

scanpy is the reference. Every implemented algorithm is cross-checked against it in
`tests/test_reference.py`, which runs each check twice — on a 240-cell synthetic
matrix, and on real PBMC 3k (2 638 cells) under the `reference` marker. The form of
agreement is chosen per algorithm and fixed in
[API_CONTRACT.md](API_CONTRACT.md#scanpy-is-the-reference); the numbers below are
what the tests *recorded*, not what the thresholds demand.

Reproduce them with:

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

## What is not validated

The 24 unimplemented functions have no results because they raise
`NotImplementedError`; the reference tests for them `skip` rather than pass. The
hand-written Metal kernels in `crates/scrust-gpu` are validated in Rust against
their `scrust-core` counterparts (`cargo test`), not here, and are not yet reachable
from Python. See [API.md](API.md) for the full implemented/not-implemented split.
