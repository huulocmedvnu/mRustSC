# API contract

Fifteen branches are developed in parallel against this document. Every symbol
below already exists on `main` as a stub — `todo!()` in Rust,
`NotImplementedError` in Python — so the workspace always compiles and imports.
A branch fills in bodies and adds tests, and never renames anything here.

## Goal of this round

Three things at once, and all three matter:

1. **A library people install.** `pip install scrust` must work on Apple silicon
   and give a usable, typed, documented package.
2. **scanpy's feature set, in Rust.** Anything a scanpy user reaches for should
   be here. Plotting stays scanpy's job — we write the AnnData slots its
   plotting reads.
3. **Big data.** Single-cell matrices are 90-95% zeros and no longer fit in
   memory. Sparse-native GPU kernels and streaming row blocks are what make the
   difference, not a faster dense loop.

**The GPU is the reason this project exists.** Every new algorithm must say, in
its report, either how it uses the device or why the work is not tensor-shaped.
"I accepted `device` and ignored it" is acceptable only with a reason.

## Ownership

| Branch | Files | Delivers |
| --- | --- | --- |
| `feat/leiden` | `core/cluster.rs`, `python/scrust/tl/_cluster.py` | `tl.leiden`, `tl.louvain`, modularity |
| `feat/qc-metrics` | `core/qc.rs`, `python/scrust/pp/_qc.py` | `pp.calculate_qc_metrics`, `sqrt`, `normalize_per_cell`, `filter_genes_dispersion` |
| `feat/sampling` | `core/sampling.rs`, `python/scrust/pp/_sampling.py` | `pp.subsample`, `pp.sample`, `pp.downsample_counts` |
| `feat/regress-combat` | `core/batch.rs`, `python/scrust/pp/_batch.py` | `pp.regress_out`, `pp.combat` |
| `feat/diffusion` | `core/diffusion.rs`, diffmap and dpt in `tl/_trajectory.py` | `tl.diffmap`, `tl.dpt` |
| `feat/paga` | `core/paga.rs`, paga only in `tl/_trajectory.py` | `tl.paga` |
| `feat/scoring` | `core/scoring.rs`, `tl/_score.py`, `filter_rank_genes_groups` in `tl/_de.py` | `tl.score_genes`, `score_genes_cell_cycle`, `marker_gene_overlap` |
| `feat/de-methods` | `core/de/parametric.rs`, the `method` argument in `tl/_de.py` | `rank_genes_groups` with `t-test`, `t-test_overestim_var`, `logreg` |
| `feat/metrics` | `core/autocorrelation.rs`, `python/scrust/metrics/` | `metrics.morans_i`, `gearys_c`, `confusion_matrix`, `modularity` |
| `feat/layout` | `core/layout.rs`, `tl/_layout.py` | `tl.dendrogram`, `tl.draw_graph`, `tl.embedding_density` |
| `feat/sparse-gpu` | `gpu/kernels/spmm.rs` | CSR kernels on Metal: SpMM, transposed SpMM, column moments, row scaling |
| `feat/out-of-core` | `core/chunked.rs`, new `python/scrust/_backed.py` | streaming row blocks, backed h5ad, incremental PCA |
| `feat/accessors` | `python/scrust/get/`, `python/scrust/settings.py` | `get.obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate`, settings and logging |
| `feat/packaging` | `pyproject.toml`, `python/scrust/py.typed`, `.github/workflows/release.yml`, `docs/INSTALL.md` | wheels, typing marker, PyPI metadata, release CI |
| `feat/docs-bench` | `docs/`, `benches/`, `README.md`, `examples/` | tutorial mirroring scanpy's, API reference, refreshed benchmarks |

Files owned by `main` and never edited on a branch: every `Cargo.toml`,
`.cargo/config.toml`, `crates/scrust-core/src/{lib,error,device,sparse}.rs`,
`crates/scrust-gpu/src/{lib,context}.rs`, every `mod.rs` and `__init__.py`,
`python/scrust/_shared.py`, and any file the table assigns to another branch.
`pyproject.toml` is `feat/packaging`'s alone — if you need a dependency, say so
in your report instead of adding it.

## Bindings

Each algorithm branch adds its own `#[pyfunction]` wrappers. To keep the binding
crate conflict-free, put them in the existing file that matches the area
(`preprocess.rs`, `embedding.rs`, `de.rs`) **only if you own that area**;
otherwise create `crates/scrust-py/src/<your-area>.rs` and say so in your report
so `main` can register it. Do not edit `crates/scrust-py/src/lib.rs`.

Bindings convert and nothing else: no defaults, no AnnData knowledge, no
algorithm. Release the GIL around the core call.

## Conventions

Unchanged from the previous round, and still binding:

- Matrices are **cells by genes**. `CsrMatrix` crosses the Python boundary as
  `(indptr, indices, values, n_cols)` with `uint32` indices and `float32` values.
- `f32` throughout, except **p-values, which are `f64`** — a rank-sum p-value
  underflows `f32` to exactly zero.
- Every algorithm takes `&candle_core::Device`; the CPU path is the same code
  and is the oracle the GPU path is tested against.
- Metal kernels are **optimisations, not separate algorithms**: each must return
  what its `scrust-core` counterpart returns, and its tests must assert that.
- Randomness takes an explicit seed. Same seed, same bytes, unless a kernel
  documents a deliberate race.
- Errors are `scrust_core::Error`; nothing panics on user input.
- The Python layer is AnnData plumbing and defaults only. No arithmetic.

## scanpy is the reference

Correctness is agreement with scanpy on the same input. Add your tests to a file
of your own — `tests/test_<area>.py` — never to an existing test file. Use the
fixtures in `tests/conftest.py` and the helpers in `tests/reference_metrics.py`.

What agreement means differs per algorithm, and picking the wrong form is the
mistake that has cost this project the most time. Three corrections already
made, all still binding:

- **Stochastic embeddings** (`umap`, `draw_graph`): compare against the band the
  reference reaches against *itself* reseeded, not an absolute number. Measured:
  umap-learn agrees with itself only ~44% on PBMC 3k.
- **An explicit objective beats a proxy.** t-SNE is judged on the KL divergence
  it minimises, not on landing in scanpy's local optimum. If your algorithm
  minimises something, assert on that.
- **Degenerate directions.** Where the reference cannot reproduce itself — PCA's
  trailing components — assert the quantity that *is* determined (the spectrum),
  not the one that is not (the eigenvectors).

For clustering, label ids are arbitrary: compare with the adjusted Rand index or
normalised mutual information against scanpy's labels, never label by label.

If you find the criterion in this document is itself unreachable, **measure the
ceiling and say so** rather than lowering a threshold until it passes. That has
happened three times already and each time the measurement was the real result.

## Big data

`feat/sparse-gpu` and `feat/out-of-core` carry this, but it constrains everyone:

- Never densify a whole matrix. Densify a row block, or use the sparse kernels.
  If your algorithm cannot avoid a dense `(n_cells, n_genes)` intermediate, say
  so in your report with the size it implies.
- State the peak memory your implementation needs as a function of the input,
  and return `Error::InvalidParameter` above a documented limit rather than
  exhausting memory.
- `settings.max_memory_gb` is the budget the chunked paths size blocks against.
