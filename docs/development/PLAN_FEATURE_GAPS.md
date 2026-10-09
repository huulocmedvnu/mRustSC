# Plan: closing the feature gaps (drafted 2026-10-08, not started)

Gap analysis against scanpy 1.12.4, the reference the test suite already checks against. Metalcyte 0.3.0
has the full standard pipeline but lacks several functions and many options. This plan orders the work by
how often users need it. Nothing here is implemented yet.

## Principles (same as the rest of the code base)

- Heavy computation in the Rust core (`crates/metalcyte-core`), GPU only where a kernel measurably helps, the
  Python layer stays thin and writes results to the standard AnnData slots.
- Every new function or option gets a test against the reference implementation with a stated tolerance
  (`tests/`), a row in `docs/VALIDATION.md`, an entry in `docs/API.md` and `CHANGELOG.md`.
- Randomness takes a seed. Out-of-core support where the algorithm allows it.
- Documentation describes Metalcyte on its own terms (see the 2026-10-07 presentation rule).

## Phase 1: options users hit first (target release 0.4.0)

| item | scope | where | test against reference |
|---|---|---|---|
| 1.1 filtering upper bounds | `filter_cells(max_genes, max_counts)`, `filter_genes(max_cells, max_counts)` | `preprocess/filter.rs`, `pp/_basics.py`, out-of-core pass 1 | identical masks |
| 1.2 variable genes | `flavor="seurat_v3"` (variance of raw counts against a LOESS fit of mean, span 0.3, clipped standardised values), `batch_key` (per-batch selection, ranked by the number of batches then the median rank), `subset`, cut-off mode (`min_mean`, `max_mean`, `min_disp`, `max_disp` with `n_top_genes=None`), `layer` | `preprocess/hvg.rs` (+ a LOESS routine), `pp/_basics.py`, out-of-core head | gene-set Jaccard >= 0.95 for seurat_v3 (LOESS implementations differ slightly), identical sets for the cut-off mode and for batch_key on seurat/cell_ranger |
| 1.3 marker tests | `pts` (fraction of cells expressing each gene in the group and in the rest), `tie_correct`, `corr_method` (Benjamini-Hochberg, Bonferroni), `n_genes`, `rankby_abs`, `key_added`, `layer` / `use_raw`; `pts` also in the streamed test | `de/wilcoxon.rs`, `de/parametric.rs`, `tl/_de.py` | identical `pts`, p-values to 1e-6 |
| 1.4 normalisation | `exclude_highly_expressed` + `max_fraction`, `key_added`, `layer` | `preprocess/normalize.rs` | identical size factors |
| 1.5 common options | `layer`, `key_added`, `copy` on the functions that lack them where cheap | Python layer | round-trip tests |

## Phase 2: algorithms (0.4.x)

| item | scope | notes |
|---|---|---|
| 2.1 Leiden options | `restrict_to` (re-cluster selected groups on the induced subgraph, labels "c,0"), `use_weights`, CPM partition | `cluster.rs`; restrict_to is the common sub-clustering workflow |
| 2.2 neighbour options | `metric` = cosine and correlation (row-normalise or centre then reuse the Euclidean kernels, exact and NN-descent), `n_pcs`, `key_added` with `neighbors_key` honoured downstream | `neighbors.rs`, `nndescent.rs`, GPU kernel unchanged after normalisation |
| 2.3 UMAP options | `init_pos` (spectral, random, PAGA positions, array), explicit `a`, `b`, `key_added` | `umap.rs` |
| 2.4 doublet detection | Scrublet-style: simulate doublets by summing random cell pairs, shared PCA, kNN doublet score, automatic threshold from the bimodal score histogram; batch-aware | new `doublets.rs`; reuses PCA and NN-descent; validate by score correlation and call agreement with the reference |
| 2.5 Pearson residuals | analytic Pearson residuals normalisation (Lause et al. 2021) and the matching HVG flavour; two passes over the data, so it also fits the out-of-core head | new module; closed form from gene and cell sums |
| 2.6 label transfer | `ingest`-style mapping of a query onto a reference (project onto the reference PCA, kNN label vote, UMAP transform) | needs a UMAP transform step; lowest priority in this phase |

## Phase 3: input and plotting (0.5.0)

| item | scope | notes |
|---|---|---|
| 3.1 readers | `read_10x_mtx`, `read_10x_h5`; let `preprocess_backed` stream a 10x h5 file directly | 10x h5 stores one column per cell, which streams like CSR rows |
| 3.2 plots | dotplot, violin, stacked violin, matrixplot, heatmap, tracksplot, scatter, QC plots (highest expressed genes, variable genes), PAGA graph | matplotlib in the house style; heatmaps of many cells through the Metal rasteriser |
| 3.3 example data | a `datasets.pbmc3k()` download helper (the tutorial already needs it) | cached under the user cache directory |

## Out of scope for now

- `recipe_*` helpers (a few lines each; documented as examples), `sim`.
- Wrappers of third-party tools (bbknn, scanorama, MAGIC, Palantir, PHATE); users call those packages on the
  same AnnData object.
- Spatial readers and plots (a separate project).

## Order and effort (rough)

1. Phase 1 in full: one to two sessions. Main risk: matching the reference LOESS closely enough for seurat_v3.
2. 2.1 to 2.3: one session. 2.4 doublets: one session. 2.5: one session. 2.6: one session.
3. Phase 3 readers: half a session. Plots: two sessions.

Release 0.4.0 after Phase 1 and 2.1 to 2.4; 0.5.0 after Phase 3. Each release goes through PyPI and Zenodo as
described in `docs/development/` and the release workflow (never re-release an existing tag).

## Manuscripts

The submitted papers describe version 0.3.0 and stay as they are. The CMPB version can state the missing parts
in its limitations (plotting beyond embeddings, doublet detection, direct 10x reading).

## Housekeeping

- Done in 0.3.2 (2026-10-09): the Metal shaders live in `crates/metalcyte-gpu/src/shaders/*.metal`.
