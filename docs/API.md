# API reference

Every function takes an `AnnData` and writes into the slot scanpy uses, so scanpy's
plotting reads the result unchanged. Signatures below are the ones the installed
package actually exposes; anything scanpy has that scrust does not is listed too,
with what the call does today.

Names are grouped as scanpy groups them: `pp`, `tl`, `metrics`, `get`.

**Status at 0.2.0**: 16 of the 40 public functions have an implementation behind
them. The other 24 exist, import, type-check and raise `NotImplementedError` naming the
branch that owes them. Nothing is a silent no-op.

Contents: [pp](#pp--preprocessing) · [tl](#tl--tools) · [metrics](#metrics) ·
[get](#get--accessors) · [settings](#settings) · [devices](#devices) ·
[out-of-core](#out-of-core-reading)

---

## `pp` — preprocessing

### Implemented

#### `pp.filter_cells(adata, *, min_genes=None, min_counts=None, inplace=True)`

Drops cells with fewer than `min_genes` expressed genes or fewer than `min_counts`
total counts. At least one of the two is required; passing neither is a
`ValueError`. With `inplace=True` the AnnData is subset in place and `None` is
returned; with `inplace=False` you get the boolean keep-mask and the object is left
alone.

Unlike scanpy, it does not add `obs["n_genes"]` or `obs["n_counts"]` — it returns
the mask and nothing else.

#### `pp.filter_genes(adata, *, min_cells=None, min_counts=None, inplace=True)`

The same, over genes. Also does not write the `var["n_cells"]` column scanpy adds.

#### `pp.normalize_total(adata, *, target_sum=None, inplace=True)`

Scales each cell to `target_sum` counts, or to the median cell count when
`target_sum` is `None`. Writes `adata.X` (CSR, `float32`), or returns the new matrix
under `inplace=False`.

scanpy's `exclude_highly_expressed`, `max_fraction`, `key_added` and `layer` are not
accepted.

#### `pp.log1p(adata, *, inplace=True)`

`log(1 + x)` on the stored entries. Writes `adata.X` and sets
`adata.uns["log1p"] = {"base": None}`, which is what downstream scanpy tools look
for. No `base` argument: natural log only.

#### `pp.highly_variable_genes(adata, *, n_top_genes=2000, flavor="seurat", inplace=True)`

Flags the `n_top_genes` most variable genes. `flavor` is `"seurat"` or
`"cell_ranger"`; anything else is a `ValueError` from the Rust side. Writes three
`var` columns:

| column | meaning |
| --- | --- |
| `highly_variable` | bool, the selection |
| `means` | per-gene mean |
| `dispersions_norm` | dispersion normalised within its mean bin |

scanpy also writes `dispersions`, `variances` and `highly_variable_rank`; scrust
does not. The Rust core computes raw `dispersions`, but the Python layer drops the
field, so anything reading `var["dispersions"]` will `KeyError`.

`flavor="cell_ranger"` bins genes by percentiles of the mean and raises `ValueError`
when enough genes share a mean value that two bin edges coincide — `pandas.cut`
refuses repeated edges, and binning differently from scanpy silently would be worse
than failing.

#### `pp.scale(adata, *, zero_center=True, max_value=None, inplace=True)`

Centres each gene and scales it to unit variance, clipping at `±max_value` when
given. **The result is dense.** `adata.X` becomes a `float32` numpy array of shape
`(n_cells, n_genes)` — 400 MB at 50 000 x 2 000 — which is also what scanpy does
with `zero_center=True`. Subset to the highly variable genes before calling it.

#### `pp.pca(adata, *, n_comps=50, zero_center=True, random_state=0, device="auto")`

Randomised SVD. scanpy's default solver is deterministic `arpack`, so the two agree
only where a randomised SVD is itself reproducible — see
[VALIDATION.md](VALIDATION.md), which measures where that stops. Writes:

- `obsm["X_pca"]`, `(n_cells, n_comps)`
- `varm["PCs"]`, `(n_genes, n_comps)` (the core returns the transpose; the Python
  layer flips it to scanpy's orientation)
- `uns["pca"]` with `variance`, `variance_ratio` and `params`

No `svd_solver`, `use_highly_variable` or `mask_var` argument.

#### `pp.neighbors(adata, *, n_neighbors=15, use_rep="X_pca", device="auto")`

Exact k-nearest neighbours followed by UMAP's fuzzy simplicial set. `n_neighbors`
counts the cell itself, as scanpy's does, so the core is asked for `n_neighbors - 1`
neighbours; below 2 it is a `ValueError`. Writes `obsp["distances"]`,
`obsp["connectivities"]` and `uns["neighbors"]`.

`use_rep="X"` uses the matrix itself. There is no `n_pcs` argument: slice `obsm`
yourself, or pass a representation you have already truncated. No approximate
method — the search is exact, which is why its results match scanpy's exactly and
why it costs what it costs at large n.

### Not implemented

`pp.calculate_qc_metrics`, `pp.normalize_per_cell`, `pp.sqrt`,
`pp.filter_genes_dispersion` (`feat/qc-metrics`) · `pp.regress_out`, `pp.combat`
(`feat/regress-combat`) · `pp.subsample`, `pp.sample`, `pp.downsample_counts`
(`feat/sampling`). Each raises `NotImplementedError` naming its branch.

---

## `tl` — tools

### Implemented

#### `tl.umap(adata, *, n_components=2, min_dist=0.5, spread=1.0, n_epochs=None, random_state=0, device="auto")`

Lays out `obsp["connectivities"]` and writes `obsm["X_umap"]`. `n_epochs` defaults
to 200 — umap-learn's rule of 500 for small data and 200 for large is not
reproduced. `min_dist` must lie in `[0, 3 * spread]`.

UMAP does not reproduce itself across seeds, so "agrees with scanpy" is a band, not
an equality; the measured numbers are in [VALIDATION.md](VALIDATION.md).

#### `tl.tsne(adata, *, n_pcs=50, perplexity=30.0, early_exaggeration=12.0, learning_rate=None, random_state=0, device="auto")`

Exact t-SNE over the first `n_pcs` columns of `obsm["X_pca"]`; writes
`obsm["X_tsne"]`. Two limits, both raised as `ValueError` rather than discovered as
an out-of-memory kill:

- **at most 20 000 cells.** The formulation is exact, not Barnes-Hut, so the
  `(n, n)` affinity matrix is materialised: 1.6 GB at 20 000 cells, and the gradient
  step holds three more buffers of that shape, for a peak near 6.5 GB.
- **perplexity below the cell count**, which is scikit-learn's own precondition.
  A t-SNE is hard to read once the perplexity approaches a third of the cell
  count, but that is advice about the plot, not a limit of the implementation,
  and it is not enforced — scanpy does not enforce it either.

`learning_rate=None` uses scikit-learn's current `auto` rule,
`max(n / early_exaggeration / 4, 50)`. scanpy still passes its legacy 1000, so the
two libraries called with defaults are not doing the same amount of work — pass
`learning_rate` explicitly to compare them.

> **Limitation: this does not scale.** Being exact rather than Barnes-Hut makes it
> `O(n^2)`, and it loses to scanpy badly as soon as the data is not small: measured
> 1.42x *faster* than scanpy at 499 cells, 0.25x at 2 638, and **0.06x at 10 000
> cells — 271.7 s against scanpy's 15.4 s**. Above 20 000 cells it refuses.
> Use `sc.tl.tsne` for anything above a couple of thousand cells; it writes the same
> `obsm["X_tsne"]`, so the rest of a scrust pipeline is unaffected. For a large
> dataset prefer `tl.umap`, which is 4.84x faster than scanpy at 10 000 cells.
> Details in [BENCHMARKS.md](BENCHMARKS.md#tltsne-does-not-scale).

#### `tl.rank_genes_groups(adata, groupby, *, groups="all", reference="rest", method="wilcoxon", device="auto")`

Wilcoxon rank-sum per group with Benjamini-Hochberg correction. `method` accepts
only `"wilcoxon"`; `"t-test"`, `"t-test_overestim_var"` and `"logreg"` raise
`ValueError` (they belong to `feat/de-methods`). Ties are **not** corrected, matching
scanpy's `tie_correct=False` default.

Writes `uns["rank_genes_groups"]` with `params` and one record array per field —
`names`, `scores`, `logfoldchanges`, `pvals`, `pvals_adj` — each with one column per
group, ranked by score. p-values are `float64`; everything else `float32`.

Cells outside the selected groups are dropped before the call rather than encoded as
a label.

#### `tl.paga(adata, groups=None, *, model="v1.2", device="auto")`

Coarse-grains `obsp["distances"]` over a categorical `obs` column. `groups=None`
looks for `obs["leiden"]` then `obs["louvain"]` — neither of which scrust can
produce yet, so in practice you pass the key. `model` accepts only `"v1.2"`.

Writes `uns["paga"]` with `connectivities`, `connectivities_tree` (both `float64`
CSR of shape `(n_groups, n_groups)`) and `groups`, preserving any other key already
in that slot, such as a saved layout position.

`device` is accepted and deliberately ignored: this is one memory-bound pass over
the stored edges into a group-sized matrix, and there is nothing for a GPU to do.

### Not implemented

`tl.leiden`, `tl.louvain` (`feat/leiden`) · `tl.diffmap`, `tl.dpt`
(`feat/diffusion`) · `tl.score_genes`, `tl.score_genes_cell_cycle`,
`tl.marker_gene_overlap`, `tl.filter_rank_genes_groups` (`feat/scoring`) ·
`tl.dendrogram`, `tl.draw_graph`, `tl.embedding_density` (`feat/layout`).

Clustering being absent is the gap you will hit first: the PBMC tutorial reaches
`tl.leiden` and stops. `examples/pbmc3k.py` gets past it by calling `sc.tl.leiden`,
and marks the line where it does.

---

## `metrics`

Nothing here is implemented. `metrics.morans_i`, `metrics.gearys_c`,
`metrics.confusion_matrix` and `metrics.modularity` all raise
`NotImplementedError("feat/metrics")`.

---

## `get` — accessors

All four are implemented, in pure Python — there is no Rust behind them.

#### `get.obs_df(adata, keys=(), *, obsm_keys=(), layer=None)`

One row per cell, with columns taken from `obs` or from gene expression, in the
order you asked for them. A key that is both a gene and an `obs` column is a
`ValueError` rather than a guess. `obsm_keys` takes `(key, column_index)` pairs, so
`("X_pca", 0)` becomes a column named `X_pca-0`.

No `gene_symbols` or `use_raw` argument.

#### `get.var_df(adata, keys=(), *, varm_keys=())`

The transpose: one row per gene, columns from `var` or from named cells.

#### `get.rank_genes_groups_df(adata, group, *, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None)`

Flattens `uns[key]` into a tidy frame. `group=None` returns every group with a
`group` column; a single group name drops that column, as scanpy does. The cutoffs
filter rows. A `logreg` result — which scrust cannot produce yet, but can read if
scanpy wrote it — has only `names` and `scores`.

#### `get.aggregate(adata, by, func, *, axis=0, layer=None, device="auto")`

Groups cells (or genes, with `axis=1`) and reduces. `func` is one or several of
`count_nonzero`, `mean`, `median`, `sum`, `var`; `var` uses ddof 1, as scanpy does.
Returns a new AnnData with one layer per function and `obs["n_obs_aggregated"]`.

`device` is accepted for signature parity and ignored: these are scipy sparse
products and per-group medians, not core algorithms.

---

## Settings

`scrust.settings` is a dataclass singleton, validated on assignment:

| attribute | default | effect |
| --- | --- | --- |
| `verbosity` | `Verbosity.warning` | how much `settings.log` lets through |
| `device` | `"auto"` | default device — but see below |
| `max_memory_gb` | `4.0` | budget the streaming block size is derived from |
| `n_jobs` | `0` | CPU threads; `0` leaves the choice to the core |
| `chunk_size` | `0` | rows per block; `0` derives one from `max_memory_gb` |

`settings.device` accepts `"auto"`, `"cpu"`, `"gpu"` or `"metal"` and rejects
anything else where you wrote it, not several calls later.

**Known discrepancy.** `settings.device` is *not* read by every function.
`normalize_total`, `highly_variable_genes` and `scale` resolve it through
`_default_device()`, so they follow the setting. `pca`, `neighbors`, `umap`, `tsne`
and `rank_genes_groups` declare `device="auto"` as their own default and pass that
string straight through, so they ignore `settings.device` unless you pass `device=`
at the call. `filter_cells`, `filter_genes` and `log1p` take no device at all.
To pin a device today, pass it per call.

## Devices

`device` is `"auto"` (Metal if a device initialises, CPU otherwise), `"cpu"`, or
`"gpu"`/`"metal"` (an error if no Metal device is found). `scrust.gpu_available()`
reports whether Metal came up.

The CPU and GPU paths are the same candle source, so results agree; only the speed
differs. What the GPU actually buys you per operation is measured in
[BENCHMARKS.md](BENCHMARKS.md), and it is not uniformly positive.

## Out-of-core reading

`scrust._backed` is private — it is not exported from the package root and no `pp`
function consumes it yet — but it is the only way to touch a matrix larger than
memory today:

```python
from scrust._backed import open_backed

with open_backed("atlas.h5ad") as backed:
    for start, block in backed.blocks():
        ...  # a scipy CSR block of at most `backed.block_size()` cells
```

`open_backed` refuses a non-`.h5ad` path, a missing file, and a CSC `X` (one row
block would need the whole file read). `block_size()` sizes a block against
`settings.max_memory_gb`, charging each row for both its CSR entries and the dense
buffer a caller densifies it into. What that saves is measured in
[BENCHMARKS.md](BENCHMARKS.md).
