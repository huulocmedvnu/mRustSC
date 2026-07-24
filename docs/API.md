# API reference

Every function takes an `AnnData` and writes into the slot scanpy uses, so scanpy's
plotting reads the result unchanged. Signatures below are the ones the installed
package actually exposes; anything scanpy has that scrust does not is listed too,
with what the call does today.

Names are grouped as scanpy groups them: `pp`, `tl`, `metrics`, `get`.

**Status at 0.2.0**: all 40 public functions have an implementation behind them —
17 in `pp`, 15 in `tl`, 4 each in `metrics` and `get`. The feature branches that owed
the other 24 are merged, `grep -rn 'todo!' crates/` returns nothing, and as of v0.2.0 the
package has **no `NotImplementedError` left**: `tl.dpt(n_branchings>0)`, the last one, now
runs native branch detection. v0.2.0 also adds `pp.harmony_integrate` (Harmony batch
integration, mirroring `sc.external.pp.harmony_integrate`).

Where scrust and scanpy disagree the divergence is stated under the function it
affects, and each one is pinned by a test in `tests/test_*_audit.py` rather than left
to drift.

Contents: [pp](#pp--preprocessing) · [tl](#tl--tools) · [metrics](#metrics) ·
[get](#get--accessors) · [settings](#settings) · [devices](#devices) ·
[out-of-core](#out-of-core-reading)

---

## `pp` — preprocessing

#### `pp.filter_cells(adata, *, min_genes=None, min_counts=None, inplace=True)`

Drops cells with fewer than `min_genes` expressed genes or fewer than `min_counts`
total counts. At least one of the two is required; passing neither is a
`ValueError`. With `inplace=True` the AnnData is subset in place and `None` is
returned; with `inplace=False` you get the boolean keep-mask and the object is left
alone.

Unlike scanpy, it does not add `obs["n_genes"]` or `obs["n_counts"]` — it returns
the mask and nothing else.

`min_genes` counts stored entries **greater than zero**, while
`pp.calculate_qc_metrics` counts stored entries **not equal to zero**. On counts the
two are the same number; on centred or otherwise signed data they are not. Both
follow scanpy, which draws the same distinction between the two modules.

#### `pp.filter_genes(adata, *, min_cells=None, min_counts=None, inplace=True)`

The same, over genes. Also does not write the `var["n_cells"]` column scanpy adds.

#### `pp.normalize_total(adata, *, target_sum=None, inplace=True)`

Scales each cell to `target_sum` counts, or to the median cell count when
`target_sum` is `None`. Writes `adata.X` (CSR, `float32`), or returns the new matrix
under `inplace=False`.

scanpy's `exclude_highly_expressed`, `max_fraction`, `key_added` and `layer` are not
accepted.

With `target_sum=None` scanpy picks the median differently depending on how `X` is
stored: `np.median` over *every* cell for CSR, and over only the cells with a
non-zero count for anything else (`_normalization.py:93-117`). The core always takes
a CSR matrix, so it always follows the first rule. The two agree unless some cell has
no counts at all.

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

At exactly **two genes per bin** `cell_ranger` is degenerate and the selection can
differ from scanpy's. The flavour centres each bin on its median and scales by its
MAD; with two dispersions in a bin the median is their midpoint and both normalise to
exactly `±0.6744897501960817`, so every gene ties. The core's arithmetic is right and
scanpy's lands a few ulps either side of that constant, which is enough for its
`>= cutoff` selection to stop at a different gene. From three genes per bin upward
the two agree exactly.

#### `pp.scale(adata, *, zero_center=True, max_value=None, inplace=True)`

Centres each gene and scales it to unit variance, clipping at `±max_value` when
given. The per-gene mean and variance are reduced in `float64` even though the matrix
is `float32`: reduced in `float32`, a constant gene's mean came out one ulp wide and
the whole column was returned as a constant `-sqrt((n-1)/n)` instead of 0 — an error
of order `1e7` without zero-centering.

**The result is dense.** `adata.X` becomes a `float32` numpy array of shape
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

#### `pp.calculate_qc_metrics(adata, *, qc_vars=(), percent_top=(50, 100, 200, 500), log1p=True, inplace=True)`

Per-cell and per-gene QC metrics under scanpy's column names. `qc_vars` names
boolean `var` columns — `"mt"` for the mitochondrial genes — and each one adds
`total_counts_<name>` and `pct_counts_<name>` to `obs`. `percent_top` is sorted
before use, as scanpy sorts it, and the columns come out in that order. With
`log1p=True` a `log1p_<column>` is inserted directly after each of the count columns.

| frame | columns |
| --- | --- |
| `obs` | `n_genes_by_counts`, `total_counts`, `pct_counts_in_top_<k>_genes`, plus a pair per `qc_vars` entry |
| `var` | `n_cells_by_counts`, `mean_counts`, `pct_dropout_by_counts`, `total_counts` |

`inplace=False` returns the two frames instead of writing them. Occupancy counts
entries **not equal to zero**, which is not the rule `pp.filter_cells` uses — see
there. `percent_top` for a cell with no counts is `0/0` and comes back `NaN` rather
than a plausible-looking 0.

#### `pp.normalize_per_cell(adata, *, counts_per_cell_after=None, inplace=True)`

scanpy's legacy per-cell normalisation. It is `normalize_total` with two habits of
its own, both kept: the pre-normalisation totals are written to `obs["n_counts"]`,
and cells with no counts at all are dropped rather than left unscaled. Both happen
whatever `inplace` says — only the matrix is returned instead of stored. Dropping the
empty cells first is also what makes the default target the median over the
*remaining* cells, which is how scanpy takes it.

#### `pp.sqrt(adata, *, inplace=True)`

`sqrt(x)` on the stored entries.

#### `pp.filter_genes_dispersion(adata, *, flavor="seurat", n_top_genes=None, inplace=True)`

The pre-`highly_variable_genes` selection scanpy still ships, over the same
dispersions, so the two agree by construction. Without `n_top_genes` it applies
scanpy's cut-off rule instead of a fixed count: `0.0125 < mean < 3.0` and
`dispersions_norm > 0.5`, with an undefined dispersion treated as no dispersion.

Unlike scanpy's version it never subsets `adata` — the flag goes to
`var["highly_variable"]`, which is scanpy's `subset=False` behaviour.

#### `pp.regress_out(adata, keys, *, device=None, inplace=True)`

Regresses every gene on the `obs` columns named by `keys` and keeps the residuals.
`keys` may be a single name. **The result is dense**, as `pp.scale`'s is.

`device=None` means `settings.device`, not `"auto"` — this and `pp.combat` are the
only two functions that take a device *and* follow the setting by default.

#### `pp.combat(adata, key="batch", *, covariates=None, device=None, inplace=True)`

Empirical-Bayes batch correction over the categorical `obs[key]`, with optional
`covariates` in the design. The batch key cannot also be a covariate, and covariates
must be unique; both are `ValueError`. Dense result, and the same `device=None` rule.

#### `pp.subsample(adata, fraction=None, *, n_obs=None, random_state=0, copy=False)`

#### `pp.sample(adata, fraction=None, *, n=None, replace=False, random_state=0, copy=False)`

Keep a random subset of cells. `subsample` is `sample` with `replace=False`. Exactly
one of `fraction` and `n` is required — both or neither is a `TypeError`, and a
fraction that cannot be honoured is a `ValueError`, which is the pair of exception
types scanpy raises. `copy=False` subsets in place and returns `None`.

#### `pp.downsample_counts(adata, *, counts_per_cell=None, total_counts=None, random_state=0, replace=False, copy=False)`

Thins the counts themselves rather than the cells. Exactly one of `counts_per_cell`
and `total_counts` is required.

---

## `tl` — tools

#### `tl.umap(adata, *, n_components=2, min_dist=0.5, spread=1.0, n_epochs=None, random_state=0, device="auto")`

Lays out `obsp["connectivities"]` and writes `obsm["X_umap"]`. `n_epochs` defaults
to 200 — umap-learn's rule of 500 for small data and 200 for large is not
reproduced. `min_dist` must lie in `[0, 3 * spread]`.

UMAP does not reproduce itself across seeds, so "agrees with scanpy" is a band, not
an equality; the measured numbers are in [VALIDATION.md](VALIDATION.md).

`device` is accepted and ignored: the layout runs on the CPU whichever device you
name.

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

Differential expression per group with Benjamini-Hochberg correction. All four of
scanpy's non-deprecated methods work: `"wilcoxon"` (the default),
`"t-test"`, `"t-test_overestim_var"` and `"logreg"`. Anything else is a `ValueError`.

- The two t-tests differ only in which sample size stands in for the reference, which
  reaches the answer twice over — through the standard error and through the degrees
  of freedom.
- `logreg` is one multinomial fit over every labelled cell at sklearn's `max_iter=100`,
  which scanpy does not override. A named `reference` takes part as a further class
  rather than as a comparison.
- Wilcoxon ties are **not** corrected, matching scanpy's `tie_correct=False` default.

Writes `uns["rank_genes_groups"]` with `params` and one record array per field —
`names`, `scores`, `logfoldchanges`, `pvals`, `pvals_adj` — each with one column per
group, ranked by score. p-values are `float64`; everything else `float32`.

**`logreg` writes only `names` and `scores`.** A coefficient is not a test statistic,
so scanpy reports neither p-values nor fold changes for it; the core returns `NaN`
there and the wrapper drops the fields rather than passing a `NaN` column off as a
result. Code that reads `uns["rank_genes_groups"]["pvals"]` unconditionally will
`KeyError` after a `logreg` run.

`scores` is stored as `float32`, but the p-values are computed from the unrounded
`float64` score, so `2 * scipy.stats.norm.sf(abs(reported score))` is not the reported
p-value. scanpy does the same thing for the same reason.

Cells outside the selected groups are dropped before the call rather than encoded as
a label. **With a one-cell group scanpy raises and the core computes.** The rank test
is well defined at `n_active = 1`; scanpy's guard is protecting its own per-group
variance, which the core does not need, so a group that filtering has reduced to a
single cell gets a number here where scanpy gives an error.

`pts` and `pts_rest` are not written, so `tl.filter_rank_genes_groups` always
recomputes the expression fractions from `X`.

`logfoldchanges` always undoes a **natural** log. scanpy uses
`expm1(x * log(base))` when `uns["log1p"]["base"]` is set; the Rust binding takes a
matrix rather than an AnnData and never sees that key. `sc.pp.log1p` and `pp.log1p`
both leave the base unset, so this only bites if you set it by hand.

#### `tl.paga(adata, groups=None, *, model="v1.2", device="auto")`

Coarse-grains `obsp["distances"]` over a categorical `obs` column. `groups=None`
looks for `obs["leiden"]` then `obs["louvain"]`, both of which `tl.leiden` and
`tl.louvain` now write. `model` accepts only `"v1.2"`.

Writes `uns["paga"]` with `connectivities`, `connectivities_tree` (both `float64`
CSR of shape `(n_groups, n_groups)`) and `groups`, preserving any other key already
in that slot, such as a saved layout position.

Every **stored** entry of `obsp["distances"]` is an edge, including one stored as
0.0: scanpy binarises the graph before it counts (`ones.data = np.ones(len(ones.data))`,
`_paga.py:182-183`), so a stored zero counts too. The core skipped those until this
session. On 120 cells of which 60 were duplicates, `sc.pp.neighbors(n_neighbors=10)`
stores 540 zeros out of 1080 entries and the connectivities were overstated by up to
0.096. It now agrees with scanpy to about 2e-8.

Two divergences remain. `uns["<groups>_sizes"]` is **not** written, which
`sc.tl.paga` does write and `sc.pl.paga` reads to size the nodes. And the spanning
tree differs from scanpy's where connectivities tie, which the `min(..., 1)` cap makes
routine; both are valid maximum spanning trees and their total weight agrees to 1e-6.

`device` is accepted and deliberately ignored: this is one memory-bound pass over
the stored edges into a group-sized matrix, and there is nothing for a GPU to do.

#### `tl.filter_rank_genes_groups(adata, *, key="rank_genes_groups", groupby=None, key_added="rank_genes_groups_filtered", min_in_group_fraction=0.25, max_out_group_fraction=0.5, min_fold_change=2.0)`

Blanks out the genes that fail the expression-fraction filters, keeping the shape of
`uns[key]` and replacing the failing names with `NaN` — which is what scanpy's
plotting expects to find. Because `tl.rank_genes_groups` writes no `pts`/`pts_rest`,
the fractions are always recomputed from `X`; the stored `logfoldchanges` are reused
only when they describe the comparison being filtered (same `groupby`,
`reference="rest"`).

The fold change it computes itself does honour `uns["log1p"]["base"]`, as scanpy's
does — unlike the `logfoldchanges` written by `tl.rank_genes_groups`, which are always
natural-log. The two coincide unless you have set the base by hand.

#### `tl.leiden(adata, resolution=1.0, *, key_added="leiden", neighbors_key="neighbors", n_iterations=2, random_state=0, device="auto")`

#### `tl.louvain(adata, resolution=1.0, *, key_added="louvain", neighbors_key="neighbors", random_state=0, device="auto")`

Community detection on `obsp["connectivities"]` under the RBConfiguration objective at
`resolution`, which is what `scanpy.tl.leiden` drives `leidenalg` with. Writes `obs[key_added]` as a `Categorical` and
`uns[key_added]` with `params` and the achieved `modularity`. Communities are
numbered `0..n-1` by descending size, so listing the categories in numeric order is
already scanpy's natural sort.

Leiden is a randomised local search, so labels are not comparable element-wise with
scanpy's even at the same seed; what is comparable is the modularity, and
`metrics.modularity` scores a labelling on the graph it was found on.

`device` is accepted and ignored (`_device` in `cluster.rs`): the graph has about
fifteen neighbours a row, and a dispatch per move costs more than the move.

#### `tl.dendrogram(adata, groupby, *, n_pcs=50, use_rep="X_pca", key_added=None)`

Hierarchical clustering of the group means over the first `n_pcs` columns of
`use_rep`, on the correlation distance. Writes `uns["dendrogram_<groupby>"]` with the
keys `pl.dendrogram`, `pl.matrixplot`, `pl.dotplot` and `pl.correlation_matrix` read:
`linkage`, `categories_ordered`, `categories_idx_ordered`, `dendrogram_info`,
`correlation_matrix`, `cor_method`, `linkage_method`, `groupby`, `use_rep`.

**The linkage is `complete`, which changed this release.** It used to be `average`,
and `uns["linkage_method"]` recorded that; `sc.tl.dendrogram` defaults to `complete`,
so the old value was a silent divergence rather than a choice. The two methods agree
on PBMC 3k, which is as far as the original justification reached — on centroids
where they part company the leaf order differs outright (`[4, 1, 3, 2, 0, 5]` and
`[4, 0, 5, 2, 1, 3]` on the six centroids in `tests/test_layout_audit.py`) and merge
heights differ by up to 0.52. **A stored AnnData
written by an earlier version carries `linkage_method="average"` and a tree to
match.**

`groupby` must be categorical with at least 2 categories, no unlabelled cells and no
empty category; each is a distinct error. At most 1024 groups — the clustering is the
textbook `O(n^3)` form, which is free at the tens of groups this is ever called with.
No `device`: every tensor involved is smaller than the dispatch that would launch it.

#### `tl.draw_graph(adata, *, layout="fa", neighbors_key="neighbors", n_iterations=500, random_state=0, device="auto")`

ForceAtlas2 over the neighbour graph. Writes `obsm["X_draw_graph_fa"]` and
`uns["draw_graph"]["params"]`. `layout` accepts only `"fa"` — the igraph layouts
scanpy also offers are a different package, not a different argument.

The graph is read as undirected, both for the attractive forces and for the masses
that scale repulsion. Reading only the upper triangle, as this did until this
session, gives a lower-triangular graph no attraction at all and a silent
pure-repulsion layout. The edge list is sorted, so the result is a function of the
graph rather than of the order its entries happened to be stored in.

#### `tl.embedding_density(adata, *, basis="umap", groupby=None, key_added=None)`

Gaussian kernel density of the cells in the first two components of `obsm["X_<basis>"]`,
written to `obs["<basis>_density_<groupby>"]` with its parameters in
`uns["<covariate>_params"]`. `basis="fa"` is spelled `draw_graph_fa`, as scanpy
spells it.

Densities are scaled to `[0, 1]` **within** each group, so they compare cells inside a
group and not across groups — scanpy's convention, and the reason the `groupby` is
stored beside the values. Takes no `device` argument and follows `settings.device`.

#### `tl.diffmap(adata, n_comps=15, *, neighbors_key="neighbors", device="auto")`

Diffusion map of the neighbour graph. Writes `obsm["X_diffmap"]` and
`uns["diffmap_evals"]`. The trivial first component is **kept**, as scanpy keeps it —
only `sc.pl.diffmap` drops it, and `tl.dpt` reads it back and uses it. The
`(n_cells, n_cells)` transition matrix is never formed: PBMC 3k at `n_comps=15` costs
0.7 MB for the operator and about 4 MB of dense blocks.

Two divergences. `n_comps >= n_cells` is a `ValueError` where scanpy clamps to
`n_cells - 1`, and `n_comps <= 2` is accepted where scanpy refuses.

A **disconnected** graph is also an error. Every component contributes its own
eigenvalue 1, so the leading eigenspace is degenerate and its basis arbitrary, and
pseudotime between components is infinite — both are silent wrong answers. The guard
counts only edges that carry weight; walking the sparsity pattern, as it did until
this session, let explicitly stored zeros bridge separate components and returned a
degenerate map with spectrum `[1.0000001, 1.0, ...]` instead of raising. This is
deliberately stricter than scipy and scanpy, and the reasoning is in `diffusion.rs`.

#### `tl.dpt(adata, *, n_dcs=10, n_branchings=0, min_group_size=0.01, device="auto")`

Diffusion pseudotime from `uns["iroot"]`, written to `obs["dpt_pseudotime"]`. Without
a stored `X_diffmap` it computes one at 15 components, which is `tl.diffmap`'s own
default rather than `n_dcs` — scanpy does the same, and a later `dpt` with a larger
`n_dcs` fails there too. `n_branchings > 0` runs native branch detection (a port of
scanpy's Haghverdi 2016 algorithm) and writes `obs["dpt_groups"]`; see
[VALIDATION.md](VALIDATION.md) for the ARI-1.0 parity check.

Two divergences from scanpy, both pinned. A cell outside the root's component gets an
ordinary large **finite** pseudotime where scanpy writes `inf`, so it is not something
a caller can spot with `isinf`. And when every cell coincides with the root it returns
0 for all of them where scanpy returns `NaN`.

#### `tl.score_genes(adata, gene_list, *, ctrl_size=50, n_bins=25, score_name="score", random_state=0, device="auto")`

Mean expression of a gene set minus the mean of a control set drawn from the same
expression bins, written to `obs[score_name]` as `float64` — scanpy's dtype, though
the arithmetic is `float32` on both sides. The legacy Mersenne Twister and Fisher-Yates
shuffle are reimplemented in Rust so the control draw is scanpy's *exactly*, not a
draw with the same distribution. Genes missing from `var_names` are dropped with a
`UserWarning`; an empty result is a `ValueError`.

**`random_state` does nothing below about 1200 genes**, on both sides. Bins hold about
`n_genes / (n_bins - 1)` genes and `ctrl_size` are drawn only `if ctrl_size < len(bin)`;
below that the whole bin is taken and there is no draw. At the defaults `ctrl_size=50`
and `n_bins=25` that threshold is around 1200 genes — under it, on a subsetted panel
or a marker matrix, the score is deterministic whatever seed you pass. The seed is
wired up; there is nothing for it to do.

#### `tl.score_genes_cell_cycle(adata, *, s_genes, g2m_genes, device="auto")`

Two `score_genes` calls at `ctrl_size = min(len(s_genes), len(g2m_genes))`, writing
`obs["S_score"]`, `obs["G2M_score"]` and `obs["phase"]` — `S` unless G2M outscores it,
and `G1` when neither programme beats its control.

#### `tl.marker_gene_overlap(adata, reference_markers, *, key="rank_genes_groups", method="overlap_count", top_n_markers=None)`

Overlap between the called markers in `uns[key]` and a reference mapping, as a frame
of reference sets by called groups. `method` is `overlap_count`, `overlap_coef` or
`jaccard`; `top_n_markers` defaults to scanpy's 100, and a value below 1 is treated as
1 with a `UserWarning`.

---

## `metrics`

All four are implemented.

#### `metrics.morans_i(adata, *, vals=None, use_graph="connectivities", device="auto")`

#### `metrics.gearys_c(adata, *, vals=None, use_graph="connectivities", device="auto")`

Spatial autocorrelation over the neighbour graph. `vals=None` scores every gene of
`adata.X`; otherwise it names one gene or one `obs` column, names several, or is an
explicit array. As in scanpy an explicit 2-D array is `(n_features, n_cells)` — the
transpose of `X` — and a single feature returns a scalar rather than a length-1 array.
A sparse `vals` stays sparse: densifying it would materialise exactly the
`(n_cells, n_genes)` intermediate the core avoids.

High Moran's I and low Geary's C both mean strong spatial correlation. A constant
feature has no statistic and comes back `nan`.

#### `metrics.confusion_matrix(orig, new, data=None, *, normalize=True)`

Contingency table of two labellings, rows the original labels and columns the new
ones. `orig` and `new` are label arrays, or column names in `data`. One shared label
set covers both axes, so a label occurring in only one of the two labellings still
gets a row and a column. Axes are in category order where the labels are categorical
and natural-sort order otherwise. `normalize` divides each row by its own total.

#### `metrics.modularity(adata, keys, *, neighbors_key="neighbors")`

Newman modularity — resolution 1.0, scanpy's default when it hands a graph to igraph
— of `obs[keys]` on the connectivities `neighbors_key` points at, so a labelling and
the graph it was found on are always scored together. Takes no `device`.

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
filter rows. A `logreg` result — recognised by `params["method"]`, and now something
scrust produces as well as reads — has only `names` and `scores`; the three cutoffs
have nothing to filter on there and are skipped rather than raising.

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

**Known discrepancy.** `settings.device` is *not* read by every function. Three
behaviours are in the package:

| behaviour | functions |
| --- | --- |
| follows `settings.device` | `normalize_total`, `highly_variable_genes`, `scale`, `embedding_density`, and `regress_out`/`combat` at their `device=None` default |
| declares `device="auto"` itself, so the setting is ignored unless you pass `device=` at the call | `pca`, `neighbors`, `umap`, `tsne`, `rank_genes_groups`, `leiden`, `louvain`, `diffmap`, `dpt`, `paga`, `draw_graph`, `score_genes`, `score_genes_cell_cycle`, `morans_i`, `gearys_c`, `get.aggregate` |
| takes no device at all | `filter_cells`, `filter_genes`, `log1p`, `calculate_qc_metrics`, `normalize_per_cell`, `sqrt`, `filter_genes_dispersion`, the three sampling functions, `dendrogram`, `filter_rank_genes_groups`, `marker_gene_overlap`, `confusion_matrix`, `modularity`, the three `get` frames |

To pin a device today, pass it per call.

Accepting a device is not the same as using one. The core binds it to a `_device`
parameter — that is, runs on the CPU whatever you ask for — in `umap`, `leiden`,
`louvain`, `normalize_total`, `highly_variable_genes`, `wilcoxon` and the two t-tests;
`paga` and `get.aggregate` do the same and say so above. It is genuinely used by
`pca`, `neighbors`, `tsne`, `scale`, `diffmap`, `draw_graph`, `embedding_density`,
`score_genes`, `regress_out`, `combat`, `logreg` and the two autocorrelation
statistics.

## Devices

`device` is `"auto"` (Metal if a device initialises, CPU otherwise), `"cpu"`, or
`"gpu"`/`"metal"` (an error if no Metal device is found). `scrust.gpu_available()`
reports whether Metal came up.

The CPU and GPU paths are the same candle source. That makes them the same algorithm;
it does not make them bit-identical, and the difference is worth knowing about because
`"auto"` means most callers are on the GPU without having chosen it.

Floating-point addition is not associative, so a reduction that a GPU splits across
threads lands a few ulps away from the sequential one. Usually that is invisible. It
was not in `pp.neighbors`: `|a - b|^2` is computed as `|a|^2 + |b|^2 - 2 a.b`, which
cancels to exactly zero for two identical cells on the CPU but left a sub-ulp positive
on Metal, and the square root amplified that to `1e-3`. Identical cells then had a
non-zero `rho`, which is subtracted when the fuzzy simplicial set is built, so their
connectivities stopped being 1. Squared distances below the expansion's own resolution
— `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` — are now snapped to zero, so the
two devices agree. The floor is far below anything real: on PBMC 3k's 50 PCs it is
0.049 against a *smallest* nearest-neighbour distance of 6.40, and it snaps 0 of
39 570 neighbours.

`tests/test_device_parity.py` holds the two devices against each other, but it skips
entirely where no Metal device comes up — which includes GitHub's hosted macOS
runners. **CI passing is not evidence the GPU path works**; that check only happens on
a machine with a GPU. `SCRUST_TEST_DEVICE` (default `"cpu"`, set it to `"auto"`)
selects the device the audit suite runs against, and both legs pass locally.

Of the four hand-written Metal kernels in `crates/scrust-gpu`, one — `knn` — is now on
the call path: `crates/scrust-py` depends on the crate and routes a Metal caller's k-NN
(behind `pp.neighbors`) to it, with the candle CPU path as the fallback and the oracle.
It reproduces `neighbors::knn`'s mean-centering and squared-distance snapping, so
`tests/test_device_parity.py` holds the two devices' neighbour lists equal. The other
three (`spmm`, `tsne_gradient`, `umap_sgd`) are not reachable — `spmm` has no plain
sparse×dense caller and `umap_sgd` is Hogwild and left unwired on purpose — so every GPU
operation described here except k-NN goes through candle.

The general rule to work from: expect agreement to `f32` precision, not equality, and
treat any quantity that is *defined* by an exact cancellation as a place where the two
can part company. What the GPU actually buys you per operation is measured in
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
