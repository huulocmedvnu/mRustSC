# API reference

Every function takes an `AnnData` object and stores its result in a fixed slot of it.
Each section below names the slots a function reads and writes. The signatures are the
ones the installed package exposes.

Functions are grouped in five modules: `pp` (preprocessing), `tl` (tools), `metrics`,
`get` (tables) and `pl` (plots).

All 42 public functions are implemented: 19 in `pp`, 15 in `tl`, and 4 each in
`metrics` and `get`. No function raises `NotImplementedError`.

Edge cases and design choices are described under the function concerned. A test in
`tests/test_*_audit.py` checks each of them. [VALIDATION.md](VALIDATION.md) reports the
tests against a reference implementation, scanpy.

Contents: [pp](#pp-preprocessing) · [tl](#tl-tools) · [metrics](#metrics) ·
[get](#get-accessors) · [pl](#pl-plotting) · [settings](#settings) ·
[devices](#devices) · [out-of-core](#out-of-core-pppreprocess_backed)

---

## `pp`: preprocessing

#### `pp.filter_cells(adata, *, min_genes=None, min_counts=None, inplace=True)`

Removes cells that have fewer than `min_genes` expressed genes or fewer than
`min_counts` total counts. You must give at least one of the two. Giving neither
raises a `ValueError`. With `inplace=True` the function removes the cells from the
AnnData itself and returns `None`. With `inplace=False` it leaves the object unchanged
and returns a boolean mask of the cells to keep.

The function does not add `obs["n_genes"]` or `obs["n_counts"]` columns.

`min_genes` counts stored entries that are **greater than zero**.
`pp.calculate_qc_metrics` counts stored entries that are **not equal to zero**. On raw
counts the two numbers are the same. On centred data, or other data with negative
values, they differ.

#### `pp.filter_genes(adata, *, min_cells=None, min_counts=None, inplace=True)`

The same filter, applied to genes. It does not write a `var["n_cells"]` column.

#### `pp.normalize_total(adata, *, target_sum=None, inplace=True)`

Scales the counts of each cell so that they add up to `target_sum`. When `target_sum`
is `None`, the target is the median total count over cells. The result goes to
`adata.X` as a sparse CSR matrix of `float32` values. With `inplace=False` the function
returns the new matrix instead.

There are no `exclude_highly_expressed`, `max_fraction`, `key_added` or `layer`
arguments.

With `target_sum=None`, the median is taken over every cell, including cells with no
counts. This differs from the median over cells with counts only when some cell has no
counts at all.

#### `pp.log1p(adata, *, inplace=True)`

Computes `log(1 + x)` on the stored entries. It writes `adata.X` and sets
`adata.uns["log1p"] = {"base": None}`, which marks the data as log-transformed for
later steps. There is no
`base` argument. The function uses the natural logarithm only.

#### `pp.highly_variable_genes(adata, *, n_top_genes=2000, flavor="seurat", inplace=True)`

Marks the `n_top_genes` most variable genes. `flavor` is
`"seurat"` or `"cell_ranger"`. Any other value raises a `ValueError`. The function
writes three `var` columns:

| column | meaning |
| --- | --- |
| `highly_variable` | bool, the selection |
| `means` | per-gene mean |
| `dispersions_norm` | dispersion normalised within its mean bin |

The function does not write `dispersions`, `variances` or `highly_variable_rank`. The
core computes raw `dispersions`, but the Python layer discards them. Code that reads
`var["dispersions"]` will raise a `KeyError`.

`flavor="seurat"` follows Satija et al. 2015. It bins genes by mean expression and
normalises each gene's dispersion within its bin. `flavor="cell_ranger"` follows Zheng
et al. 2017. It bins genes by percentiles of their mean expression. When many genes
share the same mean, two bin edges can coincide. In that case the function raises a
`ValueError`. `pandas.cut` refuses repeated edges, and binning the genes some other way
would give a wrong answer without warning.

When a bin holds exactly **two genes**, `cell_ranger` cannot separate them. This
flavour centres each bin on its median and divides by its median absolute deviation
(MAD). With two genes in a bin, the median is their midpoint and both genes get the
value `±0.6744897501960817`. Every gene in such bins then has the same score, and the
`>= cutoff` rule decides between tied genes. Metalcyte computes this constant exactly.
A value a few ulps off would make the rule stop at a
different gene. With three or more genes per bin there are no such ties.

#### `pp.scale(adata, *, zero_center=True, max_value=None, inplace=True)`

Shifts each gene to mean zero and scales it to variance one. With `max_value`, values
beyond `±max_value` are clipped. The matrix is `float32`, but Metalcyte computes the
per-gene mean and variance in `float64`. In `float32`, the mean of a constant gene is
off by one ulp. The whole column would then become the constant `-sqrt((n-1)/n)`
instead of 0. Without zero-centering, the error would be of order `1e7`.

**The result is dense.** `adata.X` becomes a `float32` numpy array of shape
`(n_cells, n_genes)`. At 50 000 cells by 2 000 genes this takes 400 MB. Keep only the
highly variable genes before you call it.

#### `pp.pca(adata, *, n_comps=50, zero_center=True, random_state=0, device=None)`

PCA by randomised SVD. `random_state` fixes the random start, so the same seed gives the same
components. The leading components are stable. The trailing ones, with small and close
singular values, can change with the seed. [VALIDATION.md](VALIDATION.md) measures
where that starts. The function writes:

- `obsm["X_pca"]`, `(n_cells, n_comps)`
- `varm["PCs"]`, `(n_genes, n_comps)` (the core returns the transpose and the Python
  layer flips it)
- `uns["pca"]` with `variance`, `variance_ratio` and `params`

There is no `svd_solver`, `use_highly_variable` or `mask_var` argument.

#### `pp.neighbors(adata, *, n_neighbors=15, use_rep="X_pca", method="auto", random_state=0, device=None)`

Builds a k-nearest-neighbour graph weighted by UMAP's fuzzy simplicial set (McInnes et
al. 2018).
`n_neighbors` counts the cell itself, so Metalcyte searches for `n_neighbors - 1` other
cells. A value below 2 raises a `ValueError`. The function writes `obsp["distances"]`,
`obsp["connectivities"]` and `uns["neighbors"]`. The `params` entry records the search
method used, under `knn_method`.

`method` chooses how neighbours are found.

- `"exact"` compares every pair of cells. On data without duplicate cells, it finds
  the true k nearest neighbours of every cell. Its run time grows with the square of
  the number of cells: 2 s at 117 000 cells and 120 s at a million on the GPU. On a
  Metal GPU it runs the hand-written `knn` kernel. On the CPU it
  uses every core.
- `"approximate"` uses NN-descent (Dong et al. 2011), initialised from a
  random-projection forest. It runs on
  every core, and its run time grows roughly in proportion to the number of cells: 0.8 s
  at 117 000 cells and 16 s at 953 000. Recall is the fraction of the true k nearest
  neighbours that the search finds. On PCA embeddings at k = 15, recall is 0.98 at
  100 000 cells and 0.96 at 953 000. The same `random_state` gives the same graph.
- `"auto"`, the default, uses the exact search up to `pp.APPROXIMATE_FROM` cells
  (200 000) and the approximate search above that.

`use_rep="X"` uses the expression matrix itself. There is no `n_pcs` argument. To use
fewer components, slice `obsm` yourself or pass a representation you have already cut
down. See [Devices](#devices).

#### `pp.calculate_qc_metrics(adata, *, qc_vars=(), percent_top=(50, 100, 200, 500), log1p=True, inplace=True)`

Computes QC metrics per cell and per gene. `qc_vars` names boolean
`var` columns, such as `"mt"` for mitochondrial genes. Each one adds
`total_counts_<name>` and `pct_counts_<name>` to `obs`. `percent_top` is sorted before
use, and the columns follow that order. With
`log1p=True`, a `log1p_<column>` column is placed directly after each count column.

| frame | columns |
| --- | --- |
| `obs` | `n_genes_by_counts`, `total_counts`, `pct_counts_in_top_<k>_genes`, plus a pair per `qc_vars` entry |
| `var` | `n_cells_by_counts`, `mean_counts`, `pct_dropout_by_counts`, `total_counts` |

`inplace=False` returns the two tables instead of writing them. A gene counts as
detected when its entry is **not equal to zero**. `pp.filter_cells` uses a different
rule (see that function). For a cell with no counts, `percent_top` is `0/0` and is
returned as `NaN`. Metalcyte does not replace it with a misleading 0.

#### `pp.normalize_per_cell(adata, *, counts_per_cell_after=None, inplace=True)`

An older form of per-cell normalisation, kept for existing pipelines. It works like
`normalize_total` with two extra steps. It writes each cell's total count before
normalisation to `obs["n_counts"]`. It also removes cells with no counts at all. Both
happen whatever the value of `inplace`. `inplace=False` only changes whether the matrix
is returned or stored. Because empty cells are removed first, the default target is the
median over the remaining cells.

#### `pp.sqrt(adata, *, inplace=True)`

Computes `sqrt(x)` on the stored entries.

#### `pp.filter_genes_dispersion(adata, *, flavor="seurat", n_top_genes=None, inplace=True)`

An older gene-selection function, kept for existing pipelines. It uses the same
dispersions as `highly_variable_genes`, so the two functions agree by design. Without
`n_top_genes`, it keeps every gene that passes fixed thresholds:
`0.0125 < mean < 3.0` and `dispersions_norm > 0.5`. A dispersion that cannot be
computed counts as zero dispersion.

The function never removes genes from `adata`. It writes the selection to
`var["highly_variable"]`.

#### `pp.regress_out(adata, keys, *, device=None, inplace=True)`

Fits a linear regression of every gene on the `obs` columns named in `keys` and keeps
the residuals. `keys` may be a single name. **The result is dense**, as with
`pp.scale`. Metalcyte reads the sparse input one block of genes at a time, so the
result is the only full cell-by-gene array in memory. The result must fit in 60% of the
computer's memory. Metalcyte checks this before it allocates anything and raises a
`ValueError` if it does not fit.

#### `pp.combat(adata, key="batch", *, covariates=None, device=None, inplace=True)`

Removes batch effects with ComBat (Johnson et al. 2007), an empirical-Bayes method. The batches are the
categories of `obs[key]`. You can add other `obs` columns as `covariates`. The batch key
cannot also be a covariate, and each covariate may appear only once. Breaking either
rule raises a `ValueError`. The result is dense, with the same memory rule as
`regress_out`. Metalcyte reads the sparse input twice, one block of genes at a time.
Between the two passes it runs the empirical-Bayes step on per-batch sufficient
statistics. Only the result is held in
memory as a whole.

#### `pp.harmony_integrate(adata, key="batch", *, basis="X_pca", adjusted_basis="X_pca_harmony", theta=2.0, sigma=0.1, lamb=None, alpha=0.2, batch_prop_cutoff=1e-5, n_clusters=None, max_iter_harmony=10, max_iter_kmeans=20, random_state=0, device=None)`

Removes batch effects from a low-dimensional embedding with Harmony (Korsunsky et al.
2019). `key` names the `obs` column that holds the batch
labels. Metalcyte converts that column to categorical. Cells without a label raise a
`ValueError`. A missing `basis` or `key` raises a `KeyError`. The function corrects the
coordinates in `obsm[basis]` and writes them to `obsm[adjusted_basis]` as `float32`. It
also writes `uns["harmony"]`, which holds `objective` (the Harmony objective at each
iteration), `key`, `basis` and `adjusted_basis`.

With `lamb=None`, Metalcyte sets the ridge penalty for each cluster and batch to
`alpha` times the batch's soft count in that cluster. Harmony 1.2 and harmonypy 2 do the
same by default. A number fixes one penalty for every batch, as in the original
Harmony. If a batch makes up less than `batch_prop_cutoff` of a cluster, Metalcyte does
not correct that batch in that cluster.

Harmony is iterative and starts from a k-means clustering. So Metalcyte
does not reproduce `harmonypy` exactly. [VALIDATION.md](VALIDATION.md)
measures batch mixing and convergence instead. On the 953 436-cell embryo embedding with
seven experiment batches, the median per-cell cosine similarity between the two results
is 0.999.

Every step that works cell by cell runs on all CPU cores. This covers the soft cluster
assignments, the block-wise update with the diversity penalty, the objective and the
k-means start. For each cluster, Metalcyte solves the ridge regression from soft batch
counts and sums collected in one pass. It never builds an array of size cells by
clusters by components. The 953 436-cell embedding takes 9 s on the M3 Pro. harmonypy
2.1 (compiled) takes 3 s on the same input.
`device` is accepted and ignored. The matrix products are too small for the GPU to
make up for the cost of copying the data.

#### `pp.subsample(adata, fraction=None, *, n_obs=None, random_state=0, copy=False)`

#### `pp.sample(adata, fraction=None, *, n=None, replace=False, random_state=0, copy=False)`

Keep a random subset of cells. `subsample` is `sample` with `replace=False`. Give
exactly one of `fraction` and `n`. Giving both or neither raises a `TypeError`. A
fraction that cannot be met raises a `ValueError`. `copy=False` changes `adata` itself and returns `None`.

#### `pp.downsample_counts(adata, *, counts_per_cell=None, total_counts=None, random_state=0, replace=False, copy=False)`

Randomly removes counts while keeping every cell. Give exactly one of
`counts_per_cell` and `total_counts`.

#### `pp.preprocess_backed(path, *, ..., keep_hvg=False)`

Runs preprocessing on a count matrix larger than memory. Metalcyte reads
the file from disk in blocks of cells. See
[Out-of-core](#out-of-core-pppreprocess_backed) at the end of this page.

With `keep_hvg=True` the function also returns `X`. `X` holds the log-normalised values
of the highly variable genes, collected during the last pass. It is a sparse
`(n_cells, n_vars)` matrix, and the columns of the other genes are empty. This is all
that the marker-gene test needs. It uses little memory: 0.17 GB for the 953 436-cell
embryo atlas with 2 000 genes. You can then run
`tl.rank_genes_groups(adata[:, adata.var.highly_variable], groupby)` in memory in 0.3 s,
without reading the file again. For every gene, or when even the variable genes do not
fit in memory, use `tl.rank_genes_groups_backed`.

---

## `tl`: tools

#### `tl.umap(adata, *, n_components=2, min_dist=0.5, spread=1.0, n_epochs=None, random_state=0, device=None, parallel=False)`

Computes a UMAP layout (McInnes et al. 2018) of the neighbour graph in `obsp["connectivities"]` and writes it
to `obsm["X_umap"]`. `n_epochs` defaults to 200. umap-learn uses 500 for small data and
200 for large data. Metalcyte does not copy that rule. `min_dist` must lie in
`[0, 3 * spread]`.

UMAP gives a different layout for each random seed. Two layouts are compared by how
well they keep the same neighbours, and they are never equal point by point.

`parallel=True` runs the optimisation on all cores with lock-free (Hogwild) SGD, like
umap-learn's `parallel=True`. This is several times faster on
large graphs. The layout then depends on thread timing as well as on `random_state`,
so it is no longer reproducible. For that reason the default stays sequential and
reproducible.

`device` is accepted and ignored. The layout always runs on the CPU.

#### `tl.tsne(adata, *, n_pcs=50, perplexity=30.0, early_exaggeration=12.0, learning_rate=None, method="auto", random_state=0, device=None)`

Computes a t-SNE layout from the first `n_pcs` columns of `obsm["X_pca"]` and writes it
to `obsm["X_tsne"]`. `method` chooses how the layout is computed.

- `"exact"` builds the full `(n, n)` matrix of cell-to-cell similarities. This takes
  1.6 GB at 20 000 cells. Each gradient step holds three more arrays of that size, so
  memory peaks near 6.5 GB. Above 20 000 cells it raises a `ValueError` before it
  allocates any memory.
- `"fft"` is FIt-SNE (Linderman et al. 2019). Each cell
  is compared only with its `3 * perplexity` nearest neighbours. Metalcyte finds 15 of
  them with NN-descent and adds their neighbours. The push between distant cells is
  computed on a grid with one FFT per iteration, using Lagrange
  interpolation between the cells and the grid. On a Metal GPU, every per-cell step of
  an iteration runs on the GPU. These steps are the attraction between neighbours, the
  placement of cells on the grid, the spreading of their charges, the FFT, the reading
  back from the grid and the update. Between kernels, the CPU sorts the cells by
  grid box. Every sum runs in a fixed order, so the same seed gives the same output, byte
  for byte. On the CPU the FFT uses Accelerate. With 1 000 iterations on
  the M3 Pro it takes 13 s for 117 308 cells and 54 s for 953 436 (20 s and about 130 s
  on the CPU alone). It produces two-dimensional layouts only.
- `"auto"`, the default, uses the exact method up to 20 000 cells and the FFT method
  above that.

One limit applies to both:

- **The perplexity must be lower than the number of cells.** A t-SNE becomes hard to
  read once the perplexity nears a third of the number of cells. That is advice about
  the plot, and Metalcyte does not enforce it.

`learning_rate=None` uses the automatic rule `max(n / early_exaggeration / 4, 50)`.
A fixed rate of 1000 is far too large for small datasets.

> **Limitation: the exact method does not scale.** Its cost is `O(n^2)` in the number
> of cells. Above 20 000 cells it refuses to run. With
> `method="auto"` any dataset above that size uses the FFT method. For a fast overview
> of a large dataset, `tl.umap` is the better choice. See
> [PERFORMANCE.md](PERFORMANCE.md#limits).

#### `tl.rank_genes_groups(adata, groupby, *, groups="all", reference="rest", method="wilcoxon", device=None)`

Tests for differential expression between groups of cells. p-values are adjusted with
Benjamini-Hochberg. Four methods
are available: `"wilcoxon"` (the default), `"t-test"`, `"t-test_overestim_var"` and
`"logreg"`. Any other value raises a `ValueError`.

- The two t-tests differ only in the sample size used for the reference group. That
  number enters the result twice, through the standard error and through the degrees of
  freedom.
- `logreg` fits one multinomial logistic regression over every labelled cell, with at
  most 100 iterations. A named `reference` group enters the fit as one more class. It
  is not used as a comparison group.
- The Wilcoxon test does **not** correct for ties.

The function writes `uns["rank_genes_groups"]` with `params` and one record array per
field (`names`, `scores`, `logfoldchanges`, `pvals`, `pvals_adj`). Each array has one
column per group, sorted by score. p-values are `float64`. Everything else is
`float32`.

**`logreg` writes only `names` and `scores`.** A regression coefficient is not a test
statistic, so the function reports neither p-values nor fold changes for it. The core returns
`NaN` for these fields, and the Python layer removes them. A column of `NaN` is then
never mistaken for a result. Code that always reads `uns["rank_genes_groups"]["pvals"]`
will raise a `KeyError` after a `logreg` run.

`scores` is stored as `float32`. The p-values are computed from the full `float64`
score. So `2 * scipy.stats.norm.sf(abs(reported score))` does not give back the reported
p-value.

Cells outside the selected groups are removed before the test. They are not treated as
a separate label. **A group with one cell gets a result.** The rank test is well
defined with one cell (`n_active = 1`). So when filtering leaves a group with a single
cell, the function still returns its statistics and does not raise an error.

`pts` and `pts_rest` are not written. So `tl.filter_rank_genes_groups` always
recomputes the fraction of expressing cells from `X`.

`logfoldchanges` always assumes the data was log-transformed with the **natural** log.
The Rust function receives a matrix and not an AnnData, so it never sees
`uns["log1p"]["base"]`. `pp.log1p` leaves the base unset. This only matters if you set
the base by hand.

#### `tl.paga(adata, groups=None, *, model="v1.2", device=None)`

PAGA (partition-based graph abstraction, Wolf et al. 2019) summarises the cell graph in `obsp["distances"]` as a graph of cell groups. The
groups come from a categorical `obs` column. With `groups=None` it looks for
`obs["leiden"]`, then `obs["louvain"]`, which `tl.leiden` and `tl.louvain` write.
`model` accepts only `"v1.2"`.

The function writes `uns["paga"]` with `connectivities`, `connectivities_tree` (both
`float64` CSR matrices of shape `(n_groups, n_groups)`) and `groups`. It keeps any other
key already stored there, such as a saved layout position.

Every **stored** entry of `obsp["distances"]` counts as an edge, including one stored
as 0.0. Two identical cells have a distance of 0.0, and they are still neighbours. Take
120 cells of which 60 are duplicates. There, `pp.neighbors(n_neighbors=10)` stores 540
zeros out of 1080 entries. Skipping them would overstate the connectivities by up to
0.096.

The function does **not** write `uns["<groups>_sizes"]`. Where two connectivities are
equal, more than one maximum spanning tree exists, and the function returns one of
them. The `min(..., 1)` cap makes such ties common.

`device` is accepted and ignored on purpose. The work is one pass over the stored edges
into a small group-by-group matrix. It is memory-bound, and a GPU would not help.

#### `tl.rank_genes_groups_backed(path, adata, groupby, *, genes="highly_variable", groups="all", reference="rest", target_sum=1e4, gene_block=4096, block_size=None, key_added="rank_genes_groups")`

Runs `rank_genes_groups(method="wilcoxon")` on a counts file on disk, without loading
the whole matrix into memory. This was the last step of the pipeline that still needed
the full matrix. `adata` is the object that `pp.preprocess_backed` returned for `path`.
Its `obs` lists the cells kept and `obs[groupby]` their groups. Its `var` lists the
file's genes.

Metalcyte reads the file in blocks of cells. It normalises and log-transforms each block
as it reads it, so the test sees the same values as the in-memory version. For each
tested gene it collects the stored values together with the group of each cell. After
the last block it ranks them. The zeros of each gene form one tied group, whose ranks
are computed directly by formula.

`genes` is `"highly_variable"` (the flags in `adata.var`), `"all"`, or a list of gene
names. Above `gene_block` genes, the test runs on one block of genes at a time, with
one pass over the file per block. Memory then stays at the stored entries of one gene
block. The statistics equal those of `rank_genes_groups` on the same values
(`tests/test_streaming.py`). The function writes the same `uns` entry.

#### `tl.filter_rank_genes_groups(adata, *, key="rank_genes_groups", groupby=None, key_added="rank_genes_groups_filtered", min_in_group_fraction=0.25, max_out_group_fraction=0.5, min_fold_change=2.0)`

Hides the marker genes that fail the filters on the fraction of expressing cells and
the fold change. A gene passes when it is expressed in more than
`min_in_group_fraction` of the group, in less than `max_out_group_fraction` of the
other cells, and has a fold change above `min_fold_change`. The table in `uns[key]`
keeps its shape. The names of failing genes are replaced by `NaN`, and the result goes
to `uns[key_added]`. `tl.rank_genes_groups` writes no `pts`/`pts_rest`, so
the fractions are always recomputed from `X`. The stored `logfoldchanges` are reused
only when they describe the same comparison (same `groupby`, `reference="rest"`).

When this function computes the fold change itself, it does use
`uns["log1p"]["base"]`. The `logfoldchanges` written by
`tl.rank_genes_groups` always use the natural log. The two are the same unless you set
the base by hand.

#### `tl.leiden(adata, resolution=1.0, *, key_added="leiden", neighbors_key="neighbors", n_iterations=2, random_state=0, device=None)`

#### `tl.louvain(adata, resolution=1.0, *, key_added="louvain", neighbors_key="neighbors", random_state=0, device=None)`

Finds clusters of cells in the graph `obsp["connectivities"]`, with the
Leiden algorithm (Traag et al. 2019) or the Louvain algorithm (Blondel et al. 2008).
The quality measure is the Reichardt-Bornholdt configuration model (RBConfiguration) at
the given `resolution`. The function writes `obs[key_added]` as a `Categorical`, and
`uns[key_added]` with `params` and the `modularity` reached. Clusters are numbered
`0..n-1` from largest to smallest, so cluster 0 is the largest.

Leiden is a randomised search. Cluster numbers are arbitrary, so two clusterings are
compared by their agreement (for example the adjusted Rand index) or their modularity.
`metrics.modularity` scores a clustering on the graph it was found on.

`device` is accepted and ignored. Each row of the graph has about fifteen neighbours.
Starting a GPU job for each step would cost more than the step itself.

#### `tl.dendrogram(adata, groupby, *, n_pcs=50, use_rep="X_pca", key_added=None)`

Builds a hierarchical clustering tree of the groups. It uses the mean of each group over
the first `n_pcs` columns of `use_rep`, and the correlation distance between groups. It
writes `uns["dendrogram_<groupby>"]` with the keys `linkage`, `categories_ordered`,
`categories_idx_ordered`, `dendrogram_info`, `correlation_matrix`, `cor_method`,
`linkage_method`, `groupby` and `use_rep`.

**The linkage is `complete`**, and the correlation is Pearson's.
`uns["linkage_method"]` records the linkage. `complete` and `average` linkage give the same tree
on PBMC 3k. On other group means they can differ. On the six group means in
`tests/test_layout_audit.py` the leaf orders are `[4, 1, 3, 2, 0, 5]` and
`[4, 0, 5, 2, 1, 3]`, and merge heights differ by up to 0.52. **An AnnData saved by an
earlier version of Metalcyte has `linkage_method="average"` and a tree built that way.**

`groupby` must be categorical, with at least 2 categories, no unlabelled cells and no
empty category. Each problem raises its own error. At most 1024 groups are allowed. The
clustering uses the textbook `O(n^3)` algorithm. With the tens of groups this function
sees, that costs nothing. There is no `device` argument. Every array involved is
smaller than the cost of starting a GPU job.

#### `tl.draw_graph(adata, *, layout="fa", neighbors_key="neighbors", n_iterations=500, random_state=0, device=None)`

Draws the neighbour graph with ForceAtlas2 (Jacomy et al. 2014), a force-directed
layout. It writes `obsm["X_draw_graph_fa"]` and `uns["draw_graph"]["params"]`. `layout`
accepts only `"fa"`.

The graph is treated as undirected, both for the pull between connected cells and for
the masses that scale the push between all cells. If only the upper triangle of the
matrix were read, a graph stored as a lower triangle would have no pull at all. The
layout would then be pure push, with no warning. The edge list is sorted, so the result
depends only on the graph and not on the order of its stored entries.

#### `tl.embedding_density(adata, *, basis="umap", groupby=None, key_added=None)`

Estimates how densely cells are packed in the first two components of
`obsm["X_<basis>"]`, using a Gaussian kernel. The result goes to
`obs["<basis>_density_<groupby>"]` and its parameters to `uns["<covariate>_params"]`.
For a ForceAtlas2 layout, write `basis="draw_graph_fa"` (or `"fa"`).

Densities are scaled to `[0, 1]` **within** each group. So they compare cells inside one
group, and not across groups. For this reason `groupby` is stored next to the values. The function has no `device` argument and uses
`settings.device`.

#### `tl.diffmap(adata, n_comps=15, *, neighbors_key="neighbors", device=None)`

Computes a diffusion map (Coifman and Lafon 2006, Haghverdi et al. 2015) of the
neighbour graph. It writes `obsm["X_diffmap"]` and `uns["diffmap_evals"]`. The trivial
first component (eigenvalue 1) is **kept**, because `tl.dpt` reads it back and uses it.
Drop it before you plot the map.
Metalcyte never builds the full `(n_cells, n_cells)` transition matrix. For PBMC 3k at
`n_comps=15`, the operator takes 0.7 MB and the dense blocks about 4 MB.

`n_comps >= n_cells` raises a `ValueError`. `n_comps <= 2` is accepted.

A graph made of **disconnected** parts also raises an error. Each part contributes its
own eigenvalue of 1. The leading components are then not uniquely defined, and the
pseudotime between parts is infinite. Both problems would give wrong answers without any
warning. The check counts only edges with a non-zero weight, so a stored zero cannot
join two separate parts. A check on stored entries alone would miss that case. The
reasoning is in `diffusion.rs`.

#### `tl.dpt(adata, *, n_dcs=10, n_branchings=0, min_group_size=0.01, device=None)`

Computes diffusion pseudotime (DPT, Haghverdi et al. 2016) from the root cell in `uns["iroot"]`. The result goes to
`obs["dpt_pseudotime"]`. If no `X_diffmap` is stored, the function computes one with 15
components. That is `tl.diffmap`'s own default, and it does not depend on `n_dcs`. So a
later `dpt` call with a larger `n_dcs` fails. `n_branchings > 0` also detects branches
with the Haghverdi et al. 2016 algorithm and writes `obs["dpt_groups"]`.

Two edge cases are tested. A cell that is not connected to the root gets a large
**finite** pseudotime, not `inf`. You cannot find such a cell with `isinf`. And when
every cell is at the same position as the root, the function returns 0 for all of
them.

#### `tl.score_genes(adata, gene_list, *, ctrl_size=50, n_bins=25, score_name="score", random_state=0, device=None)`

Scores each cell for a gene set. The score is the mean expression of the set minus the
mean of a control set. The control genes are drawn from genes with similar expression
levels (the same expression bins). The score goes to `obs[score_name]` as `float64`,
although the computation runs in `float32`. The control genes are drawn with the numpy
legacy Mersenne Twister and a Fisher-Yates shuffle, both written in Rust. So a given
`random_state` always draws the same control genes. Genes missing from `var_names` are dropped
with a `UserWarning`. If no genes remain, the function raises a `ValueError`.

**`random_state` has no effect below about 1200 genes**. Each bin
holds about `n_genes / (n_bins - 1)` genes. Control genes are drawn only
`if ctrl_size < len(bin)`. Otherwise the whole bin is taken and nothing is drawn at
random. With the defaults `ctrl_size=50` and `n_bins=25`, the threshold is around 1200
genes. Below it, for example on a small gene panel, the score is the same whatever seed
you pass. The seed is passed through correctly. It simply has nothing to choose.

#### `tl.score_genes_cell_cycle(adata, *, s_genes, g2m_genes, device=None)`

Calls `score_genes` twice, with `ctrl_size = min(len(s_genes), len(g2m_genes))`. It
writes `obs["S_score"]`, `obs["G2M_score"]` and `obs["phase"]`. The phase is `S` unless
the G2M score is higher, in which case it is `G2M`. It is `G1` when neither score is
above its control.

#### `tl.marker_gene_overlap(adata, reference_markers, *, key="rank_genes_groups", method="overlap_count", top_n_markers=None)`

Compares the marker genes found in `uns[key]` with a reference list of markers. It
returns a table with one row per reference set and one column per group. `method` is
`overlap_count`, `overlap_coef` or `jaccard`. `top_n_markers` defaults to 100.
A value below 1 is treated as 1, with a `UserWarning`.

---

## `metrics`

All four are implemented.

#### `metrics.morans_i(adata, *, vals=None, use_graph="connectivities", device=None)`

#### `metrics.gearys_c(adata, *, vals=None, use_graph="connectivities", device=None)`

Spatial autocorrelation of a value on the neighbour graph.
`vals=None` scores every gene in `adata.X`. Otherwise `vals` names one gene or one `obs`
column, names several, or is an array. A 2-D array has shape
`(n_features, n_cells)`, the transpose of `X`. A single feature returns a number and not
an array of length 1. A sparse `vals` stays sparse. Making it dense would create the
full `(n_cells, n_genes)` array that Metalcyte avoids.

A high Moran's I and a low Geary's C both mean that neighbouring cells have similar
values. A constant feature has no statistic and returns `nan`.

#### `metrics.confusion_matrix(orig, new, data=None, *, normalize=True)`

Counts how the cells of each original label are spread over the new labels. Rows are
the original labels and columns the new ones. `orig` and `new` are arrays of labels, or
column names in `data`. Both axes use one shared set of labels. A label that appears in
only one of the two labellings still gets a row and a column. Labels are in category
order when they are categorical, and in natural sort order otherwise. `normalize`
divides each row by its own total.

#### `metrics.modularity(adata, keys, *, neighbors_key="neighbors")`

Computes the Newman modularity of the clustering in `obs[keys]`. The resolution is 1.0.
The graph is the one that
`neighbors_key` points to, so a clustering is always scored on the graph it came from.
There is no `device` argument.

---

## `get`: accessors

All four are pure Python.

#### `get.obs_df(adata, keys=(), *, obsm_keys=(), layer=None)`

Returns a table with one row per cell. Columns come from `obs` or from gene expression,
in the order you list them. A key that is both a gene name and an `obs` column raises a
`ValueError`. Metalcyte does not guess which one you meant. `obsm_keys` takes
`(key, column_index)` pairs, so `("X_pca", 0)` becomes a column named `X_pca-0`.

There is no `gene_symbols` or `use_raw` argument.

#### `get.var_df(adata, keys=(), *, varm_keys=())`

The same for genes: one row per gene, with columns from `var` or from named cells.

#### `get.rank_genes_groups_df(adata, group, *, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None)`

Turns the marker-gene results in `uns[key]` into one flat table. `group=None` returns
every group, with a `group` column. With a single group name the `group` column is
dropped. The cutoffs remove rows. A `logreg` result has only `names` and
`scores`. The function recognises it from `params["method"]`. Metalcyte can produce such
results as well as read them. The three cutoffs have nothing to filter there, so they
are skipped without an error.

#### `get.aggregate(adata, by, func, *, axis=0, layer=None, device=None)`

Groups cells (or genes, with `axis=1`) and summarises each group. `func` is one or more
of `count_nonzero`, `mean`, `median`, `sum`, `var`. `var` uses ddof 1.
The function returns a new AnnData with one layer per function and an
`obs["n_obs_aggregated"]` column.

`device` is accepted so that all functions share one signature, and it is ignored. The work
is done by scipy sparse products and per-group medians, outside the core.

---

## `pl`: plotting

`metalcyte.pl` draws figures from the AnnData entries that Metalcyte writes. It uses
matplotlib, and seaborn for colour palettes when seaborn is installed. The module loads only when you first use it, so `import metalcyte` does not load
matplotlib. Install the `plot` extra (`matplotlib>=3.7`, `seaborn>=0.13`) to use it.
Every function takes `save` (a file path for the figure) and `show` (display the
figure). `show` defaults to `None`. The figure is then displayed only when it is not
saved, so a script that saves figures never waits on an open window. `show=True` with
`save` does both.

#### `pl.pca_variance_ratio(adata, n_pcs=30, *, show=None, save=None)`

Plots the variance explained by each principal component, from
`uns["pca"]["variance_ratio"]`. It shows one bar per component and a line for the
running total.

#### `pl.embedding(adata, basis="X_umap", color=None, *, title=None, palette="plotly", cmap="plasma", vmin=None, vmax=None, frameon=False, alpha=1.0, size=None, legend_loc="right margin", legend_fontsize=8, figsize=(7, 6), dpi=300, ncols=3, xlim=None, ylim=None, device=None, show=None, save=None)`

Draws a scatter plot of `obsm[basis]`, with the points drawn on the GPU. Metal draws all
the points into one RGBA image of `figsize * dpi` pixels. When no GPU is available, the
CPU draws them. matplotlib then adds the axes, legend and colour bar around the image.
So a million cells take a few milliseconds to draw, and the figure holds one
image instead of a million separate points.

`color` is an `obs` column or a gene, or a list of them for one panel each (`ncols`
panels per row). A categorical column draws one colour per category. The colours come
from `uns[f"{color}_colors"]` when present, and from the palette otherwise. The legend
goes in the right margin. With `legend_loc="on data"`, each category's label is placed
at the median position of its cells. A numeric column or a gene draws a colour bar over
`cmap`, between `vmin` and `vmax`. `size` is the point diameter in pixels. By default it
is chosen from the number of cells. `xlim`/`ylim` zoom into a window. The GPU and CPU
drawings agree up to rounding of the colour blending (`tests/test_plotting_gpu.py`).

#### `pl.umap(adata, color=None, **kwargs)`, `pl.tsne(...)`, `pl.pca(...)`

Call `pl.embedding` on `obsm["X_umap"]`, `obsm["X_tsne"]` and the first two columns of
`obsm["X_pca"]`.

#### `pl.render_embedding(adata, basis="X_umap", color=None, *, width=2100, height=1800, size=None, alpha=1.0, palette="plotly", cmap="plasma", vmin=None, vmax=None, xlim=None, ylim=None, background="white", device=None)`

The function that draws the image behind the plots. It returns the image as a
`(height, width, 4)` `uint8` array, together with a description of the colouring. The
description holds `kind`, `levels` and `colours`, or `vmin`/`vmax` and `cmap`, plus the
`xlim`/`ylim` drawn. Use it when you want the pixels themselves, for example for a web
viewer or an image file made without matplotlib.

#### `pl.rank_genes_groups(adata, n_genes=10, n_cols=4, *, show=None, save=None)`

Draws one bar chart per group, showing the top `n_genes` marker genes by score, from
`uns["rank_genes_groups"]`.

---

## Settings

`metalcyte.settings` is a single shared settings object. It checks each value when you
assign it:

| attribute | default | effect |
| --- | --- | --- |
| `verbosity` | `Verbosity.warning` | how much `settings.log` lets through |
| `device` | `"auto"`, or `METALCYTE_DEVICE` when that is set | the device every call runs on unless it is given one |
| `max_memory_gb` | `4.0` | budget the streaming block size is derived from |
| `n_jobs` | `0` | CPU threads, `0` leaves the choice to the core |
| `chunk_size` | `0` | rows per block, `0` derives one from `max_memory_gb` |

`settings.device` accepts `"auto"`, `"cpu"`, `"gpu"` or `"metal"`. Any other value
raises an error on the line where you set it, and not several calls later. To keep a
whole session on the CPU, set `METALCYTE_DEVICE=cpu` in the environment before you
import Metalcyte.

Every function with a `device` argument defaults it to `None`. `None` means the value
of `settings.device` at the time of the call. Pass `device=` to override it for one
call. `normalize_total`, `highly_variable_genes`, `scale` and `embedding_density` have
no `device` argument and read `settings.device` themselves. The other functions
(`filter_cells`, `filter_genes`, `log1p`, `calculate_qc_metrics`,
`normalize_per_cell`, `sqrt`, `filter_genes_dispersion`, the three sampling
functions, `dendrogram`, `filter_rank_genes_groups`, `marker_gene_overlap`,
`confusion_matrix`, `modularity` and the three `get` table functions) do not use a
device at all.

Some functions accept a device and still run on the CPU. In the core they receive it
as an unused `_device` parameter. This applies to `umap`, `leiden`, `louvain`,
`normalize_total`, `highly_variable_genes`, `wilcoxon` and the two t-tests. `paga` and
`get.aggregate` also ignore it, as their sections say. The device is used by `pca`,
`neighbors`, `tsne`, `scale`, `diffmap`, `draw_graph`, `embedding_density`,
`score_genes`, `regress_out`, `combat`, `harmony_integrate`, `logreg` and the two
autocorrelation statistics.

## Devices

`device` is `"auto"` (the Metal GPU if it starts, the CPU otherwise), `"cpu"`, or
`"gpu"`/`"metal"` (an error if no Metal GPU is found). `metalcyte.gpu_available()`
tells you whether the Metal GPU started.

The CPU and GPU versions are built from the same source code in candle, a Rust tensor
library. So they run the same algorithm. They do not give bit-for-bit
identical results, though. This matters because with `"auto"`, most users run on the
GPU without having chosen it.

A GPU splits a sum across many threads, so its result can differ from the CPU's by a
few ulps. Usually this is invisible.
In `pp.neighbors` you could. The squared distance `|a - b|^2` is computed as
`|a|^2 + |b|^2 - 2 a.b`. For two identical cells this gives exactly zero on the CPU. On
Metal it leaves a tiny positive value, and the square root enlarges it to about `1e-3`.
Identical cells would then have a non-zero `rho`, the distance subtracted when the
graph weights are built, and their connectivities would no longer be 1. To prevent this,
Metalcyte sets to zero any squared distance below the precision of the calculation,
`(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)`. The two devices then agree. This
threshold is far below any real distance. On the 50 PCs of PBMC 3k it is 0.049, and the
smallest nearest-neighbour distance is 6.40. It sets 0 of 39 570 neighbours to zero.

`tests/test_device_parity.py` compares the two devices. It skips all its tests when no
Metal GPU starts, so this check runs only on a machine with a GPU.
`METALCYTE_TEST_DEVICE` (default `"cpu"`, set it to `"auto"`) chooses the device that
the audit tests run on. Both settings pass on Apple silicon.

`crates/metalcyte-gpu` contains four hand-written Metal kernels. Only one of them, `knn`, is used. `crates/metalcyte-py` depends on that
crate and sends the k-nearest-neighbour search behind `pp.neighbors` to `knn` when the
device is Metal. The candle CPU code is the fallback and the reference. `knn` repeats
the core's mean-centering and its zeroing of tiny squared distances. So
`tests/test_device_parity.py` finds the same neighbour lists on both devices. The other
three kernels (`spmm`, `tsne_gradient`, `umap_sgd`) are never called. `spmm` has no
caller that multiplies a plain sparse matrix by a dense one. `umap_sgd` uses lock-free
"Hogwild" updates and is left out on purpose. So every GPU operation on this page except
k-NN runs through candle.

In general, expect the two devices to agree to `f32` precision, and do not expect exact
equality. Be careful with any quantity that depends on two numbers cancelling exactly,
because there the two devices can differ. [PERFORMANCE.md](PERFORMANCE.md) measures
how much the GPU speeds up each operation. For some operations it is slower.

## Out-of-core: `pp.preprocess_backed`

Use this function when the count matrix is larger than memory. It reads the matrix from
disk in blocks of cells.

```python
adata = mc.pp.preprocess_backed(
    "atlas_counts.h5ad",     # counts in X, CSR, any size
    n_top_genes=2000, n_comps=50, target_sum=1e4,
    min_genes=200, min_cells=3, max_value=10.0,
    block_size=None,         # rows per block; None sizes it from settings.max_memory_gb
    obs_columns=("cell_type",),
    device=None,             # settings.device unless given
)
adata.obsm["X_pca"]          # (cells kept, n_comps)
adata.var["highly_variable"] # over every gene of the file
adata.uns["streaming"]       # block size, device, per-pass timings, cells and genes dropped
mc.pp.neighbors(adata, use_rep="X_pca"); mc.tl.umap(adata, parallel=True); mc.tl.leiden(adata)
```

Full signature: `pp.preprocess_backed(path, *, n_top_genes=2000, n_comps=50,
target_sum=1e4, min_genes=200, min_cells=3, max_value=10.0, flavor="seurat",
block_size=None, random_state=0, device=None, obs_columns=(), progress=None)`.
`progress` is an optional function `(stage, seconds)`. Metalcyte calls it when each
stage finishes, with the time the stage took.

The returned `AnnData` has no `X`. `obs` holds `n_genes`, `total_counts` and the
requested `obs_columns`. `var` holds the `highly_variable`, `means` and
`dispersions_norm` columns for every gene in the file. `obsm["X_pca"]`, `varm["PCs"]`
and `uns["pca"]` are written as `pp.pca` writes them. Cells that fail `min_genes` are
not in `obs`.

The function reads the blocks of `X` from disk four times:

1. The first pass removes cells below `min_genes`. It normalises and log-transforms each
   block and adds up, per gene, the number of cells that express it and the sums needed
   by `highly_variable_genes`. The variable genes are then chosen from those sums.
2. The second pass collects the mean and variance of the log values of the variable
   genes.
3. The third pass scales each block with those values into a dense array. It adds up
   the `(genes, genes)` scatter matrix on the device. PCA comes from the
   eigendecomposition of this covariance matrix.
4. The fourth pass projects each block onto those axes.

Peak memory is one block plus the embedding. `tests/test_streaming.py` checks the
result against an exact in-memory PCA. [PERFORMANCE.md](PERFORMANCE.md#a-million-cells-on-18-gb)
describes the million-cell run.

`normalize_total` and `log1p` also accept an AnnData opened from disk in backed mode
(`anndata.read_h5ad(path, backed="r+")`). They rewrite `X` on disk one block at a time.
The block reader is `metalcyte._backed.open_backed`:

```python
from metalcyte._backed import open_backed

with open_backed("atlas.h5ad") as backed:
    for start, block in backed.blocks():
        ...  # a scipy CSR block of at most `backed.block_size()` cells
```

`open_backed` refuses a file that is not `.h5ad`, a missing file, and an `X` stored as
CSC. With CSC, reading one block of cells would require reading the whole file.
`block_size()` chooses the block size from `settings.max_memory_gb`. For each cell it
counts both its sparse CSR entries and the dense array a caller may convert the block
into. [PERFORMANCE.md](PERFORMANCE.md) measures how much memory this saves.
