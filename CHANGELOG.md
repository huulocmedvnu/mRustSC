# Changelog

## 0.3.0 (2026-10-07)

First release on PyPI. Ready-built packages (wheels) are available for macOS on Apple silicon,
Python 3.11 to 3.13.

### Added

- A faster, approximate neighbour search for large datasets. It uses NN-descent, which improves a
  rough first guess of each cell's neighbours step by step. A random-projection forest makes the
  first guess. `pp.neighbors(method="auto")` switches to it above 200 000 cells.
- A fast t-SNE for more than 20 000 cells (FIt-SNE). It approximates the forces between cells on
  a grid with the fast Fourier transform (FFT). All main steps of each iteration run on the GPU: the
  attractive forces, spreading values onto the grid, the FFT convolution, reading values back from
  the grid, and the position update.
- Harmony integration, running on all CPU cores. It estimates its penalty settings as Harmony 1.2
  does.
- Batch correction (`pp.regress_out`, `pp.combat`) that works on blocks of genes, with no fixed
  memory limit.
- `pl.embedding`, `pl.umap`, `pl.tsne` and `pl.pca` now draw the points on the GPU through Metal.
  They use plotly's colour palettes for categories and `plasma` for numbers by default.
- `tl.rank_genes_groups_backed`, a Wilcoxon marker test that reads the counts file from disk in
  blocks.
- `pp.preprocess_backed(keep_hvg=True)` keeps the log-normalised values of the highly variable
  genes in memory, so you can run a marker test on them.

### Changed

- The package and its crates are named Metalcyte (formerly scrust, then Silicell).
- Plotting functions take `show=None` by default. A figure opens in a window only when it is not
  saved. Scripts that pass `save` no longer stop and wait for a window to close.
- A plot with one panel keeps its size when the legend is long. The legend wraps into columns of
  24 entries.
- The minimum Rust version is 1.88.

### Performance (Apple M3 Pro, 18 GB)

- 117 308 cells, counts to clusters: 12 s on the GPU, for 168 J of energy.
- 953 436 cells: 103 s on the GPU with the approximate neighbour search.
- 4 062 980 cells from a 28.9 GB counts file: 538 s, with peak memory of about 9 GB and about
  6 kJ of energy.

### Known issues

- The approximate neighbour search can give slightly different neighbours on each run. So the
  number of Leiden clusters can vary a little between runs on the same file.

## 0.2.0

Preprocessing and PCA for datasets larger than memory (`pp.preprocess_backed`), reading the counts
from disk in blocks. GPU programs (Metal kernels) for the neighbour search and the Wilcoxon test.
