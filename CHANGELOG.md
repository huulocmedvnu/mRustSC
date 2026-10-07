# Changelog

## 0.3.0 (2026-10-07)

First release on PyPI. Wheels are built for macOS on Apple silicon, Python 3.11 to 3.13.

### Added

- Approximate neighbour search (NN-descent seeded by a random-projection forest).
  `pp.neighbors(method="auto")` switches to it above 200 000 cells.
- FFT-accelerated t-SNE (FIt-SNE interpolation) above 20 000 cells, with the attractive term,
  the grid spreading, the FFT convolution, the gather and the update on the GPU.
- Harmony integration across all cores, with Harmony 1.2 penalty estimation.
- Batch correction (`pp.regress_out`, `pp.combat`) by gene blocks, with no fixed memory cap.
- A Metal point rasteriser behind `pl.embedding`, `pl.umap`, `pl.tsne` and `pl.pca`, with
  plotly's qualitative palettes and `plasma` by default.
- `tl.rank_genes_groups_backed`, a Wilcoxon marker test streamed over the counts file on disk.
- `pp.preprocess_backed(keep_hvg=True)` keeps the log-normalised variable genes for an
  in-memory marker test.

### Changed

- The package and its crates are named Metalcyte (formerly scrust, then Silicell).
- Plotting functions take `show=None` by default: a figure is displayed only when it is not
  saved, so scripts that pass `save` do not block on a window.
- A single plot panel keeps its size under a long legend, which wraps into columns of 24.
- The minimum Rust version is 1.88.

### Performance (Apple M3 Pro, 18 GB)

- 117 308 cells, counts to clusters: 12 s on Metal, for one seventh of scanpy's energy.
- 953 436 cells: 103 s on Metal with the approximate neighbour search.
- 4 062 980 cells from a 28.9 GB counts file: 538 s at a peak of about 9 GB, for about 6 kJ.

### Known issues

- The approximate neighbour search is not bit-reproducible across runs, so the Leiden cluster
  count can vary slightly between runs on the same file.

## 0.2.0

Out-of-core preprocessing and PCA (`pp.preprocess_backed`), Metal kernels for the neighbour
search and the Wilcoxon test, and scanpy-compatible results on the bone-marrow atlas.
