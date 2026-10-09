# Changelog

## 0.3.4 (2026-10-09)

- `pp.preprocess_backed` is about twice as fast at four million cells (70 s against 152 s), with
  bit-identical results. The second pass keeps the variable-gene blocks in memory when they fit
  in half of the machine's memory (0.64 GB at four million cells), so the file is read twice.
  `hvg_in_memory=False` restores four reads.
- For an uncompressed `.h5ad`, the Rust core reads each block straight from the file's chunks on
  all cores, and reads the next block while the current one is processed. Compressed files are
  still read through anndata.
- `benches/prepare_counts.py` writes gene indices as int32. The four-million-cell counts file
  shrinks from 28.9 GB to 19.4 GB.

## 0.3.3 (2026-10-09)

- `tl.umap(parallel=True)` now optimises the layout on the GPU when one is available. It is about
  9 times faster than all CPU cores at a million cells (4.6 s against 41.7 s) and 11 times at
  four million (19.8 s against 222.6 s), with the same layout quality. `device="cpu"` keeps it on
  the CPU. The default `parallel=False` is unchanged: sequential, deterministic, on the CPU.
- Fixed: the GPU UMAP kernel failed to compile from the Python wheel, because the wheel's macOS 11
  target selects an older Metal language version. Kernels can now request a version.

## 0.3.2 (2026-10-09)

Maintenance release. Results and performance are unchanged.

- The Metal Shading Language source of every GPU kernel now lives in its own file under
  `crates/metalcyte-gpu/src/shaders/`. The Rust code embeds the files at build time, so the shaders
  sent to the GPU are byte for byte the same as in 0.3.1 and no Xcode is needed.
- `docs/ARCHITECTURE.md` lists each shader file and whether Python reaches it.

## 0.3.1 (2026-10-07)

Documentation release. The code is unchanged from 0.3.0.

- Shorter, more precise user-facing text in the README, the documentation, the tutorial and the
  docstrings.
- The software is archived on Zenodo (DOI 10.5281/zenodo.23210563), and CITATION.cff records the
  DOIs.
- The raw power samples of the two 4 062 980-cell energy runs are in `benches/results`.

## 0.3.0 (2026-10-07)

First release on PyPI. Prebuilt wheels are available for macOS on Apple silicon,
Python 3.11 to 3.13.

### Added

- A faster, approximate neighbour search for large datasets, using NN-descent initialised from a
  random-projection forest. `pp.neighbors(method="auto")` switches to it above 200 000 cells.
- A fast t-SNE for more than 20 000 cells (FIt-SNE). It approximates the forces between cells on
  a grid with an FFT. All main steps of each iteration run on the GPU: the
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
from disk in blocks. Metal kernels for the neighbour search and the Wilcoxon test.
