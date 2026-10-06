"""The head of the pipeline for a matrix that never fits in memory.

`preprocess_backed` takes a counts `.h5ad` on disk and returns an in-memory `AnnData`
holding what the rest of the pipeline needs — the per-cell QC columns, the
highly-variable flags and the PCA embedding — without ever holding the count matrix
whole. Three passes over the row blocks of `X`:

1. **moments**: per-cell `n_genes` and `total_counts`, per-gene `n_cells`, and the
   `sum` / `sum of squares` behind `highly_variable_genes`, each block normalised and
   log-transformed on the fly (the same core kernels as the in-memory path, so the
   numbers match it).
2. **scatter**: the highly-variable columns of each block are scaled against the
   per-gene moments from pass 1 and their `(g, g)` scatter is accumulated on the GPU.
   The principal axes are the top eigenvectors of that scatter, which is scanpy's
   `svd_solver="covariance_eigh"`.
3. **project**: each block is scaled again and multiplied by the loadings, giving its
   rows of `X_pca`.

Peak memory is one block (`silicell.settings.max_memory_gb`) plus the `(n_cells, n_comps)`
embedding, so a million cells by 2 000 genes need the 400 MB embedding and a block,
where the dense scaled matrix alone would be 8 GB. Apple's unified memory means the
block the CPU just scaled is the buffer the GPU multiplies: nothing is copied to a
device. A cell is dropped before any statistic if it has fewer than `min_genes`
genes, and a gene before the variable-gene step if it is in fewer than `min_cells`
cells, as `pp.filter_cells` / `pp.filter_genes` would.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from silicell._backed import block_size_for, open_backed
from silicell._shared import _extension
from silicell.settings import settings

if TYPE_CHECKING:
    import os

_VALUE_DTYPE = np.float32


def _blocks(backed: Any, block_size: int):
    for start, block in backed.blocks(block_size):
        yield start, block.astype(_VALUE_DTYPE, copy=False)


def _transform(ext: Any, block: sp.csr_matrix, target_sum: float, log: bool) -> sp.csr_matrix:
    """normalize_total then log1p on a block, in place on its own value buffer."""
    values = np.ascontiguousarray(block.data, dtype=_VALUE_DTYPE)
    ext.normalize_total_inplace(block.indptr, values, float(target_sum))
    if log:
        ext.log1p_inplace(values)
    block.data = values
    return block


def preprocess_backed(
    path: str | os.PathLike[str],
    *,
    n_top_genes: int = 2000,
    n_comps: int = 50,
    target_sum: float = 1e4,
    min_genes: int = 200,
    min_cells: int = 3,
    max_value: float | None = 10.0,
    flavor: str = "seurat",
    block_size: int | None = None,
    random_state: int = 0,
    device: str | None = None,
    obs_columns: tuple[str, ...] = (),
    progress: Callable[[str, float], None] | None = None,
) -> anndata.AnnData:
    """QC, normalisation, feature selection, scaling and PCA for an on-disk matrix.

    Returns an `AnnData` with no `X`: `obs` carries `n_genes` and `total_counts` and the
    requested `obs_columns`; `var` the `highly_variable`, `means` and `dispersions_norm`
    columns for every gene of the file; `obsm["X_pca"]`, `varm["PCs"]` and `uns["pca"]`
    as `pp.pca` writes them. Cells failing `min_genes` are absent from `obs`.
    """
    ext = _extension()
    dev = settings.resolve_device(device)
    path = Path(path)
    timings: dict[str, float] = {}

    def tick(stage: str, started: float) -> None:
        timings[stage] = time.perf_counter() - started
        if progress is not None:
            progress(stage, timings[stage])

    with open_backed(path) as backed:
        n_obs, n_vars = backed.shape
        var_names = backed.var.index.copy()
        obs_index = backed.obs.index.copy()
        kept_obs = {c: backed.obs[c].to_numpy() for c in obs_columns if c in backed.obs}
        size = block_size or settings.chunk_size or block_size_for(n_vars, backed.density)

        # ---- pass 1: per-cell QC, per-gene presence, HVG sums -------------------
        t = time.perf_counter()
        n_genes = np.zeros(n_obs, dtype=np.int64)
        total_counts = np.zeros(n_obs, dtype=np.float64)
        gene_cells = np.zeros(n_vars, dtype=np.int64)
        sums = np.zeros(n_vars, dtype=np.float64)
        squares = np.zeros(n_vars, dtype=np.float64)
        for start, block in _blocks(backed, size):
            stop = start + block.shape[0]
            counts_per_cell = np.diff(block.indptr)
            n_genes[start:stop] = counts_per_cell
            total_counts[start:stop] = np.add.reduceat(
                np.concatenate([block.data.astype(np.float64), [0.0]]), block.indptr[:-1]
            ) * (counts_per_cell > 0)
            keep = counts_per_cell >= min_genes
            if not keep.all():
                block = block[keep]
            gene_cells += ext.column_nnz(block.indices, n_vars)
            block = _transform(ext, block, target_sum, log=True)
            s, q = ext.hvg_partial_sums(
                block.indptr, block.indices, block.data, n_vars, flavor == "seurat"
            )
            sums += s
            squares += q
        cell_mask = n_genes >= min_genes
        n_cells_kept = int(cell_mask.sum())
        gene_mask = gene_cells >= min_cells
        tick("pass1_moments", t)

        # ---- HVG on the genes that pass the cell filter --------------------------
        t = time.perf_counter()
        hvg = ext.highly_variable_genes_from_sums(
            np.ascontiguousarray(sums[gene_mask]),
            np.ascontiguousarray(squares[gene_mask]),
            n_cells_kept,
            n_top_genes,
            flavor,
        )
        highly_variable = np.zeros(n_vars, dtype=bool)
        highly_variable[np.flatnonzero(gene_mask)] = np.asarray(hvg["highly_variable"], dtype=bool)
        means = np.full(n_vars, np.nan, dtype=_VALUE_DTYPE)
        means[gene_mask] = np.asarray(hvg["means"], dtype=_VALUE_DTYPE)
        disp = np.full(n_vars, np.nan, dtype=_VALUE_DTYPE)
        disp[gene_mask] = np.asarray(hvg["normalised_dispersions"], dtype=_VALUE_DTYPE)
        hv_columns = np.flatnonzero(highly_variable)
        n_hv = int(hv_columns.size)
        tick("hvg", t)

        # ---- per-gene moments of the log data on the HVG columns -----------------
        # `scale` standardises the log-normalised values, whose moments are not the
        # expm1 sums above; a cheap extra pass gathers them for the HVG columns only.
        t = time.perf_counter()
        hv_sums = np.zeros(n_hv, dtype=np.float64)
        hv_squares = np.zeros(n_hv, dtype=np.float64)
        col_map = np.full(n_vars, -1, dtype=np.int64)
        col_map[hv_columns] = np.arange(n_hv)

        def hv_block(block: sp.csr_matrix) -> sp.csr_matrix:
            keep_entries = col_map[block.indices] >= 0
            rows = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))[keep_entries]
            return sp.csr_matrix(
                (block.data[keep_entries], (rows, col_map[block.indices[keep_entries]])),
                shape=(block.shape[0], n_hv),
                dtype=_VALUE_DTYPE,
            )

        for start, block in _blocks(backed, size):
            stop = start + block.shape[0]
            keep = cell_mask[start:stop]
            if not keep.any():
                continue
            if not keep.all():
                block = block[keep]
            block = hv_block(_transform(ext, block, target_sum, log=True))
            s, q = ext.hvg_partial_sums(block.indptr, block.indices, block.data, n_hv, False)
            hv_sums += s
            hv_squares += q
        n = float(n_cells_kept)
        hv_mean = hv_sums / n
        hv_var = (hv_squares - n * hv_mean**2) / (n - 1.0)
        hv_std = np.sqrt(np.maximum(hv_var, 0.0))
        hv_std[hv_std == 0.0] = 1.0
        hv_mean32 = np.ascontiguousarray(hv_mean, dtype=_VALUE_DTYPE)
        hv_std32 = np.ascontiguousarray(hv_std, dtype=_VALUE_DTYPE)
        tick("pass2_hv_moments", t)

        # ---- pass 3: scatter of the scaled HVG block, on the device ---------------
        t = time.perf_counter()
        scatter = np.zeros((n_hv, n_hv), dtype=np.float64)
        for start, block in _blocks(backed, size):
            stop = start + block.shape[0]
            keep = cell_mask[start:stop]
            if not keep.any():
                continue
            if not keep.all():
                block = block[keep]
            block = hv_block(_transform(ext, block, target_sum, log=True))
            dense = ext.scale_dense_with(
                block.indptr, block.indices, block.data, n_hv, hv_mean32, hv_std32, True, max_value
            )
            scatter += ext.gram_dense(dense, dev)
        fitted = ext.pca_from_scatter(scatter, n_cells_kept, n_comps, int(random_state), dev)
        components = np.asarray(fitted["components"], dtype=_VALUE_DTYPE)
        tick("pass3_scatter_eigh", t)

        # ---- pass 4: project every block onto the loadings ------------------------
        t = time.perf_counter()
        embedding = np.empty((n_cells_kept, n_comps), dtype=_VALUE_DTYPE)
        cursor = 0
        for start, block in _blocks(backed, size):
            stop = start + block.shape[0]
            keep = cell_mask[start:stop]
            if not keep.any():
                continue
            if not keep.all():
                block = block[keep]
            block = hv_block(_transform(ext, block, target_sum, log=True))
            dense = ext.scale_dense_with(
                block.indptr, block.indices, block.data, n_hv, hv_mean32, hv_std32, True, max_value
            )
            scores = ext.project_dense(dense, components, dev)
            embedding[cursor : cursor + scores.shape[0]] = scores
            cursor += scores.shape[0]
        tick("pass4_project", t)

    obs = pd.DataFrame(
        {
            "n_genes": n_genes[cell_mask],
            "total_counts": total_counts[cell_mask].astype(_VALUE_DTYPE),
        },
        index=obs_index[cell_mask],
    )
    for name, column in kept_obs.items():
        obs[name] = column[cell_mask]
    var = pd.DataFrame(
        {
            "n_cells": gene_cells,
            "highly_variable": highly_variable,
            "means": means,
            "dispersions_norm": disp,
        },
        index=var_names,
    )
    result = anndata.AnnData(obs=obs, var=var)
    result.obsm["X_pca"] = embedding
    pcs = np.zeros((n_vars, n_comps), dtype=_VALUE_DTYPE)
    pcs[hv_columns] = components.T
    result.varm["PCs"] = pcs
    result.uns["pca"] = {
        "variance": np.asarray(fitted["explained_variance"], dtype=_VALUE_DTYPE),
        "variance_ratio": np.asarray(fitted["explained_variance_ratio"], dtype=_VALUE_DTYPE),
        "params": {"zero_center": True, "use_highly_variable": True, "solver": "covariance_eigh"},
    }
    result.uns["streaming"] = {
        "block_size": int(size),
        "device": dev,
        "n_passes": 4,
        "timings": timings,
        "cells_dropped": int(n_obs - n_cells_kept),
        "genes_dropped": int((~gene_mask).sum()),
    }
    return result


__all__ = ["preprocess_backed"]
