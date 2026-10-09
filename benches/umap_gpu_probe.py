"""Probe: UMAP layout optimisation on the GPU (Hogwild kernel, buffers resident) against
the multi-core CPU optimiser, on the same neighbour graph and initial layout.

    PYTHONPATH=$PWD/python .venv/bin/python benches/umap_gpu_probe.py \
        benches/results/embryo1m_metalcyte_metal.h5ad --json benches/results/umap_gpu_probe_1m.json

Quality is measured on a fixed random sample of cells: the share of each cell's 15 nearest
neighbours in PCA space that are also among its 15 nearest neighbours in the layout, and
the share of cells whose 15 layout neighbours vote for their own annotated cell type.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from sklearn.neighbors import NearestNeighbors

import metalcyte as mc
from metalcyte._shared import _csr_args, _extension

K = 15
SAMPLE = 20_000


def quality(layout, pca_idx, labels, sample):
    nn = NearestNeighbors(n_neighbors=K + 1).fit(layout)
    _, idx = nn.kneighbors(layout[sample])
    idx = idx[:, 1:]
    keep = np.mean([len(set(a) & set(b)) / K for a, b in zip(idx, pca_idx, strict=True)])
    codes = labels[idx]
    vote = np.array([np.bincount(r).argmax() for r in codes])
    return float(keep), float(np.mean(vote == labels[sample]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("h5ad", type=Path)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument(
        "--save-layouts", type=Path, help="write each layout to <dir>/<optimiser>_seed<k>.npy"
    )
    a = p.parse_args()
    with h5py.File(a.h5ad) as f:
        pca = np.ascontiguousarray(f["obsm/X_pca"][...], dtype=np.float32)
        obs = read_elem(f["obs"])
    adata = ad.AnnData(obs=obs, obsm={"X_pca": pca})
    t = time.perf_counter()
    mc.pp.neighbors(adata, n_neighbors=K)
    t_nn = time.perf_counter() - t
    graph = adata.obsp["connectivities"].tocsr()
    labels = pd.Categorical(adata.obs["cell_type"]).codes.astype(np.int64)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(adata.n_obs, size=min(SAMPLE, adata.n_obs), replace=False))
    pca_idx = NearestNeighbors(n_neighbors=K + 1).fit(pca).kneighbors(pca[sample])[1][:, 1:]
    ext = _extension()
    args = (*_csr_args(graph), 2, a.epochs, 0.5, 1.0, 1.0, 5)
    # warm-up: compile the GPU pipeline on a tiny graph
    small = graph[:2000, :2000].tocsr()
    ext.umap_metal(*_csr_args(small), 2, 5, 0.5, 1.0, 1.0, 5, 0)
    rows = []
    for name, fn in (("cpu_parallel", ext.umap_parallel), ("gpu_hogwild", ext.umap_metal)):
        for seed in a.seeds:
            t = time.perf_counter()
            layout = np.asarray(fn(*args, seed), dtype=np.float32)
            dt = time.perf_counter() - t
            keep, vote = quality(layout, pca_idx, labels, sample)
            if a.save_layouts:
                a.save_layouts.mkdir(parents=True, exist_ok=True)
                np.save(a.save_layouts / f"{name}_seed{seed}.npy", layout)
            finite = bool(np.isfinite(layout).all())
            rows.append(
                dict(
                    optimiser=name,
                    seed=seed,
                    seconds=dt,
                    knn_preservation=keep,
                    celltype_vote=vote,
                    finite=finite,
                    spread=float(np.ptp(layout, axis=0).max()),
                )
            )
            print(
                f"{name:13s} seed {seed}: {dt:7.1f} s  kNN kept {keep:.3f}  "
                f"cell-type vote {vote:.3f}  finite {finite}",
                flush=True,
            )
    out = dict(
        n_cells=int(adata.n_obs),
        n_edges=int(graph.nnz),
        epochs=a.epochs,
        neighbors_seconds=t_nn,
        rows=rows,
    )
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
