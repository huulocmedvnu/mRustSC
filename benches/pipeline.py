#!/usr/bin/env python3
"""One real dataset, the whole standard pipeline, one library at a time.

Run from the repo root:

    PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data.h5ad --library scrust
    PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline.py data.h5ad --library scanpy

`benches/benchmark.py` times each operation on its own, with every stage prepared by
scanpy, on PBMC 3k bootstrapped up to the requested size. This script is the other
measurement: a real count matrix (a CELLxGENE `.h5ad`, whose `raw.X` holds the counts)
pushed through QC, normalisation, feature selection, PCA, the neighbour graph, UMAP,
Leiden and marker genes by *one* library from start to finish, so each step sees the
output of the previous one from the same library. Per-step wall time and the memory the
step added (physical footprint) are printed and, with `--json`, written out.

`--device cpu` pins scrust to the CPU path (`scrust.settings.device`), `--device auto`
takes Metal where there is one. `--cells N` subsamples the matrix first.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import scipy.sparse as sp

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from benchmark import PeakRss  # noqa: E402

TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_COMPS = 50
N_NEIGHBORS = 15
MIN_GENES = 200
MIN_CELLS = 3


def load_counts(path: Path, n_cells: int | None, seed: int) -> anndata.AnnData:
    """The raw counts of a CELLxGENE file as `X`, with the author cell types kept."""
    adata = anndata.read_h5ad(path)
    counts = adata.raw.to_adata() if adata.raw is not None else adata
    keep_obs = [c for c in ("cell_type", "author_celltype", "donor_id") if c in adata.obs]
    counts.obs = adata.obs[keep_obs].copy()
    if "feature_name" in counts.var:
        counts.var_names = counts.var["feature_name"].astype(str).to_numpy()
    counts.var_names_make_unique()
    counts.var = counts.var[[]]
    counts.obsm.clear()
    counts.uns.clear()
    counts.X = sp.csr_matrix(counts.X, dtype=np.float32)
    if counts.X.nnz < 2**31:  # one index dtype, or scipy's eliminate_zeros refuses the matrix
        counts.X.indptr = counts.X.indptr.astype(np.int32)
        counts.X.indices = counts.X.indices.astype(np.int32)
    if n_cells is not None and n_cells < counts.n_obs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(counts.n_obs, n_cells, replace=False))
        counts = counts[keep].copy()
    counts.var["mt"] = counts.var_names.str.startswith("MT-")
    return counts


def _group_key(adata: anndata.AnnData) -> str:
    for key in ("cell_type", "author_celltype"):
        if key in adata.obs and adata.obs[key].nunique() > 1:
            adata.obs[key] = adata.obs[key].astype(str).astype("category")
            return key
    raise KeyError("no cell-type column to rank marker genes by")


def _drop_rare_groups(adata: anndata.AnnData, key: str, minimum: int = 10) -> None:
    counts = adata.obs[key].value_counts()
    rare = counts[counts < minimum].index
    if len(rare):
        adata.obs[key] = adata.obs[key].cat.remove_categories(list(rare))
        adata.obs[key] = adata.obs[key].cat.add_categories(["other"]).fillna("other")


def steps(
    lib: Any, library: str, umap_parallel: bool, tuned: bool = False
) -> list[tuple[str, Any]]:
    """The pipeline as (name, callable) pairs; each callable mutates `adata` in place.

    `tuned` is the fastest scanpy configuration rather than its defaults: the
    `covariance_eigh` PCA solver, UMAP without a fixed seed (which lets umap-learn run its
    optimiser across cores, as scrust's `parallel=True` does) and the `igraph` Leiden
    backend with two iterations. Every comparison against scanpy should show both.
    """
    umap_kwargs: dict[str, Any] = {"random_state": 0}
    pca_kwargs: dict[str, Any] = {"n_comps": N_COMPS, "random_state": 0}
    leiden_kwargs: dict[str, Any] = {"random_state": 0}
    if library == "scrust" and umap_parallel:
        umap_kwargs["parallel"] = True
    if library == "scanpy" and tuned:
        umap_kwargs = {"random_state": None}
        pca_kwargs["svd_solver"] = "covariance_eigh"
        leiden_kwargs.update({"flavor": "igraph", "n_iterations": 2, "directed": False})

    def hvg(adata: anndata.AnnData) -> None:
        lib.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat")

    def subset(adata: anndata.AnnData) -> anndata.AnnData:
        adata.raw = adata
        return adata[:, adata.var["highly_variable"].to_numpy()].copy()

    def markers(adata: anndata.AnnData) -> None:
        key = _group_key(adata)
        _drop_rare_groups(adata, key)
        # scrust ranks on `X` (the 2 000 variable genes); scanpy would silently switch to
        # `adata.raw` (all genes) and test ten times as many, so pin it to `X` too.
        extra = {"use_raw": False} if library == "scanpy" else {}
        lib.tl.rank_genes_groups(adata, key, method="wilcoxon", **extra)

    return [
        ("pp.calculate_qc_metrics", lambda a: lib.pp.calculate_qc_metrics(a, qc_vars=["mt"])),
        ("pp.filter_cells", lambda a: lib.pp.filter_cells(a, min_genes=MIN_GENES)),
        ("pp.filter_genes", lambda a: lib.pp.filter_genes(a, min_cells=MIN_CELLS)),
        ("pp.normalize_total", lambda a: lib.pp.normalize_total(a, target_sum=TARGET_SUM)),
        ("pp.log1p", lambda a: lib.pp.log1p(a)),
        ("pp.highly_variable_genes", hvg),
        ("subset", subset),
        ("pp.scale", lambda a: lib.pp.scale(a, max_value=10)),
        ("pp.pca", lambda a: lib.pp.pca(a, **pca_kwargs)),
        ("pp.neighbors", lambda a: lib.pp.neighbors(a, n_neighbors=N_NEIGHBORS, use_rep="X_pca")),
        ("tl.umap", lambda a: lib.tl.umap(a, **umap_kwargs)),
        ("tl.leiden", lambda a: lib.tl.leiden(a, **leiden_kwargs)),
        ("tl.rank_genes_groups", markers),
    ]


def run(
    path: Path,
    library: str,
    device: str,
    n_cells: int | None,
    umap_parallel: bool,
    tuned: bool = False,
) -> dict:
    if library == "scanpy":
        import scanpy as lib

        lib.settings.verbosity = 0
        resolved = "cpu"
    else:
        import scrust as lib

        lib.settings.device = device
        resolved = "metal" if device == "auto" and lib.gpu_available() else "cpu"

    t0 = time.perf_counter()
    adata = load_counts(path, n_cells, seed=0)
    load_s = time.perf_counter() - t0
    shape0 = adata.shape

    records = []
    total = 0.0
    for name, call in steps(lib, library, umap_parallel, tuned):
        with PeakRss() as peak:
            start = time.perf_counter()
            result = call(adata)
            seconds = time.perf_counter() - start
        if isinstance(result, anndata.AnnData):
            adata = result
        added = None
        if peak.peak_bytes is not None and peak.start_bytes is not None:
            added = (peak.peak_bytes - peak.start_bytes) / 1024**2
        records.append(
            {"step": name, "seconds": seconds, "added_mb": added, "shape": list(adata.shape)}
        )
        total += seconds
        print(f"  {name:28s} {seconds:9.3f} s  +{(added or 0):8.0f} MB  {adata.shape}", flush=True)

    clusters = int(adata.obs["leiden"].nunique())
    return {
        "library": library,
        "device": resolved,
        "umap_parallel": bool(umap_parallel and library == "scrust"),
        "tuned": bool(tuned and library == "scanpy"),
        "file": str(path),
        "input_shape": list(shape0),
        "load_seconds": load_s,
        "steps": records,
        "total_seconds": total,
        "n_clusters": clusters,
        "versions": _versions(),
    }


def _versions() -> dict[str, str]:
    out = {}
    for name in ("scanpy", "scrust", "anndata", "numpy"):
        with contextlib.suppress(Exception):
            out[name] = __import__(name).__version__
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--library", choices=["scanpy", "scrust"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--umap-parallel", action="store_true")
    parser.add_argument(
        "--tuned", action="store_true", help="scanpy's fastest settings, not its defaults"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    print(
        f"{args.library} ({args.device}{', tuned' if args.tuned else ''}) on {args.h5ad.name}",
        flush=True,
    )
    result = run(args.h5ad, args.library, args.device, args.cells, args.umap_parallel, args.tuned)
    print(
        f"  {'total':28s} {result['total_seconds']:9.3f} s   ({result['n_clusters']} Leiden clusters)"  # noqa: E501
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
