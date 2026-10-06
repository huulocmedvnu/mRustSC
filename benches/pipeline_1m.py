#!/usr/bin/env python3
"""A million cells on a laptop: the streamed head, then the graph steps in memory.

    PYTHONPATH=$PWD/python .venv/bin/python benches/pipeline_1m.py counts.h5ad --json out.json

`counts.h5ad` is a counts-only file (see `benches/prepare_counts.py`). The head of the
pipeline (QC, filters, normalise, log1p, HVG, scale, PCA) runs through
`silicell.pp.preprocess_backed`, reading row blocks off the disk and never holding the
matrix; the remaining steps (neighbours, UMAP, Leiden) run on the `(n_cells, 50)`
embedding in memory. Per-step wall time and the memory each step added are reported.

With `--library scanpy` the same pipeline is attempted in memory with scanpy, which is
what the comparison needs; on a machine that cannot hold the dense scaled matrix the
process is expected to be killed, and the script reports that rather than a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from benchmark import PeakRss  # noqa: E402


def _timed(records: list, name: str, call, *args):
    with PeakRss() as peak:
        t = time.perf_counter()
        out = call(*args)
        s = time.perf_counter() - t
    added = None
    if peak.peak_bytes is not None and peak.start_bytes is not None:
        added = (peak.peak_bytes - peak.start_bytes) / 1024**2
    records.append({"step": name, "seconds": s, "added_mb": added})
    print(f"  {name:28s} {s:9.2f} s  +{(added or 0):8.0f} MB", flush=True)
    return out


def run_silicell(path: Path, device: str, umap_parallel: bool, save: Path | None = None) -> dict:
    import silicell as si

    si.settings.device = device
    records: list = []

    def head():
        return si.pp.preprocess_backed(
            path,
            obs_columns=("cell_type",),
            device=device,
            progress=lambda s, t: print(f"      {s:20s} {t:8.2f} s", flush=True),
        )

    adata = _timed(records, "pp.preprocess_backed", head)
    _timed(records, "pp.neighbors", lambda: si.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca"))
    _timed(records, "tl.umap", lambda: si.tl.umap(adata, random_state=0, parallel=umap_parallel))
    _timed(records, "tl.leiden", lambda: si.tl.leiden(adata, random_state=0))
    if save is not None:
        import anndata

        kept = anndata.AnnData(obs=adata.obs.copy())
        kept.obsm["X_umap"] = adata.obsm["X_umap"]
        kept.obsm["X_pca"] = adata.obsm["X_pca"]
        kept.write_h5ad(save)
    return {
        "library": "silicell",
        "device": "metal" if device == "auto" and si.gpu_available() else "cpu",
        "shape": [int(adata.n_obs), int((adata.var["highly_variable"]).sum())],
        "streaming": {
            k: (dict(v) if hasattr(v, "items") else v) for k, v in adata.uns["streaming"].items()
        },
        "steps": records,
        "total_seconds": sum(r["seconds"] for r in records),
        "n_clusters": int(adata.obs["leiden"].nunique()),
    }


def run_scanpy(path: Path) -> dict:
    import anndata
    import scanpy as sc

    sc.settings.verbosity = 0
    records: list = []
    adata = _timed(records, "read_h5ad", lambda: anndata.read_h5ad(path))
    _timed(records, "pp.filter_cells", lambda: sc.pp.filter_cells(adata, min_genes=200))
    _timed(records, "pp.filter_genes", lambda: sc.pp.filter_genes(adata, min_cells=3))
    _timed(records, "pp.normalize_total", lambda: sc.pp.normalize_total(adata, target_sum=1e4))
    _timed(records, "pp.log1p", lambda: sc.pp.log1p(adata))
    _timed(
        records,
        "pp.highly_variable_genes",
        lambda: sc.pp.highly_variable_genes(adata, n_top_genes=2000),
    )
    sub = _timed(
        records, "subset", lambda: adata[:, adata.var["highly_variable"].to_numpy()].copy()
    )
    _timed(records, "pp.scale", lambda: sc.pp.scale(sub, max_value=10))
    _timed(records, "pp.pca", lambda: sc.pp.pca(sub, n_comps=50, random_state=0))
    _timed(records, "pp.neighbors", lambda: sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca"))
    _timed(records, "tl.umap", lambda: sc.tl.umap(sub, random_state=0))
    _timed(
        records,
        "tl.leiden",
        lambda: sc.tl.leiden(sub, random_state=0, flavor="igraph", n_iterations=2),
    )
    return {
        "library": "scanpy",
        "device": "cpu",
        "shape": list(sub.shape),
        "steps": records,
        "total_seconds": sum(r["seconds"] for r in records),
        "n_clusters": int(sub.obs["leiden"].nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--library", choices=["silicell", "scanpy"], default="silicell")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--umap-parallel", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--save", type=Path, help="write obs, X_umap and leiden to this .h5ad (silicell only)"
    )
    args = parser.parse_args()
    print(f"{args.library} on {args.h5ad.name}", flush=True)
    result = (
        run_silicell(args.h5ad, args.device, args.umap_parallel)
        if args.library == "silicell"
        else run_scanpy(args.h5ad)
    )
    print(
        f"  {'total':28s} {result['total_seconds']:9.2f} s   ({result['n_clusters']} Leiden clusters)"  # noqa: E501
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
