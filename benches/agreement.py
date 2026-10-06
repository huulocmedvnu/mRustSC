#!/usr/bin/env python3
"""Do metalcyte and scanpy reach the same biology on a real atlas?

    PYTHONPATH=$PWD/python .venv/bin/python benches/agreement.py data/bone_marrow_117k_counts.h5ad \
        --json benches/results/agreement_bm117k.json

Both libraries run the same pipeline on the same counts with the same seeds, and every
intermediate is compared, not only the end:

- highly variable genes: Jaccard overlap of the two 2 000-gene sets;
- PCA: canonical correlations between the two 50-dimensional embeddings on the shared
  gene set (metalcyte's set), i.e. do the subspaces agree, component by component;
- neighbour graph: mean fraction of each cell's 15 neighbours shared, on each library's
  own PCA and on a common PCA (so the graph step is judged on its own);
- Leiden: adjusted Rand index and normalised mutual information between the two
  clusterings, and of each against the author's cell-type labels;
- marker genes: for each cell type, Spearman correlation of the two libraries' Wilcoxon
  scores over the tested genes, and overlap of the top-50 lists.

`--cells N` subsamples first. Nothing here is a timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from pipeline import _drop_rare_groups, _group_key, load_counts  # noqa: E402


def run_library(lib, counts, library: str):
    a = counts.copy()
    lib.pp.filter_cells(a, min_genes=200)
    lib.pp.filter_genes(a, min_cells=3)
    lib.pp.normalize_total(a, target_sum=1e4)
    lib.pp.log1p(a)
    lib.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    hv = set(a.var_names[a.var["highly_variable"].to_numpy()])
    a.raw = a
    b = a[:, a.var["highly_variable"].to_numpy()].copy()
    lib.pp.scale(b, max_value=10)
    lib.pp.pca(b, n_comps=50, random_state=0)
    lib.pp.neighbors(b, n_neighbors=15, use_rep="X_pca")
    if library == "scanpy":
        lib.tl.leiden(b, random_state=0, flavor="igraph", n_iterations=2, directed=False)
    else:
        lib.tl.leiden(b, random_state=0)
    key = _group_key(b)
    _drop_rare_groups(b, key)
    extra = {"use_raw": False} if library == "scanpy" else {}
    lib.tl.rank_genes_groups(b, key, method="wilcoxon", **extra)
    return a, b, hv, key


def canonical(x, y, k):
    qx, _ = np.linalg.qr(x[:, :k] - x[:, :k].mean(0))
    qy, _ = np.linalg.qr(y[:, :k] - y[:, :k].mean(0))
    return np.linalg.svd(qx.T @ qy, compute_uv=False)


def knn_overlap(d1, d2) -> float:
    a = (d1 != 0).tocsr()
    b = (d2 != 0).tocsr()
    shared = a.multiply(b).sum(axis=1).A1
    k = np.maximum(a.sum(axis=1).A1, 1)
    return float(np.mean(shared / k))


def marker_agreement(b1, b2, key):
    from scipy.stats import spearmanr

    r1, r2 = b1.uns["rank_genes_groups"], b2.uns["rank_genes_groups"]
    groups = [g for g in r1["names"].dtype.names if g in r2["names"].dtype.names]
    out = {}
    for g in groups:
        s1 = dict(zip(r1["names"][g], r1["scores"][g], strict=True))
        s2 = dict(zip(r2["names"][g], r2["scores"][g], strict=True))
        common = [n for n in s1 if n in s2]
        rho = (
            spearmanr([s1[n] for n in common], [s2[n] for n in common]).correlation
            if len(common) > 2
            else np.nan
        )
        top1 = set(list(r1["names"][g])[:50])
        top2 = set(list(r2["names"][g])[:50])
        out[g] = {
            "spearman": float(rho),
            "top50_overlap": len(top1 & top2) / 50.0,
            "n_genes": len(common),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    import metalcyte as mc

    sc.settings.verbosity = 0
    counts = load_counts(args.h5ad, args.cells, seed=0)
    print("scanpy ...", flush=True)
    a_sc, b_sc, hv_sc, key = run_library(sc, counts, "scanpy")
    print("metalcyte ...", flush=True)
    _a_sr, b_sr, hv_sr, _ = run_library(mc, counts, "metalcyte")
    assert (b_sc.obs_names == b_sr.obs_names).all()

    report = {"file": str(args.h5ad), "n_cells": int(b_sc.n_obs)}
    report["hvg_jaccard"] = len(hv_sc & hv_sr) / len(hv_sc | hv_sr)

    # PCA on metalcyte's gene set for both, so the subspaces are comparable.
    shared = a_sc[:, sorted(hv_sr)].copy()
    sc.pp.scale(shared, max_value=10)
    sc.pp.pca(shared, n_comps=50, svd_solver="covariance_eigh")
    cc = canonical(b_sr.obsm["X_pca"], shared.obsm["X_pca"], 50)
    report["pca_canonical_min_top10"] = float(
        canonical(b_sr.obsm["X_pca"], shared.obsm["X_pca"], 10).min()
    )
    report["pca_canonical_min_top30"] = float(
        canonical(b_sr.obsm["X_pca"], shared.obsm["X_pca"], 30).min()
    )
    report["pca_canonical_min_top50"] = float(cc.min())
    report["pca_variance_ratio_metalcyte"] = [
        float(v) for v in b_sr.uns["pca"]["variance_ratio"][:10]
    ]
    report["pca_variance_ratio_scanpy_eigh"] = [
        float(v) for v in shared.uns["pca"]["variance_ratio"][:10]
    ]

    report["knn_overlap_own_pca"] = knn_overlap(b_sc.obsp["distances"], b_sr.obsp["distances"])
    # The graph step alone: metalcyte neighbours on scanpy's PCA versus scanpy's own.
    common = b_sc.copy()
    mc.pp.neighbors(common, n_neighbors=15, use_rep="X_pca")
    report["knn_overlap_common_pca"] = knn_overlap(b_sc.obsp["distances"], common.obsp["distances"])

    l_sc = b_sc.obs["leiden"].astype(str).to_numpy()
    l_sr = b_sr.obs["leiden"].astype(str).to_numpy()
    truth = b_sc.obs[key].astype(str).to_numpy()
    report["leiden"] = {
        "n_clusters_scanpy": len(set(l_sc)),
        "n_clusters_metalcyte": len(set(l_sr)),
        "ari_scanpy_vs_metalcyte": float(adjusted_rand_score(l_sc, l_sr)),
        "nmi_scanpy_vs_metalcyte": float(normalized_mutual_info_score(l_sc, l_sr)),
        "ari_scanpy_vs_celltype": float(adjusted_rand_score(truth, l_sc)),
        "ari_metalcyte_vs_celltype": float(adjusted_rand_score(truth, l_sr)),
        "nmi_scanpy_vs_celltype": float(normalized_mutual_info_score(truth, l_sc)),
        "nmi_metalcyte_vs_celltype": float(normalized_mutual_info_score(truth, l_sr)),
    }
    markers = marker_agreement(b_sc, b_sr, key)
    report["markers"] = markers
    report["markers_summary"] = {
        "median_spearman": float(np.nanmedian([m["spearman"] for m in markers.values()])),
        "median_top50_overlap": float(np.median([m["top50_overlap"] for m in markers.values()])),
        "n_groups": len(markers),
    }
    summary = {
        k: v
        for k, v in report.items()
        if k not in ("markers", "pca_variance_ratio_metalcyte", "pca_variance_ratio_scanpy_eigh")
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
