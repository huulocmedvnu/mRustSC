"""Exact against approximate neighbour search on subsamples of a PCA embedding.

    PYTHONPATH=$PWD/python .venv/bin/python benches/knn_methods.py \
        benches/results/embryo1m_metalcyte_metal.h5ad --json benches/results/knn_methods_embryo.json

For each size: wall time of the exact search on the GPU and on the CPU, of the
approximate search (NN-descent, CPU), and the recall of the approximate lists against
the exact ones. Sizes above the GPU's practical range skip the exact search where the
caller says so.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

SIZES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 953_436)
K = 14  # 15 neighbours in scanpy's count, which includes the cell itself


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path, help="file with obsm/X_pca")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument(
        "--exact-up-to", type=int, default=10**9, help="largest size for the exact CPU search"
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    from metalcyte import gpu_available
    from metalcyte._shared import _extension

    ext = _extension()
    with h5py.File(args.h5ad) as f:
        full = np.ascontiguousarray(f["obsm/X_pca"][...], dtype=np.float32)
    rng = np.random.default_rng(0)
    order = rng.permutation(full.shape[0])
    rows = []
    for size in args.sizes:
        size = min(size, full.shape[0])
        points = np.ascontiguousarray(full[np.sort(order[:size])])
        row = {"size": size}
        t = time.perf_counter()
        ia, _ = ext.knn_approximate(points, K, 0)
        row["approximate_s"] = time.perf_counter() - t
        ia = np.asarray(ia)
        truth = None
        if gpu_available():
            t = time.perf_counter()
            ie, _ = ext.knn(points, K, "metal")
            row["exact_metal_s"] = time.perf_counter() - t
            truth = np.asarray(ie)
        if size <= args.exact_up_to:
            t = time.perf_counter()
            ic, _ = ext.knn(points, K, "cpu")
            row["exact_cpu_s"] = time.perf_counter() - t
            truth = np.asarray(ic) if truth is None else truth
        if truth is not None:
            sample = range(0, size, max(1, size // 20_000))
            hits = sum(len(set(ia[i]) & set(truth[i])) for i in sample)
            row["recall"] = hits / (K * len(sample))
        rows.append(row)
        print(
            f"  {size:>8} cells  approximate {row['approximate_s']:6.1f} s"
            + (f"  exact GPU {row['exact_metal_s']:6.1f} s" if "exact_metal_s" in row else "")
            + (f"  exact CPU {row['exact_cpu_s']:6.1f} s" if "exact_cpu_s" in row else "")
            + (f"  recall {row['recall']:.3f}" if "recall" in row else ""),
            flush=True,
        )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"k": K + 1, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
