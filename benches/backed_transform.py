#!/usr/bin/env python3
"""Peak memory of `pp.normalize_total` + `pp.log1p` in memory versus streamed on disk.

Run from the repo root:

    PYTHONPATH=$PWD/python .venv/bin/python benches/backed_transform.py

Both modes apply the same head of a single-cell pipeline -- normalise to 10 000 counts,
then log1p -- and the only difference is how much of `X` is resident at once. The
`whole` mode reads the `.h5ad` into memory and transforms it there, so its peak carries
the entire matrix. The `streamed` mode opens the file backed and lets `sr.pp.*` rewrite
`X` a row block at a time (`scrust.settings.chunk_size`), so its peak carries one block.

Each mode runs in its own process on its own copy of the file (the streamed mode
rewrites `X` in place), so a peak reading belongs to one mode and the two never collide.
A checksum of the transformed values is printed for both: the memory reduction is only
meaningful if both produced the same matrix, which they must, bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import scipy.sparse as sp

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[0]
CACHE = REPO_ROOT / ".cache" / "backed_bench"
BYTES_PER_GB = 1024**3
sys.path.insert(0, str(BENCH_DIR))

from benchmark import PeakRss  # noqa: E402


def matrix_path(n_cells: int, n_genes: int, density: float) -> Path:
    return CACHE / f"counts_{n_cells}x{n_genes}_d{density}.h5ad"


def build(n_cells: int, n_genes: int, density: float, seed: int = 0) -> Path:
    """Write a synthetic integer-count matrix once, a row block at a time."""
    path = matrix_path(n_cells, n_genes, density)
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per_row = max(1, round(density * n_genes))
    blocks = []
    for start in range(0, n_cells, 5_000):
        rows = min(5_000, n_cells - start)
        indices = rng.integers(0, n_genes, size=rows * per_row).astype(np.int32)
        values = (rng.poisson(3.0, size=rows * per_row) + 1).astype(np.float32)
        indptr = np.arange(0, rows * per_row + 1, per_row, dtype=np.int32)
        blocks.append(sp.csr_matrix((values, indices, indptr), shape=(rows, n_genes)))
    anndata.AnnData(sp.vstack(blocks, format="csr")).write_h5ad(path)
    return path


def _checksum(matrix: Any) -> float:
    return float(np.asarray(matrix.tocsr().data, dtype=np.float64).sum())


def run_whole(path: Path) -> dict[str, Any]:
    """Read the file into memory and transform it there: the baseline peak to beat."""
    import scrust as sr

    with PeakRss() as peak:
        started = time.perf_counter()
        adata = anndata.read_h5ad(path)
        sr.pp.normalize_total(adata, target_sum=1e4)
        sr.pp.log1p(adata)
        checksum = _checksum(adata.X)
        elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "peak_mb": (peak.peak_bytes or 0) / 1e6, "checksum": checksum}


def run_streamed(path: Path, block_size: int) -> dict[str, Any]:
    """The same transform, but backed and rewritten one row block at a time."""
    import scrust as sr

    sr.settings.chunk_size = block_size
    with PeakRss() as peak:
        started = time.perf_counter()
        adata = anndata.read_h5ad(path, backed="r")
        sr.pp.normalize_total(adata, target_sum=1e4)
        sr.pp.log1p(adata)
        checksum = _checksum(adata.to_memory().X)
        elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "peak_mb": (peak.peak_bytes or 0) / 1e6,
        "checksum": checksum,
        "block_size": block_size,
    }


def _call_worker(mode: str, path: Path, block_size: int) -> dict[str, Any]:
    """Run one mode in its own process on its own copy of the file."""
    scratch = path.with_name(f"{mode}_{path.name}")
    shutil.copyfile(path, scratch)
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
             "--path", str(scratch), "--block-size", str(block_size)],
            capture_output=True, text=True, check=False,
        )
    finally:
        pass
    scratch.unlink(missing_ok=True)
    for line in process.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    tail = (process.stderr.strip().splitlines() or ["no stderr"])[-1][:200]
    return {"error": f"exited with {process.returncode}: {tail}"}


def run(n_cells: int, n_genes: int, density: float, block_size: int) -> int:
    path = build(n_cells, n_genes, density)
    nnz_gb = path.stat().st_size / BYTES_PER_GB
    print(
        f"{n_cells} cells x {n_genes} genes at density {density}\n"
        f"file {path.name}, {nnz_gb:.3f} GB on disk; streamed block = {block_size} cells"
    )
    whole = _call_worker("whole", path, block_size)
    streamed = _call_worker("streamed", path, block_size)

    header = f"{'mode':<12}{'seconds':>10}{'peak MB':>10}{'checksum':>18}"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, result in (("whole", whole), ("streamed", streamed)):
        if "error" in result:
            print(f"{name:<12}{'-':>10}{'-':>10}{'did not finish':>18}")
            continue
        print(
            f"{name:<12}{result['seconds']:>10.2f}{result['peak_mb']:>10.0f}"
            f"{result['checksum']:>18.4f}"
        )
    if "peak_mb" in whole and "peak_mb" in streamed:
        same = abs(whole["checksum"] - streamed["checksum"]) < 1e-3
        reduction = whole["peak_mb"] - streamed["peak_mb"]
        print(
            f"\npeak reduction: {reduction:.0f} MB "
            f"({streamed['peak_mb'] / whole['peak_mb']:.2f}x of the in-memory peak)"
        )
        print(f"same matrix produced (checksums agree): {same}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=40_000)
    parser.add_argument("--genes", type=int, default=6_000)
    parser.add_argument("--density", type=float, default=0.08)
    parser.add_argument("--block-size", type=int, default=2_000)
    parser.add_argument("--worker", choices=["whole", "streamed"], help=argparse.SUPPRESS)
    parser.add_argument("--path", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker == "whole":
        print(json.dumps(run_whole(Path(arguments.path))), flush=True)
        return 0
    if arguments.worker == "streamed":
        print(json.dumps(run_streamed(Path(arguments.path), arguments.block_size)), flush=True)
        return 0
    return run(arguments.cells, arguments.genes, arguments.density, arguments.block_size)


if __name__ == "__main__":
    raise SystemExit(main())
