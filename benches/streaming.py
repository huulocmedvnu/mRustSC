#!/usr/bin/env python3
"""What streaming row blocks costs against densifying the whole matrix.

Run from the repo root:

    PYTHONPATH=$PWD/python .venv/bin/python benches/streaming.py

Both modes compute the same thing — the per-gene sum of a 50 000 x 20 000 matrix —
and the only difference is how much of it is resident at once. The whole-matrix mode
is what a numpy-shaped implementation does: read the file, densify, reduce. The
streamed mode reads one row block at a time through `scrust._backed.open_backed`,
so its peak is the block rather than the dataset.

Each mode runs in its own process, so a peak reading belongs to one mode only, and a
mode killed by the operating system is reported as that rather than as a crash.

The matrix is synthetic: real files of this size are not something a benchmark can
download, and the quantity being measured is bytes resident, which depends on the
shape and the density and not on what the counts mean.
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(BENCH_DIR))

from benchmark import PeakRss  # noqa: E402

CACHE = REPO_ROOT / ".cache" / "bench"
DEFAULT_CELLS = 50_000
DEFAULT_GENES = 20_000
# Roughly PBMC 3k's density after gene filtering, so the file is sparse in the way a
# real count matrix is.
DEFAULT_DENSITY = 0.02
BYTES_PER_GB = 1024**3


def matrix_path(n_cells: int, n_genes: int, density: float) -> Path:
    return CACHE / f"synthetic_{n_cells}x{n_genes}_d{density}.h5ad"


def build(n_cells: int, n_genes: int, density: float, seed: int = 0) -> Path:
    """Write the synthetic matrix once and reuse it on later runs."""
    path = matrix_path(n_cells, n_genes, density)
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Built a row block at a time so that making the file never needs more memory
    # than reading it does.
    blocks = []
    per_row = max(1, round(density * n_genes))
    for start in range(0, n_cells, 5_000):
        rows = min(5_000, n_cells - start)
        indices = rng.integers(0, n_genes, size=rows * per_row).astype(np.int32)
        values = rng.poisson(3.0, size=rows * per_row).astype(np.float32) + 1.0
        indptr = np.arange(0, rows * per_row + 1, per_row, dtype=np.int32)
        blocks.append(sp.csr_matrix((values, indices, indptr), shape=(rows, n_genes)))
    adata = anndata.AnnData(sp.vstack(blocks, format="csr"))
    adata.write_h5ad(path)
    return path


# -------------------------------------------------------------------------- workers


def run_whole(path: Path) -> dict[str, Any]:
    """Read the file into memory, densify it, reduce it: the baseline to beat."""
    with PeakRss() as peak:
        start = time.perf_counter()
        adata = anndata.read_h5ad(path)
        dense = np.asarray(adata.X.todense(), dtype=np.float32)
        sums = dense.sum(axis=0)
        elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "peak_mb": (peak.peak_bytes or 0) / 1e6, "checksum": _sum(sums)}


def run_streamed(path: Path, budget_gb: float) -> dict[str, Any]:
    """The same reduction over row blocks read straight from the file."""
    import scrust
    from scrust._backed import open_backed

    scrust.settings.max_memory_gb = budget_gb
    with PeakRss() as peak:
        start = time.perf_counter()
        with open_backed(path) as backed:
            block_size = backed.block_size()
            sums = np.zeros(backed.n_vars, dtype=np.float64)
            for _, block in backed.blocks(block_size):
                sums += np.asarray(block.todense(), dtype=np.float32).sum(axis=0)
        elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "peak_mb": (peak.peak_bytes or 0) / 1e6,
        "checksum": _sum(sums),
        "block_size": block_size,
    }


def _sum(values: Any) -> float:
    """A single number both modes must agree on, so the comparison is of equal work."""
    return float(np.asarray(values, dtype=np.float64).sum())


# --------------------------------------------------------------------------- driver


def _call_worker(mode: str, path: Path, budget_gb: float) -> dict[str, Any]:
    """Run one mode in its own process; a mode the OS kills reports that, not a stack."""
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--path",
            str(path),
            "--budget-gb",
            str(budget_gb),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in process.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    tail = (process.stderr.strip().splitlines() or ["no stderr"])[-1][:200]
    return {"error": f"exited with {process.returncode}: {tail}"}


def run(n_cells: int, n_genes: int, density: float, budget_gb: float) -> int:
    path = build(n_cells, n_genes, density)
    dense_gb = n_cells * n_genes * 4 / BYTES_PER_GB
    print(
        f"{n_cells} cells x {n_genes} genes at density {density}\n"
        f"file {path.name}, {path.stat().st_size / BYTES_PER_GB:.2f} GB on disk;"
        f" dense f32 would be {dense_gb:.2f} GB"
    )

    results = {mode: _call_worker(mode, path, budget_gb) for mode in ("whole", "streamed")}
    header = f"{'mode':<26}{'seconds':>10}{'peak GB':>10}{'per-gene sum':>18}"
    print(f"\n{header}\n{'-' * len(header)}")
    for mode, result in results.items():
        if "error" in result:
            print(f"{mode:<26}{'-':>10}{'-':>10}{'did not finish':>18}")
            continue
        print(
            f"{mode:<26}{result['seconds']:>10.2f}"
            f"{result['peak_mb'] / 1024:>10.2f}{result['checksum']:>18.6g}"
        )

    for mode, result in results.items():
        if "error" in result:
            print(f"\n{mode} did not finish: {result['error']}")
    streamed = results["streamed"]
    if "block_size" in streamed:
        print(
            f"\nstreamed used blocks of {streamed['block_size']} rows, sized by "
            f"scrust.settings.max_memory_gb = {budget_gb} GB."
        )
    if all("error" not in result for result in results.values()):
        gap = abs(results["whole"]["checksum"] - streamed["checksum"])
        print(f"the two per-gene sums differ by {gap:.3g}; the same work was measured twice.")
    print(
        "\npeak GB is the highest resident size of the worker process, sampled every 5 ms.\n"
        "It includes the interpreter and the imports, which are the same in both modes,\n"
        "so the difference between the rows is the matrix and nothing else."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--genes", type=int, default=DEFAULT_GENES)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--budget-gb", type=float, default=1.0)
    parser.add_argument("--worker", choices=["whole", "streamed"], help=argparse.SUPPRESS)
    parser.add_argument("--path", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker == "whole":
        print(json.dumps(run_whole(Path(arguments.path))), flush=True)
        return 0
    if arguments.worker == "streamed":
        print(json.dumps(run_streamed(Path(arguments.path), arguments.budget_gb)), flush=True)
        return 0
    return run(arguments.cells, arguments.genes, arguments.density, arguments.budget_gb)


if __name__ == "__main__":
    raise SystemExit(main())
