#!/usr/bin/env python3
"""Time every implemented algorithm against scanpy, at several dataset sizes.

Run from the repo root:

    PYTHONPATH=$PWD/python .venv/bin/python benches/benchmark.py

Each library runs in its own subprocess so that peak memory is attributable, and
every operation reports either a timing or the reason it could not run. Nothing is
silently omitted: an operation that fails, refuses its input or is not implemented
yet still gets a row, and the reason is repeated in the footer.

Sizes above PBMC 3k's 2 638 cells are built by resampling its cells with
replacement and binomially thinning the copies, so the matrix keeps the sparsity
and depth of real data while staying distinct row by row. It is not biologically
meaningful at those sizes: the result is a cost model, not an analysis.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
# The test suite already owns dataset caching; reuse it rather than keeping a second copy.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import configure_datasetdir  # noqa: E402

TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_COMPS = 50
N_NEIGHBORS = 15
# Fraction of each count kept when a bootstrapped cell is thinned, so that no two
# rows of a grown matrix are identical.
THINNING = 0.9
DEFAULT_SIZES = (500, 2638, 10000, 50000)
DEFAULT_REPEATS = 3
# Seconds after which an operation stops being repeated, however few runs it has had.
REPEAT_BUDGET = 30.0


def _auto_learning_rate(n_obs: int, early_exaggeration: float = 12.0) -> float:
    """scikit-learn's `learning_rate="auto"`; scanpy still defaults to its legacy 1000.

    Passed to both libraries so the two are timed on the same amount of work.
    """
    return max(n_obs / early_exaggeration / 4.0, 50.0)


@dataclass(frozen=True)
class Op:
    """One algorithm, the prepared input it consumes and the arguments it takes."""

    path: str
    stage: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Arguments that depend on the number of cells, applied per run.
    sized_kwargs: Any = None
    # Above this many cells scrust refuses the input, so scanpy is not timed either.
    max_cells: int | None = None
    max_cells_reason: str = ""


OPS = (
    Op("pp.filter_cells", "raw", kwargs={"min_genes": 200}),
    Op("pp.filter_genes", "raw", kwargs={"min_cells": 3}),
    Op("pp.normalize_total", "counts", kwargs={"target_sum": TARGET_SUM}),
    Op("pp.log1p", "normalized"),
    Op("pp.highly_variable_genes", "lognorm", kwargs={"n_top_genes": N_TOP_GENES}),
    Op("pp.scale", "subset", kwargs={"zero_center": True, "max_value": 10}),
    Op("pp.pca", "scaled", kwargs={"n_comps": N_COMPS, "random_state": 0}),
    Op("pp.neighbors", "embedded", kwargs={"n_neighbors": N_NEIGHBORS, "use_rep": "X_pca"}),
    Op("tl.umap", "neighbored", kwargs={"random_state": 0}),
    Op(
        "tl.tsne",
        "embedded",
        kwargs={"n_pcs": N_COMPS, "perplexity": 30.0, "random_state": 0},
        sized_kwargs=lambda n: {"learning_rate": _auto_learning_rate(n)},
        max_cells=20000,
        max_cells_reason=(
            "scrust's t-SNE is exact and refuses more than 20 000 cells, "
            "so timing scanpy here would compare nothing"
        ),
    ),
    Op("tl.rank_genes_groups", "lognorm", args=("group",), kwargs={"method": "wilcoxon"}),
)


# --------------------------------------------------------------------------- data


def load_pbmc3k() -> AnnData:
    configure_datasetdir()
    counts = sc.datasets.pbmc3k()
    counts.var_names_make_unique()
    published = sc.datasets.pbmc3k_processed()
    adata = counts[published.obs_names].copy()
    adata.obs["group"] = published.obs["louvain"].values
    return adata


def resize(adata: AnnData, n_cells: int, seed: int = 0) -> AnnData:
    """Take `n_cells` cells: subsample below PBMC 3k's size, bootstrap above it."""
    rng = np.random.default_rng(seed)
    if n_cells <= adata.n_obs:
        keep = np.sort(rng.choice(adata.n_obs, n_cells, replace=False))
        return _with_comparable_groups(adata[keep].copy())

    picks = rng.integers(0, adata.n_obs, n_cells - adata.n_obs)
    replicas = adata[picks].copy()
    counts = sparse.csr_matrix(replicas.X)
    counts.data = rng.binomial(counts.data.astype(np.int64), THINNING).astype(np.float32)
    counts.eliminate_zeros()
    replicas.X = counts
    grown = anndata.concat([adata, replicas], index_unique="-")
    grown.var_names = adata.var_names
    return grown


def _with_comparable_groups(adata: AnnData, minimum: int = 2) -> AnnData:
    """Drop cell types left with fewer than `minimum` cells.

    Subsampling can leave a rare type — PBMC 3k's megakaryocytes — with a single
    cell, and scanpy's rank-sum test then refuses the whole call. Dropping them
    keeps the differential-expression row measurable at every size.
    """
    counts = adata.obs["group"].value_counts()
    keep = adata.obs["group"].isin(counts[counts >= minimum].index).to_numpy()
    if keep.all():
        return adata
    trimmed = adata[keep].copy()
    trimmed.obs["group"] = trimmed.obs["group"].cat.remove_unused_categories()
    return trimmed


def prepare(adata: AnnData) -> dict[str, AnnData]:
    """The input each algorithm expects, all of it built by scanpy so both libraries
    are timed on identical data."""
    counts = adata.copy()
    sc.pp.filter_genes(counts, min_cells=3)
    normalized = counts.copy()
    sc.pp.normalize_total(normalized, target_sum=TARGET_SUM)
    lognorm = normalized.copy()
    sc.pp.log1p(lognorm)
    subset = lognorm.copy()
    sc.pp.highly_variable_genes(subset, n_top_genes=N_TOP_GENES)
    subset = subset[:, subset.var["highly_variable"].to_numpy()].copy()
    scaled = subset.copy()
    sc.pp.scale(scaled, max_value=10)
    embedded = scaled.copy()
    sc.pp.pca(embedded, n_comps=N_COMPS, random_state=0)
    neighbored = embedded.copy()
    sc.pp.neighbors(neighbored, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    return {
        "raw": adata,
        "counts": counts,
        "normalized": normalized,
        "lognorm": lognorm,
        "subset": subset,
        "scaled": scaled,
        "embedded": embedded,
        "neighbored": neighbored,
    }


# ------------------------------------------------------------------------- memory


class _TaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_uint64 * 2),
        ("system_time", ctypes.c_uint64 * 2),
        ("policy", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
    ]


_MACH_TASK_BASIC_INFO = 20


def _rss_reader() -> Any:
    """Return a callable giving this process's resident size in bytes, or None.

    The mach call rather than `getrusage`, which only reports a high-water mark and
    so cannot say what a *single* operation needed.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        task = libc.mach_task_self()
        info = _TaskBasicInfo()
        count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
        if libc.task_info(task, _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)):
            return None
    except OSError:
        return None

    def read() -> int:
        libc.task_info(task, _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count))
        return int(info.resident_size)

    return read


class PeakRss:
    """Sample resident size while a block runs and report the highest reading.

    Sampling every 5 ms means a call shorter than that may under-report; a call that
    short is too cheap for its memory to matter.
    """

    INTERVAL = 0.005

    def __init__(self) -> None:
        self._read = _rss_reader()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.INTERVAL):
            self.peak_bytes = max(self.peak_bytes or 0, self._read())

    def __enter__(self) -> PeakRss:
        if self._read is None:
            return self
        self.peak_bytes = self._read()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._read is None or self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes or 0, self._read())


# -------------------------------------------------------------------------- worker


def _resolve(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = getattr(root, part)
    return root


def _time_call(function: Any, stage: AnnData, op: Op, repeats: int) -> dict[str, Any]:
    """Best of up to `repeats` runs, with the peak memory of the worst of them.

    Best-of rather than mean: both libraries share one machine with a scheduler and
    a JIT, so the noise is one-sided. Repetition stops early once `REPEAT_BUDGET`
    seconds have gone into an operation, which keeps the large sizes affordable.
    """
    kwargs = dict(op.kwargs)
    if op.sized_kwargs is not None:
        kwargs.update(op.sized_kwargs(stage.n_obs))

    best: float | None = None
    peak_bytes = 0
    spent = 0.0
    for _ in range(repeats):
        adata = stage.copy()
        with PeakRss() as peak:
            start = time.perf_counter()
            try:
                function(adata, *op.args, **kwargs)
            except BaseException as exc:  # a refusal or a panicking stub is data, not a crash
                return {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}
            elapsed = time.perf_counter() - start
        del adata
        best = elapsed if best is None else min(best, elapsed)
        peak_bytes = max(peak_bytes, peak.peak_bytes or 0)
        spent += elapsed
        if spent >= REPEAT_BUDGET:
            break
    return {"seconds": best, "peak_mb": peak_bytes / 1e6 if peak_bytes else None}


def run_worker(library_name: str, n_cells: int, repeats: int) -> int:
    """Run every operation for one library, printing one JSON line per operation.

    Lines are flushed as they are produced, so a worker killed by the operating
    system still leaves everything it managed to measure.
    """
    if library_name == "scanpy":
        library: Any = sc
    else:
        import scrust

        library = scrust

    stages = prepare(resize(load_pbmc3k(), n_cells))
    read_rss = _rss_reader()
    print(
        json.dumps({"kind": "baseline", "rss_mb": (read_rss() / 1e6) if read_rss else None}),
        flush=True,
    )

    for op in OPS:
        stage = stages[op.stage]
        record: dict[str, Any] = {
            "kind": "result",
            "path": op.path,
            "cells": int(stage.n_obs),
            "genes": int(stage.n_vars),
            "seconds": None,
            "peak_mb": None,
            "error": "",
        }
        if op.max_cells is not None and stage.n_obs > op.max_cells and library_name == "scanpy":
            record["error"] = f"not attempted: {op.max_cells_reason}"
            print(json.dumps(record), flush=True)
            continue

        try:
            function = _resolve(library, op.path)
        except AttributeError as exc:
            record["error"] = f"AttributeError: {exc}"
            print(json.dumps(record), flush=True)
            continue

        record.update(_time_call(function, stage, op, repeats))
        print(json.dumps(record), flush=True)

    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(json.dumps({"kind": "high_water", "rss_mb": high_water}), flush=True)
    return 0


# -------------------------------------------------------------------------- driver


def _call_worker(
    library: str, n_cells: int, repeats: int
) -> tuple[dict[str, Any], str, float | None]:
    """Run one worker in its own process and collect its per-operation records."""
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            library,
            "--cells",
            str(n_cells),
            "--repeats",
            str(repeats),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    results: dict[str, Any] = {}
    baseline: float | None = None
    for line in process.stdout.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record["kind"] == "result":
            results[record["path"]] = record
        elif record["kind"] == "baseline":
            baseline = record["rss_mb"]
    note = ""
    if process.returncode != 0:
        tail = (process.stderr.strip().splitlines() or ["no stderr"])[-1][:200]
        note = f"the {library} worker at {n_cells} cells exited with {process.returncode}: {tail}"
    return results, note, baseline


def _seconds(record: dict[str, Any] | None) -> str:
    if record is None:
        return "no run"
    if record["seconds"] is None:
        return (record["error"].split(":")[0] or "failed")[:9]
    return f"{record['seconds']:.3f}"


def _memory(record: dict[str, Any] | None) -> str:
    if record is None or record["peak_mb"] is None:
        return "-"
    return f"{record['peak_mb']:.0f}"


def _baseline_line(pairs: tuple[tuple[str, float | None], ...]) -> str:
    return ", ".join(
        f"{name} {value:.0f} MB" if value else f"{name} unknown" for name, value in pairs
    )


def run(sizes: list[int], repeats: int) -> int:
    try:
        load_pbmc3k()
    except Exception as exc:
        print(f"cannot benchmark: PBMC 3k is unavailable ({type(exc).__name__}: {exc})")
        return 1

    header = (
        f"{'algorithm':<24}{'cells':>7}{'genes':>7}"
        f"{'scanpy s':>10}{'scrust s':>10}{'speedup':>9}"
        f"{'scanpy MB':>11}{'scrust MB':>11}"
    )
    unavailable: list[str] = []
    for n_cells in sizes:
        reference, reference_note, reference_base = _call_worker("scanpy", n_cells, repeats)
        ours, our_note, our_base = _call_worker("scrust", n_cells, repeats)
        unavailable.extend(note for note in (reference_note, our_note) if note)

        baselines = _baseline_line((("scanpy", reference_base), ("scrust", our_base)))
        print(f"\n=== {n_cells} cells requested; resident before timing: {baselines}")
        print(header)
        print("-" * len(header))
        for op in OPS:
            theirs, mine = reference.get(op.path), ours.get(op.path)
            shape = mine or theirs
            cells = f"{shape['cells']:>7}" if shape else f"{'?':>7}"
            genes = f"{shape['genes']:>7}" if shape else f"{'?':>7}"
            for library, record in (("scanpy", theirs), ("scrust", mine)):
                if record is None:
                    unavailable.append(
                        f"{library} {op.path} at {n_cells} cells: no record returned"
                    )
                elif record["seconds"] is None:
                    unavailable.append(
                        f"{library} {op.path} at {record['cells']} cells: {record['error']}"
                    )
            their_s = theirs["seconds"] if theirs else None
            our_s = mine["seconds"] if mine else None
            speedup = f"{their_s / our_s:>8.2f}x" if their_s and our_s else f"{'-':>9}"
            print(
                f"{op.path:<24}{cells}{genes}"
                f"{_seconds(theirs):>10}{_seconds(mine):>10}{speedup}"
                f"{_memory(theirs):>11}{_memory(mine):>11}"
            )

    print(
        "\nspeedup is scanpy seconds / scrust seconds; above 1.00x scrust is faster."
        "\npeak MB is the highest resident size of the worker process while the call ran,"
        "\nsampled every 5 ms; it includes the input matrix the call was handed."
        f"\neach timing is the best of up to {repeats} runs, and an operation stops"
        f" repeating after {REPEAT_BUDGET:.0f} s."
    )
    if unavailable:
        print("\nnot measured, with the reason:")
        for line in unavailable:
            print(f"  {line}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--worker", choices=["scanpy", "scrust"], help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cells", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        return run_worker(arguments.worker, arguments.cells, arguments.repeats)
    return run(arguments.sizes, arguments.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
