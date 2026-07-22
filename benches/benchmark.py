#!/usr/bin/env python3
"""Time every algorithm against scanpy on the same input, across a few dataset sizes.

Run from the repo root:

    .venv/bin/python benches/benchmark.py --sizes 500 1500 2700

Anything that cannot be timed is reported with its reason, both in the table and in a
footer, so an incomplete run can never be mistaken for a fast one.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
from anndata import AnnData

REPO_ROOT = Path(__file__).resolve().parents[1]
# The test suite already owns dataset caching; reuse it rather than keeping a second copy.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import configure_datasetdir  # noqa: E402

TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_COMPS = 50
N_NEIGHBORS = 15


@dataclass(frozen=True)
class Op:
    """One algorithm, the prepared input it consumes and the arguments it takes."""

    path: str
    stage: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


OPS = (
    Op("pp.filter_cells", "raw", kwargs={"min_genes": 200}),
    Op("pp.filter_genes", "raw", kwargs={"min_cells": 3}),
    Op("pp.normalize_total", "counts", kwargs={"target_sum": TARGET_SUM}),
    Op("pp.log1p", "normalized"),
    Op("pp.highly_variable_genes", "lognorm", kwargs={"n_top_genes": N_TOP_GENES}),
    Op("pp.scale", "lognorm", kwargs={"zero_center": True, "max_value": 10}),
    Op("pp.pca", "scaled", kwargs={"n_comps": N_COMPS, "random_state": 0}),
    Op("pp.neighbors", "embedded", kwargs={"n_neighbors": N_NEIGHBORS, "use_rep": "X_pca"}),
    Op("tl.umap", "neighbored", kwargs={"random_state": 0}),
    Op("tl.tsne", "embedded", kwargs={"n_pcs": N_COMPS, "perplexity": 30.0, "random_state": 0}),
    Op("tl.rank_genes_groups", "lognorm", args=("group",), kwargs={"method": "wilcoxon"}),
)


def load_pbmc3k() -> AnnData:
    configure_datasetdir()
    counts = sc.datasets.pbmc3k()
    counts.var_names_make_unique()
    published = sc.datasets.pbmc3k_processed()
    adata = counts[published.obs_names].copy()
    adata.obs["group"] = published.obs["louvain"].values
    return adata


def subsample(adata: AnnData, n_cells: int, seed: int = 0) -> AnnData:
    if n_cells >= adata.n_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    return adata[rng.choice(adata.n_obs, n_cells, replace=False)].copy()


def prepare(adata: AnnData) -> dict[str, AnnData]:
    """The input each algorithm expects, all of it built by scanpy so both libraries
    are timed on identical data."""
    counts = adata.copy()
    sc.pp.filter_genes(counts, min_cells=3)
    normalized = counts.copy()
    sc.pp.normalize_total(normalized, target_sum=TARGET_SUM)
    lognorm = normalized.copy()
    sc.pp.log1p(lognorm)
    scaled = lognorm.copy()
    sc.pp.highly_variable_genes(scaled, n_top_genes=N_TOP_GENES)
    scaled = scaled[:, scaled.var["highly_variable"].to_numpy()].copy()
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
        "scaled": scaled,
        "embedded": embedded,
        "neighbored": neighbored,
    }


def _resolve(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = getattr(root, part)
    return root


def time_call(library: Any, op: Op, stage: AnnData) -> tuple[float | None, str]:
    """Seconds taken, or None plus the reason it could not run."""
    try:
        fn = _resolve(library, op.path)
    except (ImportError, AttributeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    adata = stage.copy()
    start = time.perf_counter()
    try:
        fn(adata, *op.args, **op.kwargs)
    except BaseException as exc:  # a panicking stub is reported, never fatal
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return time.perf_counter() - start, ""


def run(sizes: list[int]) -> int:
    try:
        pbmc3k = load_pbmc3k()
    except Exception as exc:
        print(f"cannot benchmark: PBMC 3k is unavailable ({type(exc).__name__}: {exc})")
        return 1

    try:
        import scrust
    except ImportError as exc:
        print(f"cannot benchmark: scrust is not importable ({exc}); scanpy timings only")
        scrust = None

    header = (
        f"{'algorithm':<26}{'cells':>7}{'genes':>8}{'scanpy s':>11}{'scrust s':>11}{'speedup':>9}"
    )
    print(header)
    print("-" * len(header))
    unavailable: list[str] = []
    for n_cells in sizes:
        stages = prepare(subsample(pbmc3k, n_cells))
        for op in OPS:
            stage = stages[op.stage]
            ref_time, ref_why = time_call(sc, op, stage)
            our_time, our_why = (None, "scrust not importable")
            if scrust is not None:
                our_time, our_why = time_call(scrust, op, stage)
            for library, seconds, why in (
                ("scrust", our_time, our_why),
                ("scanpy", ref_time, ref_why),
            ):
                if seconds is None:
                    unavailable.append(f"{library} {op.path} at {stage.n_obs} cells: {why}")
            speedup = f"{ref_time / our_time:>8.2f}x" if ref_time and our_time else f"{'-':>9}"
            print(
                f"{op.path:<26}{stage.n_obs:>7}{stage.n_vars:>8}"
                f"{_cell(ref_time, ref_why):>11}{_cell(our_time, our_why):>11}{speedup}"
            )
        print()

    if unavailable:
        print("not measured, with the reason:")
        for line in unavailable:
            print(f"  {line}")
    return 0


def _cell(seconds: float | None, why: str) -> str:
    """A timing, or the exception type — the footer carries the full reason."""
    return f"{seconds:.3f}" if seconds is not None else (why.split(":")[0] or "failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 2700])
    return run(parser.parse_args().sizes)


if __name__ == "__main__":
    raise SystemExit(main())
