#!/usr/bin/env python3
"""Seconds against cells: the same real pipeline at every size, both libraries.

    PYTHONPATH=$PWD/python .venv/bin/python benches/scaling.py data/embryo_1m_counts.h5ad \
        --json benches/results/scaling_embryo.json

Each (configuration, size) runs `benches/pipeline.py --cells N` in its own process on a
random subsample of one real atlas, so every point is the same biology at a different
scale (not a bootstrap). A configuration that fails or is killed at a size is marked and
not tried at any larger size; a swap watchdog kills a run that drives the machine past
`--swap-limit-gb`, which is what scanpy does on this laptop above a few hundred thousand
cells. The metalcyte point at the full size comes from `pipeline_1m.py` (the streamed head),
because the in-memory pipeline cannot hold the dense scaled matrix either.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent

SIZES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
CONFIGS = {
    "scanpy": ["--library", "scanpy"],
    "scanpy_tuned": ["--library", "scanpy", "--tuned"],
    "metalcyte_cpu": ["--library", "metalcyte", "--device", "cpu", "--umap-parallel"],
    "metalcyte_metal": ["--library", "metalcyte", "--device", "auto", "--umap-parallel"],
}


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def run_one(cmd: list[str], out: Path, swap_limit_mb: float, timeout_s: float) -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "python"))
    log = out.with_suffix(".log")
    with log.open("w") as handle:
        proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        started = time.time()
        # Swap is judged by what this run adds: macOS keeps a previous run's swap for a
        # while, so an absolute threshold would charge one run for another's memory.
        swap_at_start = swap_used_mb()
        while proc.poll() is None:
            time.sleep(5)
            if swap_used_mb() - swap_at_start > swap_limit_mb:
                proc.kill()
                return {"outcome": f"killed: added more than {swap_limit_mb / 1024:.0f} GB of swap"}
            if time.time() - started > timeout_s:
                proc.kill()
                return {"outcome": f"killed: over {timeout_s / 60:.0f} min"}
    if proc.returncode != 0:
        tail = log.read_text().strip().splitlines()[-1:] or ["no output"]
        return {"outcome": f"failed: {tail[0][:160]}"}
    result = json.loads(out.read_text())
    return {
        "outcome": "ok",
        "total_seconds": result["total_seconds"],
        "steps": {s["step"]: s["seconds"] for s in result["steps"]},
        "n_cells": (result["steps"][-1].get("shape") or result.get("shape") or [None])[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--only", nargs="*", help="configuration names")
    parser.add_argument("--swap-limit-gb", type=float, default=12.0)
    parser.add_argument("--timeout-min", type=float, default=40.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    runs_dir = args.json.parent / "scaling_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report = (
        json.loads(args.json.read_text())
        if args.json.exists()
        else {"file": str(args.h5ad), "points": []}
    )
    done = {(p["config"], p["size"]) for p in report["points"]}
    dead: set[str] = {p["config"] for p in report["points"] if p["outcome"] != "ok"}

    for size in sorted(args.sizes):
        for name, extra in CONFIGS.items():
            if args.only and name not in args.only:
                continue
            if (name, size) in done or name in dead:
                continue
            out = runs_dir / f"{name}_{size}.json"
            if size >= 1_000_000 and name.startswith("metalcyte"):
                cmd = [
                    sys.executable,
                    str(BENCH_DIR / "pipeline_1m.py"),
                    str(args.h5ad),
                    "--json",
                    str(out),
                ]
                cmd += ["--device", "cpu" if name == "metalcyte_cpu" else "auto", "--umap-parallel"]
            else:
                cmd = [
                    sys.executable,
                    str(BENCH_DIR / "pipeline.py"),
                    str(args.h5ad),
                    "--cells",
                    str(size),
                    "--json",
                    str(out),
                    *extra,
                ]
            print(f"== {name} at {size:,} cells", flush=True)
            point = {
                "config": name,
                "size": size,
                **run_one(cmd, out, args.swap_limit_gb * 1024, args.timeout_min * 60),
            }
            print(
                f"   {point['outcome']}"
                + (f", {point['total_seconds']:.1f} s" if point["outcome"] == "ok" else ""),
                flush=True,
            )
            report["points"].append(point)
            if point["outcome"] != "ok":
                dead.add(name)
            args.json.write_text(json.dumps(report, indent=2))
    print("SCALING_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
