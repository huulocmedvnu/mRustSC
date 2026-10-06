#!/usr/bin/env python3
"""Switch off one Apple-specific choice at a time and time the pipeline again.

    PYTHONPATH=$PWD/python .venv/bin/python benches/ablation.py \
        data/bone_marrow_117k_counts.h5ad --json benches/results/ablation_bm117k.json
    PYTHONPATH=$PWD/python .venv/bin/python benches/ablation.py \
        data/embryo_1m_counts.h5ad --million --json benches/results/ablation_embryo1m.json

Every configuration runs `benches/pipeline.py` (or `pipeline_1m.py` with `--million`) in
its own process, so a knob set through the environment holds for that run only:

| configuration | what is off | how |
|---|---|---|
| `all`             | nothing                         | Metal, Accelerate, 11 threads, zero-copy |
| `no_metal`        | the GPU                         | `--device cpu` |
| `p_cores_only`    | the 6 efficiency cores          | `RAYON_NUM_THREADS=5` |
| `one_core`        | every core but one              | `RAYON_NUM_THREADS=1` |
| `no_zero_copy`    | the in-place numpy borrows      | `SILICELL_FORCE_COPY=1` |
| `no_accelerate`   | Apple's BLAS (AMX)              | `.venv-noaccel` wheel, no `accelerate` |
| `sequential_umap` | the Hogwild optimiser           | no `--umap-parallel` |

A configuration whose prerequisite is missing (no `.venv-noaccel`) is reported as
skipped rather than silently measured with the wrong binary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
NOACCEL = ROOT / ".venv-noaccel" / "bin" / "python"

CONFIGS = [
    ("all", {}, [], None),
    ("no_metal", {}, ["--device", "cpu"], None),
    ("p_cores_only", {"RAYON_NUM_THREADS": "5"}, [], None),
    ("one_core", {"RAYON_NUM_THREADS": "1"}, [], None),
    ("no_zero_copy", {"SILICELL_FORCE_COPY": "1"}, [], None),
    ("no_accelerate", {}, [], NOACCEL),
    ("sequential_umap", {}, ["--no-umap-parallel"], None),
]


def run_one(name, env_extra, args, python, h5ad, million, out_dir):
    python = python or Path(sys.executable)
    if not python.exists():
        return {"config": name, "skipped": f"{python} not found"}
    script = BENCH_DIR / ("pipeline_1m.py" if million else "pipeline.py")
    out = out_dir / f"ablation_{name}.json"
    cmd = [str(python), str(script), str(h5ad), "--json", str(out)]
    if not million:
        cmd += ["--library", "silicell"]
    if "--no-umap-parallel" not in args:
        cmd.append("--umap-parallel")
    cmd += [a for a in args if a != "--no-umap-parallel"]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "python"))
    env.update(env_extra)
    print(
        f"== {name}: {' '.join(f'{k}={v}' for k, v in env_extra.items())} {' '.join(args)}",
        flush=True,
    )
    done = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["no stderr"]
        print(f"   failed: {tail[0][:200]}", flush=True)
        return {"config": name, "failed": tail[0][:500]}
    result = json.loads(out.read_text())
    steps = {s["step"]: s["seconds"] for s in result["steps"]}
    print(f"   total {result['total_seconds']:.1f} s", flush=True)
    return {
        "config": name,
        "env": env_extra,
        "args": args,
        "total_seconds": result["total_seconds"],
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--million", action="store_true", help="use pipeline_1m.py (streamed head)")
    parser.add_argument("--only", nargs="*", help="configuration names to run")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out_dir = (args.json.parent if args.json else BENCH_DIR / "results") / "ablation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, env_extra, extra, python in CONFIGS:
        if args.only and name not in args.only:
            continue
        rows.append(run_one(name, env_extra, extra, python, args.h5ad, args.million, out_dir))
    report = {"file": str(args.h5ad), "million": args.million, "rows": rows}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    base = next((r for r in rows if r.get("config") == "all" and "total_seconds" in r), None)
    if base:
        print("\nconfiguration        total s   vs all")
        for r in rows:
            if "total_seconds" in r:
                ratio = r["total_seconds"] / base["total_seconds"]
                print(f"{r['config']:20s} {r['total_seconds']:8.1f}   {ratio:5.2f}x")
            else:
                print(f"{r['config']:20s} {'skipped' if 'skipped' in r else 'failed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
