#!/usr/bin/env python3
"""Joules, not just seconds: the energy a pipeline run draws from the package.

Run from the repo root, with sudo because `powermetrics` reads the power counters:

    sudo PYTHONPATH=$PWD/python .venv/bin/python benches/energy.py data.h5ad \
        --library metalcyte --device auto --json results/energy_metalcyte.json

Apple silicon exposes per-rail package power (CPU, GPU, ANE and their sum) through
`powermetrics`. This script samples those rails every 100 ms while `benches/pipeline.py`
runs the whole pipeline in a child process, integrates power over the run and subtracts
the idle draw measured for `--idle-seconds` just before, so the result is the energy the
run *added*. Both the gross and the net figures are reported, with the mean power.

`powermetrics` is only available on macOS. The child runs as the invoking user when
`SUDO_USER` is set, so caches and settings stay theirs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
PIPELINE = BENCH_DIR / "pipeline.py"

_ELAPSED = re.compile(r"\((\d+(?:\.\d+)?)ms elapsed\)")
_RAIL = re.compile(r"^(CPU|GPU|ANE|Combined) Power(?: \(.*?\))?: (\d+(?:\.\d+)?) mW", re.M)


def _powermetrics(log: Path, interval_ms: int) -> subprocess.Popen:
    cmd = [
        "powermetrics",
        "--samplers",
        "cpu_power,gpu_power,thermal",
        "-i",
        str(interval_ms),
        "-o",
        str(log),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _integrate(log: Path) -> dict[str, Any]:
    """Energy per rail in joules and the covered wall time from a powermetrics log."""
    text = log.read_text(errors="replace")
    samples = text.split("*** Sampled system activity")
    joules: dict[str, float] = {}
    seconds = 0.0
    for sample in samples[1:]:
        elapsed = _ELAPSED.search(sample)
        if not elapsed:
            continue
        dt = float(elapsed.group(1)) / 1000.0
        seconds += dt
        for rail, milliwatts in _RAIL.findall(sample):
            joules[rail] = joules.get(rail, 0.0) + float(milliwatts) / 1000.0 * dt
    return {"seconds": seconds, "joules": joules, "n_samples": len(samples) - 1}


def _measure(log: Path, interval_ms: int, run: Any) -> dict[str, Any]:
    proc = _powermetrics(log, interval_ms)
    time.sleep(1.0)  # let the first sample land before the work starts
    try:
        payload = run()
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    result = _integrate(log)
    result["payload"] = payload
    return result


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(BENCH_DIR.parent / "python"))
    return env


def _as_user_prefix() -> list[str]:
    """Run the pipeline as the invoking user, with the environment sudo strips restored."""
    user = os.environ.get("SUDO_USER")
    if not user:
        return []
    home = Path("~" + user).expanduser()
    return [
        "sudo",
        "-u",
        user,
        "env",
        f"HOME={home}",
        f"PYTHONPATH={_child_env()['PYTHONPATH']}",
        f"SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', '')}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--library", choices=["scanpy", "metalcyte"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--umap-parallel", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("powermetrics exists only on macOS", file=sys.stderr)
        return 2
    if os.geteuid() != 0:
        print("run with sudo: powermetrics needs root to read the power counters", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        # The pipeline runs as the invoking user (`sudo -u`), who must be able to write here.
        os.chmod(tmp, 0o777)
        idle_log = Path(tmp) / "idle.txt"
        run_log = Path(tmp) / "run.txt"
        pipeline_json = Path(tmp) / "pipeline.json"

        print(f"idle for {args.idle_seconds:.0f} s ...", flush=True)
        idle = _measure(idle_log, args.interval_ms, lambda: time.sleep(args.idle_seconds))

        cmd = [
            *_as_user_prefix(),
            sys.executable,
            str(PIPELINE),
            str(args.h5ad),
            "--library",
            args.library,
            "--device",
            args.device,
            "--json",
            str(pipeline_json),
        ]
        if args.cells:
            cmd += ["--cells", str(args.cells)]
        if args.umap_parallel:
            cmd.append("--umap-parallel")

        def run_pipeline() -> dict[str, Any]:
            subprocess.run(cmd, check=True, env=_child_env())
            return json.loads(pipeline_json.read_text())

        print(f"{args.library} ({args.device}) under powermetrics ...", flush=True)
        work = _measure(run_log, args.interval_ms, run_pipeline)
        raw_log = run_log.read_text(errors="replace") if args.json else ""

    idle_watts = (
        {k: v / idle["seconds"] for k, v in idle["joules"].items()} if idle["seconds"] else {}
    )
    net = {k: v - idle_watts.get(k, 0.0) * work["seconds"] for k, v in work["joules"].items()}
    mean_w = {k: v / work["seconds"] for k, v in work["joules"].items()} if work["seconds"] else {}

    report = {
        "library": args.library,
        "device": work["payload"]["device"],
        "pipeline_seconds": work["payload"]["total_seconds"],
        "sampled_seconds": work["seconds"],
        "gross_joules": work["joules"],
        "idle_watts": idle_watts,
        "net_joules": net,
        "mean_watts": mean_w,
        "pipeline": work["payload"],
    }
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "library",
                    "device",
                    "pipeline_seconds",
                    "gross_joules",
                    "net_joules",
                    "mean_watts",
                    "idle_watts",
                )
            },
            indent=2,
        )
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        # The raw samples (per-rail power, cluster residency) feed the utilisation timeline.
        raw_path = args.json.with_suffix(".powermetrics.txt")
        raw_path.write_text(raw_log)
        if user := os.environ.get("SUDO_USER"):
            subprocess.run(["chown", user, str(args.json), str(raw_path)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
