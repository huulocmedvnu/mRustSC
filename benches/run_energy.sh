#!/bin/sh
# Energy of the 117k-cell pipeline, scanpy against scrust, under powermetrics.
# Needs root for the power counters:
#
#     sudo sh benches/run_energy.sh data/bone_marrow_117k_counts.h5ad
#
# Writes benches/results/energy_bm117k_{scanpy,scrust_metal,scrust_cpu}.json and prints
# the net joules of each run. Nothing else should be running on the machine.
set -eu
DATA="${1:?path to the counts .h5ad}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HERE/.venv/bin/python"
export PYTHONPATH="$HERE/python"
for spec in "scanpy auto" "scrust auto" "scrust cpu"; do
  set -- $spec
  lib=$1; dev=$2
  tag="$lib"; [ "$lib" = scrust ] && tag="scrust_$([ "$dev" = auto ] && echo metal || echo cpu)"
  echo "== $tag"
  "$PY" "$HERE/benches/energy.py" "$DATA" --library "$lib" --device "$dev" --umap-parallel \
      --json "$HERE/benches/results/energy_bm117k_$tag.json"
done
