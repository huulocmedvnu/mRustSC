#!/bin/sh
# Energy of the out-of-core pipeline on a counts file larger than memory, Metal then CPU,
# under powermetrics. scanpy is not run: it cannot hold the file. Needs root:
#
#     sudo sh benches/run_energy_4m.sh /path/to/embryo_4m_counts.h5ad
#
# Writes benches/results/energy_4m_metalcyte_{metal,cpu}.json. Nothing else should be running.
set -eu
DATA="${1:?path to the counts-only .h5ad}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HERE/.venv/bin/python"
export PYTHONPATH="$HERE/python"
for dev in auto cpu; do
  tag="metalcyte_$([ "$dev" = auto ] && echo metal || echo cpu)"
  echo "== $tag"
  "$PY" "$HERE/benches/energy.py" "$DATA" --library metalcyte --device "$dev" --umap-parallel \
      --backed --json "$HERE/benches/results/energy_4m_$tag.json"
done
echo ENERGY4M_DONE
