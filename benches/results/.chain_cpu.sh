#!/bin/sh
cd /Users/mac/Downloads/mRustSC_work/mRustSC
export PYTHONPATH=$PWD/python
PY=.venv/bin/python
D2=$HOME/Downloads/mRustSC_work/data/bone_marrow_117k_counts.h5ad
D1=$HOME/Downloads/mRustSC_work/data/embryo_1m_counts.h5ad
echo "== 117k scrust cpu (real)"; $PY benches/pipeline.py $D2 --library scrust --device cpu --umap-parallel --json benches/results/bm117k_scrust_cpu.json 2>&1 | grep -v Warn
echo "== 117k scrust metal (same binary, for a paired comparison)"; $PY benches/pipeline.py $D2 --library scrust --device auto --umap-parallel --json benches/results/bm117k_scrust_metal_umap_parallel.json 2>&1 | grep -v Warn
echo "== 1M scrust cpu (real)"; $PY benches/pipeline_1m.py $D1 --device cpu --umap-parallel --json benches/results/embryo1m_scrust_cpu.json 2>&1 | grep -v Warn
echo CHAIN_CPU_DONE
