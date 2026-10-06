#!/bin/sh
cd /Users/mac/Downloads/mRustSC_work/mRustSC
export PYTHONPATH=$PWD/python
PY=.venv/bin/python
echo "== 1M scrust cpu (parallel search)"; $PY benches/pipeline_1m.py $HOME/Downloads/mRustSC_work/data/embryo_1m_counts.h5ad --device cpu --umap-parallel --json benches/results/embryo1m_scrust_cpu.json 2>&1 | grep -v Warn
echo "== 117k scrust cpu (parallel search)"; $PY benches/pipeline.py $HOME/Downloads/mRustSC_work/data/bone_marrow_117k_counts.h5ad --library scrust --device cpu --umap-parallel --json benches/results/bm117k_scrust_cpu.json 2>&1 | grep -v Warn
echo CHAIN2_DONE
