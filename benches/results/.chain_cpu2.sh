#!/bin/sh
cd /Users/mac/Downloads/metalcyte_work/metalcyte
export PYTHONPATH=$PWD/python
PY=.venv/bin/python
echo "== 1M metalcyte cpu (parallel search)"; $PY benches/pipeline_1m.py $HOME/Downloads/metalcyte_work/data/embryo_1m_counts.h5ad --device cpu --umap-parallel --json benches/results/embryo1m_scrust_cpu.json 2>&1 | grep -v Warn
echo "== 117k metalcyte cpu (parallel search)"; $PY benches/pipeline.py $HOME/Downloads/metalcyte_work/data/bone_marrow_117k_counts.h5ad --library metalcyte --device cpu --umap-parallel --json benches/results/bm117k_scrust_cpu.json 2>&1 | grep -v Warn
echo CHAIN2_DONE
