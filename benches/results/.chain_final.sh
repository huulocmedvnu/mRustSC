#!/bin/sh
cd /Users/mac/Downloads/mRustSC_work/mRustSC
export PYTHONPATH=$PWD/python
PY=.venv/bin/python
D2=$HOME/Downloads/mRustSC_work/data/bone_marrow_117k_counts.h5ad
D1=$HOME/Downloads/mRustSC_work/data/embryo_1m_counts.h5ad
echo "== agreement 117k"; $PY benches/agreement.py $D2 --json benches/results/agreement_bm117k.json 2>&1 | grep -v Warn
mkdir -p benches/results/repeats
for i in 1 2 3; do
  echo "== repeat $i scrust metal"; $PY benches/pipeline.py $D2 --library scrust --device auto --umap-parallel --json benches/results/repeats/scrust_metal_$i.json 2>&1 | grep "total "
  echo "== repeat $i scrust cpu"; $PY benches/pipeline.py $D2 --library scrust --device cpu --umap-parallel --json benches/results/repeats/scrust_cpu_$i.json 2>&1 | grep "total "
  echo "== repeat $i scanpy tuned"; $PY benches/pipeline.py $D2 --library scanpy --tuned --json benches/results/repeats/scanpy_tuned_$i.json 2>&1 | grep "total "
done
for i in 1 2 3; do
  echo "== repeat $i scanpy defaults"; $PY benches/pipeline.py $D2 --library scanpy --json benches/results/repeats/scanpy_$i.json 2>&1 | grep "total "
done
echo "== 1M save"; $PY benches/pipeline_1m.py $D1 --umap-parallel --save benches/results/embryo1m_scrust_metal.h5ad --json benches/results/embryo1m_scrust_metal.json 2>&1 | grep -v Warn | tail -6
echo CHAIN_FINAL_DONE
