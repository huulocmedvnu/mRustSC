# mlxde

Differential expression analysis for RNA-seq count data, running the numerical
work on the GPU of Apple M-series chips through [MLX](https://github.com/ml-explore/mlx).

The model is the one used by DESeq2: median-of-ratios normalisation, a per-gene
negative binomial GLM fitted by iteratively reweighted least squares, a Wald test
on the contrast of interest, and Benjamini-Hochberg correction. Every gene is
fitted in the same batched kernel, which is what makes the GPU worth using.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,gpu,plots]"
```

`mlx` only installs on Apple silicon; without it the package falls back to the
NumPy backend automatically.

## Use

```bash
mlxde run --counts counts.csv --metadata samples.csv \
          --condition condition --reference control --output results.csv
```

```python
from mlxde.factory import build_default_pipeline
from mlxde.io.readers import CsvCountReader
from mlxde.io.design import build_design_matrix

counts = CsvCountReader().read("counts.csv", "samples.csv")
design = build_design_matrix(counts.sample_metadata, "condition", reference_level="control")
result = build_default_pipeline().run(counts, design, design.contrast("condition[treated]"))
print(result.significant(alpha=0.05, min_abs_log2_fold_change=1.0))
```

## Layout

```
src/mlxde/
  contracts.py    protocols and data types shared by every layer
  backend/        compute devices (MLX GPU, NumPy CPU) behind one interface
  io/             reading counts, building design matrices, writing results
  preprocess/     size factors and gene filtering
  stats/          dispersion, GLM, hypothesis tests, multiple testing
  pipeline/       orchestration, depends only on protocols
  report/         plots and text summaries
  cli/            command line entry point
  factory.py      composition root wiring concrete classes together
```

See `docs/ARCHITECTURE.md` for the layering rules and `docs/API_CONTRACT.md` for
the frozen public API.

## Performance and limits

GLM fit over 60 000 genes: 0.043 s on the GPU vs 0.291 s on the CPU (6.8x, M3 Pro).
The GPU only wins once the gene batch is large enough to hide kernel launch
latency — below ~5 000 genes the two are comparable.

On real data (10x PBMC 3k, pseudobulked by cell type) the pipeline calls 14/14
canonical marker genes in the correct direction and shares 91 of its top 100
monocyte genes with scanpy's Wilcoxon ranking.

Adjusted p-values are well calibrated at the family level (0-1 discoveries under
the global null) but slightly optimistic per gene on small designs, because the
Wald test and method-of-moments dispersion are used without Cox-Reid adjustment.
Measured numbers and their consequences are in `docs/VALIDATION.md`.

## Develop

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest
PYTHONPATH=src .venv/bin/python scripts/benchmark.py
```

The ten feature branches (`feat/*`) were developed in parallel against the frozen
interfaces in `docs/API_CONTRACT.md` and merged into `main`; the history keeps
them separate.
