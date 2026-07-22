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

## Develop

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest
```
