# API contract

Every module below implements protocols from `mlxde.contracts`. The contract is
frozen: implementations may change freely, these names and signatures may not.
Ten branches are developed in parallel against this document, so a branch that
renames a symbol here breaks the branches it is merged with.

| Branch | Module | Public API |
| --- | --- | --- |
| `feat/backend-mlx` | `mlxde/backend/mlx_backend.py` | `MLXBackend()` — `ComputeBackend` on the Apple GPU |
| `feat/io` | `mlxde/io/readers.py`, `writers.py`, `design.py` | `CsvCountReader()`, `CsvResultWriter()`, `build_design_matrix(...)` |
| (post-merge) | `mlxde/io/pseudobulk.py` | `build_pseudobulk(...)` |
| `feat/preprocess` | `mlxde/preprocess/normalization.py`, `filtering.py` | `MedianOfRatiosSizeFactors()`, `TotalCountSizeFactors()`, `MinimumCountFilter(...)` |
| `feat/dispersion` | `mlxde/stats/dispersion.py` | `MethodOfMomentsDispersion(...)`, `TrendedDispersion(...)` |
| `feat/glm` | `mlxde/stats/glm.py` | `NegativeBinomialGLM(backend, ...)` |
| `feat/hypothesis` | `mlxde/stats/hypothesis.py` | `WaldTest()`, `LikelihoodRatioTest(fitter)` |
| `feat/multiple-testing` | `mlxde/stats/multiple_testing.py` | `BenjaminiHochberg()`, `Bonferroni()` |
| `feat/pipeline` | `mlxde/pipeline/differential_expression.py` | `DifferentialExpressionPipeline(...)` |
| `feat/cli` | `mlxde/cli/main.py` | `app` (Typer) with the `run` command |
| `feat/report` | `mlxde/report/plots.py`, `summary.py` | `volcano_plot(...)`, `ma_plot(...)`, `summarize(...)` |

## Signatures

```python
# mlxde/backend/mlx_backend.py
class MLXBackend:                       # ComputeBackend, runs on the Apple GPU
    def __init__(self, dtype: str = "float32") -> None: ...

# mlxde/io/design.py
def build_design_matrix(
    sample_metadata: pd.DataFrame,
    condition_column: str,
    reference_level: str | None = None,
    covariate_columns: Sequence[str] = (),
) -> DesignMatrix: ...
# Column 0 is the intercept, named "intercept"; treatment columns are named
# f"{condition_column}[{level}]" so callers can request them via
# DesignMatrix.contrast(name).

# mlxde/io/pseudobulk.py
def build_pseudobulk(
    counts: pd.DataFrame,             # cells x genes
    cell_labels: pd.Series,           # cell id -> population label
    conditions: Mapping[str, str],    # population label -> condition
    n_replicates: int = 5,
    seed: int = 0,
) -> CountMatrix: ...

# mlxde/io/readers.py
class CsvCountReader:                   # CountReader
    def __init__(self, gene_id_column: str = "gene_id",
                 sample_id_column: str = "sample_id") -> None: ...
    def read(self, counts_path: Path, metadata_path: Path) -> CountMatrix: ...

# mlxde/io/writers.py
class CsvResultWriter:                  # ResultWriter
    def write(self, result: DifferentialExpressionResult, path: Path) -> None: ...

# mlxde/preprocess/normalization.py
class MedianOfRatiosSizeFactors: ...    # SizeFactorEstimator (DESeq2-style)
class TotalCountSizeFactors: ...        # SizeFactorEstimator (library size / mean)

# mlxde/preprocess/filtering.py
class MinimumCountFilter:               # GeneFilter
    def __init__(self, min_count: int = 10, min_samples: int = 3) -> None: ...

# mlxde/stats/dispersion.py
class MethodOfMomentsDispersion:        # DispersionEstimator
    def __init__(self, backend: ComputeBackend, minimum: float = 1e-8) -> None: ...
class TrendedDispersion:                # DispersionEstimator, shrinks towards a mean-trend
    def __init__(self, base: DispersionEstimator, shrinkage_weight: float = 0.5) -> None: ...

# mlxde/stats/glm.py
class NegativeBinomialGLM:              # GLMFitter, batched IRLS on the backend
    def __init__(self, backend: ComputeBackend, max_iterations: int = 100,
                 tolerance: float = 1e-6) -> None: ...

# mlxde/stats/hypothesis.py
class WaldTest: ...                     # HypothesisTest, normal approximation
class LikelihoodRatioTest:              # HypothesisTest, chi-squared
    def __init__(self, fitter: GLMFitter, design: DesignMatrix,
                 counts: np.ndarray, size_factors: np.ndarray) -> None: ...

# mlxde/stats/multiple_testing.py
class BenjaminiHochberg: ...            # MultipleTestingCorrection
class Bonferroni: ...                   # MultipleTestingCorrection

# mlxde/pipeline/differential_expression.py
class DifferentialExpressionPipeline:
    def __init__(self, size_factors: SizeFactorEstimator, gene_filter: GeneFilter,
                 dispersions: DispersionEstimator, fitter: GLMFitter,
                 hypothesis_test: HypothesisTest,
                 correction: MultipleTestingCorrection) -> None: ...
    def run(self, count_matrix: CountMatrix, design: DesignMatrix,
            contrast: np.ndarray) -> DifferentialExpressionResult: ...

# mlxde/report/plots.py
def volcano_plot(result: DifferentialExpressionResult, alpha: float = 0.05,
                 min_abs_log2_fold_change: float = 1.0) -> Figure: ...
def ma_plot(result: DifferentialExpressionResult, alpha: float = 0.05) -> Figure: ...

# mlxde/report/summary.py
def summarize(result: DifferentialExpressionResult, alpha: float = 0.05) -> str: ...
```

## Numerical conventions

- Counts are `(n_genes, n_samples)`; design matrices are `(n_samples, n_coefficients)`.
- Coefficients and contrasts live on the **natural-log** scale; only the final
  result table is converted to log2.
- Size factors are strictly positive and normalised to a geometric mean of 1.
- `base_mean` is the mean of size-factor-normalised counts per gene.
- Genes removed by the `GeneFilter` are excluded from multiple-testing correction.
