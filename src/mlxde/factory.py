"""Composition root.

The only place where concrete implementations are wired together. Imports are
function-local so that adding a layer never widens the import graph of callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlxde.pipeline.differential_expression import DifferentialExpressionPipeline


def build_default_pipeline(backend_name: str | None = None) -> DifferentialExpressionPipeline:
    """DESeq2-style pipeline: median-of-ratios, NB GLM, Wald test, BH correction."""
    from mlxde.backend import get_backend
    from mlxde.pipeline.differential_expression import DifferentialExpressionPipeline
    from mlxde.preprocess.filtering import MinimumCountFilter
    from mlxde.preprocess.normalization import MedianOfRatiosSizeFactors
    from mlxde.stats.dispersion import MethodOfMomentsDispersion, TrendedDispersion
    from mlxde.stats.glm import NegativeBinomialGLM
    from mlxde.stats.hypothesis import WaldTest
    from mlxde.stats.multiple_testing import BenjaminiHochberg

    backend = get_backend(backend_name)
    return DifferentialExpressionPipeline(
        size_factors=MedianOfRatiosSizeFactors(),
        gene_filter=MinimumCountFilter(),
        dispersions=TrendedDispersion(MethodOfMomentsDispersion(backend)),
        fitter=NegativeBinomialGLM(backend),
        hypothesis_test=WaldTest(),
        correction=BenjaminiHochberg(),
    )
