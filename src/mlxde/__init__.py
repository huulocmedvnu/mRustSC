"""GPU-accelerated differential expression analysis for Apple silicon."""

from mlxde.backend import available_backends, get_backend
from mlxde.contracts import (
    CountMatrix,
    DesignMatrix,
    DifferentialExpressionResult,
    GLMFit,
    TestStatistics,
)

__version__ = "0.1.0"

__all__ = [
    "CountMatrix",
    "DesignMatrix",
    "DifferentialExpressionResult",
    "GLMFit",
    "TestStatistics",
    "__version__",
    "available_backends",
    "get_backend",
]
