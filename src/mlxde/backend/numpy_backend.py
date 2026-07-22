"""CPU reference backend.

Serves as the correctness baseline the GPU backend is tested against, and as the
fallback on machines without Apple silicon.
"""

from __future__ import annotations

import numpy as np

from mlxde.contracts import Array


class NumpyBackend:
    """ComputeBackend implemented with NumPy on the CPU."""

    @property
    def name(self) -> str:
        return "numpy"

    def is_available(self) -> bool:
        return True

    def asarray(self, values: np.ndarray) -> Array:
        return np.asarray(values, dtype=np.float64)

    def asnumpy(self, values: Array) -> np.ndarray:
        return np.asarray(values)

    def exp(self, values: Array) -> Array:
        return np.exp(values)

    def log(self, values: Array) -> Array:
        return np.log(values)

    def sqrt(self, values: Array) -> Array:
        return np.sqrt(values)

    def abs(self, values: Array) -> Array:
        return np.abs(values)

    def clip(self, values: Array, low: float, high: float) -> Array:
        return np.clip(values, low, high)

    def maximum(self, values: Array, other: Array | float) -> Array:
        return np.maximum(values, other)

    def where(self, condition: Array, if_true: Array, if_false: Array) -> Array:
        return np.where(condition, if_true, if_false)

    def sum(self, values: Array, axis: int | None = None) -> Array:
        return np.sum(values, axis=axis)

    def mean(self, values: Array, axis: int | None = None) -> Array:
        return np.mean(values, axis=axis)

    def matmul(self, left: Array, right: Array) -> Array:
        return np.matmul(left, right)

    def transpose(self, values: Array, axes: tuple[int, ...] | None = None) -> Array:
        return np.transpose(values, axes)

    def solve(self, matrices: Array, vectors: Array) -> Array:
        return np.linalg.solve(matrices, vectors)

    def inverse(self, matrices: Array) -> Array:
        return np.linalg.inv(matrices)
