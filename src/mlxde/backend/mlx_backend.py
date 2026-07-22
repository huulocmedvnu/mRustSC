"""Apple GPU backend.

Pure device wrapper: it knows about arrays and linear algebra, never about
counts, dispersions or p-values. The statistics layers stack every gene into one
batch and hand it here, so each call is a single large kernel launch.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlxde.contracts import Array

_DTYPES: dict[str, mx.Dtype] = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


class MLXBackend:
    """ComputeBackend implemented with MLX on the Apple M-series GPU."""

    def __init__(self, dtype: str = "float32") -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; available: {sorted(_DTYPES)}")
        self._dtype = _DTYPES[dtype]

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        return mx.is_available(mx.gpu)

    def asarray(self, values: np.ndarray) -> Array:
        return mx.array(values, dtype=self._dtype)

    def asnumpy(self, values: Array) -> np.ndarray:
        # MLX is lazy: np.array materialises the graph before the caller sees it.
        return np.array(values)

    def exp(self, values: Array) -> Array:
        return mx.exp(values)

    def log(self, values: Array) -> Array:
        return mx.log(values)

    def sqrt(self, values: Array) -> Array:
        return mx.sqrt(values)

    def abs(self, values: Array) -> Array:
        return mx.abs(values)

    def clip(self, values: Array, low: float, high: float) -> Array:
        return mx.clip(values, low, high)

    def maximum(self, values: Array, other: Array | float) -> Array:
        return mx.maximum(values, other)

    def where(self, condition: Array, if_true: Array, if_false: Array) -> Array:
        return mx.where(condition, if_true, if_false)

    def sum(self, values: Array, axis: int | None = None) -> Array:
        return mx.sum(values, axis=axis)

    def mean(self, values: Array, axis: int | None = None) -> Array:
        return mx.mean(values, axis=axis)

    def matmul(self, left: Array, right: Array) -> Array:
        return mx.matmul(left, right)

    def transpose(self, values: Array, axes: tuple[int, ...] | None = None) -> Array:
        return mx.transpose(values, axes)

    def solve(self, matrices: Array, vectors: Array) -> Array:
        """Solve ``matrices @ x == vectors`` for a batch of small systems.

        ``matrices`` is (batch, p, p) and ``vectors`` is (batch, p).
        """
        return self._gauss_jordan(matrices, vectors[..., None])[..., 0]

    def inverse(self, matrices: Array) -> Array:
        """Invert a batch of (batch, p, p) matrices."""
        size = matrices.shape[-1]
        identity = mx.broadcast_to(mx.eye(size, dtype=matrices.dtype), matrices.shape)
        return self._gauss_jordan(matrices, identity)

    @staticmethod
    def _gauss_jordan(matrices: Array, right_hand_sides: Array) -> Array:
        """Batched Gauss-Jordan elimination without pivoting.

        ``mlx.core.linalg`` (0.32) runs solve/inv/cholesky on the CPU stream only,
        which for the batch sizes here (up to ~50k systems) is 6-30x slower than
        elimination expressed with GPU primitives, and would force a device
        round-trip in the middle of an otherwise GPU-resident IRLS iteration.
        Pivoting is unnecessary because the systems are symmetric positive
        definite Fisher information matrices, whose leading minors are positive.
        """
        size = matrices.shape[-1]
        augmented = mx.concatenate([matrices, right_hand_sides], axis=-1)
        row_indices = mx.arange(size).reshape(1, size, 1)

        for pivot_index in range(size):
            pivot_row = augmented[:, pivot_index, :] / augmented[:, pivot_index, pivot_index, None]
            eliminated = augmented - augmented[:, :, pivot_index, None] * pivot_row[:, None, :]
            # The pivot row eliminates itself to zero, so restore it normalised.
            augmented = mx.where(
                row_indices == pivot_index,
                mx.broadcast_to(pivot_row[:, None, :], eliminated.shape),
                eliminated,
            )
        return augmented[:, :, size:]
