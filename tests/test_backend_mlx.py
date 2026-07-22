"""Parity tests: the GPU backend must agree with the CPU reference."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from mlxde.backend.numpy_backend import NumpyBackend
from mlxde.contracts import Array, ComputeBackend

mx = pytest.importorskip("mlx.core")

from mlxde.backend.mlx_backend import MLXBackend  # noqa: E402  (needs the mlx guard above)

RELATIVE_TOLERANCE = 1e-5
"""Loose enough for float32 accumulation on the GPU, tight enough to catch bugs."""


@pytest.fixture
def mlx_backend() -> MLXBackend:
    return MLXBackend()


def _spd_matrices(batch: int, size: int, seed: int = 0) -> np.ndarray:
    """Well-conditioned symmetric positive definite batch, as IRLS produces."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((batch, size, size))
    return factors @ np.transpose(factors, (0, 2, 1)) + size * np.eye(size)


def _assert_matches_numpy(
    mlx_backend: MLXBackend,
    operation: Callable[[ComputeBackend, Array], Array],
    values: np.ndarray,
) -> None:
    numpy_backend = NumpyBackend()
    expected = numpy_backend.asnumpy(operation(numpy_backend, numpy_backend.asarray(values)))
    actual = mlx_backend.asnumpy(operation(mlx_backend, mlx_backend.asarray(values)))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=1e-5)


def test_backend_satisfies_the_protocol() -> None:
    assert isinstance(MLXBackend(), ComputeBackend)


def test_backend_is_named_mlx(mlx_backend: MLXBackend) -> None:
    assert mlx_backend.name == "mlx"


def test_backend_is_available_on_apple_silicon(mlx_backend: MLXBackend) -> None:
    assert mlx_backend.is_available() is mx.is_available(mx.gpu)


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        MLXBackend(dtype="float64")


def test_asarray_uses_the_configured_dtype() -> None:
    values = np.arange(6.0).reshape(2, 3)
    assert MLXBackend().asarray(values).dtype == mx.float32
    assert MLXBackend(dtype="float16").asarray(values).dtype == mx.float16


def test_asnumpy_forces_evaluation(mlx_backend: MLXBackend) -> None:
    lazy = mlx_backend.exp(mlx_backend.asarray(np.zeros(4)))
    result = mlx_backend.asnumpy(lazy)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.ones(4))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda backend, values: backend.exp(values), id="exp"),
        pytest.param(lambda backend, values: backend.log(backend.abs(values) + 1.0), id="log"),
        pytest.param(lambda backend, values: backend.sqrt(backend.abs(values)), id="sqrt"),
        pytest.param(lambda backend, values: backend.abs(values), id="abs"),
        pytest.param(lambda backend, values: backend.clip(values, -0.5, 0.5), id="clip"),
        pytest.param(lambda backend, values: backend.maximum(values, 0.0), id="maximum-scalar"),
        pytest.param(lambda backend, values: backend.maximum(values, values * 2), id="maximum"),
        pytest.param(
            lambda backend, values: backend.where(values > 0, values, backend.abs(values)),
            id="where",
        ),
        pytest.param(lambda backend, values: backend.sum(values), id="sum-all"),
        pytest.param(lambda backend, values: backend.sum(values, axis=1), id="sum-axis"),
        pytest.param(lambda backend, values: backend.mean(values), id="mean-all"),
        pytest.param(lambda backend, values: backend.mean(values, axis=0), id="mean-axis"),
        pytest.param(lambda backend, values: backend.transpose(values), id="transpose"),
        pytest.param(
            lambda backend, values: backend.transpose(values, (1, 0)), id="transpose-axes"
        ),
        pytest.param(
            lambda backend, values: backend.matmul(values, backend.transpose(values)),
            id="matmul",
        ),
    ],
)
def test_elementwise_and_reduction_parity(
    mlx_backend: MLXBackend, operation: Callable[[ComputeBackend, Array], Array]
) -> None:
    values = np.random.default_rng(1).standard_normal((7, 5))
    _assert_matches_numpy(mlx_backend, operation, values)


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("batch", [1, 64])
def test_batched_solve_parity(mlx_backend: MLXBackend, batch: int, size: int) -> None:
    matrices = _spd_matrices(batch, size, seed=size)
    vectors = np.random.default_rng(size).standard_normal((batch, size))

    expected = np.linalg.solve(matrices, vectors[..., None])[..., 0]
    actual = mlx_backend.asnumpy(
        mlx_backend.solve(mlx_backend.asarray(matrices), mlx_backend.asarray(vectors))
    )

    assert actual.shape == (batch, size)
    np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=1e-5)


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("batch", [1, 64])
def test_batched_inverse_parity(mlx_backend: MLXBackend, batch: int, size: int) -> None:
    matrices = _spd_matrices(batch, size, seed=size + 100)

    expected = np.linalg.inv(matrices)
    actual = mlx_backend.asnumpy(mlx_backend.inverse(mlx_backend.asarray(matrices)))

    assert actual.shape == (batch, size, size)
    np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=1e-5)


def test_solve_agrees_with_the_numpy_backend(mlx_backend: MLXBackend) -> None:
    numpy_backend = NumpyBackend()
    matrices = _spd_matrices(32, 6, seed=7)
    vectors = np.random.default_rng(7).standard_normal((32, 6))

    # NumPy's gufunc needs the right-hand side as an explicit column, hence the
    # extra axis the MLX backend adds internally.
    expected = numpy_backend.asnumpy(
        numpy_backend.solve(
            numpy_backend.asarray(matrices), numpy_backend.asarray(vectors)[..., None]
        )
    )[..., 0]
    actual = mlx_backend.asnumpy(
        mlx_backend.solve(mlx_backend.asarray(matrices), mlx_backend.asarray(vectors))
    )

    np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=1e-5)


def test_inverse_is_the_matrix_inverse(mlx_backend: MLXBackend) -> None:
    matrices = mlx_backend.asarray(_spd_matrices(16, 5, seed=3))
    product = mlx_backend.asnumpy(mlx_backend.matmul(matrices, mlx_backend.inverse(matrices)))
    np.testing.assert_allclose(product, np.broadcast_to(np.eye(5), (16, 5, 5)), atol=1e-4)


@pytest.mark.skipif(not mx.is_available(mx.gpu), reason="requires an Apple GPU")
def test_linear_algebra_runs_on_the_gpu(mlx_backend: MLXBackend) -> None:
    """mlx raises for CPU-only ops on a GPU stream, so completing proves the device."""
    matrices = _spd_matrices(8, 4, seed=11)
    vectors = np.random.default_rng(11).standard_normal((8, 4))

    with mx.stream(mx.gpu):
        solution = mlx_backend.solve(mlx_backend.asarray(matrices), mlx_backend.asarray(vectors))
        inverses = mlx_backend.inverse(mlx_backend.asarray(matrices))
        mx.eval(solution, inverses)

    expected = np.linalg.solve(matrices, vectors[..., None])[..., 0]
    np.testing.assert_allclose(np.array(solution), expected, rtol=RELATIVE_TOLERANCE, atol=1e-5)
