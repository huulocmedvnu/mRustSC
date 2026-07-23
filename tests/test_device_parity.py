"""The CPU and the GPU have to agree, because callers do not choose between them.

`settings.device` defaults to `"auto"`, and `DeviceKind::Auto` resolves to Metal
wherever one exists (crates/scrust-core/src/device.rs). So the device a caller gets is
a property of their machine, not of their code, and any quantity that differs between
the two is a result that changes when the code moves.

Every audit in this directory pins behaviour against scanpy on one device at a time.
This file pins the two devices against *each other*, which is a different question and
the one that went unasked: a Metal-only divergence in the k-NN distances survived a
full audit because every test named `"cpu"`.

The whole file skips where there is no GPU -- which includes GitHub's hosted macOS
runners, so CI does not cover this. It runs on developer hardware and on a self-hosted
Apple-silicon runner, and nowhere else. Do not read a green CI as evidence that these
pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from scrust_call import scrust_call


def _gpu_available() -> bool:
    try:
        import scrust
    except ImportError:
        return False
    return bool(scrust._scrust.gpu_available())


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="no Metal device: there is nothing to compare against"
)

K = 14


def embedding(n_cells: int, n_dims: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_cells, n_dims)).astype(np.float32)


def knn(x: np.ndarray, k: int, device: str):
    return [np.asarray(v) for v in scrust_call("_scrust.knn", x.astype(np.float32), k, device)]


def test_duplicate_cells_are_at_distance_zero_on_both_devices():
    """The regression this file exists for.

    `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b` cancels to exactly zero for identical rows on
    the CPU, but leaves a sub-ulp positive on Metal -- 9.5e-7 against a norm scale of
    12 -- and the square root turns that into 9.8e-4. `rho` is then non-zero for a
    duplicated cell, and since `rho` is subtracted when the fuzzy simplicial set is
    built, every weight out of that cell stops being 1.

    `expansion_resolution` now snaps anything below the expansion's own resolution to
    zero, so both devices agree. Identical cells are common enough in real data --
    doublets, empty droplets, a gene panel too small to separate two cells -- that this
    is not a corner case.
    """
    x = embedding(200, 6, 9)
    x[:20] = x[0]  # 20 exact copies, more than K, so the whole row is at distance zero

    for device in ("cpu", "auto"):
        _, distances = knn(x, K, device)
        duplicated = distances[:20]
        assert np.count_nonzero(duplicated) == 0, (
            f"{device}: {np.count_nonzero(duplicated)} of {duplicated.size} distances "
            f"among identical cells are not zero, largest {duplicated.max():.6g}"
        )


@pytest.mark.parametrize(("n_cells", "n_dims", "seed"), [(400, 20, 3), (250, 50, 11)])
def test_the_neighbour_graph_does_not_depend_on_the_device(n_cells, n_dims, seed):
    """Same cells, same k, same neighbours -- and distances to f32 precision.

    The neighbour *lists* have to match exactly: a different neighbour is a different
    graph, and everything downstream of `pp.neighbors` reads that graph. The distances
    are held to f32 rather than to equality, because the two devices sum the dot
    product in a different order and nothing can make that bit-identical.
    """
    x = embedding(n_cells, n_dims, seed)
    cpu_indices, cpu_distances = knn(x, K, "cpu")
    gpu_indices, gpu_distances = knn(x, K, "auto")

    np.testing.assert_array_equal(
        cpu_indices, gpu_indices, err_msg="the two devices chose different neighbours"
    )
    np.testing.assert_allclose(cpu_distances, gpu_distances, rtol=1e-5, atol=1e-6)


def test_a_cluster_tighter_than_the_expansions_resolution_collapses_the_same_way():
    """Points closer together than the expansion can resolve are indistinguishable from
    identical ones, and must be treated the same way on both devices rather than one
    device rounding them to zero and the other not.

    This is the boundary the fix draws, so it is asserted rather than left implied.
    """
    x = embedding(150, 8, 21)
    # A tight knot, well inside the resolution of the expansion at this norm scale.
    x[:10] = x[0] + np.float32(1e-7) * np.arange(10, dtype=np.float32)[:, None]

    cpu_indices, cpu_distances = knn(x, 5, "cpu")
    gpu_indices, gpu_distances = knn(x, 5, "auto")
    np.testing.assert_array_equal(cpu_indices[:10], gpu_indices[:10])
    np.testing.assert_array_equal(
        cpu_distances[:10] == 0.0,
        gpu_distances[:10] == 0.0,
        err_msg="one device collapsed the knot to zero and the other did not",
    )
