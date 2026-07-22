"""Time the GLM fit, the dominant cost of a run, on each available backend.

Usage: PYTHONPATH=src python scripts/benchmark.py [n_genes ...]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from mlxde.backend import available_backends, get_backend
from mlxde.contracts import DesignMatrix
from mlxde.stats.glm import NegativeBinomialGLM

SAMPLES_PER_GROUP = 6
DISPERSION = 0.2


def make_counts(n_genes: int, rng: np.random.Generator) -> np.ndarray:
    means = rng.lognormal(4.0, 1.0, size=(n_genes, 1)) * np.ones(2 * SAMPLES_PER_GROUP)
    return rng.negative_binomial(
        1.0 / DISPERSION, 1.0 / (1.0 + DISPERSION * means), size=means.shape
    ).astype(np.float64)


def time_fit(backend_name: str, counts: np.ndarray, design: DesignMatrix) -> float:
    fitter = NegativeBinomialGLM(get_backend(backend_name))
    size_factors = np.ones(counts.shape[1])
    dispersions = np.full(counts.shape[0], DISPERSION)

    fitter.fit(counts[:64], design, size_factors, dispersions[:64])  # warm up kernels
    started = time.perf_counter()
    fitter.fit(counts, design, size_factors, dispersions)
    return time.perf_counter() - started


def main() -> None:
    sizes = [int(argument) for argument in sys.argv[1:]] or [5_000, 20_000, 60_000]
    rng = np.random.default_rng(0)
    is_treated = np.repeat([0.0, 1.0], SAMPLES_PER_GROUP)
    design = DesignMatrix(
        matrix=np.column_stack([np.ones(2 * SAMPLES_PER_GROUP), is_treated]),
        coefficient_names=("intercept", "condition[treated]"),
    )
    backends = available_backends()

    print(f"genes    {'  '.join(f'{name:>10}' for name in backends)}   speedup")
    for n_genes in sizes:
        counts = make_counts(n_genes, rng)
        seconds = [time_fit(name, counts, design) for name in backends]
        speedup = max(seconds) / min(seconds) if len(seconds) > 1 else 1.0
        print(f"{n_genes:<8} {'  '.join(f'{value:>10.3f}' for value in seconds)}   {speedup:.1f}x")


if __name__ == "__main__":
    main()
