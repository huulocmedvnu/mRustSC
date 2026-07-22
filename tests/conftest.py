"""Shared fixtures. Owned by `main`; feature branches consume, never edit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mlxde.backend.numpy_backend import NumpyBackend
from mlxde.contracts import CountMatrix, DesignMatrix


@dataclass(frozen=True)
class SyntheticDataset:
    """Negative-binomial counts with a known subset of differential genes."""

    count_matrix: CountMatrix
    design: DesignMatrix
    true_log2_fold_change: np.ndarray
    differential_genes: np.ndarray


def make_synthetic_dataset(
    n_genes: int = 200,
    n_samples_per_group: int = 4,
    n_differential: int = 20,
    dispersion: float = 0.2,
    effect_log2: float = 2.0,
    seed: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_samples_per_group
    is_treated = np.repeat([0.0, 1.0], n_samples_per_group)

    baseline = rng.lognormal(mean=4.0, sigma=1.0, size=n_genes)
    true_log2_fold_change = np.zeros(n_genes)
    differential_genes = rng.choice(n_genes, size=n_differential, replace=False)
    true_log2_fold_change[differential_genes] = effect_log2 * rng.choice(
        [-1.0, 1.0], size=n_differential
    )

    fold_change = 2.0 ** np.outer(true_log2_fold_change, is_treated)
    means = baseline[:, None] * fold_change
    counts = rng.negative_binomial(
        n=1.0 / dispersion, p=1.0 / (1.0 + dispersion * means), size=means.shape
    ).astype(np.float64)

    sample_ids = np.array([f"sample_{index:02d}" for index in range(n_samples)])
    metadata = pd.DataFrame(
        {"condition": ["control"] * n_samples_per_group + ["treated"] * n_samples_per_group},
        index=sample_ids,
    )
    count_matrix = CountMatrix(
        counts=counts,
        gene_ids=np.array([f"gene_{index:04d}" for index in range(n_genes)]),
        sample_ids=sample_ids,
        sample_metadata=metadata,
    )
    design = DesignMatrix(
        matrix=np.column_stack([np.ones(n_samples), is_treated]),
        coefficient_names=("intercept", "condition[treated]"),
    )
    return SyntheticDataset(count_matrix, design, true_log2_fold_change, differential_genes)


@pytest.fixture
def synthetic_dataset() -> SyntheticDataset:
    return make_synthetic_dataset()


@pytest.fixture
def numpy_backend() -> NumpyBackend:
    return NumpyBackend()
