from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlxde.backend import available_backends, get_backend
from mlxde.backend.numpy_backend import NumpyBackend
from mlxde.contracts import ComputeBackend, CountMatrix, DesignMatrix


def test_count_matrix_rejects_mismatched_gene_ids():
    with pytest.raises(ValueError, match="gene ids"):
        CountMatrix(
            counts=np.zeros((3, 2)),
            gene_ids=np.array(["a", "b"]),
            sample_ids=np.array(["s1", "s2"]),
            sample_metadata=pd.DataFrame(index=["s1", "s2"]),
        )


def test_count_matrix_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        CountMatrix(
            counts=np.array([[-1.0, 2.0]]),
            gene_ids=np.array(["a"]),
            sample_ids=np.array(["s1", "s2"]),
            sample_metadata=pd.DataFrame(index=["s1", "s2"]),
        )


def test_select_genes_keeps_masked_rows(synthetic_dataset):
    mask = np.zeros(synthetic_dataset.count_matrix.n_genes, dtype=bool)
    mask[:5] = True

    subset = synthetic_dataset.count_matrix.select_genes(mask)

    assert subset.n_genes == 5
    assert subset.n_samples == synthetic_dataset.count_matrix.n_samples


def test_contrast_isolates_named_coefficient(synthetic_dataset):
    contrast = synthetic_dataset.design.contrast("condition[treated]")

    assert contrast.tolist() == [0.0, 1.0]


def test_contrast_rejects_unknown_coefficient(synthetic_dataset):
    with pytest.raises(KeyError):
        synthetic_dataset.design.contrast("does_not_exist")


def test_design_matrix_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="coefficient_names"):
        DesignMatrix(matrix=np.ones((4, 2)), coefficient_names=("intercept",))


def test_numpy_backend_satisfies_compute_backend_protocol():
    assert isinstance(NumpyBackend(), ComputeBackend)


def test_registry_always_offers_a_working_backend():
    assert "numpy" in available_backends()
    assert get_backend("numpy").name == "numpy"


def test_registry_rejects_unknown_backend():
    with pytest.raises(KeyError):
        get_backend("quantum")
