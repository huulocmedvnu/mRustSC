"""Integration tests: the wired pipeline must recover a planted signal.

Thresholds come from measured behaviour on the synthetic generator, not from
theory — see `docs/VALIDATION.md` for the numbers and their interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlxde.backend import available_backends
from mlxde.factory import build_default_pipeline
from mlxde.io.design import build_design_matrix
from mlxde.io.readers import CsvCountReader
from mlxde.io.writers import CsvResultWriter
from mlxde.report.summary import summarize
from tests.conftest import make_synthetic_dataset


def _treatment_contrast(design):
    return design.contrast("condition[treated]")


@pytest.mark.parametrize("backend_name", available_backends())
def test_pipeline_recovers_planted_differential_genes(backend_name):
    dataset = make_synthetic_dataset(
        n_genes=500, n_samples_per_group=6, n_differential=50, effect_log2=3.0, seed=7
    )

    result = build_default_pipeline(backend_name).run(
        dataset.count_matrix, dataset.design, _treatment_contrast(dataset.design)
    )

    called = set(result.significant(alpha=0.05, min_abs_log2_fold_change=1.0)["gene_id"])
    planted = set(dataset.count_matrix.gene_ids[dataset.differential_genes])
    recall = len(called & planted) / len(planted)
    false_discovery_rate = len(called - planted) / max(len(called), 1)

    assert recall > 0.9, f"{backend_name} recovered only {recall:.0%} of the planted genes"
    assert false_discovery_rate < 0.25, (
        f"{backend_name} false discoveries {false_discovery_rate:.0%}"
    )


@pytest.mark.parametrize("backend_name", available_backends())
def test_no_discoveries_under_the_global_null(backend_name):
    dataset = make_synthetic_dataset(n_genes=1000, n_samples_per_group=6, n_differential=0, seed=11)

    result = build_default_pipeline(backend_name).run(
        dataset.count_matrix, dataset.design, _treatment_contrast(dataset.design)
    )

    discoveries = np.nansum(result.adjusted_p_value <= 0.05)

    assert discoveries <= 5, f"{backend_name} made {discoveries} discoveries with no signal present"


@pytest.mark.parametrize("backend_name", available_backends())
def test_estimated_fold_changes_track_the_truth(backend_name):
    dataset = make_synthetic_dataset(
        n_genes=400, n_samples_per_group=8, n_differential=120, seed=11
    )

    result = build_default_pipeline(backend_name).run(
        dataset.count_matrix, dataset.design, _treatment_contrast(dataset.design)
    )

    tested = np.isfinite(result.log2_fold_change)
    correlation = np.corrcoef(
        result.log2_fold_change[tested], dataset.true_log2_fold_change[tested]
    )[0, 1]

    assert correlation > 0.9


def test_backends_agree_on_the_same_dataset():
    if len(available_backends()) < 2:
        pytest.skip("only one backend is available on this machine")
    dataset = make_synthetic_dataset(n_genes=300, seed=3)
    contrast = _treatment_contrast(dataset.design)

    reference, *others = [
        build_default_pipeline(name).run(dataset.count_matrix, dataset.design, contrast)
        for name in available_backends()
    ]

    for other in others:
        np.testing.assert_allclose(
            other.log2_fold_change, reference.log2_fold_change, rtol=1e-3, atol=1e-4
        )


def test_full_run_from_files_to_report(tmp_path):
    dataset = make_synthetic_dataset(n_genes=200, seed=5)
    counts_path, metadata_path = tmp_path / "counts.csv", tmp_path / "samples.csv"
    output_path = tmp_path / "results.csv"
    counts_table = {"gene_id": dataset.count_matrix.gene_ids} | {
        sample_id: dataset.count_matrix.counts[:, index].astype(int)
        for index, sample_id in enumerate(dataset.count_matrix.sample_ids)
    }
    pd.DataFrame(counts_table).to_csv(counts_path, index=False)
    dataset.count_matrix.sample_metadata.rename_axis("sample_id").to_csv(metadata_path)

    count_matrix = CsvCountReader().read(counts_path, metadata_path)
    design = build_design_matrix(count_matrix.sample_metadata, "condition", "control")
    result = build_default_pipeline().run(count_matrix, design, _treatment_contrast(design))
    CsvResultWriter().write(result, output_path)

    written = pd.read_csv(output_path)
    assert len(written) == count_matrix.n_genes
    assert set(written["gene_id"]) == set(count_matrix.gene_ids)
    assert "significant" in summarize(result).lower()
