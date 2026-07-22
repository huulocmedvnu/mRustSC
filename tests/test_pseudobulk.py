from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlxde.io.pseudobulk import build_pseudobulk


@pytest.fixture
def cells() -> tuple[pd.DataFrame, pd.Series]:
    cell_ids = [f"cell_{index:03d}" for index in range(60)]
    counts = pd.DataFrame(
        np.arange(60 * 4).reshape(60, 4),
        index=cell_ids,
        columns=["gene_a", "gene_b", "gene_c", "gene_d"],
    )
    labels = pd.Series(["monocyte"] * 30 + ["b_cell"] * 20 + ["other"] * 10, index=cell_ids)
    return counts, labels


CONDITIONS = {"monocyte": "treated", "b_cell": "control"}


def test_pools_every_cell_of_the_named_populations_exactly_once(cells):
    counts, labels = cells

    pseudobulk = build_pseudobulk(counts, labels, CONDITIONS, n_replicates=5)

    pooled_total = pseudobulk.counts.sum()
    expected_total = counts.loc[labels.isin(CONDITIONS)].to_numpy().sum()
    assert pooled_total == expected_total


def test_ignores_cells_outside_the_named_populations(cells):
    counts, labels = cells

    pseudobulk = build_pseudobulk(counts, labels, CONDITIONS)

    assert counts.loc[labels == "other"].to_numpy().sum() > 0
    assert pseudobulk.counts.sum() < counts.to_numpy().sum()


def test_produces_one_sample_per_replicate_and_condition(cells):
    counts, labels = cells

    pseudobulk = build_pseudobulk(counts, labels, CONDITIONS, n_replicates=4)

    assert pseudobulk.n_samples == 8
    assert pseudobulk.n_genes == 4
    assert sorted(pseudobulk.sample_metadata["condition"].unique()) == ["control", "treated"]
    assert pseudobulk.sample_metadata["condition"].value_counts().to_dict() == {
        "treated": 4,
        "control": 4,
    }


def test_pooling_is_reproducible_and_seed_dependent(cells):
    counts, labels = cells

    first = build_pseudobulk(counts, labels, CONDITIONS, seed=1)
    same_seed = build_pseudobulk(counts, labels, CONDITIONS, seed=1)
    other_seed = build_pseudobulk(counts, labels, CONDITIONS, seed=2)

    np.testing.assert_array_equal(first.counts, same_seed.counts)
    assert not np.array_equal(first.counts, other_seed.counts)


def test_rejects_unusable_inputs(cells):
    counts, labels = cells

    with pytest.raises(ValueError, match="at least 2 replicates"):
        build_pseudobulk(counts, labels, CONDITIONS, n_replicates=1)
    with pytest.raises(KeyError, match="no cells labelled"):
        build_pseudobulk(counts, labels, {"neutrophil": "treated"})
    with pytest.raises(ValueError, match="too few"):
        build_pseudobulk(counts, labels, CONDITIONS, n_replicates=25)
