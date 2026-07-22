"""Gene-set scoring against scanpy: `tl.score_genes`, cell cycle, marker overlap, filtering.

`score_genes` reproduces scanpy's control draw exactly — scanpy samples control
genes with `pandas.Series.sample`, which is numpy's legacy MT19937 stream, and
the core replays that stream — so the comparison is element wise rather than
statistical. The correlation and the rank agreement are reported anyway, because
they are the numbers that would matter if the draw ever stopped matching.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse

from scrust_call import scrust_call

ELEMENTWISE = {"rtol": 1e-5, "atol": 1e-6}

# Real markers for PBMC 3k, and the planted markers of the synthetic fixture.
GENE_SETS = {
    "pbmc3k": [
        "IL7R",
        "CD14",
        "LYZ",
        "MS4A1",
        "CD8A",
        "GNLY",
        "NKG7",
        "FCGR3A",
        "MS4A7",
        "FCER1A",
        "CST3",
        "PPBP",
        "CD3D",
        "CD3E",
    ],
    "synthetic": [f"gene{index}" for index in range(0, 12)],
}
# Two disjoint gene sets standing in for the S and G2M programmes.
CELL_CYCLE_SETS = {
    "pbmc3k": (
        ["MCM5", "PCNA", "TYMS", "RRM2", "MCM2", "UHRF1", "CDC45", "RRM1", "GINS2", "CHAF1B"],
        ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "TOP2A", "TPX2", "CKS2", "CDC20", "TTK", "CENPF"],
    ),
    "synthetic": (
        [f"gene{index}" for index in range(0, 10)],
        [f"gene{index}" for index in range(20, 30)],
    ),
}


def gene_set(adata: AnnData) -> list[str]:
    """A marker set that exists in this fixture, and in enough cells to score."""
    return [gene for gene in GENE_SETS[adata.uns["dataset_id"]] if gene in adata.var_names]


def test_score_genes_matches_scanpy(
    lognorm: AnnData, record_property: Callable[[str, object], None]
) -> None:
    genes = gene_set(lognorm)
    scrust_call("tl.score_genes", lognorm, genes, score_name="ours")
    expected = lognorm.copy()
    sc.tl.score_genes(expected, genes, score_name="theirs")

    ours = lognorm.obs["ours"].to_numpy()
    theirs = expected.obs["theirs"].to_numpy()
    dataset = lognorm.uns["dataset_id"]
    correlation = float(np.corrcoef(ours, theirs)[0, 1])
    rank_correlation = float(np.corrcoef(pd.Series(ours).rank(), pd.Series(theirs).rank())[0, 1])
    record_property(f"score_genes.{dataset}.pearson", round(correlation, 6))
    record_property(f"score_genes.{dataset}.spearman", round(rank_correlation, 6))
    print(
        f"\nscore_genes on {dataset}: pearson {correlation:.6f}, "
        f"cell ranking {rank_correlation:.6f}"
    )

    assert_allclose(ours, theirs, **ELEMENTWISE, err_msg="scores differ from scanpy's")


def test_a_constant_gene_set_scores_zero() -> None:
    """Every gene at the same level in every cell: the set and its control agree."""
    adata = AnnData(sparse.csr_matrix(np.full((30, 40), 2.5, dtype=np.float32)))
    adata.var_names = [f"gene{index}" for index in range(adata.n_vars)]
    genes = [f"gene{index}" for index in range(5)]

    scrust_call("tl.score_genes", adata, genes, ctrl_size=10, score_name="ours")
    sc.tl.score_genes(adata, genes, ctrl_size=10, score_name="theirs")

    assert_allclose(adata.obs["ours"].to_numpy(), np.zeros(adata.n_obs), atol=1e-6)
    assert_allclose(adata.obs["ours"], adata.obs["theirs"], **ELEMENTWISE)


def test_score_genes_cell_cycle_assigns_the_same_phase(
    lognorm: AnnData, record_property: Callable[[str, object], None]
) -> None:
    s_genes, g2m_genes = CELL_CYCLE_SETS[lognorm.uns["dataset_id"]]
    s_genes = [gene for gene in s_genes if gene in lognorm.var_names]
    g2m_genes = [gene for gene in g2m_genes if gene in lognorm.var_names]

    scrust_call("tl.score_genes_cell_cycle", lognorm, s_genes=s_genes, g2m_genes=g2m_genes)
    expected = lognorm.copy()
    sc.tl.score_genes_cell_cycle(expected, s_genes=s_genes, g2m_genes=g2m_genes)

    agreement = float((lognorm.obs["phase"] == expected.obs["phase"]).mean())
    dataset = lognorm.uns["dataset_id"]
    record_property(f"score_genes_cell_cycle.{dataset}.phase_agreement", round(agreement, 6))
    print(f"\nscore_genes_cell_cycle on {dataset}: phases agree on {agreement:.4%} of cells")

    for score in ("S_score", "G2M_score"):
        assert_allclose(lognorm.obs[score], expected.obs[score], **ELEMENTWISE, err_msg=score)
    assert agreement == 1.0


@pytest.fixture
def called_markers() -> AnnData:
    """A hand-built `rank_genes_groups` slot: three groups over a known gene order."""
    adata = AnnData(np.zeros((3, 1), dtype=np.float32))
    groups = {
        "A": ["CD3D", "CD3E", "IL7R", "LYZ"],
        "B": ["LYZ", "CD14", "CST3", "FCER1A"],
        "C": ["NKG7", "GNLY", "CD3D", "PPBP"],
    }
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "group", "reference": "rest", "use_raw": False},
        "names": np.rec.fromarrays(list(groups.values()), dtype=[(name, "O") for name in groups]),
    }
    return adata


REFERENCE_MARKERS = {
    "T cells": {"CD3D", "CD3E", "IL7R"},
    "Monocytes": {"LYZ", "CD14", "FCGR3A"},
    "NK cells": {"NKG7", "GNLY"},
    "Nothing in common": {"HBB", "HBA1"},
}


@pytest.mark.parametrize("method", ["overlap_count", "overlap_coef", "jaccard"])
@pytest.mark.parametrize("top_n_markers", [None, 2])
def test_marker_gene_overlap_matches_scanpy(
    called_markers: AnnData, method: str, top_n_markers: int | None
) -> None:
    ours = scrust_call(
        "tl.marker_gene_overlap",
        called_markers,
        REFERENCE_MARKERS,
        method=method,
        top_n_markers=top_n_markers,
    )
    theirs = sc.tl.marker_gene_overlap(
        called_markers, REFERENCE_MARKERS, method=method, top_n_markers=top_n_markers
    )
    pd.testing.assert_frame_equal(ours, theirs)


def test_filter_rank_genes_groups_blanks_the_same_genes(lognorm: AnnData) -> None:
    """The reference `rank_genes_groups` slot is scanpy's, so only the filter differs."""
    sc.tl.rank_genes_groups(lognorm, "group", method="wilcoxon")
    thresholds = {
        "min_in_group_fraction": 0.25,
        "max_out_group_fraction": 0.5,
        "min_fold_change": 2.0,
    }
    scrust_call("tl.filter_rank_genes_groups", lognorm, **thresholds)
    expected = lognorm.copy()
    sc.tl.filter_rank_genes_groups(expected, **thresholds)

    ours = pd.DataFrame(lognorm.uns["rank_genes_groups_filtered"]["names"])
    theirs = pd.DataFrame(expected.uns["rank_genes_groups_filtered"]["names"])
    pd.testing.assert_frame_equal(ours, theirs)
    assert ours.isna().to_numpy().any(), "the filter blanked nothing, so it proves nothing"


def test_filter_rank_genes_groups_keeps_the_rest_of_the_slot(lognorm: AnnData) -> None:
    sc.tl.rank_genes_groups(lognorm, "group", method="wilcoxon")
    scrust_call("tl.filter_rank_genes_groups", lognorm)
    filtered = lognorm.uns["rank_genes_groups_filtered"]
    assert set(filtered) == set(lognorm.uns["rank_genes_groups"])
    assert_allclose(
        pd.DataFrame(filtered["pvals"]).to_numpy(),
        pd.DataFrame(lognorm.uns["rank_genes_groups"]["pvals"]).to_numpy(),
    )


def test_an_empty_gene_set_is_rejected(lognorm: AnnData) -> None:
    with pytest.raises(ValueError, match="No valid genes"):
        scrust_call("tl.score_genes", lognorm, [])
    with pytest.raises(ValueError, match="No valid genes"):
        scrust_call("tl.score_genes", lognorm, ["not-a-gene"])


def test_genes_missing_from_the_matrix_are_dropped_with_a_warning(lognorm: AnnData) -> None:
    genes = gene_set(lognorm)
    with pytest.warns(UserWarning, match="not in var_names"):
        scrust_call("tl.score_genes", lognorm, [*genes, "not-a-gene"], score_name="ours")
    expected = lognorm.copy()
    sc.tl.score_genes(expected, genes, score_name="theirs")
    assert_allclose(lognorm.obs["ours"], expected.obs["theirs"], **ELEMENTWISE)


def test_a_gene_set_covering_the_matrix_leaves_no_control(lognorm: AnnData) -> None:
    """scanpy raises too: every gene is scored, so every bin empties out."""
    everything = [*lognorm.var_names, "not-a-gene"]
    with pytest.raises((ValueError, RuntimeError)):
        scrust_call("tl.score_genes", lognorm, everything)


def test_ctrl_size_larger_than_a_bin_takes_the_whole_bin(lognorm: AnnData) -> None:
    """With more controls asked for than a bin holds, scanpy keeps the bin as it is."""
    genes = gene_set(lognorm)
    ctrl_size = lognorm.n_vars  # larger than any bin, so no sampling happens at all
    scrust_call("tl.score_genes", lognorm, genes, ctrl_size=ctrl_size, score_name="ours")
    expected = lognorm.copy()
    sc.tl.score_genes(expected, genes, ctrl_size=ctrl_size, score_name="theirs")
    assert_allclose(lognorm.obs["ours"], expected.obs["theirs"], **ELEMENTWISE)
