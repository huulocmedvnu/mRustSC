"""Regression tests against scanpy on real 10x PBMC 3k data.

scanpy's `rank_genes_groups` (Wilcoxon over individual cells) is the reference:
it is an established, independent implementation on the same biology. It is a
different test on different units — cells, not pseudobulk replicates — so the
agreement asserted here is on *direction* and *ranking*, never on p-values.

Thresholds are the measured agreement across six cell-type pairs, with margin.
Marked `realdata`: they download ~29 MB on first run. Skip with `-m "not realdata"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from mlxde.factory import build_default_pipeline
from mlxde.io.design import build_design_matrix
from mlxde.io.pseudobulk import build_pseudobulk

pytestmark = pytest.mark.realdata

sc = pytest.importorskip("scanpy", reason="install the 'validate' extra for real-data tests")

CACHE_DIRECTORY = ".cache/scanpy"
N_REPLICATES = 5
CELL_TYPE_PAIRS = [
    ("CD14+ Monocytes", "B cells"),
    ("CD4 T cells", "B cells"),
    ("NK cells", "CD14+ Monocytes"),
    ("CD8 T cells", "B cells"),
    ("CD4 T cells", "CD14+ Monocytes"),
    ("FCGR3A+ Monocytes", "CD4 T cells"),
]

# Measured agreement over those pairs: rank correlation 0.833-0.961, top-100
# overlap 53-91, direction on scanpy's top 50 exactly 1.00 everywhere.
MINIMUM_RANK_CORRELATION = 0.75
MINIMUM_TOP_100_OVERLAP = 45
N_REFERENCE_GENES = 50


@pytest.fixture(scope="session")
def pbmc() -> tuple[pd.DataFrame, pd.Series, object]:
    """Raw UMI counts, published cell-type labels, and the annotated object."""
    Path(CACHE_DIRECTORY).mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = CACHE_DIRECTORY
    try:
        raw = sc.datasets.pbmc3k()
        labelled = sc.datasets.pbmc3k_processed()
    except OSError as error:  # no network on this machine is a skip, not a defect
        pytest.skip(f"PBMC 3k dataset unavailable: {error}")

    raw.var_names_make_unique()
    shared_cells = raw.obs_names.intersection(labelled.obs_names)
    return (
        raw[shared_cells].to_df(),
        labelled.obs.loc[shared_cells, "louvain"].astype(str),
        labelled,
    )


def run_pipeline(counts, cell_labels, treated: str, reference: str) -> pd.DataFrame:
    """Our calls for `treated` vs `reference`, indexed by gene, tested genes only."""
    count_matrix = build_pseudobulk(
        counts, cell_labels, {treated: "treated", reference: "reference"}, N_REPLICATES
    )
    design = build_design_matrix(count_matrix.sample_metadata, "condition", "reference")
    result = build_default_pipeline().run(
        count_matrix, design, design.contrast("condition[treated]")
    )
    table = result.to_dataframe().set_index("gene_id")
    return table[np.isfinite(table["p_value"])]


def run_scanpy(labelled, treated: str, reference: str) -> pd.DataFrame:
    """scanpy's Wilcoxon ranking for the same comparison, best score first."""
    subset = labelled[labelled.obs["louvain"].isin([treated, reference])].copy()
    sc.tl.rank_genes_groups(
        subset, "louvain", groups=[treated], reference=reference, method="wilcoxon"
    )
    ranking = subset.uns["rank_genes_groups"]
    return pd.DataFrame(
        {"score": ranking["scores"][treated]}, index=ranking["names"][treated]
    ).sort_values("score", ascending=False)


@pytest.fixture(scope="session")
def comparison(request, pbmc) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, cell_labels, labelled = pbmc
    treated, reference = request.param
    return run_pipeline(counts, cell_labels, treated, reference), run_scanpy(
        labelled, treated, reference
    )


def parametrise_over_pairs(test):
    return pytest.mark.parametrize(
        "comparison",
        CELL_TYPE_PAIRS,
        ids=[f"{treated}_vs_{reference}" for treated, reference in CELL_TYPE_PAIRS],
        indirect=True,
    )(test)


@parametrise_over_pairs
def test_direction_matches_scanpy_on_its_strongest_genes(comparison):
    ours, scanpy_ranking = comparison
    strongest = [gene for gene in scanpy_ranking.index[:N_REFERENCE_GENES] if gene in ours.index]

    disagreeing = [gene for gene in strongest if ours.loc[gene, "log2_fold_change"] <= 0]

    assert len(strongest) >= N_REFERENCE_GENES * 0.9, "too few of scanpy's top genes were tested"
    assert not disagreeing, f"opposite direction to scanpy for {disagreeing}"


@parametrise_over_pairs
def test_gene_ranking_correlates_with_scanpy(comparison):
    ours, scanpy_ranking = comparison
    shared = ours.index.intersection(scanpy_ranking.index)

    correlation = spearmanr(
        ours.loc[shared, "statistic"], scanpy_ranking.loc[shared, "score"]
    ).statistic

    assert correlation > MINIMUM_RANK_CORRELATION, f"rank correlation only {correlation:.3f}"


@parametrise_over_pairs
def test_top_genes_overlap_scanpy(comparison):
    ours, scanpy_ranking = comparison
    our_top = set(ours[ours["log2_fold_change"] > 0].sort_values("adjusted_p_value").index[:100])
    scanpy_top = set(scanpy_ranking.index[:100])

    overlap = len(our_top & scanpy_top)

    assert overlap >= MINIMUM_TOP_100_OVERLAP, f"only {overlap}/100 genes shared with scanpy"


def test_canonical_markers_are_called(pbmc):
    """Direction and significance for markers whose biology is not in dispute."""
    counts, cell_labels, _ = pbmc
    ours = run_pipeline(counts, cell_labels, "CD14+ Monocytes", "B cells")
    up_in_monocytes = ("LYZ", "S100A8", "S100A9", "CD14", "FCN1", "VCAN", "CST3", "FTL")
    up_in_b_cells = ("MS4A1", "CD79A", "CD79B", "TCL1A", "BANK1", "CD19")

    miscalled = [
        gene
        for gene, expected_sign in (
            [(marker, 1.0) for marker in up_in_monocytes]
            + [(marker, -1.0) for marker in up_in_b_cells]
        )
        if gene not in ours.index
        or np.sign(ours.loc[gene, "log2_fold_change"]) != expected_sign
        or ours.loc[gene, "adjusted_p_value"] > 0.05
    ]

    assert not miscalled, f"canonical markers not called as expected: {miscalled}"
