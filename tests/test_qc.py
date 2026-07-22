"""QC metrics and the legacy normalisation helpers, against scanpy.

Every metric here is a deterministic reduction over the same counts, so the form
of agreement is element-wise: same column names, same numbers to `rtol` 1e-5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from numpy.testing import assert_allclose

from scrust_call import scrust_call

RTOL = 1e-5

# scanpy refuses a `percent_top` deeper than the matrix has genes, and the
# synthetic fixture has 300, so the default is trimmed to what fits.
PERCENT_TOP = (50, 100, 200, 500)

# A gene subset that exists on both fixtures, whatever the genes are called.
QC_VAR = "mt"

# The tutorial's normalisation target, as in `conftest`.
TARGET_SUM = 1e4


def percent_top_for(adata: AnnData) -> tuple[int, ...]:
    return tuple(n for n in PERCENT_TOP if n <= adata.n_vars)


def mark_subset(adata: AnnData) -> np.ndarray:
    """Flag every seventh gene, standing in for the mitochondrial ones."""
    flags = np.arange(adata.n_vars) % 7 == 0
    adata.var[QC_VAR] = flags
    return flags


def tiny() -> AnnData:
    """3 cells x 4 genes, small enough to check by hand.

    cell 0: 1 3 0 0   total 4,  top-1 3/4,  top-2 4/4, top-3 and top-4 also 4/4
    cell 1: 0 0 0 0   empty
    cell 2: 5 0 2 3   total 10, top-1 5/10, top-2 8/10
    """
    dense = np.array([[1, 3, 0, 0], [0, 0, 0, 0], [5, 0, 2, 3]], dtype=np.float32)
    return AnnData(sp.csr_matrix(dense))


def assert_frames_agree(ours, reference, label: str) -> None:
    assert list(ours.columns) == list(reference.columns), f"{label} columns"
    for column in reference.columns:
        assert_allclose(
            np.asarray(ours[column], dtype=np.float64),
            np.asarray(reference[column], dtype=np.float64),
            rtol=RTOL,
            err_msg=f"{label} column {column!r}",
        )


def test_qc_metrics_match_scanpy(counts: AnnData) -> None:
    percent_top = percent_top_for(counts)
    mark_subset(counts)
    ours = scrust_call(
        "pp.calculate_qc_metrics",
        counts,
        qc_vars=[QC_VAR],
        percent_top=percent_top,
        inplace=False,
    )
    reference = sc.pp.calculate_qc_metrics(
        counts, qc_vars=[QC_VAR], percent_top=percent_top, inplace=False
    )

    assert_frames_agree(ours[0], reference[0], "obs")
    assert_frames_agree(ours[1], reference[1], "var")


def test_qc_metrics_land_in_obs_and_var(counts: AnnData) -> None:
    percent_top = percent_top_for(counts)
    scrust_call("pp.calculate_qc_metrics", counts, percent_top=percent_top)
    assert "total_counts" in counts.obs
    assert "pct_dropout_by_counts" in counts.var
    assert counts.obs["total_counts"].shape == (counts.n_obs,)


def test_gene_subset_percentage(counts: AnnData) -> None:
    """`pct_counts_mt` is the subset's share of each cell's counts."""
    flags = mark_subset(counts)
    obs, _ = scrust_call(
        "pp.calculate_qc_metrics",
        counts,
        qc_vars=[QC_VAR],
        percent_top=percent_top_for(counts),
        inplace=False,
    )

    subset_totals = np.asarray(counts.X[:, flags].sum(axis=1)).ravel()
    assert_allclose(obs[f"total_counts_{QC_VAR}"], subset_totals, rtol=RTOL)
    assert_allclose(
        obs[f"pct_counts_{QC_VAR}"],
        subset_totals / np.asarray(counts.X.sum(axis=1)).ravel() * 100,
        rtol=RTOL,
    )


def test_percent_top_is_hand_computable() -> None:
    adata = tiny()
    # Cell 0 expresses only 2 genes, so its top 3 already hold everything.
    obs, _ = scrust_call(
        "pp.calculate_qc_metrics", adata, percent_top=(1, 2, 3), log1p=False, inplace=False
    )

    assert_allclose(obs["pct_counts_in_top_1_genes"], [75.0, np.nan, 50.0], rtol=RTOL)
    assert_allclose(obs["pct_counts_in_top_2_genes"], [100.0, np.nan, 80.0], rtol=RTOL)
    assert_allclose(obs["pct_counts_in_top_3_genes"], [100.0, np.nan, 100.0], rtol=RTOL)


def test_percent_top_matches_scanpy_on_a_tiny_matrix() -> None:
    percent_top = (1, 2, 3, 4)
    ours = scrust_call("pp.calculate_qc_metrics", tiny(), percent_top=percent_top, inplace=False)
    reference = sc.pp.calculate_qc_metrics(tiny(), percent_top=percent_top, inplace=False)
    assert_frames_agree(ours[0], reference[0], "obs")
    assert_frames_agree(ours[1], reference[1], "var")


def test_empty_percent_top_produces_no_columns() -> None:
    obs, _ = scrust_call("pp.calculate_qc_metrics", tiny(), percent_top=(), inplace=False)
    assert not [column for column in obs.columns if "in_top" in column]


def test_all_zero_cell_and_gene() -> None:
    """Cell 1 has no counts and gene 3 is seen in no cell."""
    dense = np.array([[1, 2, 0, 0], [0, 0, 0, 0], [3, 0, 4, 0]], dtype=np.float32)
    adata = AnnData(sp.csr_matrix(dense))
    obs, var = scrust_call("pp.calculate_qc_metrics", adata, percent_top=(2,), inplace=False)

    assert obs["total_counts"].to_numpy()[1] == 0.0
    assert obs["n_genes_by_counts"].to_numpy()[1] == 0
    # No counts means no share of them to report, which is scanpy's NaN.
    assert np.isnan(obs["pct_counts_in_top_2_genes"].to_numpy()[1])
    assert var["n_cells_by_counts"].to_numpy()[3] == 0
    assert var["total_counts"].to_numpy()[3] == 0.0
    assert var["pct_dropout_by_counts"].to_numpy()[3] == 100.0


def test_sqrt_matches_scanpy(counts: AnnData) -> None:
    reference = counts.copy()
    scrust_call("pp.sqrt", counts)
    sc.pp.sqrt(reference)
    assert_allclose(counts.X.toarray(), reference.X.toarray(), rtol=RTOL)


def test_normalize_per_cell_matches_scanpy(counts: AnnData) -> None:
    reference = counts.copy()
    scrust_call("pp.normalize_per_cell", counts)
    with pytest.warns(FutureWarning):
        sc.pp.normalize_per_cell(reference)

    assert counts.shape == reference.shape  # both drop the cells without counts
    assert_allclose(counts.obs["n_counts"], reference.obs["n_counts"], rtol=RTOL)
    assert_allclose(counts.X.toarray(), reference.X.toarray(), rtol=RTOL)


def test_normalize_per_cell_with_an_explicit_target(counts: AnnData) -> None:
    reference = counts.copy()
    scrust_call("pp.normalize_per_cell", counts, counts_per_cell_after=1e4)
    with pytest.warns(FutureWarning):
        sc.pp.normalize_per_cell(reference, counts_per_cell_after=1e4)
    assert_allclose(counts.X.toarray(), reference.X.toarray(), rtol=RTOL)


def test_filter_genes_dispersion_matches_the_legacy_cutoffs(counts: AnnData) -> None:
    """Without `n_top_genes` the selection is scanpy's cut-off rule.

    The input convention is the modern one — `seurat` reads log data — so the
    reference is scanpy's legacy call on the counts that log data came from.
    """
    reference = counts.copy()
    sc.pp.filter_genes(reference, min_cells=3)
    sc.pp.normalize_total(reference, target_sum=TARGET_SUM)
    logged = reference.copy()
    sc.pp.log1p(logged)

    scrust_call("pp.filter_genes_dispersion", logged)
    with warnings.catch_warnings():  # the legacy entry point announces its age
        warnings.simplefilter("ignore")
        sc.pp.filter_genes_dispersion(reference, subset=False)

    assert_allclose(logged.var["means"], reference.var["means"], rtol=RTOL)
    assert_allclose(
        logged.var["dispersions_norm"],
        reference.var["dispersions_norm"],
        rtol=RTOL,
        atol=1e-5,
    )
    assert (logged.var["highly_variable"] == reference.var["highly_variable"]).all()


def test_filter_genes_dispersion_flags_genes(lognorm: AnnData) -> None:
    n_top_genes = 50
    scrust_call("pp.filter_genes_dispersion", lognorm, n_top_genes=n_top_genes)

    assert lognorm.var["highly_variable"].sum() >= n_top_genes
    for column in ("means", "dispersions_norm", "highly_variable"):
        assert column in lognorm.var
    # The legacy entry point is the modern one underneath, so they must select
    # the same genes from the same data.
    modern = lognorm.copy()
    scrust_call("pp.highly_variable_genes", modern, n_top_genes=n_top_genes)
    assert (lognorm.var["highly_variable"] == modern.var["highly_variable"]).all()
