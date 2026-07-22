"""The whole scanpy tutorial pipeline, run through scrust and through scanpy.

Single-step tests hand every step the same reference input, so none of them can see
error accumulating from one step to the next. This one chains the steps inside each
library and compares stage by stage, which is where drift shows up.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose, assert_array_equal

from conftest import CEILING_FRACTION, K_CAND, K_REF, check_pca_agreement
from reference_metrics import (
    DE_TOLERANCES,
    as_dense,
    de_comparison,
    neighbor_sets,
    per_row_overlap,
    preservation_band,
    set_overlap,
)
from scrust_call import scrust_call

# The pipeline's own settings, shared by the run under test and the ceiling run.
_N_COMPS = 50
_N_NEIGHBORS = 15

# The single-step DE test hands both libraries the same matrix, so it holds
# p-values to 1e-6. Here the matrices differ by ~1e-6 absolute after eight
# chained f32 steps, and a rank-sum statistic is discrete: one swapped rank
# moves a p-value by more than that. Measured worst case 8.0e-3 on one gene.
_PIPELINE_DE_TOLERANCES = {**DE_TOLERANCES, "pvals": 1e-2, "pvals_adj": 1e-2}

pytestmark = pytest.mark.reference

Call = Callable[..., None]


def _scanpy_call(path: str, adata: AnnData, *args: object, **kwargs: object) -> None:
    module, name = path.split(".")
    getattr(getattr(sc, module), name)(adata, *args, **kwargs)


def _scrust_call(path: str, adata: AnnData, *args: object, **kwargs: object) -> None:
    scrust_call(path, adata, *args, **kwargs)


def _run_pipeline(adata: AnnData, call: Call) -> dict[str, AnnData]:
    """The tutorial pipeline, snapshotted after every stage.

    One body for both libraries: the API contract says the scrust signatures mirror
    scanpy's, so any divergence in the calls themselves is a contract violation.
    """
    call("pp.filter_cells", adata, min_genes=200)
    call("pp.filter_genes", adata, min_cells=3)
    call("pp.normalize_total", adata, target_sum=1e4)
    call("pp.log1p", adata)
    stages = {"lognorm": adata.copy()}

    call("pp.highly_variable_genes", adata, n_top_genes=2000, flavor="seurat")
    stages["hvg"] = adata.copy()

    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    call("pp.scale", adata, zero_center=True, max_value=10)
    stages["scaled"] = adata.copy()

    call("pp.pca", adata, n_comps=_N_COMPS, random_state=0)
    stages["pca"] = adata.copy()

    call("pp.neighbors", adata, n_neighbors=_N_NEIGHBORS, use_rep="X_pca")
    stages["neighbors"] = adata.copy()

    call("tl.umap", adata, random_state=0)
    stages["umap"] = adata.copy()

    # Differential expression belongs on log-normalised counts, not on the scaled HVG
    # subset, exactly as the tutorial runs it against `.raw`.
    de = stages["lognorm"].copy()
    call("tl.rank_genes_groups", de, "group", method="wilcoxon")
    stages["de"] = de
    return stages


def _common_genes(ours: AnnData, theirs: AnnData) -> tuple[AnnData, AnnData]:
    shared = [g for g in theirs.var_names if g in set(ours.var_names)]
    return ours[:, shared], theirs[:, shared]


def test_full_pipeline(pbmc3k: AnnData, record_property: Callable[[str, object], None]) -> None:
    # scrust first: an unimplemented step skips before the slow reference run.
    ours = _run_pipeline(pbmc3k.copy(), _scrust_call)
    theirs = _run_pipeline(pbmc3k.copy(), _scanpy_call)

    assert_array_equal(ours["lognorm"].obs_names, theirs["lognorm"].obs_names)
    assert_array_equal(ours["lognorm"].var_names, theirs["lognorm"].var_names)
    assert_allclose(
        as_dense(ours["lognorm"].X), as_dense(theirs["lognorm"].X), rtol=1e-5, atol=1e-6
    )

    hvg_overlap = set_overlap(
        ours["hvg"].var_names[ours["hvg"].var["highly_variable"].to_numpy()],
        theirs["hvg"].var_names[theirs["hvg"].var["highly_variable"].to_numpy()],
    )
    assert hvg_overlap >= 0.95, f"HVG overlap {hvg_overlap:.3f} < 0.95"

    # The HVG sets may differ slightly, so scaling is compared where both kept the gene.
    # atol is a decade looser than in the single-step test: standardising a nearly
    # constant gene divides by a tiny standard deviation, which turns the float32 rounding
    # the two pipelines accumulated differently into a large *relative* error on a value
    # that is itself ~1e-2. The data is unit variance, so 1e-5 absolute is still far below
    # anything that could matter downstream. Measured worst case: 1.4e-6 on 3 of 5.3M.
    ours_scaled, theirs_scaled = _common_genes(ours["scaled"], theirs["scaled"])
    assert_allclose(as_dense(ours_scaled.X), as_dense(theirs_scaled.X), rtol=1e-5, atol=1e-5)

    check_pca_agreement(
        theirs["scaled"], ours["pca"], theirs["pca"], label="pipeline", record=record_property
    )

    # Neighbours here are built on our PCA against theirs, and the two disagree in
    # the degenerate tail of the spectrum by design. The ceiling is therefore what
    # scanpy itself reaches when its PCA is re-run with the randomised solver — the
    # same class of disagreement — rather than the 0.90 the single-step test uses,
    # where both sides share one representation.
    ceiling_pipeline = theirs["scaled"].copy()
    sc.pp.pca(ceiling_pipeline, n_comps=_N_COMPS, random_state=0, svd_solver="randomized")
    sc.pp.neighbors(ceiling_pipeline, n_neighbors=_N_NEIGHBORS, use_rep="X_pca")
    reference_sets = neighbor_sets(theirs["neighbors"].obsp["distances"])
    ceiling = per_row_overlap(
        neighbor_sets(ceiling_pipeline.obsp["distances"]), reference_sets
    ).mean()
    overlaps = per_row_overlap(neighbor_sets(ours["neighbors"].obsp["distances"]), reference_sets)
    record_property("pipeline.neighbor_overlap", round(float(overlaps.mean()), 4))
    record_property("pipeline.neighbor_overlap_ceiling", round(float(ceiling), 4))
    assert overlaps.mean() >= CEILING_FRACTION * ceiling, (
        f"mean neighbour overlap {overlaps.mean():.3f} against a ceiling of {ceiling:.3f} "
        f"(worst cell {overlaps.min():.3f})"
    )

    # UMAP does not reproduce itself across seeds on PBMC 3k, so the bar is a fraction of
    # the ceiling scanpy reaches against its own re-run, measured here rather than assumed.
    # The ceiling must carry the same divergence as the run under test: a UMAP laid
    # out on a graph built from a *different* PCA, not a reseeded run on the same
    # graph. Re-running scanpy end to end with its randomised solver is exactly that.
    sc.tl.umap(ceiling_pipeline, random_state=0)
    preserved, ceiling = preservation_band(
        theirs["umap"].obsm["X_umap"],
        ceiling_pipeline.obsm["X_umap"],
        ours["umap"].obsm["X_umap"],
        k_ref=K_REF,
        k_cand=K_CAND,
    )
    record_property("pipeline.umap.preservation", round(preserved, 4))
    record_property("pipeline.umap.ceiling", round(ceiling, 4))
    print(f"\npipeline umap: preservation {preserved:.3f}, ceiling {ceiling:.3f}")
    assert preserved >= CEILING_FRACTION * ceiling, (
        f"UMAP preservation {preserved:.3f} is below {CEILING_FRACTION:.0%} of the "
        f"{ceiling:.3f} scanpy reaches when its own PCA is re-run randomised"
    )

    ours_de = ours["de"].uns["rank_genes_groups"]
    theirs_de = theirs["de"].uns["rank_genes_groups"]
    for group in theirs_de["names"].dtype.names:
        problems, deviations = de_comparison(
            ours_de, theirs_de, group, tolerances=_PIPELINE_DE_TOLERANCES
        )
        for field, worst in deviations.items():
            record_property(f"pipeline.de.{group}.{field}", f"{worst:.2e}")
        assert not problems, (
            f"differential expression differs for {group} after the pipeline: {problems}"
        )
