"""Validate the pipeline against real 10x PBMC 3k data.

Pseudobulks CD14+ monocytes and B cells into replicates, runs the default
pipeline, and checks the calls against canonical marker genes and against
scanpy's Wilcoxon ranking. Requires the optional `validate` extra.

Usage: PYTHONPATH=src python scripts/validate_pbmc.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

from mlxde.contracts import CountMatrix
from mlxde.factory import build_default_pipeline
from mlxde.io.design import build_design_matrix

CELL_TYPES = {"CD14+ Monocytes": "monocyte", "B cells": "bcell"}
MONOCYTE_MARKERS = ("LYZ", "S100A8", "S100A9", "CD14", "FCN1", "VCAN", "CST3", "FTL")
BCELL_MARKERS = ("MS4A1", "CD79A", "CD79B", "TCL1A", "BANK1", "CD19")
N_REPLICATES = 5


def build_pseudobulk(seed: int = 0) -> CountMatrix:
    """Sum each cell type's counts into replicate pools of cells."""
    raw = sc.datasets.pbmc3k()
    labelled = sc.datasets.pbmc3k_processed()
    raw.var_names_make_unique()

    shared_cells = raw.obs_names.intersection(labelled.obs_names)
    counts = raw[shared_cells].to_df()
    cell_types = labelled.obs.loc[shared_cells, "louvain"].astype(str)

    rng = np.random.default_rng(seed)
    pools: dict[str, np.ndarray] = {}
    conditions: dict[str, str] = {}
    for cell_type, condition in CELL_TYPES.items():
        cells = np.array(cell_types.index[cell_types == cell_type])
        for replicate, pool in enumerate(np.array_split(rng.permutation(cells), N_REPLICATES)):
            sample_id = f"{condition}_{replicate}"
            pools[sample_id] = counts.loc[pool].to_numpy().sum(axis=0)
            conditions[sample_id] = condition

    sample_ids = np.array(list(pools))
    return CountMatrix(
        counts=np.column_stack([pools[sample_id] for sample_id in sample_ids]),
        gene_ids=np.array(counts.columns),
        sample_ids=sample_ids,
        sample_metadata=pd.DataFrame(
            {"condition": [conditions[sample_id] for sample_id in sample_ids]}, index=sample_ids
        ),
    )


def report_markers(table: pd.DataFrame) -> int:
    """Print each marker's call and return how many matched expectation."""
    correct = 0
    for marker, expected_up_in in [(gene, "monocyte") for gene in MONOCYTE_MARKERS] + [
        (gene, "bcell") for gene in BCELL_MARKERS
    ]:
        if marker not in table.index:
            print(f"  ABSENT {marker}")
            continue
        row = table.loc[marker]
        observed_up_in = "monocyte" if row.log2_fold_change > 0 else "bcell"
        matched = observed_up_in == expected_up_in and row.adjusted_p_value <= 0.05
        correct += matched
        print(
            f"  {'OK  ' if matched else 'MISS'} {marker:8s} log2FC={row.log2_fold_change:+7.2f} "
            f"padj={row.adjusted_p_value:.1e} (expected up in {expected_up_in})"
        )
    return correct


def main() -> None:
    count_matrix = build_pseudobulk()
    design = build_design_matrix(count_matrix.sample_metadata, "condition", "bcell")
    contrast = design.contrast("condition[monocyte]")
    print(f"pseudobulk: {count_matrix.n_genes} genes x {count_matrix.n_samples} samples")

    result = build_default_pipeline().run(count_matrix, design, contrast)
    table = result.to_dataframe().set_index("gene_id")
    tested = int(np.isfinite(result.p_value).sum())
    significant = result.significant(alpha=0.05, min_abs_log2_fold_change=1.0)
    print(f"tested {tested} genes, {len(significant)} significant (FDR 5%, |log2FC| >= 1)\n")

    n_markers = len(MONOCYTE_MARKERS) + len(BCELL_MARKERS)
    print("canonical markers:")
    print(f"  -> {report_markers(table)}/{n_markers} called in the expected direction\n")

    reference = sc.datasets.pbmc3k_processed()
    reference = reference[reference.obs["louvain"].isin(CELL_TYPES)].copy()
    sc.tl.rank_genes_groups(reference, "louvain", groups=["CD14+ Monocytes"], method="wilcoxon")
    scanpy_top = set(
        pd.DataFrame(reference.uns["rank_genes_groups"]["names"])["CD14+ Monocytes"][:100]
    )
    ours_top = set(table[table.log2_fold_change > 0].sort_values("adjusted_p_value").index[:100])
    print(f"top-100 monocyte genes shared with scanpy wilcoxon: {len(scanpy_top & ours_top)}/100")

    separated = table[np.abs(table.log2_fold_change) > 15]
    if len(separated):
        print(
            f"\n{len(separated)} gene(s) are expressed in one group only; their fold changes are "
            "unbounded and backend-dependent: " + ", ".join(separated.index)
        )


if __name__ == "__main__":
    main()
