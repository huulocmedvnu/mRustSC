"""Gene-set scoring and marker comparison. Owned by feat/scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = ["marker_gene_overlap", "score_genes", "score_genes_cell_cycle"]


def score_genes(
    adata: AnnData,
    gene_list: Sequence[str],
    *,
    ctrl_size: int = 50,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Mean expression of a gene set minus a binned control, as `scanpy.tl.score_genes`."""
    raise NotImplementedError("feat/scoring")


def score_genes_cell_cycle(
    adata: AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    device: str = "auto",
) -> None:
    """S and G2M scores plus the assigned phase, as `scanpy.tl.score_genes_cell_cycle`."""
    raise NotImplementedError("feat/scoring")


def marker_gene_overlap(
    adata: AnnData,
    reference_markers: Mapping[str, Sequence[str]],
    *,
    key: str = "rank_genes_groups",
    method: str = "overlap_count",
    top_n_markers: int | None = None,
) -> pd.DataFrame:
    """Overlap between called markers and a reference set, as `scanpy.tl.marker_gene_overlap`."""
    raise NotImplementedError("feat/scoring")
