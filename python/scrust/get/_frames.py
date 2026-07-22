"""Pulling tidy frames out of an AnnData. Owned by feat/accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = ["aggregate", "obs_df", "rank_genes_groups_df", "var_df"]


def obs_df(
    adata: AnnData,
    keys: Sequence[str] = (),
    *,
    obsm_keys: Sequence[tuple[str, int]] = (),
    layer: str | None = None,
) -> pd.DataFrame:
    """Per-cell frame of genes, `obs` columns and `obsm` slices, as `scanpy.get.obs_df`."""
    raise NotImplementedError("feat/accessors")


def var_df(
    adata: AnnData, keys: Sequence[str] = (), *, varm_keys: Sequence[tuple[str, int]] = ()
) -> pd.DataFrame:
    """Per-gene frame, as `scanpy.get.var_df`."""
    raise NotImplementedError("feat/accessors")


def rank_genes_groups_df(
    adata: AnnData,
    group: str | Sequence[str] | None,
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
) -> pd.DataFrame:
    """The differential expression result as a tidy frame, as `scanpy.get.rank_genes_groups_df`."""
    raise NotImplementedError("feat/accessors")


def aggregate(
    adata: AnnData,
    by: str | Sequence[str],
    func: str | Sequence[str],
    *,
    axis: int = 0,
    layer: str | None = None,
    device: str = "auto",
) -> AnnData:
    """Group cells and reduce, as `scanpy.get.aggregate`."""
    raise NotImplementedError("feat/accessors")
