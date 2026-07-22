"""Subsampling cells and counts. Owned by feat/sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["downsample_counts", "sample", "subsample"]


def subsample(
    adata: AnnData,
    fraction: float | None = None,
    *,
    n_obs: int | None = None,
    random_state: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Keep a random subset of cells, as `scanpy.pp.subsample`."""
    raise NotImplementedError("feat/sampling")


def sample(
    adata: AnnData,
    fraction: float | None = None,
    *,
    n: int | None = None,
    replace: bool = False,
    random_state: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """scanpy's newer sampling entry point, which also allows replacement."""
    raise NotImplementedError("feat/sampling")


def downsample_counts(
    adata: AnnData,
    *,
    counts_per_cell: int | None = None,
    total_counts: int | None = None,
    random_state: int = 0,
    replace: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Thin the counts themselves, as `scanpy.pp.downsample_counts`."""
    raise NotImplementedError("feat/sampling")
