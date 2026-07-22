"""Subsampling cells and counts. Owned by feat/sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust._shared import _csr_args, _csr_from_parts, _extension

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
    return sample(adata, fraction, n=n_obs, replace=False, random_state=random_state, copy=copy)


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
    n = _resolve_size(adata.n_obs, fraction, n, replace=replace)
    indices = np.asarray(
        _extension().subsample(adata.n_obs, n, replace, random_state), dtype=np.intp
    )
    if copy:
        # `adata[indices]` is a view; copying it carries obs and var along.
        return adata[indices].copy()
    adata._inplace_subset_obs(indices)
    return None


def _resolve_size(n_obs: int, fraction: float | None, n: int | None, *, replace: bool) -> int:
    """The number of cells to draw, from whichever of the two the caller gave.

    scanpy raises `TypeError` for both or neither and `ValueError` for a fraction
    that cannot be honoured, so the same two exception types are raised here.
    """
    if (fraction is None) is (n is None):
        given = "both" if n is not None else "neither"
        raise TypeError(f"provide exactly one of fraction or n, got {given}")
    if n is not None:
        return n
    if fraction < 0:
        raise ValueError(f"fraction needs to be nonnegative, got {fraction}")
    if fraction > 1 and not replace:
        raise ValueError(f"if replace=False, fraction needs to be within [0, 1], got {fraction}")
    return int(fraction * n_obs)


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
    if (counts_per_cell is None) is (total_counts is None):
        raise ValueError("Must specify exactly one of `total_counts` or `counts_per_cell`.")
    if copy:
        adata = adata.copy()
    parts = _extension().downsample_counts(
        *_csr_args(adata.X), counts_per_cell, total_counts, replace, random_state
    )
    adata.X = _csr_from_parts(parts, adata.shape)
    return adata if copy else None
