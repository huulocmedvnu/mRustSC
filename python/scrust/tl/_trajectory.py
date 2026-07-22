"""Diffusion maps, pseudotime and abstracted graphs. Owned by feat/diffusion and feat/paga."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["diffmap", "dpt", "paga"]


def diffmap(
    adata: AnnData, n_comps: int = 15, *, neighbors_key: str = "neighbors", device: str = "auto"
) -> None:
    """Diffusion map of the neighbour graph, as `scanpy.tl.diffmap`."""
    raise NotImplementedError("feat/diffusion")


def dpt(
    adata: AnnData,
    *,
    n_dcs: int = 10,
    n_branchings: int = 0,
    min_group_size: float = 0.01,
    device: str = "auto",
) -> None:
    """Diffusion pseudotime from `uns["iroot"]`, as `scanpy.tl.dpt`."""
    raise NotImplementedError("feat/diffusion")


def paga(
    adata: AnnData, groups: str | None = None, *, model: str = "v1.2", device: str = "auto"
) -> None:
    """Partition-based graph abstraction, as `scanpy.tl.paga`."""
    raise NotImplementedError("feat/paga")
