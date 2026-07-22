"""Community detection on the neighbour graph. Owned by feat/leiden."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["leiden", "louvain"]


def leiden(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    key_added: str = "leiden",
    neighbors_key: str = "neighbors",
    n_iterations: int = 2,
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Leiden clustering, writing `obs[key_added]` as `scanpy.tl.leiden` does."""
    raise NotImplementedError("feat/leiden")


def louvain(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    key_added: str = "louvain",
    neighbors_key: str = "neighbors",
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Louvain clustering, writing `obs[key_added]` as `scanpy.tl.louvain` does."""
    raise NotImplementedError("feat/leiden")
