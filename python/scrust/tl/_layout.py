"""Dendrograms, force-directed layouts and embedding densities. Owned by feat/layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["dendrogram", "draw_graph", "embedding_density"]


def dendrogram(
    adata: AnnData,
    groupby: str,
    *,
    n_pcs: int = 50,
    use_rep: str = "X_pca",
    key_added: str | None = None,
) -> None:
    """Hierarchical clustering of group means, as `scanpy.tl.dendrogram`."""
    raise NotImplementedError("feat/layout")


def draw_graph(
    adata: AnnData,
    *,
    layout: str = "fa",
    neighbors_key: str = "neighbors",
    n_iterations: int = 500,
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Force-directed layout of the neighbour graph, as `scanpy.tl.draw_graph`."""
    raise NotImplementedError("feat/layout")


def embedding_density(
    adata: AnnData, *, basis: str = "umap", groupby: str | None = None, key_added: str | None = None
) -> None:
    """Kernel density of cells in an embedding, as `scanpy.tl.embedding_density`."""
    raise NotImplementedError("feat/layout")
