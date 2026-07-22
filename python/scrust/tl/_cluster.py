"""Community detection on the neighbour graph. Owned by feat/leiden."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scrust._shared import _csr_args, _extension, _neighbor_graph

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["leiden", "louvain"]


def _connectivities(adata: AnnData, neighbors_key: str) -> Any:
    """The weighted graph to cluster, resolved the way scanpy resolves it.

    `uns[neighbors_key]` names the `obsp` key; without it the conventional
    `obsp["connectivities"]` is used, which is also where the error lives.
    """
    settings = adata.uns.get(neighbors_key)
    if isinstance(settings, dict):
        key = settings.get("connectivities_key", "connectivities")
        if key in adata.obsp:
            return adata.obsp[key]
    return _neighbor_graph(adata)


def _write_clusters(
    adata: AnnData,
    key_added: str,
    partition: tuple[np.ndarray, float, int],
    params: dict[str, Any],
) -> None:
    """Store a partition the way scanpy stores one.

    The core numbers communities `0..n-1` by descending size, so listing the
    categories in numeric order is already scanpy's natural sort of the labels.
    """
    labels, modularity, n_communities = partition
    adata.obs[key_added] = pd.Categorical(
        values=np.asarray(labels).astype("U"),
        categories=[str(community) for community in range(n_communities)],
    )
    adata.uns[key_added] = {"params": params, "modularity": modularity}


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
    graph = _connectivities(adata, neighbors_key)
    partition = _extension().leiden(
        *_csr_args(graph),
        resolution,
        n_iterations,
        random_state,
        device,
    )
    _write_clusters(
        adata,
        key_added,
        partition,
        {
            "resolution": resolution,
            "random_state": random_state,
            "n_iterations": n_iterations,
        },
    )


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
    graph = _connectivities(adata, neighbors_key)
    partition = _extension().louvain(
        *_csr_args(graph),
        resolution,
        random_state,
        device,
    )
    _write_clusters(
        adata,
        key_added,
        partition,
        {"resolution": resolution, "random_state": random_state},
    )
