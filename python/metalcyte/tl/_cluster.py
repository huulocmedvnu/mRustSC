"""Find clusters (communities) of cells in the neighbour graph. Owned by feat/leiden."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from metalcyte._shared import _csr_args, _extension, _neighbor_graph, _resolve_device

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["leiden", "louvain"]


def _connectivities(adata: AnnData, neighbors_key: str) -> Any:
    """The weighted graph to cluster.

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
    """Store a partition as a categorical `obs` column and its parameters in `uns`.

    The core numbers communities `0..n-1` by descending size. The categories are
    listed in that numeric order, so cluster 0 is the largest.
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
    device: str | None = None,
) -> None:
    """Leiden clustering (Traag et al. 2019) of the neighbour graph.

    Writes the cluster of each cell to `obs[key_added]` and the parameters and
    modularity to `uns[key_added]`.
    """
    device = _resolve_device(device)
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
    device: str | None = None,
) -> None:
    """Louvain clustering (Blondel et al. 2008) of the neighbour graph.

    Writes the cluster of each cell to `obs[key_added]` and the parameters and
    modularity to `uns[key_added]`.
    """
    device = _resolve_device(device)
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
