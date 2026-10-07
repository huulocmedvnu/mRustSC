"""Dendrograms, force-directed graph layouts and cell density in an embedding.

Owned by feat/layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from metalcyte._shared import (
    _VALUE_DTYPE,
    _csr_args,
    _default_device,
    _dense,
    _extension,
    _neighbor_graph,
    _representation,
    _resolve_device,
)

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["dendrogram", "draw_graph", "embedding_density"]

# What the core computes, recorded in the `uns` slot so a reader of a stored
# AnnData can see which tree they are looking at.
_COR_METHOD = "pearson"
_LINKAGE_METHOD = "complete"

# `tl.draw_graph` computes only the ForceAtlas2 layout.
_LAYOUT = "fa"

# `embedding_density` always uses the first two components of a basis.
_DENSITY_COMPONENTS = [1, 2]


def _categorical(adata: AnnData, key: str):
    """The `obs` column named `key`, which has to be categorical."""
    import pandas as pd

    if key not in adata.obs:
        raise KeyError(f"adata.obs has no {key!r}")
    column = adata.obs[key]
    if not isinstance(column.dtype, pd.CategoricalDtype):
        raise ValueError(f"adata.obs[{key!r}] is {column.dtype}, not categorical")
    return column


def dendrogram(
    adata: AnnData,
    groupby: str,
    *,
    n_pcs: int = 50,
    use_rep: str = "X_pca",
    key_added: str | None = None,
) -> None:
    """Hierarchical clustering of group means.

    It averages `use_rep` within each group, correlates the means (Pearson) and joins
    them by complete linkage. Writes `uns["dendrogram_<groupby>"]` with the linkage,
    the leaf order, the correlation matrix and the drawing coordinates.
    """
    import pandas as pd

    column = _categorical(adata, groupby)
    categories = list(column.cat.categories)
    if len(categories) < 2:
        raise ValueError(f"adata.obs[{groupby!r}] has {len(categories)} categories; 2 are needed")

    codes = column.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"adata.obs[{groupby!r}] has unlabelled cells")

    # The group means: arithmetic, but the core's entry point takes centroids, so
    # somebody upstream of it has to average. pandas does it here. Grouping by the codes
    # keeps the rows in category order, which is the order the leaf ids index.
    representation = pd.DataFrame(_representation(adata, use_rep)[:, :n_pcs])
    means = representation.groupby(codes, observed=True).mean()
    if len(means) != len(categories):
        raise ValueError(f"adata.obs[{groupby!r}] has a category with no cells")
    centroids = np.ascontiguousarray(means.to_numpy(), dtype=_VALUE_DTYPE)

    linkage, leaves = _extension().dendrogram(centroids)
    leaf_order = [int(leaf) for leaf in leaves]

    # Correlations for `pl.correlation_matrix`, and the drawing coordinates for
    # `pl.dendrogram`. Neither is the clustering, which is `linkage` and
    # `leaf_order` above. Plotting reads both, and the drawing coordinates follow
    # scipy's convention.
    correlation = np.clip(np.corrcoef(centroids.astype(np.float64)), -1.0, 1.0)
    import scipy.cluster.hierarchy as sch

    info = sch.dendrogram(linkage, labels=categories, no_plot=True)

    adata.uns[key_added or f"dendrogram_{groupby}"] = {
        "linkage": linkage,
        "groupby": [groupby],
        "use_rep": use_rep,
        "cor_method": _COR_METHOD,
        "linkage_method": _LINKAGE_METHOD,
        "categories_ordered": [categories[leaf] for leaf in leaf_order],
        "categories_idx_ordered": leaf_order,
        "dendrogram_info": info,
        "correlation_matrix": correlation,
    }


def draw_graph(
    adata: AnnData,
    *,
    layout: str = "fa",
    neighbors_key: str = "neighbors",
    n_iterations: int = 500,
    random_state: int = 0,
    device: str | None = None,
) -> None:
    """Force-directed layout of the neighbour graph with ForceAtlas2 (Jacomy et al. 2014).

    Writes the positions to `obsm["X_draw_graph_fa"]` and the parameters to
    `uns["draw_graph"]`.
    """
    device = _resolve_device(device)
    if layout != _LAYOUT:
        raise ValueError(f"layout must be 'fa' (ForceAtlas2), got {layout!r}")

    # `neighbors_key` names the `uns` entry that says where the graph lives; the
    # default entry, and a missing one, both mean `obsp["connectivities"]`.
    graph_key = adata.uns.get(neighbors_key, {}).get("connectivities_key")
    graph = adata.obsp[graph_key] if graph_key in adata.obsp else _neighbor_graph(adata)
    positions = _extension().draw_graph(*_csr_args(graph), n_iterations, random_state, device)
    adata.obsm[f"X_draw_graph_{_LAYOUT}"] = np.asarray(positions, dtype=_VALUE_DTYPE)
    adata.uns["draw_graph"] = {"params": {"layout": _LAYOUT, "random_state": random_state}}


def embedding_density(
    adata: AnnData, *, basis: str = "umap", groupby: str | None = None, key_added: str | None = None
) -> None:
    """Estimate the density of cells in an embedding with a Gaussian kernel.

    Writes `obs["<basis>_density_<groupby>"]` and its `uns` parameters. Densities
    are scaled to `[0, 1]` *within* each group. They compare cells inside a group
    and not across groups, so the `groupby` used is stored beside them.
    """
    basis = "draw_graph_fa" if basis.lower() == "fa" else basis.lower()
    if f"X_{basis}" not in adata.obsm:
        raise KeyError(f"adata.obsm has no 'X_{basis}'; compute the embedding first")

    embedding = _dense(adata.obsm[f"X_{basis}"])[:, :2]
    if embedding.shape[1] != 2:
        raise ValueError(f"adata.obsm['X_{basis}'] has {embedding.shape[1]} columns; 2 are needed")

    device = _default_device()
    density = np.zeros(adata.n_obs, dtype=np.float64)
    if groupby is None:
        density[:] = _extension().embedding_density(np.ascontiguousarray(embedding), device)
    else:
        column = _categorical(adata, groupby)
        for category in column.cat.categories:
            selected = (column == category).to_numpy()
            density[selected] = _extension().embedding_density(
                np.ascontiguousarray(embedding[selected]), device
            )

    covariate = key_added or (
        f"{basis}_density_{groupby}" if groupby is not None else f"{basis}_density"
    )
    adata.obs[covariate] = density
    parameters: dict[str, Any] = {"covariate": groupby, "components": _DENSITY_COMPONENTS}
    adata.uns[f"{covariate}_params"] = parameters
