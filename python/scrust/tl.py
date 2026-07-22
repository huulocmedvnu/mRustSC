"""Tools, mirroring `scanpy.tl`. Owned by feat/python-tl.

Like `scrust.pp` this is plumbing only; the AnnData conventions it needs live in
`scrust.pp` as private helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust.pp import _VALUE_DTYPE, _csr_args, _extension, _representation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

__all__ = ["rank_genes_groups", "tsne", "umap"]

# scanpy's dtype per `uns["rank_genes_groups"]` field; its plotting reads these.
_DE_FIELD_DTYPES = {
    "names": "O",
    "scores": "float32",
    "logfoldchanges": "float32",
    "pvals": "float64",
    "pvals_adj": "float64",
}

# Only Wilcoxon is wired up, and the core's tie handling is not exposed yet.
_SUPPORTED_METHODS = ("wilcoxon",)
_TIE_CORRECT = False


def umap(
    adata: AnnData,
    *,
    n_components: int = 2,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_epochs: int | None = None,
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Lay the neighbour graph out with UMAP, writing `obsm["X_umap"]`."""
    graph = _neighbor_graph(adata)
    embedding = _extension().umap(
        *_csr_args(graph), n_components, min_dist, spread, n_epochs, random_state, device
    )
    adata.obsm["X_umap"] = np.asarray(embedding, dtype=_VALUE_DTYPE)


def tsne(
    adata: AnnData,
    *,
    n_pcs: int = 50,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float = 200.0,
    random_state: int = 0,
    device: str = "auto",
) -> None:
    """Lay the principal components out with t-SNE, writing `obsm["X_tsne"]`."""
    embedding = _representation(adata, "X_pca")[:, :n_pcs]
    result = _extension().tsne(
        np.ascontiguousarray(embedding),
        perplexity,
        early_exaggeration,
        learning_rate,
        random_state,
        device,
    )
    adata.obsm["X_tsne"] = np.asarray(result, dtype=_VALUE_DTYPE)


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    groups: str | Sequence[str] = "all",
    reference: str = "rest",
    method: str = "wilcoxon",
    device: str = "auto",
) -> None:
    """Rank genes by differential expression, writing `uns["rank_genes_groups"]`."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {_SUPPORTED_METHODS}, got {method!r}")

    group_names, label_names = _group_order(adata, groupby, groups, reference)
    labels = _labels(adata, groupby, label_names)
    reference_index = -1 if reference == "rest" else label_names.index(reference)

    result = _extension().rank_genes_groups_wilcoxon(
        *_csr_args(adata.X),
        labels,
        len(label_names),
        reference_index,
        _TIE_CORRECT,
        device,
    )

    rows = [label_names.index(name) for name in group_names]
    gene_names = adata.var_names.to_numpy()
    ranked = np.asarray(result["names"])
    fields = {"names": [gene_names[ranked[row]] for row in rows]}
    fields.update(
        {
            key: [np.asarray(result[key])[row] for row in rows]
            for key in _DE_FIELD_DTYPES
            if key != "names"
        }
    )
    adata.uns["rank_genes_groups"] = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": method,
            "use_raw": False,
            "layer": None,
            "corr_method": "benjamini-hochberg",
        },
        **{
            key: _record_array(columns, group_names, _DE_FIELD_DTYPES[key])
            for key, columns in fields.items()
        },
    }


def _neighbor_graph(adata: AnnData):
    """Return the connectivities graph UMAP lays out."""
    if "connectivities" not in adata.obsp:
        raise KeyError("adata.obsp has no 'connectivities'; run scrust.pp.neighbors first")
    return adata.obsp["connectivities"]


def _group_order(
    adata: AnnData, groupby: str, groups: str | Sequence[str], reference: str
) -> tuple[list[str], list[str]]:
    """Return the groups to report on, and the groups the core must label.

    A named `reference` is labelled too even when it is not reported, because the
    test compares against its cells.
    """
    if groupby not in adata.obs:
        raise KeyError(f"adata.obs has no {groupby!r}")
    categories = [str(name) for name in adata.obs[groupby].astype("category").cat.categories]

    if isinstance(groups, str):
        if groups != "all":
            raise ValueError("groups must be 'all' or a sequence of group names")
        group_names = list(categories)
    else:
        group_names = [str(name) for name in groups]

    unknown = set(group_names + ([reference] if reference != "rest" else [])) - set(categories)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)} in adata.obs[{groupby!r}]")

    label_names = list(group_names)
    if reference != "rest" and reference not in label_names:
        label_names.append(reference)
    return group_names, label_names


def _labels(adata: AnnData, groupby: str, label_names: Sequence[str]) -> np.ndarray:
    """Encode the grouping as `int32` codes into `label_names`, `-1` for excluded cells."""
    positions = {name: index for index, name in enumerate(label_names)}
    observed = adata.obs[groupby].astype(str)
    return observed.map(lambda name: positions.get(name, -1)).to_numpy(dtype=np.int32)


def _record_array(columns: Sequence[np.ndarray], group_names: Sequence[str], dtype: str):
    """Build the one-field-per-group structured array scanpy's accessors expect."""
    return np.rec.fromarrays(
        [np.asarray(column, dtype=dtype) for column in columns],
        dtype=[(name, dtype) for name in group_names],
    )
