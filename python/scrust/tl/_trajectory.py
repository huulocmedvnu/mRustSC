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
    """Partition-based graph abstraction, writing `uns["paga"]`, as `scanpy.tl.paga`.

    `device` is accepted for signature consistency and ignored: coarse-graining a
    neighbour graph is a single memory-bound pass over its stored entries into a
    matrix the size of the group count, which a GPU cannot make faster.
    """
    # Imported here because this module is shared with feat/diffusion, which
    # owns every other line of it.
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from scrust._shared import _LABEL_DTYPE, _csr_args, _extension

    if model != "v1.2":
        raise ValueError(f"model must be 'v1.2', got {model!r}")
    if groups is None:
        groups = next((key for key in ("leiden", "louvain") if key in adata.obs), None)
    if groups is None:
        raise ValueError("no 'leiden' or 'louvain' in adata.obs; pass groups='an_existing_key'")
    if groups not in adata.obs:
        raise KeyError(f"adata.obs has no {groups!r}")
    if "distances" not in adata.obsp:
        raise KeyError("adata.obsp has no 'distances'; run scrust.pp.neighbors first")

    # scanpy abstracts over the *distance* graph, which is directed: an entry
    # (cell, neighbour) is one edge, and the pair is only symmetric by accident.
    column = adata.obs[groups]
    if not isinstance(column.dtype, pd.CategoricalDtype):
        column = column.astype("category")
    codes = column.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"adata.obs[{groups!r}] has unlabelled cells")

    connectivities, tree, n_groups = _extension().paga(
        *_csr_args(adata.obsp["distances"]),
        codes.astype(_LABEL_DTYPE),
        len(column.cat.categories),
    )

    # scanpy stores both as float64 CSR, the tree with each edge once.
    def _sparse(flat: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix(np.asarray(flat, dtype=np.float64).reshape(n_groups, n_groups))

    # Whatever `uns["paga"]` already holds — a layout position, say — is kept.
    slot = dict(adata.uns.get("paga", {}))
    slot["connectivities"] = _sparse(connectivities)
    slot["connectivities_tree"] = _sparse(tree)
    slot["groups"] = groups
    adata.uns["paga"] = slot
