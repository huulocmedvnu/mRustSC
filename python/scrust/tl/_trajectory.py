"""Diffusion maps, pseudotime and abstracted graphs. Owned by feat/diffusion and feat/paga."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scrust._shared import _VALUE_DTYPE, _csr_args, _dense, _extension, _neighbor_graph

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["diffmap", "dpt", "paga"]

# scanpy's `tl.dpt` falls back to `tl.diffmap`'s own default when no map is
# stored, not to `n_dcs`; a later `dpt` with a larger `n_dcs` fails there too.
_DIFFMAP_COMPONENTS = 15


def diffmap(
    adata: AnnData, n_comps: int = 15, *, neighbors_key: str = "neighbors", device: str = "auto"
) -> None:
    """Diffusion map of the neighbour graph, as `scanpy.tl.diffmap`."""
    if neighbors_key == "neighbors":
        graph = _neighbor_graph(adata)
    else:
        graph = adata.obsp[adata.uns[neighbors_key]["connectivities_key"]]
    embedding, eigenvalues = _extension().diffmap(*_csr_args(graph), n_comps, device)
    # scanpy keeps the trivial first component in `X_diffmap`; only `pl.diffmap`
    # skips it, and `tl.dpt` reads it back and uses it.
    adata.obsm["X_diffmap"] = np.asarray(embedding, dtype=_VALUE_DTYPE)
    adata.uns["diffmap_evals"] = np.asarray(eigenvalues, dtype=_VALUE_DTYPE)


def dpt(
    adata: AnnData,
    *,
    n_dcs: int = 10,
    n_branchings: int = 0,
    min_group_size: float = 0.01,
    device: str = "auto",
) -> None:
    """Diffusion pseudotime from `uns["iroot"]`, as `scanpy.tl.dpt`."""
    if n_branchings > 0:
        raise NotImplementedError(
            "branch detection is not implemented; use scrust.tl.paga for branches "
            f"(min_group_size={min_group_size} applies to branchings only)"
        )
    if "X_diffmap" not in adata.obsm:
        diffmap(adata, n_comps=_DIFFMAP_COMPONENTS, device=device)
    if "iroot" not in adata.uns:
        raise KeyError("adata.uns['iroot'] must hold the index of the root cell")
    pseudotime = _extension().dpt(
        _dense(adata.obsm["X_diffmap"]),
        np.asarray(adata.uns["diffmap_evals"], dtype=_VALUE_DTYPE),
        int(adata.uns["iroot"]),
        n_dcs,
    )
    adata.obs["dpt_pseudotime"] = np.asarray(pseudotime, dtype=_VALUE_DTYPE)


def paga(
    adata: AnnData, groups: str | None = None, *, model: str = "v1.2", device: str = "auto"
) -> None:
    """Partition-based graph abstraction, as `scanpy.tl.paga`."""
    raise NotImplementedError("feat/paga")
