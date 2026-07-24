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
    """Diffusion pseudotime from `uns["iroot"]`, as `scanpy.tl.dpt`.

    With `n_branchings > 0`, also detects branchings and writes `obs["dpt_groups"]`, a
    categorical partition, using a native port of scanpy's Haghverdi 2016 algorithm
    (`scrust.tl._dpt_branching`). Branch labels are arbitrary, so parity with scanpy is an
    adjusted Rand index, pinned in `tests/test_dpt_branching_audit.py`.
    """
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

    if n_branchings > 0:
        import pandas as pd

        from scrust.tl._dpt_branching import dpt_groups as _dpt_groups

        labels = _dpt_groups(
            adata,
            n_branchings=n_branchings,
            min_group_size=min_group_size,
            n_dcs=n_dcs,
        )
        categories = [str(label) for label in sorted(set(labels.tolist()))]
        adata.obs["dpt_groups"] = pd.Categorical(
            labels.astype(str), categories=categories
        )


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
