"""Spatial autocorrelation over the neighbour graph. Owned by feat/metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

__all__ = ["gearys_c", "morans_i"]


def gearys_c(
    adata: AnnData, *, vals: Any = None, use_graph: str = "connectivities", device: str = "auto"
) -> np.ndarray:
    """Geary's C for each gene over the neighbour graph, as `scanpy.metrics.gearys_c`."""
    raise NotImplementedError("feat/metrics")


def morans_i(
    adata: AnnData, *, vals: Any = None, use_graph: str = "connectivities", device: str = "auto"
) -> np.ndarray:
    """Moran's I for each gene over the neighbour graph, as `scanpy.metrics.morans_i`."""
    raise NotImplementedError("feat/metrics")
