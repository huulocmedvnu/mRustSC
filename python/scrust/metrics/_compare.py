"""Comparing labellings. Owned by feat/metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

__all__ = ["confusion_matrix", "modularity"]


def confusion_matrix(
    orig: Any, new: Any, data: Any = None, *, normalize: bool = True
) -> pd.DataFrame:
    """Contingency table of two labellings, as `scanpy.metrics.confusion_matrix`."""
    raise NotImplementedError("feat/metrics")


def modularity(adata: AnnData, keys: str, *, neighbors_key: str = "neighbors") -> float:
    """Newman modularity of a labelling on the neighbour graph."""
    raise NotImplementedError("feat/metrics")
