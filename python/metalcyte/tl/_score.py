"""Score cells for a gene set and compare marker gene lists. Owned by feat/scoring."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from metalcyte._shared import _INDEX_DTYPE, _csr_args, _extension, _resolve_device

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anndata import AnnData

__all__ = ["marker_gene_overlap", "score_genes", "score_genes_cell_cycle"]

# The number of top markers used when `top_n_markers` is not given.
_DEFAULT_TOP_MARKERS = 100

# The three overlap measures, each over two sets of gene names.
_OVERLAP_METHODS = {
    "overlap_count": lambda reference, called: float(len(reference & called)),
    "overlap_coef": lambda reference, called: (
        len(reference & called) / max(min(len(reference), len(called)), 1)
    ),
    "jaccard": lambda reference, called: len(reference & called) / len(reference | called),
}


def score_genes(
    adata: AnnData,
    gene_list: Sequence[str],
    *,
    ctrl_size: int = 50,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int = 0,
    device: str | None = None,
) -> None:
    """Score each cell for a gene set against matched control genes.

    The score is the mean expression of the gene set minus the mean expression of
    control genes drawn from the same expression bins. It is written to
    `obs[score_name]`.
    """
    device = _resolve_device(device)
    scores = _extension().score_genes(
        *_csr_args(adata.X),
        _gene_columns(adata, gene_list),
        ctrl_size,
        n_bins,
        random_state,
        device,
    )
    # The column is float64 even though the arithmetic is f32.
    adata.obs[score_name] = pd.Series(np.asarray(scores, dtype=np.float64), index=adata.obs_names)


def _gene_columns(adata: AnnData, gene_list: Sequence[str]) -> np.ndarray:
    """Column indices of the genes that are present in `var_names`.

    Genes missing from `var_names` are dropped with a warning. An empty result is
    an error. A caller who misspells a gene finds out here and does not get a
    quietly smaller set.
    """
    requested = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    ignored = requested.difference(adata.var_names, sort=False)
    present = requested.intersection(adata.var_names)
    if len(ignored) > 0:
        warnings.warn(
            f"genes are not in var_names and ignored: {list(ignored)}",
            UserWarning,
            stacklevel=3,
        )
    if len(present) == 0:
        raise ValueError("No valid genes were passed for scoring.")
    return adata.var_names.get_indexer(present).astype(_INDEX_DTYPE)


def score_genes_cell_cycle(
    adata: AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    device: str | None = None,
) -> None:
    """S and G2M scores plus the assigned cell-cycle phase.

    Writes `obs["S_score"]`, `obs["G2M_score"]` and `obs["phase"]`. A cell is G2M when
    its G2M score is higher, S otherwise, and G1 when both scores are below 0.
    """
    device = _resolve_device(device)
    ctrl_size = min(len(s_genes), len(g2m_genes))
    for genes, name in ((s_genes, "S_score"), (g2m_genes, "G2M_score")):
        score_genes(adata, genes, score_name=name, ctrl_size=ctrl_size, device=device)

    scores = adata.obs[["S_score", "G2M_score"]]
    # S unless G2M outscores it, and G1 when neither programme beats its control.
    phase = pd.Series("S", index=scores.index)
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"
    phase[np.all(scores < 0, axis=1)] = "G1"
    adata.obs["phase"] = phase


def marker_gene_overlap(
    adata: AnnData,
    reference_markers: Mapping[str, Sequence[str]],
    *,
    key: str = "rank_genes_groups",
    method: str = "overlap_count",
    top_n_markers: int | None = None,
) -> pd.DataFrame:
    """Overlap between the marker genes found for each group and a reference set.

    Returns a table with one row per reference cell type and one column per group.
    """
    if key not in adata.uns:
        raise ValueError(f"adata.uns has no {key!r}; run rank_genes_groups first")
    if method not in _OVERLAP_METHODS:
        raise ValueError(f"method must be one of {sorted(_OVERLAP_METHODS)}, got {method!r}")
    if top_n_markers is not None and top_n_markers < 1:
        warnings.warn(
            "`top_n_markers` was set below 1 and is treated as 1", UserWarning, stacklevel=2
        )
        top_n_markers = 1

    n_markers = _DEFAULT_TOP_MARKERS if top_n_markers is None else top_n_markers
    names = adata.uns[key]["names"]
    called = {group: set(names[group][:n_markers]) for group in names.dtype.names}
    reference = {group: set(genes) for group, genes in reference_markers.items()}

    overlap = _OVERLAP_METHODS[method]
    return pd.DataFrame(
        [[overlap(genes, called[group]) for group in called] for genes in reference.values()],
        index=list(reference),
        columns=list(called),
        dtype=float,
    )
