"""Metrics that define "agrees with scanpy" for the reference tests.

These are the measuring instruments of the whole suite, so they live apart from the
tests that use them and are unit tested against hand computed cases in
`test_reference_metrics.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import cdist


def as_dense(x: object) -> NDArray[np.float64]:
    """A dense float64 copy of a dense or sparse matrix."""
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def knn_indices(x: NDArray[np.floating], k: int) -> NDArray[np.int64]:
    """Indices of the `k` nearest neighbours of every row, excluding the row itself.

    Exact and brute force: this is the oracle the approximate implementations are
    measured against, so it must not share their heuristics.
    """
    points = as_dense(x)
    n = points.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    # Stable sort so ties resolve by index and repeated runs give the same answer.
    return np.argsort(distances, axis=1, kind="stable")[:, :k]


def set_overlap(a: Iterable[object], b: Iterable[object]) -> float:
    """Shared fraction of two sets, normalised by the larger one.

    Dividing by the larger set means a short answer cannot buy a perfect score by
    returning only the items it is sure of.
    """
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left), len(right))


def component_correlations(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.float64]:
    """`abs(corr)` of matching columns: the sign of a singular vector is arbitrary."""
    left, right = as_dense(a), as_dense(b)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return np.array(
        [abs(np.corrcoef(left[:, i], right[:, i])[0, 1]) for i in range(left.shape[1])],
        dtype=np.float64,
    )


# Tighter than the contract's rtol=1e-3 where the two implementations should agree to
# machine precision; p-values are compared as float64 because they underflow float32.
DE_TOLERANCES: dict[str, float] = {
    "scores": 1e-3,
    "pvals": 1e-6,
    "pvals_adj": 1e-6,
    "logfoldchanges": 1e-3,
}


def _by_gene(result: Mapping[str, Any], group: str, field: str) -> dict[str, float]:
    values = np.asarray(result[field][group], dtype=np.float64)
    return dict(zip(map(str, result["names"][group]), values, strict=True))


def de_comparison(
    ours: Mapping[str, Any],
    reference: Mapping[str, Any],
    group: str,
    *,
    top: int = 100,
    tolerances: Mapping[str, float] = DE_TOLERANCES,
) -> tuple[list[str], dict[str, float]]:
    """Compare one group of a `rank_genes_groups` result: `(problems, worst deviations)`.

    Values are joined **on the gene name**, never on rank. scanpy's top-n selection uses
    `np.argpartition`, which leaves no defined order among equal scores, so with a
    different tie order rank *i* holds different genes in the two results and a positional
    comparison reports a large error where there is none. What is asserted is therefore:
    the top-`top` gene set, the per-gene values, and that any position where the order
    differs carries equal scores.
    """
    ours_names = [str(name) for name in ours["names"][group]]
    reference_names = [str(name) for name in reference["names"][group]]
    ours_top, reference_top = set(ours_names[:top]), set(reference_names[:top])

    problems: list[str] = []
    if ours_top != reference_top:
        problems.append(
            f"top {top} gene set differs: {sorted(ours_top - reference_top)} "
            f"instead of {sorted(reference_top - ours_top)}"
        )

    deviations: dict[str, float] = {}
    for field, rtol in tolerances.items():
        if field not in reference or field not in ours:
            continue
        ours_values, reference_values = (
            _by_gene(ours, group, field),
            _by_gene(reference, group, field),
        )
        worst, worst_gene = 0.0, ""
        for gene in ours_top & reference_top:
            expected = reference_values[gene]
            # A zero where scanpy has a tiny p-value is a real failure, not a small one,
            # so the scale never falls below the smallest positive double.
            relative = abs(ours_values[gene] - expected) / max(abs(expected), np.finfo(float).tiny)
            if relative > worst:
                worst, worst_gene = relative, gene
        deviations[field] = worst
        if worst > rtol:
            problems.append(
                f"{field} differs by {worst:.2e} relative at {worst_gene} (rtol {rtol})"
            )

    ours_scores = np.asarray(ours["scores"][group][:top], dtype=np.float64)
    reference_scores = np.asarray(reference["scores"][group][:top], dtype=np.float64)
    misordered = [
        i
        for i in range(len(reference_scores))
        if ours_names[i] != reference_names[i]
        and not np.isclose(ours_scores[i], reference_scores[i], rtol=tolerances["scores"], atol=0.0)
    ]
    if misordered:
        problems.append(f"ranks {misordered} hold different genes with unequal scores")
    return problems, deviations


def neighbor_sets(graph: sparse.spmatrix) -> list[set[int]]:
    """Neighbour indices per row of a kNN graph, dropping self loops."""
    csr = sparse.csr_matrix(graph)
    return [
        set(csr.indices[csr.indptr[i] : csr.indptr[i + 1]].tolist()) - {i}
        for i in range(csr.shape[0])
    ]


def per_row_overlap(a: Sequence[set[int]], b: Sequence[set[int]]) -> NDArray[np.float64]:
    """`set_overlap` of each pair of rows, so callers can report the distribution."""
    return np.array([set_overlap(x, y) for x, y in zip(a, b, strict=True)], dtype=np.float64)


def neighborhood_preservation(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    k_ref: int = 15,
    k_cand: int = 30,
) -> float:
    """Mean fraction of each point's `k_ref` neighbours in `reference` that are still
    among its `k_cand` neighbours in `candidate`.

    The comparison embeddings are only defined up to rotation, reflection and scale, so
    local structure is all that can be asserted.
    """
    ref = knn_indices(reference, k_ref)
    cand = knn_indices(candidate, k_cand)
    kept = [len(set(r.tolist()) & set(c.tolist())) / k_ref for r, c in zip(ref, cand, strict=True)]
    return float(np.mean(kept))


def preservation_band(
    reference: NDArray[np.floating],
    reseeded: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    k_ref: int = 15,
    k_cand: int = 30,
) -> tuple[float, float]:
    """`(ours, ceiling)` preservation against `reference`.

    `reseeded` is the reference implementation run again with a different random seed.
    Its preservation is the ceiling: no implementation can agree with a stochastic layout
    better than that layout agrees with itself, and on real data the ceiling is far below
    1.0. Comparing against it is what keeps the threshold honest.
    """
    ours = neighborhood_preservation(reference, candidate, k_ref=k_ref, k_cand=k_cand)
    ceiling = neighborhood_preservation(reference, reseeded, k_ref=k_ref, k_cand=k_cand)
    return ours, ceiling
