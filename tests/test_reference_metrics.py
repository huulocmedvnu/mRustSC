"""Unit tests for the metrics themselves, against cases computed by hand.

These touch no scrust code, so they run today and keep the rest of the suite honest.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from reference_metrics import (
    as_dense,
    component_correlations,
    de_comparison,
    knn_indices,
    neighbor_sets,
    neighborhood_preservation,
    per_row_overlap,
    preservation_band,
    set_overlap,
)

# Four points on a line at 0, 1, 2, 3.
LINE = np.array([[0.0], [1.0], [2.0], [3.0]])


def test_as_dense_handles_sparse_and_dense() -> None:
    dense = np.array([[1, 0], [0, 2]], dtype=np.float32)
    assert_array_equal(as_dense(sparse.csr_matrix(dense)), dense)
    assert as_dense(dense).dtype == np.float64


def test_knn_indices_on_a_line() -> None:
    # 1 and 2 are each equidistant from two neighbours; the stable sort takes the lower
    # index, so the expected answer is fully determined.
    assert_array_equal(knn_indices(LINE, 1), [[1], [0], [1], [2]])
    assert_array_equal(knn_indices(LINE, 2), [[1, 2], [0, 2], [1, 3], [2, 1]])


def test_knn_indices_rejects_impossible_k() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        knn_indices(LINE, 4)


def test_set_overlap_hand_cases() -> None:
    assert set_overlap("abc", "bcd") == pytest.approx(2 / 3)
    assert set_overlap([], []) == 1.0
    assert set_overlap("ab", "cd") == 0.0
    assert set_overlap("a", "ab") == 0.5  # normalised by the larger set
    assert set_overlap("aab", "ab") == 1.0  # duplicates do not count twice


def test_component_correlations_ignores_sign_and_scale() -> None:
    a = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0]])
    assert_allclose(component_correlations(a, np.column_stack([-a[:, 0], 2 * a[:, 1]])), [1.0, 1.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        component_correlations(a, a[:, :1])


def _de_result(names: list[str], scores: list[float], **fields: list[float]) -> dict:
    """A `uns["rank_genes_groups"]`-shaped result for one group named "g"."""
    result = {"names": {"g": np.array(names)}, "scores": {"g": np.array(scores)}}
    result.update({name: {"g": np.array(values)} for name, values in fields.items()})
    return result


DE_REFERENCE = _de_result(["a", "b", "c", "d"], [3.0, 2.0, 2.0, 1.0], pvals=[1e-9, 1e-5, 1e-5, 0.5])


def test_de_comparison_forgives_tie_order() -> None:
    # b and c swap, and their scores are equal: not a ranking difference. The p-values
    # must still be joined by gene, which is what a positional compare would get wrong.
    swapped = _de_result(["a", "c", "b", "d"], [3.0, 2.0, 2.0, 1.0], pvals=[1e-9, 1e-5, 1e-5, 0.5])
    problems, deviations = de_comparison(swapped, DE_REFERENCE, "g", top=4)
    assert problems == []
    assert deviations == {"scores": 0.0, "pvals": 0.0}


def test_de_comparison_catches_a_real_reordering() -> None:
    reordered = _de_result(
        ["a", "d", "b", "c"], [3.0, 1.0, 2.0, 2.0], pvals=[1e-9, 0.5, 1e-5, 1e-5]
    )
    problems, _ = de_comparison(reordered, DE_REFERENCE, "g", top=4)
    assert any("unequal scores" in p for p in problems), problems


def test_de_comparison_catches_set_and_value_differences() -> None:
    other_gene = _de_result(["a", "b", "z"], [3.0, 2.0, 2.0], pvals=[1e-9, 1e-5, 1e-5])
    problems, _ = de_comparison(other_gene, DE_REFERENCE, "g", top=3)
    assert any("gene set differs" in p for p in problems), problems

    # A p-value that underflowed to zero is an infinite relative error, never a small one.
    underflowed = _de_result(
        ["a", "b", "c", "d"], [3.0, 2.0, 2.0, 1.0], pvals=[0.0, 1e-5, 1e-5, 0.5]
    )
    problems, deviations = de_comparison(underflowed, DE_REFERENCE, "g", top=4)
    assert any("pvals differs" in p for p in problems), problems
    assert deviations["pvals"] == 1.0


def test_neighbor_sets_drops_self_loops_and_keeps_row_order() -> None:
    graph = sparse.csr_matrix(
        np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]),
    )
    assert neighbor_sets(graph) == [{1}, {2}, {0}]


def test_per_row_overlap() -> None:
    assert_allclose(per_row_overlap([{1, 2}, {3}], [{1, 2}, {4}]), [1.0, 0.0])


def test_neighborhood_preservation_is_one_for_a_similarity_transform() -> None:
    # Reflected, scaled and shifted: neighbourhoods are unchanged.
    assert neighborhood_preservation(LINE, -3.0 * LINE + 7.0, k_ref=1, k_cand=1) == 1.0


def test_neighborhood_preservation_hand_computed() -> None:
    # Reference order 0,1,2,3; candidate pairs (0,2) and (1,3) instead.
    candidate = np.array([[0.0], [10.0], [1.0], [11.0]])
    assert neighborhood_preservation(LINE, candidate, k_ref=1, k_cand=1) == 0.0
    assert neighborhood_preservation(LINE, candidate, k_ref=1, k_cand=2) == pytest.approx(0.75)


def test_preservation_band_reports_ours_and_the_ceiling() -> None:
    candidate = np.array([[0.0], [10.0], [1.0], [11.0]])
    ours, ceiling = preservation_band(LINE, LINE, candidate, k_ref=1, k_cand=2)
    assert (ours, ceiling) == (pytest.approx(0.75), 1.0)
