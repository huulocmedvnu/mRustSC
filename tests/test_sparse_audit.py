"""Audit of `scrust_core::sparse::CsrMatrix`, the type every other module is handed.

There is no scanpy equivalent, but there is a precise reference: `scipy.sparse.csr_matrix`.
Every binding converts a scipy CSR into this type through
`crates/scrust-py/src/convert.rs::csr_from_py` -> `CsrMatrix::new`, and the two functions
that return a CSR triple (`_scrust.log1p`, `_scrust.normalize_total`) hand the same three
arrays back, so construction, validation and export are all reachable end to end.

`log1p` is the sharpest probe available: `crates/scrust-core/src/preprocess/normalize.rs`
copies `indptr` and `indices` through untouched and only maps the values, so anything the
round trip changes about the sparsity pattern was changed by `CsrMatrix`, not by the
algorithm. `normalize_total` with a target equal to the row total gives an exact identity
(factor `1.0`), which pins the values bit for bit.

What carries the most risk here:

* `test_explicit_zeros_survive_*`, because several modules in this crate read stored zeros
  and several deliberately skip them; if the container dropped them the two groups would
  silently disagree and nothing would say so.
* the three `..._is_accepted_...` tests at the bottom, because `CsrMatrix::new` performs no
  monotonicity check on `indptr` at all -- they pin a real defect, not intended behaviour.
* `test_duplicate_columns_are_kept_and_densify_takes_the_last`, because scipy sums
  duplicates and `densify_rows` overwrites, so the same triple means two different
  matrices on the two sides of the boundary.

These tests call `scrust._scrust` directly instead of going through `tests/scrust_call.py`.
That helper skips on `PanicException`, and three of the tests below exist precisely to show
that a `PanicException` is what an invalid `indptr` produces; routing them through the
helper would turn every one of those failures into a green skip.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

_scrust = pytest.importorskip("scrust._scrust", reason="the scrust extension is not built")

DEVICE = "cpu"


def u32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint32)


def f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def log1p(indptr, indices, values, n_cols):
    """Round trip a raw CSR triple through `CsrMatrix` and back.

    Nothing is caught: a `PanicException` or a `ValueError` from the core is the
    observation the caller is making.
    """
    return _scrust.log1p(u32(indptr), u32(indices), f32(values), n_cols)


def normalize_total(indptr, indices, values, n_cols, target_sum):
    return _scrust.normalize_total(
        u32(indptr), u32(indices), f32(values), n_cols, target_sum, DEVICE
    )


def scipy_csr(indptr, indices, values, shape) -> sparse.csr_matrix:
    """The same triple as scipy reads it, with no canonicalisation applied."""
    return sparse.csr_matrix((f32(values), u32(indices), u32(indptr)), shape=shape)


# --------------------------------------------------------------------------------------
# What `CsrMatrix::new` rejects.
# --------------------------------------------------------------------------------------


def test_empty_indptr_is_rejected():
    """An empty `indptr` has no row count to speak of -- `n_rows()` is `len - 1`, which
    would underflow -- so it must be refused at construction rather than produce a
    matrix whose row count is `usize::MAX`."""
    with pytest.raises(ValueError, match="indptr"):
        log1p([], [], [], 4)


def test_indices_and_values_of_different_lengths_are_rejected():
    """`indices` and `values` are read as parallel arrays by every consumer
    (`zip(indices, values)` in the column reductions, for one). A length mismatch would
    silently truncate to the shorter of the two."""
    with pytest.raises(ValueError):
        log1p([0, 2], [0, 1], [1.0], 4)
    with pytest.raises(ValueError):
        log1p([0, 1], [0], [1.0, 2.0], 4)


def test_indptr_last_entry_must_equal_the_stored_count():
    """`indptr[-1]` is the nnz by definition; if it disagrees with `len(values)` the last
    row is either truncated or reads past the end of the buffer."""
    with pytest.raises(ValueError, match="stored entries"):
        log1p([0, 2], [0], [1.0], 4)


def test_a_column_index_at_or_past_n_cols_is_rejected():
    """`densify_rows` writes `dense[offset + indices[e]]`, so an out-of-range column index
    is an out-of-bounds write into the next row's slice, i.e. silent corruption rather
    than a crash. The boundary matters in both directions: `n_cols - 1` must be legal."""
    with pytest.raises(ValueError, match="column indices below 4"):
        log1p([0, 1], [4], [1.0], 4)
    with pytest.raises(ValueError):
        log1p([0, 1], [99], [1.0], 4)

    indptr, indices, values, n_cols = log1p([0, 1], [3], [1.0], 4)
    assert_array_equal(indptr, [0, 1])
    assert_array_equal(indices, [3])
    assert n_cols == 4
    assert values == pytest.approx([np.log1p(1.0)])


# --------------------------------------------------------------------------------------
# Explicitly stored zeros. The theme of this audit.
# --------------------------------------------------------------------------------------


def test_explicit_zeros_survive_a_log1p_round_trip():
    """An explicitly stored zero must come back stored.

    scipy keeps stored zeros (`nnz` counts them; only `eliminate_zeros()` drops them), and
    modules in this crate depend on the distinction in both directions -- the QC path skips
    stored zeros by hand because scanpy eliminates them first, while the filters count
    positive entries. If `CsrMatrix` quietly compacted them, one group would change meaning
    and the other would not, with no error anywhere.

    `ln(1 + 0) == 0`, so the correct answer is the identical pattern with the zeros still
    in it, at the identical positions.
    """
    data = [0.0, 5.0, 0.0, 7.0, 0.0]
    indices = [0, 1, 2, 0, 3]
    indptr = [0, 3, 5]
    reference = scipy_csr(indptr, indices, data, (2, 4))
    assert reference.nnz == 5, "scipy itself must be storing the zeros for this to test"

    out_indptr, out_indices, out_values, out_n_cols = log1p(indptr, indices, data, 4)

    assert_array_equal(out_indptr, reference.indptr)
    assert_array_equal(out_indices, reference.indices)
    assert out_n_cols == 4
    assert len(out_values) == 5
    assert out_values == pytest.approx(np.log1p(reference.data))
    # Stated separately, because "same length" would also hold if the zeros had been
    # replaced by something non-zero at the same offsets.
    assert_array_equal(np.flatnonzero(out_values == 0.0), [0, 2, 4])


def test_explicit_zeros_survive_normalize_total_and_do_not_change_the_totals():
    """A stored zero contributes nothing to a row total and stays a stored zero after the
    row is rescaled -- rescaling touches the value array in place, so a container that
    dropped zeros would shift every later value onto the wrong column.

    Row 0 stores `[0, 4, 0, 6]`: total 10, target 20, so the factor is exactly 2 and the
    two zeros must still be zero. If the zeros were counted as entries but the factor came
    from `nnz` rather than the sum, or if they were dropped, the surviving values would
    not be `[8, 12]` at columns 1 and 3.
    """
    indptr, indices, values = [0, 4], [0, 1, 2, 3], [0.0, 4.0, 0.0, 6.0]

    out_indptr, out_indices, out_values, _ = normalize_total(indptr, indices, values, 4, 20.0)

    assert_array_equal(out_indptr, [0, 4])
    assert_array_equal(out_indices, [0, 1, 2, 3])
    assert out_values == pytest.approx([0.0, 8.0, 0.0, 12.0])


def test_a_row_made_entirely_of_stored_zeros_keeps_its_entries():
    """The degenerate case of the above: a row whose every stored value is zero is still a
    row with entries, not an empty row. `normalize_total` must leave it alone (its total is
    zero, so there is no factor) without compacting it away."""
    indptr, indices, values = [0, 3, 4], [0, 1, 2, 0], [0.0, 0.0, 0.0, 5.0]

    out_indptr, out_indices, out_values, _ = normalize_total(indptr, indices, values, 4, 5.0)

    assert_array_equal(out_indptr, [0, 3, 4])
    assert_array_equal(out_indices, [0, 1, 2, 0])
    assert out_values == pytest.approx([0.0, 0.0, 0.0, 5.0])


def test_stored_zeros_reach_the_dense_form_as_zeros():
    """Reaching `densify_rows` through `_scrust.scale`: the dense matrix a stored zero
    produces has to be indistinguishable from the one an unstored zero produces, since
    `dense` starts as a buffer of zeros and only stored entries are written.

    Two triples describing the same mathematical matrix -- one with the zeros stored, one
    without -- must scale to the same array. The gene means and deviations are computed by
    iterating `indices`/`values`, so a stored zero contributes `0` to the sum either way;
    if it did not, the two calls would differ.
    """
    with_zeros = _scrust.scale(
        u32([0, 3, 6]),
        u32([0, 1, 2, 0, 1, 2]),
        f32([1.0, 0.0, 3.0, 5.0, 0.0, 0.0]),
        3,
        True,
        None,
        DEVICE,
    )
    without_zeros = _scrust.scale(
        u32([0, 2, 3]), u32([0, 2, 0]), f32([1.0, 3.0, 5.0]), 3, True, None, DEVICE
    )
    assert_array_equal(with_zeros, without_zeros)
    # And the dense form is the one scipy produces from the same triple.
    reference = scipy_csr([0, 3, 6], [0, 1, 2, 0, 1, 2], [1.0, 0.0, 3.0, 5.0, 0.0, 0.0], (2, 3))
    dense = reference.toarray()
    assert_array_equal(dense, [[1.0, 0.0, 3.0], [5.0, 0.0, 0.0]])


# --------------------------------------------------------------------------------------
# Ordering and duplicates inside a row.
# --------------------------------------------------------------------------------------


def test_unsorted_column_indices_within_a_row_are_accepted_and_left_unsorted():
    """`CsrMatrix::new` does not require sorted column indices and does not sort them, and
    no consumer in the crate assumes sortedness (`densify_rows` scatters, the reductions
    zip). Pinned because sorting on the way in would silently reorder `values` relative to
    a caller who kept its own copy of the triple, and because scipy's `sorted_indices` is
    a *different* matrix object -- the values move with the indices.

    The pairing is what matters: column 2 must still carry `ln(1+1)`, not `ln(1+3)`.
    """
    indptr, indices, values = [0, 3], [2, 0, 1], [1.0, 2.0, 3.0]

    out_indptr, out_indices, out_values, _ = log1p(indptr, indices, values, 4)

    assert_array_equal(out_indptr, [0, 3])
    assert_array_equal(out_indices, [2, 0, 1], err_msg="indices were reordered")
    assert out_values == pytest.approx(np.log1p([1.0, 2.0, 3.0]))
    # The same triple in scipy denses to the same row, which is the invariant that would
    # break if indices and values were sorted independently of one another.
    assert_array_equal(scipy_csr(indptr, indices, values, (1, 4)).toarray(), [[2.0, 3.0, 1.0, 0.0]])


def test_duplicate_columns_are_kept_and_densify_takes_the_last():
    """DEFECT (divergence from the reference).

    scipy's `csr_matrix` allows duplicate column indices in a row and *sums* them when it
    materialises the matrix (`toarray`, `sum_duplicates`). `CsrMatrix::new` accepts them
    with no check, and `densify_rows` writes `dense[col] = value` in stored order, so the
    last duplicate wins and the earlier ones vanish. The same three arrays therefore denote
    two different matrices on the two sides of the boundary, with nothing raised.

    Reached through `_scrust.scale(zero_center=False)`, whose output is
    `densify_rows(...) / deviation`. The per-gene deviation is computed by iterating
    `indices`/`values` and so is *identical* under either reading of the duplicates; only
    the numerator differs. The ratio of the two rows in gene 0 is therefore a clean
    discriminator that does not depend on the scaling constant:

      row 0 stores gene 0 twice, as 3 then 4; row 1 stores it once, as 7.
      last-wins   -> 4/7; scipy's summing -> 7/7 == 1.
    """
    indptr, indices, values = [0, 2, 3, 3, 3], [0, 0, 0], [3.0, 4.0, 7.0]

    scaled = _scrust.scale(u32(indptr), u32(indices), f32(values), 2, False, None, DEVICE)

    assert scaled.shape == (4, 2)
    assert scaled[1, 0] > 0.0, "the discriminator is a ratio; the denominator must be real"
    ratio = float(scaled[0, 0] / scaled[1, 0])
    assert ratio == pytest.approx(4.0 / 7.0, rel=1e-5), (
        f"expected the last duplicate to win (ratio 4/7), got {ratio}"
    )
    assert ratio != pytest.approx(1.0, rel=1e-5), "would mean duplicates are summed"

    # scipy, on the identical triple, sums them: both rows read 7.0 in gene 0.
    assert_array_equal(scipy_csr(indptr, indices, values, (4, 2)).toarray()[:2, 0], [7.0, 7.0])

    # The triple itself is handed back verbatim -- neither summed nor deduplicated.
    out_indptr, out_indices, out_values, _ = log1p(indptr, indices, values, 2)
    assert_array_equal(out_indptr, indptr)
    assert_array_equal(out_indices, indices)
    assert out_values == pytest.approx(np.log1p(values))


# --------------------------------------------------------------------------------------
# Degenerate shapes.
# --------------------------------------------------------------------------------------


def test_an_all_empty_rows_matrix_keeps_its_rows():
    """`n_rows()` is `indptr.len() - 1`, so a matrix of nothing but empty rows is the case
    where the row count comes from `indptr` alone and from nothing else. If it were derived
    from the entries instead, three empty rows would collapse to zero rows and every
    per-cell result downstream would be the wrong length.

    Checked twice: the triple round trips unchanged, and `filter_cells` -- which allocates
    its output from `n_rows()` -- returns exactly three flags.
    """
    out_indptr, out_indices, out_values, out_n_cols = log1p([0, 0, 0, 0], [], [], 4)
    assert_array_equal(out_indptr, [0, 0, 0, 0])
    assert len(out_indices) == 0
    assert len(out_values) == 0
    assert out_n_cols == 4

    keep = _scrust.filter_cells(u32([0, 0, 0, 0]), u32([]), f32([]), 4, 0, None)
    assert keep.shape == (3,), f"n_rows() should be len(indptr) - 1 == 3, got {keep.shape}"
    assert_array_equal(scipy_csr([0, 0, 0, 0], [], [], (3, 4)).toarray(), np.zeros((3, 4)))


def test_a_matrix_with_zero_rows_is_accepted():
    """`indptr == [0]` is the smallest legal CSR: zero rows, zero entries. It has to be
    accepted (an empty gene-subset or an empty cell-subset produces exactly this) and it
    has to come back as zero rows, not as one row."""
    out_indptr, out_indices, out_values, out_n_cols = log1p([0], [], [], 4)
    assert_array_equal(out_indptr, [0])
    assert len(out_indices) == 0
    assert len(out_values) == 0
    assert out_n_cols == 4
    assert scipy_csr([0], [], [], (0, 4)).shape == (0, 4)


def test_a_matrix_with_zero_columns_is_accepted():
    """`n_cols == 0` makes the out-of-range check `column >= 0` vacuously true for every
    index, so the only way to build one is with no entries at all -- which is what a matrix
    with every gene filtered out is. It must survive rather than be rejected, and it must
    keep its `n_cols` of 0 through the round trip.

    The dense form is checked too, because `densify_rows` allocates `n_rows * n_cols` and
    an off-by-one there would give a non-empty buffer for an empty matrix.
    """
    out_indptr, out_indices, out_values, out_n_cols = log1p([0, 0, 0], [], [], 0)
    assert_array_equal(out_indptr, [0, 0, 0])
    assert len(out_indices) == 0
    assert len(out_values) == 0
    assert out_n_cols == 0

    dense = _scrust.scale(u32([0, 0, 0]), u32([]), f32([]), 0, True, None, DEVICE)
    assert dense.shape == (2, 0)

    # And any entry at all is out of range, since there is no legal column.
    with pytest.raises(ValueError, match="column indices below 0"):
        log1p([0, 1], [0], [1.0], 0)


def test_trailing_all_zero_columns_are_preserved():
    """`n_cols` is carried, not inferred from `max(indices) + 1`. A matrix whose last genes
    are all zero must keep its width, or every per-gene array downstream is short."""
    out_indptr, out_indices, out_values, out_n_cols = log1p([0, 2], [0, 1], [1.0, 2.0], 50)
    assert_array_equal(out_indptr, [0, 2])
    assert out_n_cols == 50
    assert_array_equal(out_indices, [0, 1])
    assert len(out_values) == 2

    dense = _scrust.scale(
        u32([0, 2, 3]), u32([0, 1, 0]), f32([1.0, 2.0, 4.0]), 50, True, None, DEVICE
    )
    assert dense.shape == (2, 50)
    assert_array_equal(dense[:, 2:], np.zeros((2, 48), dtype=np.float32))


def test_the_triple_round_trips_bit_for_bit_under_a_unit_rescale():
    """The whole export path in one assertion.

    `normalize_total` with `target_sum` equal to each row's total gives a factor of exactly
    `1.0`, so the values must come back bitwise identical -- through `CsrMatrix::new`, the
    value buffer, and `csr_to_py`. The matrix deliberately combines every awkward feature at
    once: a stored zero, an unsorted row, a duplicate column, an empty row, and a trailing
    zero column. Dtypes are asserted too, since the bindings promise u32/u32/f32 and a
    silent widening would break `csr_args`-style callers.
    """
    indptr = [0, 3, 3, 6]
    indices = [2, 0, 1, 1, 1, 0]
    values = [4.0, 0.0, 6.0, 2.0, 3.0, 5.0]  # row totals 10 and 10
    out_indptr, out_indices, out_values, out_n_cols = normalize_total(
        indptr, indices, values, 5, 10.0
    )

    assert out_indptr.dtype == np.uint32
    assert out_indices.dtype == np.uint32
    assert out_values.dtype == np.float32
    assert out_n_cols == 5
    assert_array_equal(out_indptr, indptr)
    assert_array_equal(out_indices, indices)
    assert_array_equal(out_values, f32(values))  # exact, not approximate


# --------------------------------------------------------------------------------------
# DEFECT: `CsrMatrix::new` never checks that `indptr` is monotone or in range.
# --------------------------------------------------------------------------------------


def test_a_non_monotone_indptr_is_accepted_and_panics_downstream():
    """DEFECT.

    `CsrMatrix::new` checks `indptr` is non-empty and that its *last* entry equals the
    stored count, and nothing else. A decreasing step such as `[0, 2, 1, 3]` passes, and the
    first consumer to slice `values[indptr[r]..indptr[r + 1]]` (here
    `preprocess/filter.rs:14`) panics with "slice index starts at 2 but ends at 1".

    A panic is not an error the bindings can convert: it crosses into Python as
    `pyo3_runtime.PanicException`, which derives from `BaseException`, so an `except
    Exception` around a scrust call does not catch it and the process is left with a
    poisoned thread. `scipy.sparse.csr_matrix.check_format()` reports this as a clean
    ValueError. This test pins the current behaviour so the fix is visible when it lands.
    """
    log1p([0, 2, 1, 3], [0, 1, 2], [1.0, 2.0, 3.0], 4)  # construction alone does not complain

    with pytest.raises(BaseException) as excinfo:  # PanicException is not an Exception subclass
        normalize_total([0, 2, 1, 3], [0, 1, 2], [1.0, 2.0, 3.0], 4, 10.0)
    assert type(excinfo.value).__name__ == "PanicException", (
        f"expected the unchecked indptr to panic, got {excinfo.value!r}"
    )
    assert "slice index starts at 2 but ends at 1" in str(excinfo.value)


def test_an_interior_indptr_entry_past_the_stored_count_is_accepted_and_panics():
    """DEFECT, same root cause.

    Only `indptr[-1]` is compared against `len(values)`, so `[0, 5, 3]` with three stored
    entries passes: row 0 claims entries `0..5` of a three-element buffer. The slice then
    panics with "range end index 5 out of range for slice of length 3".

    Worth pinning separately from the non-monotone case because it is the one an off-by-one
    in a caller's own CSR construction produces, and because the two panics come from
    different bounds checks.
    """
    log1p([0, 5, 3], [0, 1, 2], [1.0, 2.0, 3.0], 4)

    with pytest.raises(BaseException) as excinfo:
        normalize_total([0, 5, 3], [0, 1, 2], [1.0, 2.0, 3.0], 4, 10.0)
    assert type(excinfo.value).__name__ == "PanicException"
    assert "range end index 5 out of range" in str(excinfo.value)


def test_an_indptr_not_starting_at_zero_is_accepted_and_silently_drops_entries():
    """DEFECT, and the worst of the three: this one does not even panic.

    CSR requires `indptr[0] == 0`; scipy's `check_format` rejects anything else. `CsrMatrix`
    does not look at `indptr[0]`, so `[1, 3]` over three stored entries builds a one-row
    matrix whose row spans `values[1..3]` -- entry 0 belongs to no row at all.

    The result is that the row-oriented readers and the column-oriented readers of the very
    same matrix disagree. `row_reductions` (used by `normalize_total` and `filter_cells`)
    walks `indptr` and never sees `values[0]`; `column_reductions` and the `scale` moments
    zip `indices` with `values` and do see it. Nothing is raised either way.

    Demonstrated on `[5.0, 1.0, 1.0]` with `indptr == [1, 3]` and `target_sum == 6`:

      * as `CsrMatrix` reads it, the row total is `1 + 1 == 2`, factor `3`, giving
        `[5.0, 3.0, 3.0]` -- the orphaned `5.0` untouched, so it is *both* excluded from the
        total and still exported as part of the matrix;
      * as scipy reads the same triple, `check_format` raises; as a correct CSR would read
        it (`indptr == [0, 3]`), the total is `7`, the factor `6/7`, and every value moves.
    """
    out_indptr, out_indices, out_values, _ = normalize_total(
        [1, 3], [0, 1, 2], [5.0, 1.0, 1.0], 4, 6.0
    )

    assert_array_equal(out_indptr, [1, 3])
    assert_array_equal(out_indices, [0, 1, 2])
    assert out_values == pytest.approx([5.0, 3.0, 3.0]), (
        "the leading entry should have been part of the row total"
    )
    # The correct reading of the same data, for contrast: no value survives unscaled.
    correct = normalize_total([0, 3], [0, 1, 2], [5.0, 1.0, 1.0], 4, 6.0)[2]
    assert correct == pytest.approx([30.0 / 7.0, 6.0 / 7.0, 6.0 / 7.0])

    # scipy refuses the same triple outright.
    with pytest.raises(ValueError):
        scipy_csr([1, 3], [0, 1, 2], [5.0, 1.0, 1.0], (1, 4)).check_format(full_check=True)
