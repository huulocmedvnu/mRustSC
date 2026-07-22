//! Sparse kernels on the GPU. Owned by feat/sparse-gpu.
//!
//! Densifying a 95%-zero matrix to reach the GPU wastes both the bandwidth and
//! the memory that make the GPU worth using. These kernels consume CSR directly,
//! which is what lets a run scale past what a dense block would allow.
//!
//! # Shape of every kernel here
//! One threadgroup owns one row of stored entries. Its lanes stride that row,
//! each holding a private accumulator, and a pairwise tree in threadgroup memory
//! collapses them to the row's result. Nothing is written twice, so every
//! function in this module is bit-for-bit reproducible.
//!
//! # Accumulation in f32
//! The Apple GPU has no `f64`, so a long sum has to be shortened instead of
//! widened. Striding gives each lane only `row_length / width` terms, and the
//! tree that merges the lanes adds `log2(width)` more roundings rather than
//! `width`. A 50 000-entry column reduced by 1024 lanes therefore has a serial
//! chain of 49 + 10 additions instead of 50 000. Measured against an f64
//! reference over the test shapes: 1.0e-7 for [`spmm`], 8.9e-8 for
//! [`spmm_transposed`], and for [`column_moments`] 8.6e-8 on the sum of squares
//! and 7.3e-6 on the plain sum. That last figure is cancellation in signed test
//! data, not accumulation length — the sum it is relative to is much smaller
//! than the terms in it, which never happens for the counts real callers pass.

use std::ffi::c_void;

use metal::{ComputePipelineState, MTLCommandBufferStatus, MTLSize, NSUInteger};
use ndarray::Array2;
use scrust_core::error::{Error, Result};
use scrust_core::sparse::CsrMatrix;

use crate::context::MetalContext;

const SPMM_FUNCTION: &str = "csr_spmm";
const COLUMN_MOMENTS_FUNCTION: &str = "csr_column_moments";
const SCALE_ROWS_FUNCTION: &str = "csr_scale_rows";

/// The narrowest threadgroup worth launching. A group below one SIMD group
/// cannot fill an execution unit, so the threadgroup memory budget is spent on
/// `k` only down to this width; past it we reject `k` instead.
const MIN_THREADS: usize = 32;

/// The largest `k` [`spmm`] and [`spmm_transposed`] can serve on `context`.
///
/// Every lane of a row's threadgroup owns a private length-`k` accumulator in
/// threadgroup memory, so a group costs `width * k * 4` bytes. Holding the group
/// at [`MIN_THREADS`] turns that into a ceiling on `k`: the 32 KiB of
/// threadgroup memory on Apple silicon gives `k <= 256`. Callers ask for far
/// less — `k` is a component count or an embedding width, 2 to 50 in practice —
/// so the limit exists to fail loudly rather than to constrain anything real.
pub fn max_dense_columns(context: &MetalContext) -> usize {
    context.device().max_threadgroup_memory_length() as usize / (size_of::<f32>() * MIN_THREADS)
}

/// Sparse times dense: `(n_rows, n_cols) x (n_cols, k) -> (n_rows, k)`.
///
/// One threadgroup per row, striding its stored entries.
///
/// Peak device memory is `8 * nnz + 4 * (n_rows + 1)` for the matrix,
/// `4 * n_cols * k` for the dense operand and `4 * n_rows * k` for the result —
/// no dense `(n_rows, n_cols)` intermediate exists at any point.
pub fn spmm(
    context: &MetalContext,
    sparse: &CsrMatrix,
    dense: &Array2<f32>,
) -> Result<Array2<f32>> {
    if dense.nrows() != sparse.n_cols() {
        return Err(Error::shape(
            format!("a dense operand with {} rows", sparse.n_cols()),
            format!("{} rows", dense.nrows()),
        ));
    }
    let k = dense.ncols();
    if k == 0 || k > max_dense_columns(context) {
        return Err(Error::parameter(
            "dense columns",
            "at least 1 and within the threadgroup memory budget",
            k,
        ));
    }
    let n_rows = sparse.n_rows();
    if n_rows == 0 {
        return Ok(Array2::zeros((0, k)));
    }

    let contiguous = dense.as_standard_layout();
    let dense_values = contiguous
        .as_slice()
        .ok_or_else(|| Error::shape("a contiguous dense operand", "a strided view"))?;

    let pipeline = context.pipeline(SPMM_FUNCTION, SPMM_SOURCE)?;
    // The scratch is k floats per lane, and a row never needs more lanes than it
    // has stored entries.
    let width = threadgroup_width(context, &pipeline, k, longest_row(sparse));

    let indptr = context.buffer(sparse.indptr());
    let indices = context.buffer(non_empty(sparse.indices(), &[0]));
    let values = context.buffer(non_empty(sparse.values(), &[0.0]));
    let dense_buffer = context.buffer(non_empty(dense_values, &[0.0]));
    let out = context.empty_buffer::<f32>(n_rows * k);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&indptr), 0);
    encoder.set_buffer(1, Some(&indices), 0);
    encoder.set_buffer(2, Some(&values), 0);
    encoder.set_buffer(3, Some(&dense_buffer), 0);
    encoder.set_buffer(4, Some(&out), 0);
    set_scalar(encoder, 5, &(k as u32));
    encoder.set_threadgroup_memory_length(0, (width * k * size_of::<f32>()) as NSUInteger);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_rows as u64, 1, 1),
        MTLSize::new(width as u64, 1, 1),
    );
    encoder.end_encoding();
    run(command, SPMM_FUNCTION)?;

    // SAFETY: the buffer holds exactly n_rows * k floats and every threadgroup
    // writes its whole output row.
    let result = unsafe { MetalContext::read::<f32>(&out, n_rows * k) };
    Array2::from_shape_vec((n_rows, k), result)
        .map_err(|error| Error::shape(format!("({n_rows}, {k})"), error.to_string()))
}

/// Transposed sparse times dense, without materialising the transpose:
/// `(n_cols, n_rows) x (n_rows, k) -> (n_cols, k)`.
///
/// # The write conflict, and what is done about it
/// Every stored entry `(row, column, value)` contributes `value * dense[row]` to
/// output row `column`, so all the entries of one column of `A` — which live in
/// different CSR rows — target the same output row. Three ways out were on the
/// table:
///
/// - **Device atomics.** Cheapest to write and the fastest single pass, but
///   float atomics commit in arrival order, so two runs disagree in the last
///   bits. `docs/API_CONTRACT.md` allows that only for a documented race, and a
///   reduction that silently stops being reproducible is a bad trade for a
///   library whose GPU path is checked against a CPU oracle.
/// - **A per-threadgroup accumulator plus reduction.** Conflict-free only if
///   each accumulator owns the whole output, so it costs
///   `groups * n_cols * k * 4` bytes: at 20 000 genes, `k = 50` and 256 groups
///   that is 10 GB. Ruled out by the memory this branch exists to save.
/// - **A precomputed CSC-like index — chosen.** Counting-sort the stored entries
///   by column on the host, which is exactly `Aᵀ` in CSR, then run [`spmm`] on
///   it. Each output row is then owned by one threadgroup again, so there is no
///   conflict at all and the result is bit-for-bit reproducible.
///
/// **Cost of the choice:** one `O(nnz + n_cols)` host pass and a second copy of
/// the stored entries, `8 * nnz` bytes, live for the duration of the call. At
/// 60 M stored entries that is 480 MB and a measured 0.45 to 1.3 s of host time
/// against 160 ms of GPU time — the sort dominates. A caller multiplying by `Aᵀ`
/// repeatedly should therefore transpose once itself and call [`spmm`]; the
/// current signature cannot cache it for them.
pub fn spmm_transposed(
    context: &MetalContext,
    sparse: &CsrMatrix,
    dense: &Array2<f32>,
) -> Result<Array2<f32>> {
    if dense.nrows() != sparse.n_rows() {
        return Err(Error::shape(
            format!("a dense operand with {} rows", sparse.n_rows()),
            format!("{} rows", dense.nrows()),
        ));
    }
    spmm(context, &transposed(sparse)?, dense)
}

/// Per-column sum and sum of squares in one pass, the reduction every
/// normalisation and variance step needs.
///
/// Both moments run over the *stored* entries only. The implicit zeros
/// contribute nothing to either sum, but they do belong to the column: a caller
/// after a variance wants `sum2 / n_rows - (sum / n_rows)^2`, with `n_rows` — not
/// the stored count — as the denominator, and it already has `n_rows`.
///
/// Columns are the conflicting axis of a CSR matrix, so this shares
/// [`spmm_transposed`]'s answer to that and its cost: the entries are
/// counting-sorted by column first, `8 * nnz` bytes and one host pass, and each
/// column is then reduced by a threadgroup of its own.
pub fn column_moments(context: &MetalContext, sparse: &CsrMatrix) -> Result<(Vec<f32>, Vec<f32>)> {
    let n_cols = sparse.n_cols();
    if n_cols == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let by_column = transposed(sparse)?;

    let pipeline = context.pipeline(COLUMN_MOMENTS_FUNCTION, COLUMN_MOMENTS_SOURCE)?;
    // Two scratch floats per lane: the running sum and the running sum of
    // squares are reduced by the same tree.
    let width = threadgroup_width(context, &pipeline, 2, longest_row(&by_column));

    let indptr = context.buffer(by_column.indptr());
    let values = context.buffer(non_empty(by_column.values(), &[0.0]));
    let sums = context.empty_buffer::<f32>(n_cols);
    let squares = context.empty_buffer::<f32>(n_cols);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&indptr), 0);
    encoder.set_buffer(1, Some(&values), 0);
    encoder.set_buffer(2, Some(&sums), 0);
    encoder.set_buffer(3, Some(&squares), 0);
    encoder.set_threadgroup_memory_length(0, (2 * width * size_of::<f32>()) as NSUInteger);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_cols as u64, 1, 1),
        MTLSize::new(width as u64, 1, 1),
    );
    encoder.end_encoding();
    run(command, COLUMN_MOMENTS_FUNCTION)?;

    // SAFETY: both buffers hold n_cols floats and lane zero of every threadgroup
    // writes its column.
    let sums = unsafe { MetalContext::read::<f32>(&sums, n_cols) };
    let squares = unsafe { MetalContext::read::<f32>(&squares, n_cols) };
    Ok((sums, squares))
}

/// Scale each row by its factor, in place on the stored values.
///
/// Trivially parallel over stored entries: an implicit zero stays zero under any
/// factor, so only the `nnz` stored values move and the sparsity pattern is
/// untouched.
pub fn scale_rows(context: &MetalContext, sparse: &mut CsrMatrix, factors: &[f32]) -> Result<()> {
    if factors.len() != sparse.n_rows() {
        return Err(Error::shape(
            format!("{} row factors", sparse.n_rows()),
            format!("{} row factors", factors.len()),
        ));
    }
    let nnz = sparse.nnz();
    if nnz == 0 {
        return Ok(());
    }

    let pipeline = context.pipeline(SCALE_ROWS_FUNCTION, SCALE_ROWS_SOURCE)?;
    // No reduction here, so no scratch: the width only has to cover a row.
    let width = threadgroup_width(context, &pipeline, 0, longest_row(sparse));

    let indptr = context.buffer(sparse.indptr());
    let values = context.buffer(sparse.values());
    let factors_buffer = context.buffer(factors);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&indptr), 0);
    encoder.set_buffer(1, Some(&values), 0);
    encoder.set_buffer(2, Some(&factors_buffer), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(sparse.n_rows() as u64, 1, 1),
        MTLSize::new(width as u64, 1, 1),
    );
    encoder.end_encoding();
    run(command, SCALE_ROWS_FUNCTION)?;

    // SAFETY: the buffer was created from exactly `nnz` floats and the kernel
    // only overwrites them. Copying straight out of it avoids the intermediate
    // `Vec` a `read` would allocate.
    let scaled = unsafe { std::slice::from_raw_parts(values.contents() as *const f32, nnz) };
    sparse.values_mut().copy_from_slice(scaled);
    Ok(())
}

/// `Aᵀ` in CSR, which is `A` in CSC: a counting sort of the stored entries by
/// column, stable in the row index so the result never depends on scheduling.
fn transposed(sparse: &CsrMatrix) -> Result<CsrMatrix> {
    let n_rows = sparse.n_rows();
    let n_cols = sparse.n_cols();
    if u32::try_from(n_rows).is_err() {
        return Err(Error::parameter(
            "n_rows",
            "representable as a u32 column index",
            n_rows,
        ));
    }

    let mut indptr = vec![0u32; n_cols + 1];
    for &column in sparse.indices() {
        indptr[column as usize + 1] += 1;
    }
    for column in 0..n_cols {
        indptr[column + 1] += indptr[column];
    }

    let mut cursor = indptr.clone();
    let mut indices = vec![0u32; sparse.nnz()];
    let mut values = vec![0.0f32; sparse.nnz()];
    for row in 0..n_rows {
        let row_end = sparse.indptr()[row + 1] as usize;
        for entry in sparse.indptr()[row] as usize..row_end {
            let column = sparse.indices()[entry] as usize;
            let slot = cursor[column] as usize;
            indices[slot] = row as u32;
            values[slot] = sparse.values()[entry];
            cursor[column] += 1;
        }
    }
    CsrMatrix::new(indptr, indices, values, n_rows)
}

/// The most stored entries any single row holds, which is as wide as a
/// threadgroup ever needs to be.
fn longest_row(sparse: &CsrMatrix) -> usize {
    sparse
        .indptr()
        .windows(2)
        .map(|bounds| (bounds[1] - bounds[0]) as usize)
        .max()
        .unwrap_or(0)
}

/// The widest power-of-two threadgroup that the pipeline allows, that
/// `scratch_floats_per_lane` fits in threadgroup memory, and that the longest
/// row can actually keep busy.
///
/// A power of two is required because the tree reduction halves the active lane
/// count each step.
fn threadgroup_width(
    context: &MetalContext,
    pipeline: &ComputePipelineState,
    scratch_floats_per_lane: usize,
    work: usize,
) -> usize {
    widest_power_of_two(
        pipeline.max_total_threads_per_threadgroup() as usize,
        context.device().max_threadgroup_memory_length() as usize,
        scratch_floats_per_lane,
        work,
    )
}

/// The arithmetic behind [`threadgroup_width`], separated from the device so it
/// can be checked against limits this machine may not have.
fn widest_power_of_two(
    thread_limit: usize,
    threadgroup_memory: usize,
    scratch_floats_per_lane: usize,
    work: usize,
) -> usize {
    let by_pipeline = thread_limit.max(1);
    let by_memory = match scratch_floats_per_lane {
        0 => by_pipeline,
        floats => threadgroup_memory / (floats * size_of::<f32>()),
    };
    let allowed = by_pipeline.min(by_memory).max(1);
    let capacity = 1usize << (usize::BITS - 1 - allowed.leading_zeros());
    work.max(1).next_power_of_two().min(capacity)
}

/// Metal rejects a zero-length buffer, and an empty matrix is legal input.
fn non_empty<'a, T>(slice: &'a [T], fallback: &'a [T]) -> &'a [T] {
    if slice.is_empty() {
        fallback
    } else {
        slice
    }
}

fn set_scalar<T>(encoder: &metal::ComputeCommandEncoderRef, index: NSUInteger, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as NSUInteger,
        value as *const T as *const c_void,
    );
}

fn run(command: &metal::CommandBufferRef, name: &'static str) -> Result<()> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Error::Kernel {
            name,
            message: format!("dispatch ended in state {:?}", command.status()),
        });
    }
    Ok(())
}

const SPMM_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// C = A * B with A in CSR. One threadgroup owns one row of A: its lanes stride
// the row's stored entries, each accumulating a private length-k row of C in
// threadgroup memory, and a pairwise tree sums the lanes. A row with no stored
// entries writes zeros, which is the right answer for it.
//
// Only the k columns of B named by the row's column indices are ever read, so
// the traffic is proportional to the stored entries and not to n_cols * k.
kernel void csr_spmm(device const uint *indptr [[buffer(0)]],
                     device const uint *indices [[buffer(1)]],
                     device const float *values [[buffer(2)]],
                     device const float *dense [[buffer(3)]],
                     device float *out [[buffer(4)]],
                     constant uint &k [[buffer(5)]],
                     threadgroup float *scratch [[threadgroup(0)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_position_in_threadgroup]],
                     uint width [[threads_per_threadgroup]]) {
    threadgroup float *mine = scratch + lane * k;
    for (uint column = 0; column < k; ++column) {
        mine[column] = 0.0f;
    }

    const uint row_end = indptr[row + 1];
    for (uint entry = indptr[row] + lane; entry < row_end; entry += width) {
        const float value = values[entry];
        device const float *dense_row = dense + (ulong)indices[entry] * k;
        for (uint column = 0; column < k; ++column) {
            mine[column] = fma(value, dense_row[column], mine[column]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            threadgroup const float *other = scratch + (lane + stride) * k;
            for (uint column = 0; column < k; ++column) {
                mine[column] += other[column];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint column = lane; column < k; column += width) {
        out[(ulong)row * k + column] = scratch[column];
    }
}
"#;

const COLUMN_MOMENTS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Sum and sum of squares of one column, given the entries already grouped by
// column. One threadgroup per column, lanes striding its entries, one tree
// reducing both moments at once so the values are read exactly once. A column
// with no stored entries has both moments zero.
kernel void csr_column_moments(device const uint *indptr [[buffer(0)]],
                               device const float *values [[buffer(1)]],
                               device float *sums [[buffer(2)]],
                               device float *squares [[buffer(3)]],
                               threadgroup float *scratch [[threadgroup(0)]],
                               uint column [[threadgroup_position_in_grid]],
                               uint lane [[thread_position_in_threadgroup]],
                               uint width [[threads_per_threadgroup]]) {
    float sum = 0.0f;
    float square_sum = 0.0f;
    const uint column_end = indptr[column + 1];
    for (uint entry = indptr[column] + lane; entry < column_end; entry += width) {
        const float value = values[entry];
        sum += value;
        square_sum = fma(value, value, square_sum);
    }
    scratch[lane] = sum;
    scratch[width + lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
            scratch[width + lane] += scratch[width + lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        sums[column] = scratch[0];
        squares[column] = scratch[width];
    }
}
"#;

const SCALE_ROWS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// values[entry] *= factors[row]. One threadgroup per row so the row a stored
// entry belongs to is known without searching indptr for it.
kernel void csr_scale_rows(device const uint *indptr [[buffer(0)]],
                           device float *values [[buffer(1)]],
                           device const float *factors [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lane [[thread_position_in_threadgroup]],
                           uint width [[threads_per_threadgroup]]) {
    const float factor = factors[row];
    const uint row_end = indptr[row + 1];
    for (uint entry = indptr[row] + lane; entry < row_end; entry += width) {
        values[entry] *= factor;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic generator; `rand` is not a dependency of this crate and
    /// the tests need spread out values, not statistical quality.
    struct Xorshift(u64);

    impl Xorshift {
        fn next_unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / 16_777_216.0
        }

        fn next_below(&mut self, bound: usize) -> usize {
            (self.next_unit() as f64 * bound as f64) as usize % bound.max(1)
        }
    }

    fn dense_matrix(rows: usize, columns: usize, seed: u64) -> Array2<f32> {
        let mut random = Xorshift(seed);
        Array2::from_shape_fn((rows, columns), |_| random.next_unit() * 2.0 - 1.0)
    }

    /// A CSR matrix with `density` of its entries stored, plus one empty row and
    /// one empty column so those cases are always exercised.
    fn sparse_matrix(n_rows: usize, n_cols: usize, density: f64, seed: u64) -> CsrMatrix {
        let mut random = Xorshift(seed);
        let mut dense = vec![0.0f32; n_rows * n_cols];
        for row in 0..n_rows {
            if row == n_rows / 2 {
                continue; // an empty row
            }
            for column in 0..n_cols {
                if column == n_cols / 3 {
                    continue; // an empty column
                }
                if f64::from(random.next_unit()) < density {
                    dense[row * n_cols + column] = random.next_unit() * 4.0 - 2.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, n_rows, n_cols).unwrap()
    }

    /// f64 oracle: densify the CSR and multiply, optionally transposed.
    fn cpu_spmm(sparse: &CsrMatrix, dense: &Array2<f32>, transpose: bool) -> Array2<f64> {
        let (n_rows, n_cols) = (sparse.n_rows(), sparse.n_cols());
        let densified = sparse.densify_rows(0, n_rows);
        let k = dense.ncols();
        let out_rows = if transpose { n_cols } else { n_rows };
        let mut out = Array2::<f64>::zeros((out_rows, k));
        for row in 0..n_rows {
            for column in 0..n_cols {
                let value = f64::from(densified[row * n_cols + column]);
                if value == 0.0 {
                    continue;
                }
                let (out_row, dense_row) = if transpose {
                    (column, row)
                } else {
                    (row, column)
                };
                for slot in 0..k {
                    out[[out_row, slot]] += value * f64::from(dense[[dense_row, slot]]);
                }
            }
        }
        out
    }

    /// The largest relative deviation, taken against the largest entry of the
    /// reference so that entries passing through zero stay meaningful.
    fn worst_deviation(actual: &Array2<f32>, expected: &Array2<f64>) -> f64 {
        assert_eq!(actual.dim(), expected.dim());
        let scale = expected
            .iter()
            .fold(1e-12f64, |worst, v| worst.max(v.abs()));
        actual
            .iter()
            .zip(expected.iter())
            .fold(0.0f64, |worst, (&got, &want)| {
                worst.max((f64::from(got) - want).abs() / scale)
            })
    }

    #[test]
    fn spmm_matches_the_f64_reference_on_several_shapes() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        // 257 and 1025 rows are not multiples of any threadgroup width, k = 1 is
        // the narrowest legal dense operand, and every matrix carries an empty
        // row and an empty column.
        let shapes = [
            (1, 1, 1, 1.0),
            (257, 40, 1, 0.1),
            (64, 300, 7, 0.05),
            (1025, 33, 16, 0.2),
            (30, 30, 50, 0.5),
        ];
        let mut worst = 0.0f64;
        for (n_rows, n_cols, k, density) in shapes {
            let sparse = sparse_matrix(n_rows, n_cols, density, n_rows as u64 + 1);
            let dense = dense_matrix(n_cols, k, n_cols as u64 + 7);
            let actual = spmm(&context, &sparse, &dense).unwrap();
            let expected = cpu_spmm(&sparse, &dense, false);
            let deviation = worst_deviation(&actual, &expected);
            assert!(deviation < 1e-5, "{n_rows}x{n_cols}, k={k}: {deviation}");
            worst = worst.max(deviation);
        }
        println!("spmm worst relative deviation from the f64 reference: {worst:.3e}");
    }

    #[test]
    fn spmm_transposed_matches_the_f64_reference_on_several_shapes() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let shapes = [
            (1, 1, 1, 1.0),
            (257, 40, 1, 0.1),
            (64, 300, 7, 0.05),
            (1025, 33, 16, 0.2),
        ];
        let mut worst = 0.0f64;
        for (n_rows, n_cols, k, density) in shapes {
            let sparse = sparse_matrix(n_rows, n_cols, density, n_rows as u64 + 3);
            let dense = dense_matrix(n_rows, k, n_rows as u64 + 11);
            let actual = spmm_transposed(&context, &sparse, &dense).unwrap();
            let expected = cpu_spmm(&sparse, &dense, true);
            let deviation = worst_deviation(&actual, &expected);
            assert!(deviation < 1e-5, "{n_rows}x{n_cols}, k={k}: {deviation}");
            worst = worst.max(deviation);
        }
        println!("spmm_transposed worst relative deviation: {worst:.3e}");
    }

    #[test]
    fn a_matrix_of_only_empty_rows_gives_a_zero_product() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let sparse = CsrMatrix::new(vec![0; 5], Vec::new(), Vec::new(), 3).unwrap();
        let dense = dense_matrix(3, 4, 5);
        assert_eq!(
            spmm(&context, &sparse, &dense).unwrap(),
            Array2::<f32>::zeros((4, 4))
        );
        let dense = dense_matrix(4, 4, 6);
        assert_eq!(
            spmm_transposed(&context, &sparse, &dense).unwrap(),
            Array2::<f32>::zeros((3, 4))
        );
    }

    #[test]
    fn spmm_multiplies_a_hand_written_example() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // [[1, 0, 2], [0, 0, 0], [0, 3, 0]] times [[1, 0], [0, 1], [2, 2]].
        let sparse =
            CsrMatrix::from_dense(&[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0], 3, 3).unwrap();
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 2.0, 2.0]).unwrap();
        let expected = Array2::from_shape_vec((3, 2), vec![5.0, 4.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
        assert_eq!(spmm(&context, &sparse, &dense).unwrap(), expected);

        // The transpose is [[1, 0, 0], [0, 0, 3], [2, 0, 0]] against a 3 x 2.
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 5.0, 5.0, 0.0, 2.0]).unwrap();
        let expected = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 0.0, 6.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            spmm_transposed(&context, &sparse, &dense).unwrap(),
            expected
        );
    }

    #[test]
    fn column_moments_match_the_f64_reference() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // 1025 rows exercises a column longer than a threadgroup, and the
        // generator leaves column n_cols / 3 empty everywhere. The signed
        // matrices are the harder case on purpose: a column of real counts never
        // cancels, so its sum is never small compared with the terms in it.
        for (n_rows, n_cols, density) in [(1, 1, 1.0), (1025, 37, 0.3), (300, 900, 0.05)] {
            let sparse = sparse_matrix(n_rows, n_cols, density, n_cols as u64 + 13);
            let (sums, squares) = column_moments(&context, &sparse).unwrap();
            let densified = sparse.densify_rows(0, n_rows);

            let (mut worst_sum, mut worst_square) = (0.0f64, 0.0f64);
            for column in 0..n_cols {
                let (mut sum, mut square_sum) = (0.0f64, 0.0f64);
                for row in 0..n_rows {
                    let value = f64::from(densified[row * n_cols + column]);
                    sum += value;
                    square_sum += value * value;
                }
                let deviation = |got: f32, want: f64| (f64::from(got) - want).abs();
                worst_sum = worst_sum.max(deviation(sums[column], sum) / sum.abs().max(1e-12));
                worst_square = worst_square
                    .max(deviation(squares[column], square_sum) / square_sum.abs().max(1e-12));
            }
            assert!(worst_sum < 1e-5, "column sums deviate by {worst_sum}");
            assert!(
                worst_square < 1e-5,
                "column sums of squares deviate by {worst_square}"
            );
            // The empty column has no stored entry, so both moments are zero.
            assert_eq!(sums[n_cols / 3], 0.0);
            assert_eq!(squares[n_cols / 3], 0.0);
            println!(
                "column_moments at {n_rows}x{n_cols}: worst relative deviation \
                 {worst_sum:.3e} on the sum, {worst_square:.3e} on the sum of squares"
            );
        }
    }

    #[test]
    fn column_moments_of_a_hand_written_example() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // Column 1 is zero everywhere; column 2 mixes signs so the two moments
        // cannot be confused for one another.
        let sparse = CsrMatrix::from_dense(&[1.0, 0.0, 3.0, 2.0, 0.0, -3.0], 2, 3).unwrap();
        let (sums, squares) = column_moments(&context, &sparse).unwrap();
        assert_eq!(sums, vec![3.0, 0.0, 0.0]);
        assert_eq!(squares, vec![5.0, 0.0, 18.0]);
    }

    #[test]
    fn scale_rows_matches_an_explicit_expectation() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let mut sparse = CsrMatrix::from_dense(
            &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 6.0],
            3,
            4,
        )
        .unwrap();
        // Row 1 is scaled to zero; a zero factor must not drop stored entries,
        // because callers rely on the pattern surviving.
        scale_rows(&context, &mut sparse, &[2.0, 0.0, -1.0]).unwrap();
        assert_eq!(sparse.values(), &[2.0, 4.0, 0.0, 0.0, -5.0, -6.0]);
        assert_eq!(sparse.indptr(), &[0, 2, 4, 6]);
        assert_eq!(
            sparse.densify_rows(0, 3),
            vec![2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, -6.0]
        );
    }

    #[test]
    fn scale_rows_matches_a_host_reference_at_scale() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // 1025 rows: not a multiple of any threadgroup width, and one row is
        // empty so an empty stride range is exercised too.
        let mut sparse = sparse_matrix(1025, 60, 0.2, 17);
        let expected: Vec<f32> = (0..sparse.n_rows())
            .flat_map(|row| {
                let range = sparse.indptr()[row] as usize..sparse.indptr()[row + 1] as usize;
                sparse.values()[range]
                    .iter()
                    .map(move |value| value * row as f32)
                    .collect::<Vec<_>>()
            })
            .collect();
        let factors: Vec<f32> = (0..sparse.n_rows()).map(|row| row as f32).collect();
        scale_rows(&context, &mut sparse, &factors).unwrap();
        assert_eq!(sparse.values(), expected.as_slice());
    }

    #[test]
    fn repeated_runs_are_identical() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // No kernel here writes an output element twice, so there is no race to
        // tolerate: every result is bit-for-bit reproducible.
        let sparse = sparse_matrix(700, 90, 0.15, 29);
        let dense = dense_matrix(90, 12, 31);
        assert_eq!(
            spmm(&context, &sparse, &dense).unwrap(),
            spmm(&context, &sparse, &dense).unwrap()
        );
        let dense = dense_matrix(700, 12, 37);
        assert_eq!(
            spmm_transposed(&context, &sparse, &dense).unwrap(),
            spmm_transposed(&context, &sparse, &dense).unwrap()
        );
        assert_eq!(
            column_moments(&context, &sparse).unwrap(),
            column_moments(&context, &sparse).unwrap()
        );
    }

    #[test]
    fn rejects_input_it_cannot_serve() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let mut sparse = sparse_matrix(20, 10, 0.3, 41);
        // A dense operand whose rows do not match the contracted axis.
        assert!(matches!(
            spmm(&context, &sparse, &dense_matrix(9, 4, 1)).unwrap_err(),
            Error::Shape { .. }
        ));
        assert!(matches!(
            spmm_transposed(&context, &sparse, &dense_matrix(10, 4, 1)).unwrap_err(),
            Error::Shape { .. }
        ));
        // No columns at all, and more than the threadgroup memory allows.
        assert!(matches!(
            spmm(&context, &sparse, &dense_matrix(10, 0, 1)).unwrap_err(),
            Error::InvalidParameter { .. }
        ));
        let too_wide = dense_matrix(10, max_dense_columns(&context) + 1, 1);
        assert!(matches!(
            spmm(&context, &sparse, &too_wide).unwrap_err(),
            Error::InvalidParameter { .. }
        ));
        // One factor per row, no more and no fewer.
        assert!(matches!(
            scale_rows(&context, &mut sparse, &[1.0; 19]).unwrap_err(),
            Error::Shape { .. }
        ));
    }

    #[test]
    fn threadgroup_width_stays_within_every_limit() {
        // 1024 lanes allowed, 32 KiB of threadgroup memory: k = 8 leaves room
        // for 1024 lanes, k = 50 for 163 and so 128 after rounding down, and a
        // row of 40 entries never needs more than 64 lanes.
        let width = |floats, work| widest_power_of_two(1024, 32768, floats, work);
        assert_eq!(width(8, 100_000), 1024);
        assert_eq!(width(50, 100_000), 128);
        assert_eq!(width(256, 100_000), 32);
        assert_eq!(width(8, 40), 64);
        assert_eq!(width(8, 0), 1);
        // No scratch at all: only the pipeline limit and the row length bind.
        assert_eq!(width(0, 100_000), 1024);
    }

    /// The reason this branch exists, measured. Run with
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: allocates a 60 M entry matrix, run with --release"]
    fn beats_densify_then_matmul_at_single_cell_scale() {
        use candle_core::{Device, Tensor};

        let Ok(context) = MetalContext::new() else {
            return;
        };
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let (n_cells, n_genes, k) = (50_000usize, 20_000usize, 50usize);
        let per_row = n_genes * 6 / 100; // 6% density

        // Stratified sampling: one column per bucket keeps the indices sorted
        // and distinct without a per-row sort.
        let mut random = Xorshift(0x5eed);
        let mut indptr = Vec::with_capacity(n_cells + 1);
        let mut indices = Vec::with_capacity(n_cells * per_row);
        let mut values = Vec::with_capacity(n_cells * per_row);
        indptr.push(0u32);
        for _ in 0..n_cells {
            for bucket in 0..per_row {
                let width = n_genes / per_row;
                let column = bucket * width + random.next_below(width);
                indices.push(column as u32);
                values.push(random.next_unit() * 5.0);
            }
            indptr.push(indices.len() as u32);
        }
        let sparse = CsrMatrix::new(indptr, indices, values, n_genes).unwrap();
        let dense = dense_matrix(n_genes, k, 99);

        // Best of three throughout: the first pass over a freshly allocated
        // gigabyte pays for page faults that have nothing to do with either
        // algorithm, and that noise is larger than the difference being measured.
        let best_of_three = |mut run: Box<dyn FnMut()>| {
            run();
            (0..3)
                .map(|_| {
                    let started = std::time::Instant::now();
                    run();
                    started.elapsed()
                })
                .min()
                .unwrap()
        };

        let sparse_result = spmm(&context, &sparse, &dense).unwrap();
        let sparse_elapsed = best_of_three(Box::new(|| {
            spmm(&context, &sparse, &dense).unwrap();
        }));

        // The baseline this branch replaces: densify a row block, matmul it on
        // the same GPU through candle, repeat.
        let block_rows = 2048;
        let dense_operand = Tensor::from_slice(
            dense.as_standard_layout().as_slice().unwrap(),
            (n_genes, k),
            &device,
        )
        .unwrap();
        let densify_then_matmul = || {
            let mut blocks = Vec::new();
            for start in (0..n_cells).step_by(block_rows) {
                let end = (start + block_rows).min(n_cells);
                let block = sparse.densify_rows(start, end);
                let tensor = Tensor::from_vec(block, (end - start, n_genes), &device).unwrap();
                blocks.push(tensor.matmul(&dense_operand).unwrap());
            }
            Tensor::cat(&blocks, 0)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
        };
        let dense_result = densify_then_matmul();
        let dense_elapsed = best_of_three(Box::new(|| {
            densify_then_matmul();
        }));

        // The other two entry points at the same scale, so the counting sort
        // their doc comments charge for is a measured number and not a guess.
        let cell_side = dense_matrix(n_cells, k, 101);
        let transposed_elapsed = best_of_three(Box::new(|| {
            spmm_transposed(&context, &sparse, &cell_side).unwrap();
        }));
        let moments_elapsed = best_of_three(Box::new(|| {
            column_moments(&context, &sparse).unwrap();
        }));
        println!(
            "spmm_transposed (counting sort included): {transposed_elapsed:?}\n\
             column_moments (counting sort included):  {moments_elapsed:?}"
        );

        let nnz = sparse.nnz();
        let stored_bytes = nnz * 8 + (n_cells + 1) * 4;
        let block_bytes = block_rows * n_genes * 4;
        let flops = 2.0 * nnz as f64 * k as f64;
        println!(
            "{n_cells} cells x {n_genes} genes at {:.0}% density, k = {k} ({nnz} stored entries)\n\
             sparse spmm:            {sparse_elapsed:?}  {:.1} GFLOP/s\n\
             densify + candle matmul: {dense_elapsed:?}  {:.1}x slower\n\
             peak matrix memory: sparse {:.2} GB resident (CSR host arrays plus one \
             unified-memory copy), dense path {:.2} GB per {block_rows} row block \
             ({:.2} GB if densified whole)",
            100.0 * per_row as f64 / n_genes as f64,
            flops / sparse_elapsed.as_secs_f64() / 1e9,
            dense_elapsed.as_secs_f64() / sparse_elapsed.as_secs_f64(),
            2.0 * stored_bytes as f64 / 1e9,
            block_bytes as f64 / 1e9,
            (n_cells * n_genes * 4) as f64 / 1e9,
        );

        // The point of the comparison is that both paths compute the same thing.
        let dense_values = dense_result
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let scale = dense_values.iter().fold(1e-12f32, |w, v| w.max(v.abs()));
        let deviation = sparse_result
            .iter()
            .zip(&dense_values)
            .fold(0.0f32, |worst, (&got, &want)| {
                worst.max((got - want).abs() / scale)
            });
        println!("sparse vs densified agreement: {deviation:.3e} relative");
        assert!(deviation < 1e-4, "paths disagree by {deviation}");
    }
}
