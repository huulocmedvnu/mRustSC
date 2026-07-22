//! Processing matrices larger than memory. Owned by feat/out-of-core.
//!
//! The in-memory `CsrMatrix` is the fast path. Above a configured ceiling the
//! same operations run over row blocks streamed from disk, so a run is bounded
//! by the block size rather than by the dataset.

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// A source of row blocks, whether from memory or from a file.
///
/// Small enough that a caller needing only sequential access does not depend on
/// random access, and vice versa.
pub trait RowBlocks {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    /// The next block of at most `max_rows` rows, or `None` at the end.
    fn next_block(&mut self, max_rows: usize) -> Result<Option<CsrMatrix>>;
    fn restart(&mut self) -> Result<()>;
}

/// One stored entry costs a `u32` column index plus an `f32` value.
const BYTES_PER_STORED_ENTRY: usize = size_of::<u32>() + size_of::<f32>();
/// One row costs one `indptr` entry.
const BYTES_PER_ROW_POINTER: usize = size_of::<u32>();
/// One entry of the dense buffer a caller builds from a block.
const BYTES_PER_DENSE_ENTRY: usize = size_of::<f32>();

/// Budget the streaming statistics below size their own blocks against.
///
/// They are the one caller that knows no user budget: they take a `RowBlocks`
/// and nothing else. A few tens of megabytes is small enough to be invisible
/// next to any dataset worth streaming and large enough that block overhead
/// disappears.
const STREAMING_BUDGET_BYTES: usize = 64 << 20;

/// Sizing the streaming blocks assumes the worst case, that every entry is
/// stored. Being wrong here only makes the blocks smaller than they had to be.
const WORST_CASE_DENSITY: f64 = 1.0;

/// How many rows fit in `budget_bytes` given the matrix's density.
///
/// The model charges a row for both halves of what a block costs while it is
/// live: its share of the CSR arrays, and its row of the dense
/// `(rows, n_cols)` buffer the caller densifies it into.
///
/// ```text
/// bytes(rows) = rows * ( n_cols * 4              dense f32 block
///                      + density * n_cols * 8    CSR index + value
///                      + 4 )                     indptr entry
/// ```
///
/// The dense term dominates at the 5-10% densities single-cell data has, which
/// is why it cannot be left out: sizing against the CSR arrays alone would
/// overshoot by more than a factor of ten at the moment the caller densifies.
///
/// Assumptions, and what breaks them: `density` is the *mean* fraction of
/// stored entries per row, so a block of unusually dense rows overshoots; the
/// budget covers the block only, not what a caller accumulates across blocks
/// (those are per gene, so they do not grow with the number of cells). A
/// density that is not a number is treated as fully dense rather than rejected,
/// because this function has no error channel and the safe reading is the
/// pessimistic one. The result is never zero: a caller must make progress even
/// when one row alone exceeds the budget.
pub fn rows_per_block(n_cols: usize, density: f64, budget_bytes: usize) -> usize {
    let density = if density.is_nan() {
        WORST_CASE_DENSITY
    } else {
        density.clamp(0.0, WORST_CASE_DENSITY)
    };
    let stored_per_row = (density * n_cols as f64).ceil() as usize;
    let bytes_per_row = n_cols
        .saturating_mul(BYTES_PER_DENSE_ENTRY)
        .saturating_add(stored_per_row.saturating_mul(BYTES_PER_STORED_ENTRY))
        .saturating_add(BYTES_PER_ROW_POINTER);
    (budget_bytes / bytes_per_row).max(1)
}

/// Per-gene mean and variance accumulated over blocks, in one pass.
///
/// The variance is the sample variance (`ddof = 1`), which is what scanpy's
/// `_get_mean_var` returns and therefore what the highly-variable-gene code
/// downstream expects.
///
/// Consumes the stream from the beginning, so the result does not depend on
/// how far a previous caller had read.
pub fn streaming_gene_statistics(blocks: &mut dyn RowBlocks) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut moments = GeneMoments::new(blocks.n_cols());
    for_each_block(blocks, |block| moments.absorb(block))?;
    moments.finish()
}

/// Per-cell totals accumulated over blocks.
pub fn streaming_cell_totals(blocks: &mut dyn RowBlocks) -> Result<Vec<f32>> {
    let mut totals = Vec::with_capacity(blocks.n_rows());
    for_each_block(blocks, |block| {
        let indptr = block.indptr();
        for row in 0..block.n_rows() {
            let entries = indptr[row] as usize..indptr[row + 1] as usize;
            // f64 because a cell in a deeply sequenced dataset sums hundreds of
            // thousands of counts, where f32 has already lost the ones digit.
            let total: f64 = block.values()[entries].iter().map(|&v| v as f64).sum();
            totals.push(total as f32);
        }
        Ok(())
    })?;
    Ok(totals)
}

/// Restart the stream and hand every block to `consume`, at a block size that
/// fits [`STREAMING_BUDGET_BYTES`] whatever the source's density turns out to be.
fn for_each_block(
    blocks: &mut dyn RowBlocks,
    mut consume: impl FnMut(&CsrMatrix) -> Result<()>,
) -> Result<()> {
    let max_rows = rows_per_block(blocks.n_cols(), WORST_CASE_DENSITY, STREAMING_BUDGET_BYTES);
    blocks.restart()?;
    while let Some(block) = blocks.next_block(max_rows)? {
        if block.n_cols() != blocks.n_cols() {
            return Err(Error::shape(
                format!("blocks of {} genes", blocks.n_cols()),
                format!("a block of {} genes", block.n_cols()),
            ));
        }
        consume(&block)?;
    }
    Ok(())
}

/// Running per-gene mean and sum of squared deviations.
///
/// Blocks are merged with the Chan-Golub-LeVeque update — the batched form of
/// Welford's algorithm — rather than by accumulating a raw sum of squares. Both
/// are one pass, but `sum(x^2) - n * mean^2` subtracts two nearly equal large
/// numbers, and on data with a large mean and a small variance that cancellation
/// eats every significant digit of the answer (it can even go negative). Each
/// block's own moments are taken about the block's own mean, so nothing large is
/// ever squared, and the merge is exact in exact arithmetic.
struct GeneMoments {
    n_rows: usize,
    mean: Vec<f64>,
    squared_deviations: Vec<f64>,
}

impl GeneMoments {
    fn new(n_cols: usize) -> Self {
        Self {
            n_rows: 0,
            mean: vec![0.0; n_cols],
            squared_deviations: vec![0.0; n_cols],
        }
    }

    fn absorb(&mut self, block: &CsrMatrix) -> Result<()> {
        let n_cols = self.mean.len();
        if block.n_cols() != n_cols {
            return Err(Error::shape(
                format!("{n_cols} genes"),
                format!("{} genes", block.n_cols()),
            ));
        }
        let block_rows = block.n_rows();
        if block_rows == 0 {
            return Ok(());
        }

        let mut stored = vec![0usize; n_cols];
        let mut block_mean = vec![0.0f64; n_cols];
        for (&column, &value) in block.indices().iter().zip(block.values()) {
            block_mean[column as usize] += value as f64;
            stored[column as usize] += 1;
        }
        for mean in &mut block_mean {
            *mean /= block_rows as f64;
        }

        let mut block_deviations = vec![0.0f64; n_cols];
        for (&column, &value) in block.indices().iter().zip(block.values()) {
            let deviation = value as f64 - block_mean[column as usize];
            block_deviations[column as usize] += deviation * deviation;
        }
        // The entries not stored are zeros, and each of them deviates from the
        // block mean by exactly the block mean.
        for column in 0..n_cols {
            let implicit_zeros = (block_rows - stored[column]) as f64;
            block_deviations[column] += implicit_zeros * block_mean[column] * block_mean[column];
        }

        let total_rows = self.n_rows + block_rows;
        let weight = (self.n_rows * block_rows) as f64 / total_rows as f64;
        for column in 0..n_cols {
            let shift = block_mean[column] - self.mean[column];
            self.mean[column] += shift * block_rows as f64 / total_rows as f64;
            self.squared_deviations[column] += block_deviations[column] + shift * shift * weight;
        }
        self.n_rows = total_rows;
        Ok(())
    }

    fn finish(self) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.n_rows == 0 {
            return Err(Error::shape("at least one cell", "an empty matrix"));
        }
        let mean = self.mean.iter().map(|&value| value as f32).collect();
        // A single cell has no sample variance; zero is what scanpy reports and
        // it keeps the caller from having to special-case the degenerate input.
        let denominator = (self.n_rows.max(2) - 1) as f64;
        let variance = self
            .squared_deviations
            .iter()
            .map(|&total| (total.max(0.0) / denominator) as f32)
            .collect();
        Ok((mean, variance))
    }
}

/// [`RowBlocks`] over a matrix already in memory: a cursor over the rows, not a
/// second copy of them. Only the rows of the block asked for are copied out.
pub struct CsrRowBlocks<'a> {
    matrix: &'a CsrMatrix,
    next_row: usize,
}

impl<'a> CsrRowBlocks<'a> {
    pub fn new(matrix: &'a CsrMatrix) -> Self {
        Self {
            matrix,
            next_row: 0,
        }
    }
}

impl RowBlocks for CsrRowBlocks<'_> {
    fn n_rows(&self) -> usize {
        self.matrix.n_rows()
    }

    fn n_cols(&self) -> usize {
        self.matrix.n_cols()
    }

    fn next_block(&mut self, max_rows: usize) -> Result<Option<CsrMatrix>> {
        if max_rows == 0 {
            return Err(Error::parameter("max_rows", "at least 1", max_rows));
        }
        if self.next_row >= self.matrix.n_rows() {
            return Ok(None);
        }
        let end = (self.next_row + max_rows).min(self.matrix.n_rows());
        let block = slice_rows(self.matrix, self.next_row, end)?;
        self.next_row = end;
        Ok(Some(block))
    }

    fn restart(&mut self) -> Result<()> {
        self.next_row = 0;
        Ok(())
    }
}

/// Rows `start..end` of `matrix` as a matrix of their own.
fn slice_rows(matrix: &CsrMatrix, start: usize, end: usize) -> Result<CsrMatrix> {
    if start > end || end > matrix.n_rows() {
        return Err(Error::shape(
            format!("a row range within 0..{}", matrix.n_rows()),
            format!("{start}..{end}"),
        ));
    }
    let indptr = matrix.indptr();
    let first = indptr[start];
    let last = indptr[end] as usize;
    let rebased = indptr[start..=end]
        .iter()
        .map(|&offset| offset - first)
        .collect();
    CsrMatrix::new(
        rebased,
        matrix.indices()[first as usize..last].to_vec(),
        matrix.values()[first as usize..last].to_vec(),
        matrix.n_cols(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_COLS: usize = 8;

    /// Linear congruential generator, so the fixtures are reproducible without
    /// pulling `rand` into the test path.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f64 / (1u64 << 24) as f64
        }
    }

    /// Sparse counts: 60% zeros, small values, the shape of real data.
    fn sparse_counts(n_rows: usize) -> CsrMatrix {
        let mut rng = Lcg(11);
        let mut dense = vec![0.0f32; n_rows * N_COLS];
        for value in &mut dense {
            let uniform = rng.next();
            *value = if uniform < 0.6 {
                0.0
            } else {
                (uniform * 20.0).floor() as f32
            };
        }
        CsrMatrix::from_dense(&dense, n_rows, N_COLS).unwrap()
    }

    /// The case a raw sum of squares cannot survive: every value near 1e6, the
    /// spread around it of order one. The variance is then twelve orders of
    /// magnitude below the second moment it would be recovered from.
    fn large_mean_small_variance(n_rows: usize) -> CsrMatrix {
        let mut rng = Lcg(5);
        let mut dense = vec![0.0f32; n_rows * N_COLS];
        for (index, value) in dense.iter_mut().enumerate() {
            let offset = (index % N_COLS) as f64 * 1000.0;
            *value = (1.0e6 + offset + (rng.next() * 2.0 - 1.0)) as f32;
        }
        CsrMatrix::from_dense(&dense, n_rows, N_COLS).unwrap()
    }

    /// Exact reference: two passes over the dense matrix in f64.
    fn reference_statistics(matrix: &CsrMatrix) -> (Vec<f64>, Vec<f64>) {
        let n_rows = matrix.n_rows();
        let dense = matrix.densify_rows(0, n_rows);
        let mut mean = vec![0.0f64; N_COLS];
        for row in dense.chunks_exact(N_COLS) {
            for (total, &value) in mean.iter_mut().zip(row) {
                *total += value as f64;
            }
        }
        for total in &mut mean {
            *total /= n_rows as f64;
        }
        let mut variance = vec![0.0f64; N_COLS];
        for row in dense.chunks_exact(N_COLS) {
            for ((total, &value), &centre) in variance.iter_mut().zip(row).zip(&mean) {
                let deviation = value as f64 - centre;
                *total += deviation * deviation;
            }
        }
        for total in &mut variance {
            *total /= (n_rows - 1) as f64;
        }
        (mean, variance)
    }

    /// The formula this module exists to avoid: one pass accumulating the sum
    /// and the sum of squares, then `sum(x^2) - n * mean^2`. Measured in the f32
    /// the rest of the library uses and, for the record, in f64 as well.
    fn naive_variance<T: NaiveAccumulator>(matrix: &CsrMatrix) -> Vec<f32> {
        let n_rows = matrix.n_rows();
        let dense = matrix.densify_rows(0, n_rows);
        let mut sum = [T::ZERO; N_COLS];
        let mut squares = [T::ZERO; N_COLS];
        for row in dense.chunks_exact(N_COLS) {
            for ((total, square), &value) in sum.iter_mut().zip(&mut squares).zip(row) {
                let value = T::of(value);
                *total = *total + value;
                *square = *square + value * value;
            }
        }
        sum.iter()
            .zip(&squares)
            .map(|(&total, &square)| {
                let corrected = square - total * total / T::of(n_rows as f32);
                T::into_f32(corrected / T::of((n_rows - 1) as f32))
            })
            .collect()
    }

    /// The accumulator width `naive_variance` is measured at.
    trait NaiveAccumulator:
        Copy
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Div<Output = Self>
    {
        const ZERO: Self;
        fn of(value: f32) -> Self;
        fn into_f32(self) -> f32;
    }

    impl NaiveAccumulator for f32 {
        const ZERO: f32 = 0.0;
        fn of(value: f32) -> f32 {
            value
        }
        fn into_f32(self) -> f32 {
            self
        }
    }

    impl NaiveAccumulator for f64 {
        const ZERO: f64 = 0.0;
        fn of(value: f32) -> f64 {
            value as f64
        }
        fn into_f32(self) -> f32 {
            self as f32
        }
    }

    fn worst_relative_error(measured: &[f32], reference: &[f64]) -> f64 {
        measured
            .iter()
            .zip(reference)
            .map(|(&value, &expected)| (value as f64 - expected).abs() / expected.abs().max(1e-12))
            .fold(0.0, f64::max)
    }

    /// Drive the accumulator at an explicit block size, which the public
    /// entry point deliberately does not expose.
    fn statistics_in_blocks(matrix: &CsrMatrix, block_rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut blocks = CsrRowBlocks::new(matrix);
        let mut moments = GeneMoments::new(matrix.n_cols());
        while let Some(block) = blocks.next_block(block_rows).unwrap() {
            moments.absorb(&block).unwrap();
        }
        moments.finish().unwrap()
    }

    #[test]
    fn blocks_cover_every_row_when_the_count_is_not_a_multiple() {
        let matrix = sparse_counts(7);
        let mut blocks = CsrRowBlocks::new(&matrix);
        let mut sizes = Vec::new();
        let mut rows = Vec::new();
        while let Some(block) = blocks.next_block(3).unwrap() {
            sizes.push(block.n_rows());
            rows.extend(block.densify_rows(0, block.n_rows()));
        }
        assert_eq!(sizes, vec![3, 3, 1]);
        assert_eq!(rows, matrix.densify_rows(0, 7));
        assert!(blocks.next_block(3).unwrap().is_none());
    }

    #[test]
    fn a_block_larger_than_the_matrix_returns_it_whole() {
        let matrix = sparse_counts(5);
        let mut blocks = CsrRowBlocks::new(&matrix);
        let block = blocks.next_block(100).unwrap().unwrap();
        assert_eq!(block.n_rows(), 5);
        assert_eq!(block.nnz(), matrix.nnz());
        assert!(blocks.next_block(100).unwrap().is_none());
    }

    #[test]
    fn restart_hands_out_the_first_block_again() {
        let matrix = sparse_counts(7);
        let mut blocks = CsrRowBlocks::new(&matrix);
        let first = blocks.next_block(3).unwrap().unwrap();
        while blocks.next_block(3).unwrap().is_some() {}
        blocks.restart().unwrap();
        let again = blocks.next_block(3).unwrap().unwrap();
        assert_eq!(again.indptr(), first.indptr());
        assert_eq!(again.values(), first.values());
        assert_eq!(
            streaming_cell_totals(&mut blocks).unwrap().len(),
            matrix.n_rows()
        );
    }

    #[test]
    fn a_zero_row_budget_is_rejected() {
        let matrix = sparse_counts(4);
        let mut blocks = CsrRowBlocks::new(&matrix);
        assert!(blocks.next_block(0).is_err());
    }

    #[test]
    fn statistics_do_not_depend_on_where_the_blocks_fall() {
        let matrix = sparse_counts(101);
        let (reference_mean, reference_variance) = reference_statistics(&matrix);
        for block_rows in [1, 7, 100, 101, 500] {
            let (mean, variance) = statistics_in_blocks(&matrix, block_rows);
            assert!(
                worst_relative_error(&mean, &reference_mean) < 1e-5,
                "{block_rows} rows"
            );
            assert!(
                worst_relative_error(&variance, &reference_variance) < 1e-5,
                "{block_rows} rows"
            );
        }
    }

    #[test]
    fn the_public_entry_point_matches_the_one_shot_reference() {
        let matrix = sparse_counts(313);
        let (reference_mean, reference_variance) = reference_statistics(&matrix);
        let (mean, variance) = streaming_gene_statistics(&mut CsrRowBlocks::new(&matrix)).unwrap();
        assert!(worst_relative_error(&mean, &reference_mean) < 1e-5);
        assert!(worst_relative_error(&variance, &reference_variance) < 1e-5);
    }

    #[test]
    fn the_streaming_update_survives_what_a_sum_of_squares_does_not() {
        let matrix = large_mean_small_variance(2000);
        let (_, reference_variance) = reference_statistics(&matrix);
        let (_, variance) = statistics_in_blocks(&matrix, 128);

        let streaming_error = worst_relative_error(&variance, &reference_variance);
        let naive_f32 = worst_relative_error(&naive_variance::<f32>(&matrix), &reference_variance);
        let naive_f64 = worst_relative_error(&naive_variance::<f64>(&matrix), &reference_variance);
        println!(
            "worst relative error on the variance: streaming {streaming_error:.3e}, \
             naive sum of squares {naive_f32:.3e} in f32 and {naive_f64:.3e} in f64"
        );
        assert!(streaming_error < 1e-5, "streaming error {streaming_error}");
        assert!(
            naive_f32 > 1000.0 * streaming_error.max(f64::EPSILON),
            "streaming {streaming_error}, naive {naive_f32}"
        );
    }

    #[test]
    fn cell_totals_match_a_row_by_row_sum() {
        let matrix = sparse_counts(31);
        let totals = streaming_cell_totals(&mut CsrRowBlocks::new(&matrix)).unwrap();
        let dense = matrix.densify_rows(0, matrix.n_rows());
        let expected: Vec<f32> = dense
            .chunks_exact(N_COLS)
            .map(|row| row.iter().sum::<f32>())
            .collect();
        assert_eq!(totals, expected);
    }

    #[test]
    fn statistics_reject_an_empty_stream() {
        let matrix = CsrMatrix::new(vec![0], vec![], vec![], N_COLS).unwrap();
        assert!(streaming_gene_statistics(&mut CsrRowBlocks::new(&matrix)).is_err());
        assert!(streaming_cell_totals(&mut CsrRowBlocks::new(&matrix))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_single_cell_has_no_variance() {
        let matrix = sparse_counts(1);
        let (mean, variance) = streaming_gene_statistics(&mut CsrRowBlocks::new(&matrix)).unwrap();
        assert_eq!(mean.len(), N_COLS);
        assert!(variance.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn rows_per_block_stays_inside_its_budget() {
        for &n_cols in &[1usize, 100, 2_000, 30_000] {
            for &density in &[0.0, 0.05, 0.5, 1.0] {
                for &budget in &[1usize, 1 << 10, 1 << 20, 1 << 30] {
                    let rows = rows_per_block(n_cols, density, budget);
                    assert!(rows >= 1, "{n_cols} genes, {density}, {budget} bytes");
                    let dense = rows * n_cols * BYTES_PER_DENSE_ENTRY;
                    let stored =
                        rows * (density * n_cols as f64).ceil() as usize * BYTES_PER_STORED_ENTRY;
                    let used = dense + stored + rows * BYTES_PER_ROW_POINTER;
                    assert!(
                        rows == 1 || used <= budget,
                        "{rows} rows of {n_cols} genes need {used} bytes of {budget}"
                    );
                }
            }
        }
    }

    #[test]
    fn rows_per_block_grows_with_the_budget_and_shrinks_with_density() {
        assert!(rows_per_block(1_000, 0.1, 1 << 20) > rows_per_block(1_000, 0.1, 1 << 18));
        assert!(rows_per_block(1_000, 0.1, 1 << 20) > rows_per_block(1_000, 0.9, 1 << 20));
        // A density that is not a number is read as fully dense, not rejected.
        assert_eq!(
            rows_per_block(1_000, f64::NAN, 1 << 20),
            rows_per_block(1_000, 1.0, 1 << 20)
        );
        assert_eq!(
            rows_per_block(1_000, -1.0, 1 << 20),
            rows_per_block(1_000, 0.0, 1 << 20)
        );
    }
}
