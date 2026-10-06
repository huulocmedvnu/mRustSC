//! Subsampling cells and thinning counts. Owned by feat/sampling.
//!
//! Neither entry point takes a `Device` and neither uses the GPU. Thinning a
//! cell is a multivariate hypergeometric draw: gene `j`'s share is conditional
//! on what genes `0..j` already took, so the loop carries state that the next
//! iteration needs. That is a sequential dependency, not tensor algebra — there
//! is no `(cells, genes)` array of independent arithmetic to dispatch, and the
//! only tensor formulation would be materialising one slot per count, which is
//! exactly what this module exists to avoid.

use rand::rngs::StdRng;
use rand::seq::index;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Indices of the cells to keep.
pub fn subsample(n_cells: usize, n_keep: usize, replace: bool, seed: u64) -> Result<Vec<u32>> {
    if !replace && n_keep > n_cells {
        return Err(Error::parameter(
            "n_keep",
            "at most the number of cells unless replace is set",
            n_keep,
        ));
    }
    if n_cells == 0 && n_keep > 0 {
        return Err(Error::parameter(
            "n_keep",
            "zero when there are no cells",
            n_keep,
        ));
    }

    // Seeded here and nowhere else, so the indices depend on the seed alone.
    let mut rng = StdRng::seed_from_u64(seed);
    if replace {
        Ok((0..n_keep)
            .map(|_| rng.gen_range(0..n_cells) as u32)
            .collect())
    } else {
        Ok(index::sample(&mut rng, n_cells, n_keep)
            .into_iter()
            .map(|cell| cell as u32)
            .collect())
    }
}

/// Thin each cell to at most `counts_per_cell` by multivariate hypergeometric
/// sampling, as `scanpy.pp.downsample_counts`.
///
/// Exactly one of `counts_per_cell` and `total_counts` may be given. A cell
/// already at or below the target is left alone, so a cell's new total is the
/// target or its original total, whichever is smaller.
///
/// Peak memory is one `f32` per stored entry for the thinned values plus one
/// `u64` per cell for the row totals — no count vector is ever expanded, so a
/// cell holding tens of thousands of counts costs nothing beyond its own row.
pub fn downsample_counts(
    matrix: &CsrMatrix,
    counts_per_cell: Option<f32>,
    total_counts: Option<f32>,
    replace: bool,
    seed: u64,
) -> Result<CsrMatrix> {
    let totals = row_totals(matrix)?;
    let mut thinned = matrix.values().to_vec();

    match (counts_per_cell, total_counts) {
        (Some(target), None) => {
            let target = whole_count("counts_per_cell", target)?;
            let indptr = matrix.indptr();
            for (cell, &total) in totals.iter().enumerate() {
                if total <= target {
                    continue;
                }
                let span = indptr[cell] as usize..indptr[cell + 1] as usize;
                // Seeded from the cell's own index, so a cell draws the same
                // counts wherever it sits in the loop and however many cells
                // are processed alongside it.
                let mut rng = StdRng::seed_from_u64(stream_seed(seed, cell as u64));
                thin(&mut thinned[span], total, target, replace, &mut rng);
            }
        }
        (None, Some(target)) => {
            let target = whole_count("total_counts", target)?;
            let total: u64 = totals.iter().sum();
            if total > target {
                // One draw over the whole matrix, which is what scanpy does:
                // the stored entries are one long row for this purpose.
                let mut rng = StdRng::seed_from_u64(stream_seed(seed, 0));
                thin(&mut thinned, total, target, replace, &mut rng);
            }
        }
        (Some(_), Some(_)) => {
            return Err(Error::parameter(
                "counts_per_cell/total_counts",
                "exactly one of the two",
                "both",
            ))
        }
        (None, None) => {
            return Err(Error::parameter(
                "counts_per_cell/total_counts",
                "exactly one of the two",
                "neither",
            ))
        }
    }

    compact(matrix, &thinned)
}

/// Exact per-cell totals.
///
/// Counted in `u64` rather than reusing the `f32` reduction the filters share:
/// an `f32` accumulator stops being exact past 2^24, and a whole-matrix total
/// runs into the millions.
fn row_totals(matrix: &CsrMatrix) -> Result<Vec<u64>> {
    let indptr = matrix.indptr();
    let mut totals = Vec::with_capacity(matrix.n_rows());
    for cell in 0..matrix.n_rows() {
        let span = indptr[cell] as usize..indptr[cell + 1] as usize;
        let mut total = 0u64;
        for &value in &matrix.values()[span] {
            total += whole_count("matrix values", value)?;
        }
        totals.push(total);
    }
    Ok(totals)
}

/// A count is a whole non-negative number; anything else means the caller
/// downsampled something that is no longer counts, which scanpy silently
/// mangles and we refuse.
fn whole_count(parameter: &'static str, value: f32) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(Error::parameter(
            parameter,
            "a whole non-negative number of counts",
            value,
        ));
    }
    Ok(value as u64)
}

/// A cell's own RNG seed, mixed from the run seed and the cell index.
///
/// The splitmix64 finaliser decorrelates neighbouring indices, so cell 0 and
/// cell 1 get unrelated streams rather than adjacent ones.
fn stream_seed(seed: u64, index: u64) -> u64 {
    let mut state = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
}

/// Replace `values` in place by `target` counts drawn from the `total` they hold.
fn thin(values: &mut [f32], total: u64, target: u64, replace: bool, rng: &mut StdRng) {
    if replace {
        thin_with_replacement(values, total, target, rng);
    } else {
        thin_without_replacement(values, total, target, rng);
    }
}

/// Multivariate hypergeometric: `target` distinct counts out of the `total`.
///
/// Every count is one slot of an urn that is never built. Walking the slots in
/// storage order and keeping each with the current conditional probability
/// draws a uniform subset of size `target`, which is what a hypergeometric
/// draw is, using three counters and no allocation.
fn thin_without_replacement(values: &mut [f32], total: u64, target: u64, rng: &mut StdRng) {
    let mut pool = total; // slots not yet decided
    let mut need = target; // slots still to keep
    for value in values.iter_mut() {
        let mut undecided = *value as u64;
        let mut taken = 0u64;
        while undecided > 0 && need > 0 {
            if need == pool {
                // Every slot left in the urn is needed, so stop drawing.
                taken += undecided;
                need -= undecided;
                pool -= undecided;
                undecided = 0;
                break;
            }
            if rng.gen_range(0..pool) < need {
                taken += 1;
                need -= 1;
            }
            pool -= 1;
            undecided -= 1;
        }
        pool -= undecided; // once `need` is 0 the rest are dropped undrawn
        *value = taken as f32;
    }
}

/// Multinomial: `target` counts drawn with replacement from the `total`.
///
/// The draws are generated already sorted, largest first, so a single reverse
/// pass over the row assigns each to its entry. Sampling them unsorted would
/// mean holding all `target` positions, which for a deep cell is the count
/// vector this module refuses to materialise.
fn thin_with_replacement(values: &mut [f32], total: u64, target: u64, rng: &mut StdRng) {
    let mut draws = DescendingDraws::new(total, target);
    let mut pending = draws.next(rng);
    let mut upper = total;
    for value in values.iter_mut().rev() {
        let lower = upper - *value as u64;
        let mut taken = 0u64;
        while pending.is_some_and(|position| position >= lower as f64) {
            taken += 1;
            pending = draws.next(rng);
        }
        *value = taken as f32;
        upper = lower;
    }
}

/// Uniform draws over `[0, total)` in descending order, one at a time.
struct DescendingDraws {
    remaining: u64,
    quantile: f64,
    total: f64,
}

impl DescendingDraws {
    fn new(total: u64, target: u64) -> Self {
        Self {
            remaining: target,
            quantile: 1.0,
            total: total as f64,
        }
    }

    /// The next largest draw, or `None` once all of them have been handed out.
    fn next(&mut self, rng: &mut StdRng) -> Option<f64> {
        if self.remaining == 0 {
            return None;
        }
        // The largest of n uniforms on (0, 1] is U^(1/n), and conditioned on it
        // the largest of the remaining n-1 is that value times U^(1/(n-1)).
        // Drawing from (0, 1] rather than [0, 1) keeps the product off zero.
        let uniform: f64 = 1.0 - rng.gen::<f64>();
        self.quantile *= uniform.powf(1.0 / self.remaining as f64);
        self.remaining -= 1;
        Some(self.quantile * self.total)
    }
}

/// Rebuild the matrix from thinned values, dropping the entries that went to
/// zero — scanpy calls `eliminate_zeros` for the same reason, and a CSR full of
/// stored zeros is a different matrix to every consumer that counts nnz.
///
/// Not `preprocess::filter::subset`: that keeps or drops whole rows and columns,
/// and thinning empties individual entries of rows it otherwise keeps.
fn compact(matrix: &CsrMatrix, thinned: &[f32]) -> Result<CsrMatrix> {
    let indptr_in = matrix.indptr();
    let mut indptr = Vec::with_capacity(matrix.n_rows() + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0);
    for cell in 0..matrix.n_rows() {
        let span = indptr_in[cell] as usize..indptr_in[cell + 1] as usize;
        for (&column, &value) in matrix.indices()[span.clone()].iter().zip(&thinned[span]) {
            if value != 0.0 {
                indices.push(column);
                values.push(value);
            }
        }
        indptr.push(values.len() as u32);
    }
    CsrMatrix::new(indptr, indices, values, matrix.n_cols())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One deep cell, so a single row carries the whole distributional check.
    fn deep_cell() -> CsrMatrix {
        CsrMatrix::from_dense(&[1000.0, 2000.0, 3000.0, 4000.0], 1, 4).unwrap()
    }

    fn three_cells() -> CsrMatrix {
        CsrMatrix::from_dense(
            &[
                10.0, 0.0, 20.0, 30.0, //
                1.0, 2.0, 0.0, 0.0, //
                40.0, 40.0, 40.0, 40.0,
            ],
            3,
            4,
        )
        .unwrap()
    }

    fn row_sum(matrix: &CsrMatrix, row: usize) -> f32 {
        let span = matrix.indptr()[row] as usize..matrix.indptr()[row + 1] as usize;
        matrix.values()[span].iter().sum()
    }

    #[test]
    fn subsample_keeps_the_requested_number_of_distinct_cells() {
        let kept = subsample(100, 30, false, 0).unwrap();
        assert_eq!(kept.len(), 30);
        let mut unique = kept.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 30);
        assert!(kept.iter().all(|&cell| cell < 100));
    }

    #[test]
    fn subsample_is_reproducible_from_the_seed_alone() {
        assert_eq!(
            subsample(100, 30, false, 7).unwrap(),
            subsample(100, 30, false, 7).unwrap()
        );
        assert_ne!(
            subsample(100, 30, false, 7).unwrap(),
            subsample(100, 30, false, 8).unwrap()
        );
    }

    #[test]
    fn subsample_with_replacement_may_repeat_and_may_exceed_the_cell_count() {
        let kept = subsample(5, 50, true, 0).unwrap();
        assert_eq!(kept.len(), 50);
        let mut unique = kept.clone();
        unique.sort_unstable();
        unique.dedup();
        assert!(unique.len() < 50, "50 draws from 5 cells must repeat");
    }

    #[test]
    fn subsample_rejects_asking_for_more_cells_than_exist() {
        assert!(subsample(10, 11, false, 0).is_err());
        assert!(subsample(10, 10, false, 0).is_ok());
        assert!(subsample(0, 1, true, 0).is_err());
        assert!(subsample(0, 0, false, 0).is_ok());
    }

    #[test]
    fn rejects_zero_or_two_criteria() {
        let matrix = three_cells();
        assert!(downsample_counts(&matrix, None, None, false, 0).is_err());
        assert!(downsample_counts(&matrix, Some(5.0), Some(5.0), false, 0).is_err());
    }

    #[test]
    fn rejects_a_negative_target_and_non_integer_counts() {
        let matrix = three_cells();
        assert!(downsample_counts(&matrix, Some(-1.0), None, false, 0).is_err());
        assert!(downsample_counts(&matrix, None, Some(f32::NAN), false, 0).is_err());
        let fractional = CsrMatrix::from_dense(&[0.5, 1.5], 1, 2).unwrap();
        assert!(downsample_counts(&fractional, Some(1.0), None, false, 0).is_err());
    }

    #[test]
    fn every_cell_lands_on_the_target_or_keeps_its_own_total() {
        let matrix = three_cells();
        for &replace in &[false, true] {
            let thinned = downsample_counts(&matrix, Some(4.0), None, replace, 3).unwrap();
            assert_eq!(row_sum(&thinned, 0), 4.0);
            assert_eq!(
                row_sum(&thinned, 1),
                3.0,
                "a cell below the target is untouched"
            );
            assert_eq!(row_sum(&thinned, 2), 4.0);
            assert_eq!(thinned.n_cols(), matrix.n_cols());
            assert_eq!(thinned.n_rows(), matrix.n_rows());
        }
    }

    #[test]
    fn total_counts_thins_the_whole_matrix_to_the_target() {
        let matrix = three_cells();
        for &replace in &[false, true] {
            let thinned = downsample_counts(&matrix, None, Some(50.0), replace, 1).unwrap();
            let total: f32 = thinned.values().iter().sum();
            assert_eq!(total, 50.0);
        }
        // A matrix already under the target comes back untouched.
        let untouched = downsample_counts(&matrix, None, Some(10_000.0), false, 1).unwrap();
        assert_eq!(untouched.values(), matrix.values());
    }

    #[test]
    fn without_replacement_no_entry_grows() {
        let matrix = three_cells();
        let thinned = downsample_counts(&matrix, Some(60.0), None, false, 11).unwrap();
        // Row 2 is the only one above the target; its entries were 40 each.
        let span = thinned.indptr()[2] as usize..thinned.indptr()[3] as usize;
        assert!(thinned.values()[span].iter().all(|&value| value <= 40.0));
        assert_eq!(row_sum(&thinned, 2), 60.0);
    }

    #[test]
    fn no_stored_zero_survives_the_thinning() {
        // A target of one count leaves exactly one non-zero entry per cell, so
        // every other entry has to be gone rather than stored as a zero.
        let thinned = downsample_counts(&three_cells(), Some(1.0), None, false, 5).unwrap();
        assert!(thinned.values().iter().all(|&value| value != 0.0));
        assert_eq!(thinned.nnz(), 3);
    }

    #[test]
    fn a_cell_draws_the_same_counts_wherever_the_loop_reaches_it() {
        let full = three_cells();
        // The same first two cells, without the third in the matrix.
        let head =
            CsrMatrix::from_dense(&[10.0, 0.0, 20.0, 30.0, 1.0, 2.0, 0.0, 0.0], 2, 4).unwrap();
        let from_full = downsample_counts(&full, Some(4.0), None, false, 42).unwrap();
        let from_head = downsample_counts(&head, Some(4.0), None, false, 42).unwrap();
        let end = from_head.indptr()[2] as usize;
        assert_eq!(&from_full.values()[..end], from_head.values());
    }

    #[test]
    fn the_same_seed_gives_the_same_matrix_and_a_different_seed_does_not() {
        let matrix = deep_cell();
        let first = downsample_counts(&matrix, Some(1000.0), None, false, 4).unwrap();
        let again = downsample_counts(&matrix, Some(1000.0), None, false, 4).unwrap();
        let other = downsample_counts(&matrix, Some(1000.0), None, false, 5).unwrap();
        assert_eq!(first.values(), again.values());
        assert_ne!(first.values(), other.values());
    }

    /// Mean count per gene over `repeats` draws of `target` from `deep_cell`.
    fn mean_per_gene(target: f32, replace: bool, repeats: u64) -> Vec<f64> {
        let matrix = deep_cell();
        let mut sums = vec![0.0f64; matrix.n_cols()];
        for seed in 0..repeats {
            let thinned = downsample_counts(&matrix, Some(target), None, replace, seed).unwrap();
            for (&gene, &value) in thinned.indices().iter().zip(thinned.values()) {
                sums[gene as usize] += value as f64;
            }
        }
        sums.iter().map(|sum| sum / repeats as f64).collect()
    }

    /// The expected count of a gene is its share of the cell times the target,
    /// for both draws. The tolerance is four standard errors of the mean of the
    /// repeats, computed from the exact variance of the draw, so a correct
    /// implementation fails this about once in 16000 runs — and the seeds are
    /// fixed, so in practice it never flakes.
    fn assert_proportional(replace: bool) {
        let counts: [f64; 4] = [1000.0, 2000.0, 3000.0, 4000.0];
        let total: f64 = 10_000.0;
        let target: f64 = 1000.0;
        let repeats = 400;
        let means = mean_per_gene(target as f32, replace, repeats as u64);

        for (gene, &count) in counts.iter().enumerate() {
            let share = count / total;
            let expected = target * share;
            // Binomial variance for the multinomial draw; the hypergeometric
            // draw is the same times the finite population correction.
            let mut variance = target * share * (1.0 - share);
            if !replace {
                variance *= (total - target) / (total - 1.0);
            }
            let standard_error = (variance / repeats as f64).sqrt();
            let deviation = (means[gene] - expected).abs();
            assert!(
                deviation <= 4.0 * standard_error,
                "gene {gene}: mean {} is {deviation:.3} from the expected {expected}, \
                 beyond four standard errors ({:.3})",
                means[gene],
                4.0 * standard_error
            );
        }
    }

    #[test]
    fn expected_counts_are_proportional_without_replacement() {
        assert_proportional(false);
    }

    #[test]
    fn expected_counts_are_proportional_with_replacement() {
        assert_proportional(true);
    }

    #[test]
    fn an_empty_matrix_survives_both_modes() {
        let empty = CsrMatrix::new(vec![0, 0, 0], vec![], vec![], 4).unwrap();
        assert_eq!(
            downsample_counts(&empty, Some(5.0), None, false, 0)
                .unwrap()
                .nnz(),
            0
        );
        assert_eq!(
            downsample_counts(&empty, None, Some(5.0), true, 0)
                .unwrap()
                .nnz(),
            0
        );
    }

    #[test]
    fn a_target_of_zero_empties_the_matrix() {
        let thinned = downsample_counts(&three_cells(), Some(0.0), None, false, 0).unwrap();
        assert_eq!(thinned.nnz(), 0);
        assert_eq!(thinned.n_rows(), 3);
    }
}
