//! Gene-set scoring. Owned by feat/scoring.

use std::collections::BTreeSet;

use candle_core::{Device, Tensor};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Densified `f32` elements per row block: 4 MB, so the tile stays in cache and
/// peak memory is bounded by the block rather than by the number of cells.
const BLOCK_ELEMENTS: usize = 1 << 20;

/// Mean expression of a gene set minus a expression-binned control set, as
/// `scanpy.tl.score_genes`.
///
/// The control set is drawn from the same average-expression bins the scored
/// genes fall into, so the score measures the gene set rather than the depth of
/// the cell. `seed` reproduces scanpy's draw exactly: see [`numpy_rng`].
///
/// Peak memory is `BLOCK_ELEMENTS` floats for the densified row block plus
/// `12 * n_genes` bytes of bookkeeping — never a dense cells-by-genes matrix.
pub fn score_genes(
    matrix: &CsrMatrix,
    gene_set: &[u32],
    ctrl_size: usize,
    n_bins: usize,
    seed: u64,
    device: &Device,
) -> Result<Vec<f32>> {
    if n_bins < 2 {
        return Err(Error::parameter("n_bins", "at least 2", n_bins));
    }
    if matrix.n_rows() == 0 {
        return Err(Error::parameter("matrix", "at least one cell", 0));
    }
    let scored = column_set(gene_set, matrix.n_cols(), "gene_set")?;
    if scored.is_empty() {
        return Err(Error::parameter("gene_set", "non-empty", 0));
    }

    let bins = expression_bins(&column_means(matrix)?, n_bins)?;
    let control = control_set(&bins, &scored, ctrl_size, seed)?;

    let scored: Vec<u32> = scored.into_iter().collect();
    let scored_means = subset_row_means(matrix, &scored, device)?;
    let control_means = subset_row_means(matrix, &control, device)?;
    Ok(scored_means
        .iter()
        .zip(&control_means)
        .map(|(scored, control)| scored - control)
        .collect())
}

/// Validate column indices and drop duplicates, as pandas' `Index.intersection`
/// does on the Python side.
fn column_set(columns: &[u32], n_cols: usize, name: &'static str) -> Result<BTreeSet<u32>> {
    if let Some(&column) = columns.iter().find(|&&column| column as usize >= n_cols) {
        return Err(Error::parameter(
            name,
            "column indices inside the matrix",
            column,
        ));
    }
    Ok(columns.iter().copied().collect())
}

/// Mean of every column over all cells, in `f64`.
///
/// A column reduction of a CSR matrix is a scatter-add over the stored values,
/// not tensor algebra: expressing it with candle would mean materialising the
/// dense matrix the sparse form exists to avoid. The accumulation order and the
/// `f64` accumulator are scanpy's, so the bin edges below are bit-identical.
fn column_means(matrix: &CsrMatrix) -> Result<Vec<f64>> {
    let mut totals = vec![0.0f64; matrix.n_cols()];
    for (&column, &value) in matrix.indices().iter().zip(matrix.values()) {
        if !value.is_finite() {
            return Err(Error::parameter("matrix", "finite values", value));
        }
        totals[column as usize] += f64::from(value);
    }
    let n_rows = matrix.n_rows() as f64;
    Ok(totals.into_iter().map(|total| total / n_rows).collect())
}

/// Bin genes by average expression, as scanpy's `obs_avg.rank(method="min") // n_items`.
///
/// The bins are equal-count rather than equal-width, and the last one is short
/// whenever the gene count is not a multiple of the bin size — which is why the
/// divisor is derived from `n_bins - 1`.
fn expression_bins(means: &[f64], n_bins: usize) -> Result<Vec<usize>> {
    let n_items = (means.len() as f64 / (n_bins - 1) as f64).round_ties_even() as usize;
    if n_items == 0 {
        return Err(Error::parameter(
            "n_bins",
            "no larger than the number of genes",
            n_bins,
        ));
    }
    let ranks = minimum_ranks(means);
    Ok(ranks.into_iter().map(|rank| rank / n_items).collect())
}

/// Competition ranks (1-based, ties share the lowest rank), pandas' `method="min"`.
fn minimum_ranks(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0usize; values.len()];
    let mut position = 0;
    while position < order.len() {
        let mut tied = position + 1;
        while tied < order.len() && values[order[tied]] == values[order[position]] {
            tied += 1;
        }
        for &index in &order[position..tied] {
            ranks[index] = position + 1;
        }
        position = tied;
    }
    ranks
}

/// Draw the control genes from the bins the scored genes occupy.
///
/// Every step is scanpy's, including the ones that surprise: a bin holding at
/// most `ctrl_size` genes contributes all of itself, the scored genes are
/// removed *after* the draw so a bin can contribute fewer than `ctrl_size`
/// genes, and bins are visited in ascending order because that is the order the
/// shared random stream is consumed in.
fn control_set(
    bins: &[usize],
    scored: &BTreeSet<u32>,
    ctrl_size: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    let occupied: BTreeSet<usize> = scored.iter().map(|&gene| bins[gene as usize]).collect();
    let mut generator = numpy_rng::LegacyRandom::seed(seed)?;
    let mut control = BTreeSet::new();
    for bin in occupied {
        let mut candidates: Vec<u32> = (0..bins.len() as u32)
            .filter(|&gene| bins[gene as usize] == bin)
            .collect();
        if ctrl_size < candidates.len() {
            let order = generator.permutation(candidates.len());
            candidates = order[..ctrl_size]
                .iter()
                .map(|&position| candidates[position])
                .collect();
        }
        control.extend(candidates.iter().filter(|gene| !scored.contains(gene)));
    }
    if control.is_empty() {
        return Err(Error::parameter(
            "ctrl_size",
            "large enough to leave control genes outside the gene set",
            ctrl_size,
        ));
    }
    Ok(control.into_iter().collect())
}

/// Mean expression per cell over `columns`, block by block on `device`.
///
/// This is the device-friendly half: gathering the selected columns of a row
/// block gives a dense `(block, k)` tile whose row sums are one tensor
/// reduction, and `k` is the size of the gene set, not the number of genes.
fn subset_row_means(matrix: &CsrMatrix, columns: &[u32], device: &Device) -> Result<Vec<f32>> {
    let width = columns.len();
    let mut slot_of = vec![u32::MAX; matrix.n_cols()];
    for (slot, &column) in columns.iter().enumerate() {
        slot_of[column as usize] = slot as u32;
    }

    let rows_per_block = (BLOCK_ELEMENTS / width).max(1);
    let mut means = Vec::with_capacity(matrix.n_rows());
    for start in (0..matrix.n_rows()).step_by(rows_per_block) {
        let end = (start + rows_per_block).min(matrix.n_rows());
        let mut block = vec![0.0f32; (end - start) * width];
        for row in start..end {
            let from = matrix.indptr()[row] as usize;
            let to = matrix.indptr()[row + 1] as usize;
            for entry in from..to {
                let slot = slot_of[matrix.indices()[entry] as usize];
                if slot != u32::MAX {
                    block[(row - start) * width + slot as usize] = matrix.values()[entry];
                }
            }
        }
        let tile = Tensor::from_vec(block, (end - start, width), device)?;
        let block_means = tile.sum(1)?.affine(1.0 / width as f64, 0.0)?;
        means.extend(block_means.to_device(&Device::Cpu)?.to_vec1::<f32>()?);
    }
    Ok(means)
}

/// numpy's legacy `RandomState`, reproduced so the control set is scanpy's.
///
/// scanpy draws control genes with `pandas.Series.sample`, which is
/// `np.random.choice(n, size, replace=False)`, which is
/// `np.random.permutation(n)[:size]` off the global MT19937 stream seeded by
/// `np.random.seed`. Any other generator gives a different — equally valid, but
/// not comparable — control set, so the draw is reimplemented here rather than
/// taken from `rand`.
mod numpy_rng {
    use crate::error::{Error, Result};

    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7fff_ffff;

    pub(super) struct LegacyRandom {
        state: [u32; N],
        position: usize,
    }

    impl LegacyRandom {
        /// Seed as `np.random.seed(seed)` does, which is Knuth's `init_genrand`.
        pub(super) fn seed(seed: u64) -> Result<Self> {
            let seed = u32::try_from(seed)
                .map_err(|_| Error::parameter("seed", "below 2^32, as numpy requires", seed))?;
            let mut state = [0u32; N];
            state[0] = seed;
            for index in 1..N {
                let previous = state[index - 1];
                state[index] = 1_812_433_253u32
                    .wrapping_mul(previous ^ (previous >> 30))
                    .wrapping_add(index as u32);
            }
            Ok(Self { state, position: N })
        }

        fn next_u32(&mut self) -> u32 {
            if self.position >= N {
                self.twist();
            }
            let mut value = self.state[self.position];
            self.position += 1;
            value ^= value >> 11;
            value ^= (value << 7) & 0x9d2c_5680;
            value ^= (value << 15) & 0xefc6_0000;
            value ^ (value >> 18)
        }

        fn twist(&mut self) {
            for index in 0..N {
                let combined =
                    (self.state[index] & UPPER_MASK) | (self.state[(index + 1) % N] & LOWER_MASK);
                let mut next = self.state[(index + M) % N] ^ (combined >> 1);
                if combined & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.state[index] = next;
            }
            self.position = 0;
        }

        /// numpy's `random_interval`: masked rejection sampling on `0..=max`.
        fn interval(&mut self, max: u32) -> u32 {
            let mut mask = max;
            for shift in [1, 2, 4, 8, 16] {
                mask |= mask >> shift;
            }
            loop {
                let value = self.next_u32() & mask;
                if value <= max {
                    return value;
                }
            }
        }

        /// numpy's `permutation(n)`: a Fisher-Yates shuffle from the top down.
        pub(super) fn permutation(&mut self, n: usize) -> Vec<usize> {
            let mut order: Vec<usize> = (0..n).collect();
            for index in (1..n).rev() {
                order.swap(index, self.interval(index as u32) as usize);
            }
            order
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: Device = Device::Cpu;

    #[test]
    fn permutation_reproduces_numpy() {
        // Generated with `np.random.seed(seed); np.random.permutation(n)`.
        let cases: [(u64, usize, &[usize]); 4] = [
            (0, 10, &[2, 8, 4, 9, 1, 6, 7, 3, 0, 5]),
            (1, 5, &[2, 1, 4, 0, 3]),
            (42, 100, &[83, 53, 70, 45, 44, 39, 22, 80]),
            (3, 1000, &[642, 762, 909, 199, 586, 797, 652, 755]),
        ];
        for (seed, n, expected) in cases {
            let drawn = numpy_rng::LegacyRandom::seed(seed).unwrap().permutation(n);
            assert_eq!(&drawn[..expected.len()], expected, "seed {seed}, n {n}");
        }
    }

    #[test]
    fn rejects_a_seed_numpy_would_reject() {
        assert!(numpy_rng::LegacyRandom::seed(1 << 32).is_err());
    }

    #[test]
    fn ranks_share_the_lowest_rank_on_ties() {
        assert_eq!(minimum_ranks(&[3.0, 1.0, 3.0, 2.0]), vec![3, 1, 3, 2]);
        assert_eq!(minimum_ranks(&[1.0, 1.0, 1.0]), vec![1, 1, 1]);
    }

    /// 4 cells over `n_genes` genes; gene `g` carries `g + 1` in every cell, so
    /// the column means are strictly increasing and the bins are predictable.
    fn graded(n_genes: usize) -> CsrMatrix {
        let mut dense = Vec::new();
        for _ in 0..4 {
            dense.extend((0..n_genes).map(|gene| gene as f32 + 1.0));
        }
        CsrMatrix::from_dense(&dense, 4, n_genes).unwrap()
    }

    #[test]
    fn a_constant_gene_set_scores_zero() {
        // Every gene has the same value in every cell, so the gene set mean and
        // any control mean coincide whatever the control draw is.
        let matrix = CsrMatrix::from_dense(&[2.0; 24], 4, 6).unwrap();
        let scores = score_genes(&matrix, &[0, 1], 2, 3, 0, &CPU).unwrap();
        assert_eq!(scores.len(), 4);
        assert!(scores.iter().all(|score| score.abs() < 1e-6), "{scores:?}");
    }

    #[test]
    fn the_metal_path_returns_what_the_cpu_path_returns() {
        // A device is an optimisation, never a second algorithm, so on a machine
        // with a GPU the two must agree; on one without, there is nothing to check.
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        let matrix = graded(20);
        let on_cpu = score_genes(&matrix, &[0, 18], 2, 5, 0, &CPU).unwrap();
        let on_metal = score_genes(&matrix, &[0, 18], 2, 5, 0, &metal).unwrap();
        for (cpu, gpu) in on_cpu.iter().zip(&on_metal) {
            assert!((cpu - gpu).abs() < 1e-5, "{cpu} vs {gpu}");
        }
    }

    #[test]
    fn subset_row_means_average_the_selected_columns() {
        let means = subset_row_means(&graded(6), &[0, 5], &CPU).unwrap();
        assert_eq!(means, vec![3.5; 4]);
    }

    #[test]
    fn the_control_set_excludes_the_scored_genes() {
        let bins = vec![0, 0, 0, 1, 1, 1];
        let scored: BTreeSet<u32> = [0, 1].into_iter().collect();
        assert_eq!(control_set(&bins, &scored, 10, 0).unwrap(), vec![2]);
    }

    #[test]
    fn a_bin_smaller_than_ctrl_size_contributes_all_of_itself() {
        let bins = vec![0, 0, 1, 1];
        let scored: BTreeSet<u32> = [0].into_iter().collect();
        assert_eq!(control_set(&bins, &scored, 50, 0).unwrap(), vec![1]);
    }

    #[test]
    fn rejects_input_it_cannot_score() {
        let matrix = graded(6);
        assert!(score_genes(&matrix, &[], 2, 3, 0, &CPU).is_err());
        assert!(score_genes(&matrix, &[9], 2, 3, 0, &CPU).is_err());
        assert!(score_genes(&matrix, &[0], 2, 1, 0, &CPU).is_err());
        // Every gene is scored, so no control gene is left over.
        assert!(score_genes(&matrix, &[0, 1, 2, 3, 4, 5], 2, 3, 0, &CPU).is_err());
        // More bins than genes leaves no genes per bin.
        assert!(score_genes(&matrix, &[0], 2, 100, 0, &CPU).is_err());
    }

    #[test]
    fn a_gene_set_spanning_bins_draws_from_each_of_them() {
        let matrix = graded(20);
        let bins = expression_bins(&column_means(&matrix).unwrap(), 5).unwrap();
        assert_ne!(bins[0], bins[18]);
        let scored: BTreeSet<u32> = [0, 18].into_iter().collect();
        let control = control_set(&bins, &scored, 2, 0).unwrap();
        assert!(control.iter().any(|&gene| bins[gene as usize] == bins[0]));
        assert!(control.iter().any(|&gene| bins[gene as usize] == bins[18]));
        // The most highly expressed gene sits alone in a bin of its own: with
        // `n_bins` bins the divisor is `n_bins - 1`, so the top rank overflows
        // into an extra, one-gene bin. That is scanpy's binning, quirk included.
        assert_eq!(bins.iter().filter(|&&bin| bin == bins[19]).count(), 1);
    }
}
