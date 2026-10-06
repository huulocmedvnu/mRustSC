//! Quality-control metrics. Owned by feat/qc-metrics.
//!
//! The per-cell and the per-gene metrics are two reductions over the same stored
//! entries, so both are accumulated in one sweep: the matrix is read once and
//! never densified.
//!
//! The sweep stays on the CPU. It is a scatter-add into per-gene accumulators
//! plus a per-cell partial selection, neither of which is tensor algebra; the
//! only way to phrase it for a device would be to densify an
//! `(n_cells, n_genes)` block that is 90-95% zeros, which costs more than the
//! arithmetic it would parallelise. That is also why this function, alone among
//! the algorithms, takes no `Device`.

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Per-cell metrics, in the order `scanpy.pp.calculate_qc_metrics` reports them.
#[derive(Debug, Clone)]
pub struct CellMetrics {
    pub n_genes_by_counts: Vec<u32>,
    pub total_counts: Vec<f32>,
    /// Cumulative fraction in the top-N genes, one row per requested N.
    pub pct_counts_in_top: Vec<Vec<f32>>,
    /// Total counts falling in each requested gene subset.
    pub subset_totals: Vec<Vec<f32>>,
}

/// Per-gene metrics.
#[derive(Debug, Clone)]
pub struct GeneMetrics {
    pub n_cells_by_counts: Vec<u32>,
    pub mean_counts: Vec<f32>,
    pub pct_dropout_by_counts: Vec<f32>,
    pub total_counts: Vec<f32>,
}

/// Both halves in one pass over the stored entries.
///
/// `percent_top` is 1-indexed, as in scanpy: `50` asks for the fraction of a
/// cell's counts held by its 50 highest-expressed genes. `gene_subsets` carries
/// one flag per gene per subset, and yields the totals behind `pct_counts_mt`
/// and friends.
///
/// Peak memory is `O(n_cells * (percent_top.len() + gene_subsets.len()) + n_genes)`
/// on top of the matrix itself: nothing here is proportional to `n_cells * n_genes`.
pub fn qc_metrics(
    matrix: &CsrMatrix,
    percent_top: &[usize],
    gene_subsets: &[Vec<bool>],
) -> Result<(CellMetrics, GeneMetrics)> {
    validate(matrix, percent_top, gene_subsets)?;

    let n_cells = matrix.n_rows();
    let n_genes = matrix.n_cols();
    let indptr = matrix.indptr();
    let indices = matrix.indices();
    let values = matrix.values();

    let mut cells = CellMetrics {
        n_genes_by_counts: vec![0; n_cells],
        total_counts: vec![0.0; n_cells],
        pct_counts_in_top: vec![vec![0.0; n_cells]; percent_top.len()],
        subset_totals: vec![vec![0.0; n_cells]; gene_subsets.len()],
    };
    let mut n_cells_by_counts = vec![0u32; n_genes];
    let mut gene_totals = vec![0.0f64; n_genes];

    // Scratch buffers live outside the loop so a matrix with a million cells
    // does not mean a million allocations.
    let deepest = percent_top.iter().copied().max().unwrap_or(0);
    let mut subset_sums = vec![0.0f64; gene_subsets.len()];
    let mut ranked = Vec::with_capacity(deepest);

    for cell in 0..n_cells {
        let row = indptr[cell] as usize..indptr[cell + 1] as usize;
        let mut total = 0.0f64;
        let mut expressed = 0u32;
        subset_sums.fill(0.0);

        for entry in row.clone() {
            let value = values[entry];
            // scanpy drops explicitly stored zeros before it counts anything, so
            // one counts as neither an expressed gene nor an occupied cell.
            if value == 0.0 {
                continue;
            }
            let gene = indices[entry] as usize;
            expressed += 1;
            total += value as f64;
            n_cells_by_counts[gene] += 1;
            gene_totals[gene] += value as f64;
            for (sum, subset) in subset_sums.iter_mut().zip(gene_subsets) {
                if subset[gene] {
                    *sum += value as f64;
                }
            }
        }

        cells.n_genes_by_counts[cell] = expressed;
        cells.total_counts[cell] = total as f32;
        for (totals, sum) in cells.subset_totals.iter_mut().zip(&subset_sums) {
            totals[cell] = *sum as f32;
        }

        if deepest == 0 {
            continue;
        }
        rank_and_accumulate(&values[row], deepest, &mut ranked);
        for (fractions, &n) in cells.pct_counts_in_top.iter_mut().zip(percent_top) {
            // A cell expressing fewer than N genes holds all of its counts in
            // them, and one expressing none has no total to divide by — scanpy
            // reports NaN there, which is what 0/0 gives.
            let held = ranked.get(n - 1).or_else(|| ranked.last()).copied();
            fractions[cell] = (held.unwrap_or(0.0) / total) as f32;
        }
    }

    let denominator = n_cells as f64;
    let genes = GeneMetrics {
        mean_counts: gene_totals
            .iter()
            .map(|total| (total / denominator) as f32)
            .collect(),
        pct_dropout_by_counts: n_cells_by_counts
            .iter()
            .map(|&occupied| ((1.0 - occupied as f64 / denominator) * 100.0) as f32)
            .collect(),
        total_counts: gene_totals.iter().map(|&total| total as f32).collect(),
        n_cells_by_counts,
    };
    Ok((cells, genes))
}

/// Square root of every stored value, as `scanpy.pp.sqrt`.
///
/// `sqrt(0) == 0`, so the zeros stay implicit and the result stays sparse. A
/// negative value has no real root; this yields NaN there exactly as
/// `scipy.sparse.csr_matrix.sqrt` does, rather than rejecting a matrix scanpy
/// would accept.
pub fn sqrt(matrix: &CsrMatrix) -> Result<CsrMatrix> {
    let values = matrix.values().iter().map(|value| value.sqrt()).collect();
    CsrMatrix::new(
        matrix.indptr().to_vec(),
        matrix.indices().to_vec(),
        values,
        matrix.n_cols(),
    )
}

fn validate(matrix: &CsrMatrix, percent_top: &[usize], gene_subsets: &[Vec<bool>]) -> Result<()> {
    if let Some(&n) = percent_top.iter().find(|&&n| n == 0 || n > matrix.n_cols()) {
        return Err(Error::parameter(
            "percent_top",
            "between 1 and the number of genes",
            n,
        ));
    }
    if let Some(subset) = gene_subsets
        .iter()
        .find(|subset| subset.len() != matrix.n_cols())
    {
        return Err(Error::shape(
            format!("{} gene flags per subset", matrix.n_cols()),
            format!("{} flags", subset.len()),
        ));
    }
    Ok(())
}

/// Leave the running totals of a cell's `deepest` largest values in `ranked`.
///
/// The partial selection is what `percent_top` costs: `select_nth_unstable` is
/// linear in the row's stored entries, and only the `deepest` values it keeps
/// are then sorted, so a cell costs `O(nnz + deepest * log(deepest))` rather
/// than the `O(nnz * log(nnz))` a full sort of the row would.
fn rank_and_accumulate(row: &[f32], deepest: usize, ranked: &mut Vec<f64>) {
    ranked.clear();
    ranked.extend(row.iter().map(|&value| value as f64));
    if ranked.len() > deepest {
        ranked.select_nth_unstable_by(deepest, |a, b| b.total_cmp(a));
        ranked.truncate(deepest);
    }
    ranked.sort_unstable_by(|a, b| b.total_cmp(a));

    let mut running = 0.0;
    for value in ranked.iter_mut() {
        running += *value;
        *value = running;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 cells x 4 genes, hand-checkable throughout.
    ///
    /// cell 0: 1 3 0 0   total 4, 2 genes, top-1 3/4, top-2 4/4
    /// cell 1: 0 0 0 0   empty
    /// cell 2: 5 0 2 3   total 10, 3 genes, top-1 5/10, top-2 8/10
    const DENSE: [f32; 12] = [
        1.0, 3.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        5.0, 0.0, 2.0, 3.0,
    ];

    fn example() -> CsrMatrix {
        CsrMatrix::from_dense(&DENSE, 3, 4).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            // An undefined metric is expected as NaN, and NaN is never close to itself.
            let close = if e.is_nan() {
                a.is_nan()
            } else {
                (a - e).abs() <= 1e-5 * e.abs().max(1.0)
            };
            assert!(close, "element {i}: {a} != {e}");
        }
    }

    #[test]
    fn per_cell_totals_and_gene_counts() {
        let (cells, _) = qc_metrics(&example(), &[], &[]).unwrap();
        assert_eq!(cells.n_genes_by_counts, vec![2, 0, 3]);
        assert_close(&cells.total_counts, &[4.0, 0.0, 10.0]);
        assert!(cells.pct_counts_in_top.is_empty());
        assert!(cells.subset_totals.is_empty());
    }

    #[test]
    fn per_gene_metrics_include_the_implicit_zeros() {
        let (_, genes) = qc_metrics(&example(), &[], &[]).unwrap();
        assert_eq!(genes.n_cells_by_counts, vec![2, 1, 1, 1]);
        assert_close(&genes.total_counts, &[6.0, 3.0, 2.0, 3.0]);
        assert_close(&genes.mean_counts, &[2.0, 1.0, 2.0 / 3.0, 1.0]);
        // Gene 0 is seen in 2 of 3 cells, the others in 1 of 3.
        let third = 100.0 / 3.0;
        assert_close(
            &genes.pct_dropout_by_counts,
            &[third, 2.0 * third, 2.0 * third, 2.0 * third],
        );
    }

    #[test]
    fn percent_top_is_cumulative_and_one_indexed() {
        let (cells, _) = qc_metrics(&example(), &[1, 2], &[]).unwrap();
        assert_close(&cells.pct_counts_in_top[0], &[0.75, f32::NAN, 0.5]);
        assert_close(&cells.pct_counts_in_top[1], &[1.0, f32::NAN, 0.8]);
    }

    #[test]
    fn a_cell_with_fewer_genes_than_n_holds_everything_in_them() {
        // Cell 0 expresses 2 genes, so its top 4 hold all of its counts.
        let (cells, _) = qc_metrics(&example(), &[4], &[]).unwrap();
        assert_eq!(cells.pct_counts_in_top[0][0], 1.0);
        assert_eq!(cells.pct_counts_in_top[0][2], 1.0);
    }

    #[test]
    fn an_empty_cell_has_no_fraction_to_report() {
        let (cells, _) = qc_metrics(&example(), &[1], &[]).unwrap();
        assert!(cells.pct_counts_in_top[0][1].is_nan());
        assert_eq!(cells.total_counts[1], 0.0);
        assert_eq!(cells.n_genes_by_counts[1], 0);
    }

    #[test]
    fn a_gene_seen_nowhere_drops_out_of_every_cell() {
        let matrix = CsrMatrix::from_dense(&[1.0, 0.0, 2.0, 0.0], 2, 2).unwrap();
        let (_, genes) = qc_metrics(&matrix, &[], &[]).unwrap();
        assert_eq!(genes.n_cells_by_counts[1], 0);
        assert_eq!(genes.total_counts[1], 0.0);
        assert_eq!(genes.mean_counts[1], 0.0);
        assert_eq!(genes.pct_dropout_by_counts[1], 100.0);
    }

    #[test]
    fn gene_subsets_are_summed_per_cell() {
        let first_two = vec![true, true, false, false];
        let last = vec![false, false, false, true];
        let (cells, _) = qc_metrics(&example(), &[], &[first_two, last]).unwrap();
        assert_close(&cells.subset_totals[0], &[4.0, 0.0, 5.0]);
        assert_close(&cells.subset_totals[1], &[0.0, 0.0, 3.0]);
    }

    #[test]
    fn a_stored_zero_counts_as_no_expression() {
        // The same matrix with gene 2 of cell 0 stored explicitly as zero.
        let matrix = CsrMatrix::new(vec![0, 3], vec![0, 1, 2], vec![1.0, 3.0, 0.0], 4).unwrap();
        let (cells, genes) = qc_metrics(&matrix, &[1], &[]).unwrap();
        assert_eq!(cells.n_genes_by_counts, vec![2]);
        assert_eq!(genes.n_cells_by_counts[2], 0);
        assert_close(&cells.pct_counts_in_top[0], &[0.75]);
    }

    #[test]
    fn rejects_positions_outside_the_gene_range() {
        assert!(qc_metrics(&example(), &[0], &[]).is_err());
        assert!(qc_metrics(&example(), &[5], &[]).is_err());
    }

    #[test]
    fn rejects_a_subset_that_is_not_one_flag_per_gene() {
        assert!(qc_metrics(&example(), &[], &[vec![true, false]]).is_err());
    }

    #[test]
    fn sqrt_keeps_the_sparsity_pattern() {
        let matrix = example();
        let rooted = sqrt(&matrix).unwrap();
        assert_eq!(rooted.indptr(), matrix.indptr());
        assert_eq!(rooted.indices(), matrix.indices());
        assert_close(
            &rooted.densify_rows(0, 3),
            &[
                1.0,
                3.0f32.sqrt(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0f32.sqrt(),
                0.0,
                2.0f32.sqrt(),
                3.0f32.sqrt(),
            ],
        );
    }
}
