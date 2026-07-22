use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Sum of stored values and number of positive entries, per row.
///
/// Shared with `normalize_total`, which needs the totals: both reductions come
/// out of the same single pass over the CSR arrays.
pub(crate) fn row_reductions(matrix: &CsrMatrix) -> (Vec<f32>, Vec<usize>) {
    let mut totals = vec![0.0; matrix.n_rows()];
    let mut occupancy = vec![0; matrix.n_rows()];
    let indptr = matrix.indptr();
    for row in 0..matrix.n_rows() {
        let span = indptr[row] as usize..indptr[row + 1] as usize;
        for &value in &matrix.values()[span] {
            totals[row] += value;
            if value > 0.0 {
                occupancy[row] += 1;
            }
        }
    }
    (totals, occupancy)
}

/// Sum of stored values and number of positive entries, per column.
fn column_reductions(matrix: &CsrMatrix) -> (Vec<f32>, Vec<usize>) {
    let mut totals = vec![0.0; matrix.n_cols()];
    let mut occupancy = vec![0; matrix.n_cols()];
    for (&column, &value) in matrix.indices().iter().zip(matrix.values()) {
        totals[column as usize] += value;
        if value > 0.0 {
            occupancy[column as usize] += 1;
        }
    }
    (totals, occupancy)
}

/// Apply whichever of the two thresholds the caller supplied.
///
/// scanpy accepts exactly one criterion per call and raises otherwise, so the
/// same either-or check serves cells and genes.
fn threshold_mask(
    totals: &[f32],
    occupancy: &[usize],
    min_occupancy: Option<usize>,
    min_counts: Option<f32>,
    parameters: &'static str,
) -> Result<Vec<bool>> {
    match (min_occupancy, min_counts) {
        (Some(minimum), None) => Ok(occupancy.iter().map(|&n| n >= minimum).collect()),
        (None, Some(minimum)) => Ok(totals.iter().map(|&total| total >= minimum).collect()),
        (Some(_), Some(_)) => Err(Error::parameter(
            parameters,
            "exactly one of the two",
            "both",
        )),
        (None, None) => Err(Error::parameter(
            parameters,
            "exactly one of the two",
            "neither",
        )),
    }
}

/// Cells to keep, as `scanpy.pp.filter_cells`.
///
/// Index arithmetic over the CSR arrays only, so it stays on the CPU: a single
/// memory-bound pass has nothing for the GPU to win back over the transfer.
pub fn filter_cells(
    matrix: &CsrMatrix,
    min_genes: Option<usize>,
    min_counts: Option<f32>,
) -> Result<Vec<bool>> {
    let (totals, occupancy) = row_reductions(matrix);
    threshold_mask(
        &totals,
        &occupancy,
        min_genes,
        min_counts,
        "min_genes/min_counts",
    )
}

/// Genes to keep, as `scanpy.pp.filter_genes`.
pub fn filter_genes(
    matrix: &CsrMatrix,
    min_cells: Option<usize>,
    min_counts: Option<f32>,
) -> Result<Vec<bool>> {
    let (totals, occupancy) = column_reductions(matrix);
    threshold_mask(
        &totals,
        &occupancy,
        min_cells,
        min_counts,
        "min_cells/min_counts",
    )
}

/// Keep only the rows and columns flagged in the masks.
pub fn subset(matrix: &CsrMatrix, keep_rows: &[bool], keep_cols: &[bool]) -> Result<CsrMatrix> {
    if keep_rows.len() != matrix.n_rows() {
        return Err(Error::shape(
            format!("{} row flags", matrix.n_rows()),
            format!("{} row flags", keep_rows.len()),
        ));
    }
    if keep_cols.len() != matrix.n_cols() {
        return Err(Error::shape(
            format!("{} column flags", matrix.n_cols()),
            format!("{} column flags", keep_cols.len()),
        ));
    }

    // Position of each kept column in the reduced matrix; dropped columns get None.
    let mut renumbered = Vec::with_capacity(keep_cols.len());
    let mut n_cols = 0u32;
    for &keep in keep_cols {
        renumbered.push(keep.then(|| {
            let position = n_cols;
            n_cols += 1;
            position
        }));
    }

    let indptr_in = matrix.indptr();
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..matrix.n_rows() {
        if !keep_rows[row] {
            continue;
        }
        for entry in indptr_in[row] as usize..indptr_in[row + 1] as usize {
            if let Some(column) = renumbered[matrix.indices()[entry] as usize] {
                indices.push(column);
                values.push(matrix.values()[entry]);
            }
        }
        indptr.push(values.len() as u32);
    }
    CsrMatrix::new(indptr, indices, values, n_cols as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 6 cells x 5 genes matrix the scanpy reference script ran on.
    const DENSE: [f32; 30] = [
        2.0, 0.0, 0.0, 1.0, 2.0, //
        0.0, 0.0, 0.0, 0.0, 1.0, //
        4.0, 0.0, 1.0, 0.0, 1.0, //
        2.0, 1.0, 0.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 0.0, 1.0, //
        2.0, 0.0, 1.0, 1.0, 1.0,
    ];

    fn reference() -> CsrMatrix {
        CsrMatrix::from_dense(&DENSE, 6, 5).unwrap()
    }

    fn tiny() -> CsrMatrix {
        // Row 1 is all zero and column 1 is empty.
        CsrMatrix::from_dense(&[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 3, 3).unwrap()
    }

    #[test]
    fn hand_checked_cell_masks() {
        let matrix = tiny();
        assert_eq!(
            filter_cells(&matrix, Some(2), None).unwrap(),
            vec![true, false, true]
        );
        assert_eq!(
            filter_cells(&matrix, None, Some(4.0)).unwrap(),
            vec![false, false, true]
        );
        // The all-zero row survives only a zero threshold.
        assert_eq!(
            filter_cells(&matrix, Some(0), None).unwrap(),
            vec![true, true, true]
        );
    }

    #[test]
    fn hand_checked_gene_masks() {
        let matrix = tiny();
        assert_eq!(
            filter_genes(&matrix, Some(2), None).unwrap(),
            vec![true, false, true]
        );
        assert_eq!(
            filter_genes(&matrix, None, Some(5.0)).unwrap(),
            vec![false, false, true]
        );
    }

    #[test]
    fn matches_scanpy_masks() {
        let matrix = reference();
        assert_eq!(
            filter_cells(&matrix, Some(3), None).unwrap(),
            vec![true, false, true, true, false, true]
        );
        assert_eq!(
            filter_cells(&matrix, None, Some(4.0)).unwrap(),
            vec![true, false, true, true, false, true]
        );
        assert_eq!(
            filter_genes(&matrix, Some(3), None).unwrap(),
            vec![true, false, true, false, true]
        );
        assert_eq!(
            filter_genes(&matrix, None, Some(4.0)).unwrap(),
            vec![true, false, false, false, true]
        );
    }

    #[test]
    fn rejects_zero_or_two_criteria() {
        let matrix = tiny();
        assert!(filter_cells(&matrix, None, None).is_err());
        assert!(filter_cells(&matrix, Some(1), Some(1.0)).is_err());
        assert!(filter_genes(&matrix, None, None).is_err());
        assert!(filter_genes(&matrix, Some(1), Some(1.0)).is_err());
    }

    #[test]
    fn subset_renumbers_columns() {
        let matrix = tiny();
        let reduced = subset(&matrix, &[true, false, true], &[true, false, true]).unwrap();
        assert_eq!(reduced.n_rows(), 2);
        assert_eq!(reduced.n_cols(), 2);
        assert_eq!(reduced.densify_rows(0, 2), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn filter_then_subset_round_trip() {
        let matrix = reference();
        let keep_rows = filter_cells(&matrix, Some(3), None).unwrap();
        let keep_cols = filter_genes(&matrix, None, Some(4.0)).unwrap();
        let reduced = subset(&matrix, &keep_rows, &keep_cols).unwrap();
        assert_eq!(reduced.n_rows(), 4);
        assert_eq!(reduced.n_cols(), 2);
        assert_eq!(
            reduced.densify_rows(0, 4),
            vec![2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        );
    }

    #[test]
    fn subset_can_drop_everything() {
        let matrix = tiny();
        let reduced = subset(&matrix, &[false; 3], &[false; 3]).unwrap();
        assert_eq!(reduced.n_rows(), 0);
        assert_eq!(reduced.n_cols(), 0);
        assert_eq!(reduced.nnz(), 0);
    }

    #[test]
    fn subset_rejects_mismatched_masks() {
        let matrix = tiny();
        assert!(subset(&matrix, &[true, true], &[true; 3]).is_err());
        assert!(subset(&matrix, &[true; 3], &[true, true]).is_err());
    }
}
