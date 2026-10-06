use candle_core::Device;

use crate::error::{Error, Result};
use crate::preprocess::filter::row_reductions;
use crate::sparse::CsrMatrix;

/// Median of the per-cell totals, or `None` for a matrix without cells.
///
/// scanpy's CSR path takes the median over *all* cells, zero-count cells
/// included, so this does too.
fn median(totals: &[f32]) -> Option<f32> {
    if totals.is_empty() {
        return None;
    }
    let mut sorted = totals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    })
}

/// Scale every cell to the same total count, as `scanpy.pp.normalize_total`.
///
/// `target_sum` of `None` uses the median total count across cells.
///
/// A per-row rescale keeps the sparsity pattern untouched, so this works
/// straight on the CSR values and stays on the CPU; `device` is accepted only to
/// keep the call shape uniform across the preprocessing steps.
pub fn normalize_total(
    matrix: &CsrMatrix,
    target_sum: Option<f32>,
    _device: &Device,
) -> Result<CsrMatrix> {
    let (totals, _) = row_reductions(matrix);
    let Some(target) = target_sum.or_else(|| median(&totals)) else {
        return Ok(matrix.clone()); // no cells, nothing to scale
    };
    if target <= 0.0 || !target.is_finite() {
        return Err(Error::parameter("target_sum", "a positive count", target));
    }

    let mut values = matrix.values().to_vec();
    let indptr = matrix.indptr();
    for (row, &total) in totals.iter().enumerate() {
        // A cell with no counts has no factor to speak of; scanpy leaves it as it
        // is rather than dividing by zero.
        if total <= 0.0 {
            continue;
        }
        let factor = target / total;
        for value in &mut values[indptr[row] as usize..indptr[row + 1] as usize] {
            *value *= factor;
        }
    }
    CsrMatrix::new(
        matrix.indptr().to_vec(),
        matrix.indices().to_vec(),
        values,
        matrix.n_cols(),
    )
}

/// Natural log of one plus each stored value, as `scanpy.pp.log1p`.
///
/// `ln(1 + 0) == 0`, so the zeros stay implicit and the result stays sparse.
pub fn log1p(matrix: &CsrMatrix) -> Result<CsrMatrix> {
    let values = matrix.values().iter().map(|v| v.ln_1p()).collect();
    CsrMatrix::new(
        matrix.indptr().to_vec(),
        matrix.indices().to_vec(),
        values,
        matrix.n_cols(),
    )
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

    /// `sc.pp.normalize_total` on `DENSE`; per-cell totals are 5, 1, 6, 4, 2, 5
    /// so the median target is 4.5.
    const NORMALIZED: [f32; 30] = [
        1.8, 0.0, 0.0, 0.9, 1.8, //
        0.0, 0.0, 0.0, 0.0, 4.5, //
        3.0, 0.0, 0.75, 0.0, 0.75, //
        2.25, 1.125, 0.0, 0.0, 1.125, //
        0.0, 0.0, 2.25, 0.0, 2.25, //
        1.8, 0.0, 0.9, 0.9, 0.9,
    ];

    /// `sc.pp.normalize_total(target_sum=10)` on `DENSE`.
    const NORMALIZED_TARGET_10: [f32; 30] = [
        4.0, 0.0, 0.0, 2.0, 4.0, //
        0.0, 0.0, 0.0, 0.0, 10.0, //
        6.6666665, 0.0, 1.6666666, 0.0, 1.6666666, //
        5.0, 2.5, 0.0, 0.0, 2.5, //
        0.0, 0.0, 5.0, 0.0, 5.0, //
        4.0, 0.0, 2.0, 2.0, 2.0,
    ];

    /// `sc.pp.log1p` applied to `NORMALIZED`.
    const LOGGED: [f32; 30] = [
        1.0296195, 0.0, 0.0, 0.64185387, 1.0296195, //
        0.0, 0.0, 0.0, 0.0, 1.704748, //
        1.3862944, 0.0, 0.5596158, 0.0, 0.5596158, //
        1.178655, 0.7537718, 0.0, 0.0, 0.7537718, //
        0.0, 0.0, 1.178655, 0.0, 1.178655, //
        1.0296195, 0.0, 0.64185387, 0.64185387, 0.64185387,
    ];

    const RTOL: f32 = 1e-5;

    fn reference() -> CsrMatrix {
        CsrMatrix::from_dense(&DENSE, 6, 5).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= RTOL * e.abs().max(1.0),
                "element {i}: {a} != {e}"
            );
        }
    }

    #[test]
    fn hand_checked_totals() {
        // Two cells with totals 3 and 6; the median target is 4.5.
        let matrix = CsrMatrix::from_dense(&[1.0, 2.0, 0.0, 2.0, 0.0, 4.0], 2, 3).unwrap();
        let out = normalize_total(&matrix, None, &Device::Cpu).unwrap();
        assert_close(&out.densify_rows(0, 2), &[1.5, 3.0, 0.0, 1.5, 0.0, 3.0]);
    }

    #[test]
    fn zero_total_cell_is_left_alone() {
        let matrix = CsrMatrix::from_dense(&[1.0, 1.0, 0.0, 0.0, 4.0, 0.0], 3, 2).unwrap();
        let out = normalize_total(&matrix, Some(4.0), &Device::Cpu).unwrap();
        let dense = out.densify_rows(0, 3);
        assert_close(&dense, &[2.0, 2.0, 0.0, 0.0, 4.0, 0.0]);
        assert!(dense.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn keeps_the_sparsity_pattern() {
        let matrix = reference();
        let out = normalize_total(&matrix, None, &Device::Cpu).unwrap();
        assert_eq!(out.indptr(), matrix.indptr());
        assert_eq!(out.indices(), matrix.indices());
    }

    #[test]
    fn matches_scanpy_normalize_total() {
        let matrix = reference();
        let median_target = normalize_total(&matrix, None, &Device::Cpu).unwrap();
        assert_close(&median_target.densify_rows(0, 6), &NORMALIZED);

        let fixed_target = normalize_total(&matrix, Some(10.0), &Device::Cpu).unwrap();
        assert_close(&fixed_target.densify_rows(0, 6), &NORMALIZED_TARGET_10);
    }

    #[test]
    fn rejects_a_non_positive_target() {
        let matrix = reference();
        assert!(normalize_total(&matrix, Some(0.0), &Device::Cpu).is_err());
        assert!(normalize_total(&matrix, Some(-1.0), &Device::Cpu).is_err());
    }

    #[test]
    fn hand_checked_log1p() {
        let matrix = CsrMatrix::from_dense(&[0.0, 1.0, std::f32::consts::E - 1.0], 1, 3).unwrap();
        let out = log1p(&matrix).unwrap();
        assert_close(&out.densify_rows(0, 1), &[0.0, std::f32::consts::LN_2, 1.0]);
    }

    #[test]
    fn matches_scanpy_log1p() {
        let normalized = CsrMatrix::from_dense(&NORMALIZED, 6, 5).unwrap();
        let out = log1p(&normalized).unwrap();
        assert_eq!(out.nnz(), normalized.nnz()); // still sparse
        assert_close(&out.densify_rows(0, 6), &LOGGED);
    }

    #[test]
    fn median_of_even_and_odd_lengths() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
