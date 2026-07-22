use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Cells to keep, as `scanpy.pp.filter_cells`.
pub fn filter_cells(
    _matrix: &CsrMatrix,
    _min_genes: Option<usize>,
    _min_counts: Option<f32>,
) -> Result<Vec<bool>> {
    todo!("feat/filter")
}

/// Genes to keep, as `scanpy.pp.filter_genes`.
pub fn filter_genes(
    _matrix: &CsrMatrix,
    _min_cells: Option<usize>,
    _min_counts: Option<f32>,
) -> Result<Vec<bool>> {
    todo!("feat/filter")
}

/// Keep only the rows and columns flagged in the masks.
pub fn subset(_matrix: &CsrMatrix, _keep_rows: &[bool], _keep_cols: &[bool]) -> Result<CsrMatrix> {
    todo!("feat/filter")
}
