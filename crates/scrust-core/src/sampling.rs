//! Subsampling cells and thinning counts. Owned by feat/sampling.

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Indices of the cells to keep.
pub fn subsample(_n_cells: usize, _n_keep: usize, _replace: bool, _seed: u64) -> Result<Vec<u32>> {
    todo!("feat/sampling")
}

/// Thin each cell to at most `counts_per_cell` by multivariate hypergeometric
/// sampling, as `scanpy.pp.downsample_counts`.
pub fn downsample_counts(
    _matrix: &CsrMatrix,
    _counts_per_cell: Option<f32>,
    _total_counts: Option<f32>,
    _replace: bool,
    _seed: u64,
) -> Result<CsrMatrix> {
    todo!("feat/sampling")
}
