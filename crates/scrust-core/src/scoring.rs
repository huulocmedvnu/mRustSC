//! Gene-set scoring. Owned by feat/scoring.

use candle_core::Device;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Mean expression of a gene set minus a expression-binned control set, as
/// `scanpy.tl.score_genes`.
pub fn score_genes(
    _matrix: &CsrMatrix,
    _gene_set: &[u32],
    _ctrl_size: usize,
    _n_bins: usize,
    _seed: u64,
    _device: &Device,
) -> Result<Vec<f32>> {
    todo!("feat/scoring")
}
