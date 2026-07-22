use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Principal components of a cells-by-genes matrix.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Cell coordinates, `(n_cells, n_components)` — AnnData's `obsm["X_pca"]`.
    pub embedding: Array2<f32>,
    /// Gene loadings, `(n_components, n_genes)` — AnnData's `varm["PCs"]` transposed.
    pub components: Array2<f32>,
    pub explained_variance: Vec<f32>,
    pub explained_variance_ratio: Vec<f32>,
}

/// Truncated PCA by randomised SVD, as `scanpy.pp.pca`.
///
/// Randomised range finding turns the decomposition into a few large matmuls,
/// which is what makes it worth moving to the GPU.
pub fn pca(
    _matrix: &CsrMatrix,
    _n_components: usize,
    _zero_center: bool,
    _seed: u64,
    _device: &Device,
) -> Result<PcaResult> {
    todo!("feat/pca")
}
