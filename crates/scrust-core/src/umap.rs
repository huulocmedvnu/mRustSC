use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Layout parameters, named as in `scanpy.tl.umap`.
#[derive(Debug, Clone)]
pub struct UmapParams {
    pub n_components: usize,
    pub n_epochs: usize,
    pub min_dist: f32,
    pub spread: f32,
    pub learning_rate: f32,
    pub negative_sample_rate: usize,
    pub seed: u64,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_epochs: 200,
            min_dist: 0.5,
            spread: 1.0,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            seed: 0,
        }
    }
}

/// Optimise the UMAP layout of a connectivity graph.
pub fn umap(
    _connectivities: &CsrMatrix,
    _params: &UmapParams,
    _device: &Device,
) -> Result<Array2<f32>> {
    todo!("feat/umap")
}
