use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;

/// Layout parameters, named as in `scanpy.tl.tsne`.
#[derive(Debug, Clone)]
pub struct TsneParams {
    pub n_components: usize,
    pub perplexity: f32,
    pub early_exaggeration: f32,
    pub learning_rate: f32,
    pub n_iterations: usize,
    pub seed: u64,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            n_iterations: 1000,
            seed: 0,
        }
    }
}

/// t-SNE embedding of a cells-by-features matrix, usually PCA coordinates.
pub fn tsne(
    _embedding: &Array2<f32>,
    _params: &TsneParams,
    _device: &Device,
) -> Result<Array2<f32>> {
    todo!("feat/tsne")
}
