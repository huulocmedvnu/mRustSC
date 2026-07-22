use ndarray::Array2;
use scrust_core::error::Result;

use crate::context::MetalContext;

/// Attractive and repulsive gradient of the t-SNE objective for one iteration.
///
/// Returns the gradient and the normalisation constant Z of the low-dimensional
/// affinities, which the caller needs to scale the repulsive term.
pub fn tsne_gradient(
    _context: &MetalContext,
    _embedding: &Array2<f32>,
    _affinity_indptr: &[u32],
    _affinity_indices: &[u32],
    _affinity_values: &[f32],
    _exaggeration: f32,
) -> Result<(Array2<f32>, f32)> {
    todo!("feat/tsne-kernel")
}
