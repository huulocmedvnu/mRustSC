use ndarray::Array2;
use scrust_core::error::Result;
use scrust_core::neighbors::KnnGraph;

use crate::context::MetalContext;

/// Exact k nearest neighbours in one pass over tiles of the distance matrix.
///
/// Selection is what needs a kernel: candle can produce the distances, but
/// keeping the k smallest per row without materialising an `(n, n)` matrix
/// cannot be expressed as tensor algebra.
pub fn knn_metal(_context: &MetalContext, _embedding: &Array2<f32>, _k: usize) -> Result<KnnGraph> {
    todo!("feat/knn-kernel")
}
