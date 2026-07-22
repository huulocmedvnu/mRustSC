//! Dendrograms, force-directed layouts and densities. Owned by feat/layout.

use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// A merge tree in the form `scipy.cluster.hierarchy` produces.
#[derive(Debug, Clone)]
pub struct Dendrogram {
    /// `(n_groups - 1, 4)` linkage rows: left, right, distance, size.
    pub linkage: Vec<[f64; 4]>,
    pub leaf_order: Vec<u32>,
}

/// Average-linkage clustering of group centroids, as `scanpy.tl.dendrogram`.
pub fn dendrogram(_centroids: &Array2<f32>) -> Result<Dendrogram> {
    todo!("feat/layout")
}

/// ForceAtlas2 layout of the neighbour graph, as `scanpy.tl.draw_graph`.
pub fn force_directed_layout(
    _graph: &CsrMatrix,
    _n_iterations: usize,
    _seed: u64,
    _device: &Device,
) -> Result<Array2<f32>> {
    todo!("feat/layout")
}

/// Gaussian kernel density of cells in an embedding, scaled to [0, 1].
pub fn embedding_density(_embedding: &Array2<f32>, _device: &Device) -> Result<Vec<f32>> {
    todo!("feat/layout")
}
