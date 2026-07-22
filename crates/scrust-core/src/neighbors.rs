use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Nearest neighbours of every cell, excluding the cell itself.
#[derive(Debug, Clone)]
pub struct KnnGraph {
    /// Neighbour ids, `(n_cells, k)`, nearest first.
    pub indices: Array2<u32>,
    /// Distances to those neighbours, `(n_cells, k)`.
    pub distances: Array2<f32>,
}

/// Exact k nearest neighbours by Euclidean distance.
///
/// Exact rather than approximate: on the GPU the distance matrix is one tiled
/// matmul, so the usual reason to approximate does not apply at this scale.
pub fn knn(_embedding: &Array2<f32>, _k: usize, _device: &Device) -> Result<KnnGraph> {
    todo!("feat/neighbors")
}

/// UMAP's fuzzy simplicial set, the weighted graph `scanpy.pp.neighbors` stores
/// in `obsp["connectivities"]`.
pub fn connectivities(_graph: &KnnGraph) -> Result<CsrMatrix> {
    todo!("feat/neighbors")
}
