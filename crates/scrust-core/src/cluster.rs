//! Community detection on the neighbour graph. Owned by feat/leiden.

use candle_core::Device;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Community labels plus the modularity the partition achieves.
#[derive(Debug, Clone)]
pub struct Partition {
    pub labels: Vec<u32>,
    pub modularity: f64,
    pub n_communities: usize,
}

/// Leiden community detection, as `scanpy.tl.leiden`.
pub fn leiden(
    _graph: &CsrMatrix,
    _resolution: f64,
    _n_iterations: usize,
    _seed: u64,
    _device: &Device,
) -> Result<Partition> {
    todo!("feat/leiden")
}

/// Louvain community detection, as `scanpy.tl.louvain`.
pub fn louvain(
    _graph: &CsrMatrix,
    _resolution: f64,
    _seed: u64,
    _device: &Device,
) -> Result<Partition> {
    todo!("feat/leiden")
}

/// Newman modularity of an existing labelling.
pub fn modularity(_graph: &CsrMatrix, _labels: &[u32], _resolution: f64) -> Result<f64> {
    todo!("feat/leiden")
}
