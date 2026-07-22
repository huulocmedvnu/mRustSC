//! Partition-based graph abstraction. Owned by feat/paga.

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// The abstracted graph over groups.
#[derive(Debug, Clone)]
pub struct AbstractedGraph {
    /// `(n_groups, n_groups)` connectivity, symmetric with a zero diagonal.
    pub connectivities: Vec<f32>,
    pub connectivities_tree: Vec<f32>,
    pub n_groups: usize,
}

/// PAGA connectivities, as `scanpy.tl.paga` with `model="v1.2"`.
pub fn paga(_graph: &CsrMatrix, _groups: &[u32], _n_groups: usize) -> Result<AbstractedGraph> {
    todo!("feat/paga")
}
