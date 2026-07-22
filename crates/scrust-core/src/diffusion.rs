//! Diffusion maps and pseudotime. Owned by feat/diffusion.

use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Eigenvectors and eigenvalues of the diffusion operator.
#[derive(Debug, Clone)]
pub struct DiffusionMap {
    /// `(n_cells, n_comps)`, the first component dropped as scanpy does.
    pub embedding: Array2<f32>,
    pub eigenvalues: Vec<f32>,
}

/// Diffusion map of a connectivity graph, as `scanpy.tl.diffmap`.
pub fn diffmap(_graph: &CsrMatrix, _n_comps: usize, _device: &Device) -> Result<DiffusionMap> {
    todo!("feat/diffusion")
}

/// Diffusion pseudotime from a root cell, as `scanpy.tl.dpt`.
pub fn dpt(_map: &DiffusionMap, _root: usize, _n_dcs: usize) -> Result<Vec<f32>> {
    todo!("feat/diffusion")
}
