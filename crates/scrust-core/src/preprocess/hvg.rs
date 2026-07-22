use candle_core::Device;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Which dispersion definition to use, matching scanpy's `flavor` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvgFlavor {
    /// Binned normalised dispersion on log data, scanpy's default.
    Seurat,
    /// Loess-free variance-stabilising ranking on raw counts.
    CellRanger,
}

/// Per-gene statistics behind the highly-variable flag.
#[derive(Debug, Clone)]
pub struct HighlyVariableGenes {
    pub means: Vec<f32>,
    pub dispersions: Vec<f32>,
    pub normalised_dispersions: Vec<f32>,
    pub highly_variable: Vec<bool>,
}

/// Flag the most variable genes, as `scanpy.pp.highly_variable_genes`.
pub fn highly_variable_genes(
    _matrix: &CsrMatrix,
    _n_top_genes: usize,
    _flavor: HvgFlavor,
    _device: &Device,
) -> Result<HighlyVariableGenes> {
    todo!("feat/hvg")
}
