//! Quality-control metrics. Owned by feat/qc-metrics.

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Per-cell metrics, in the order `scanpy.pp.calculate_qc_metrics` reports them.
#[derive(Debug, Clone)]
pub struct CellMetrics {
    pub n_genes_by_counts: Vec<u32>,
    pub total_counts: Vec<f32>,
    /// Cumulative fraction in the top-N genes, one row per requested N.
    pub pct_counts_in_top: Vec<Vec<f32>>,
    /// Total counts falling in each requested gene subset.
    pub subset_totals: Vec<Vec<f32>>,
}

/// Per-gene metrics.
#[derive(Debug, Clone)]
pub struct GeneMetrics {
    pub n_cells_by_counts: Vec<u32>,
    pub mean_counts: Vec<f32>,
    pub pct_dropout_by_counts: Vec<f32>,
    pub total_counts: Vec<f32>,
}

/// Both halves in one pass over the stored entries.
pub fn qc_metrics(
    _matrix: &CsrMatrix,
    _percent_top: &[usize],
    _gene_subsets: &[Vec<bool>],
) -> Result<(CellMetrics, GeneMetrics)> {
    todo!("feat/qc-metrics")
}

/// Square root of every stored value.
pub fn sqrt(_matrix: &CsrMatrix) -> Result<CsrMatrix> {
    todo!("feat/qc-metrics")
}
