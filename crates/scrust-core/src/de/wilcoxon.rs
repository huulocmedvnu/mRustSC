use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// One row per group, one column per gene — the layout scanpy's
/// `rank_genes_groups` result is built from.
#[derive(Debug, Clone)]
pub struct GroupComparison {
    pub scores: Array2<f32>,
    pub p_values: Array2<f32>,
    pub adjusted_p_values: Array2<f32>,
    pub log2_fold_changes: Array2<f32>,
}

/// Rank-sum test of each group against the rest, as scanpy's default
/// `rank_genes_groups(method="wilcoxon")`.
///
/// `group_labels` holds one group id per cell; `reference` of `None` compares
/// each group against all other cells.
pub fn rank_genes_groups_wilcoxon(
    _matrix: &CsrMatrix,
    _group_labels: &[u32],
    _n_groups: usize,
    _reference: Option<u32>,
    _tie_correct: bool,
    _device: &Device,
) -> Result<GroupComparison> {
    todo!("feat/wilcoxon")
}
