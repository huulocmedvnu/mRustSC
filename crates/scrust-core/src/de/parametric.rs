//! t-test and logistic-regression differential expression. Owned by feat/de-methods.

use candle_core::Device;

use crate::de::wilcoxon::GroupComparison;
use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Welch's t-test per group, as scanpy's `method="t-test"`.
pub fn t_test(
    _matrix: &CsrMatrix,
    _group_labels: &[u32],
    _n_groups: usize,
    _reference: Option<u32>,
    _device: &Device,
) -> Result<GroupComparison> {
    todo!("feat/de-methods")
}

/// scanpy's `method="t-test_overestim_var"`, which uses the group size in place
/// of the Welch degrees of freedom.
pub fn t_test_overestimated_variance(
    _matrix: &CsrMatrix,
    _group_labels: &[u32],
    _n_groups: usize,
    _reference: Option<u32>,
    _device: &Device,
) -> Result<GroupComparison> {
    todo!("feat/de-methods")
}

/// Multinomial logistic regression coefficients as scores, as scanpy's
/// `method="logreg"`.
pub fn logistic_regression(
    _matrix: &CsrMatrix,
    _group_labels: &[u32],
    _n_groups: usize,
    _max_iterations: usize,
    _device: &Device,
) -> Result<GroupComparison> {
    todo!("feat/de-methods")
}
