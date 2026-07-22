use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;

/// Per-sample scaling factors by the median-of-ratios method, normalised to a
/// geometric mean of one.
pub fn size_factors_median_of_ratios(_counts: &Array2<f32>) -> Result<Vec<f32>> {
    todo!("feat/dispersion")
}

/// Per-gene negative binomial dispersions by the method of moments, taking the
/// design's residual degrees of freedom into account.
pub fn dispersions_method_of_moments(
    _counts: &Array2<f32>,
    _size_factors: &[f32],
    _design: &Array2<f32>,
    _device: &Device,
) -> Result<Vec<f32>> {
    todo!("feat/dispersion")
}

/// Shrink gene-wise dispersions towards a fitted mean-dispersion trend.
pub fn shrink_towards_trend(
    _dispersions: &[f32],
    _means: &[f32],
    _shrinkage_weight: f32,
) -> Result<Vec<f32>> {
    todo!("feat/dispersion")
}
