use candle_core::Device;
use ndarray::{Array2, Array3};

use crate::error::Result;

/// A per-gene negative binomial GLM fit on the natural-log scale.
#[derive(Debug, Clone)]
pub struct GlmFit {
    /// `(n_genes, n_coefficients)`.
    pub coefficients: Array2<f32>,
    /// `(n_genes, n_coefficients, n_coefficients)`.
    pub covariance: Array3<f32>,
    pub dispersions: Vec<f32>,
    /// `(n_genes, n_samples)`.
    pub fitted_means: Array2<f32>,
    pub converged: Vec<bool>,
    pub n_iterations: usize,
}

/// Fit one GLM per gene by iteratively reweighted least squares.
///
/// All genes advance through the same iteration as one batch: tens of thousands
/// of tiny identical problems is exactly the shape a GPU is good at.
pub fn fit_negative_binomial(
    _counts: &Array2<f32>,
    _design: &Array2<f32>,
    _size_factors: &[f32],
    _dispersions: &[f32],
    _max_iterations: usize,
    _tolerance: f32,
    _device: &Device,
) -> Result<GlmFit> {
    todo!("feat/glm")
}
