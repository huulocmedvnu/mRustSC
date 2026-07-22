use ndarray::Array1;

use crate::de::glm::GlmFit;
use crate::error::Result;

/// A test applied to one contrast of a fit.
#[derive(Debug, Clone)]
pub struct TestStatistics {
    pub statistic: Array1<f32>,
    pub p_values: Array1<f32>,
    /// Contrast estimate on the natural-log scale.
    pub effect: Array1<f32>,
    pub effect_standard_error: Array1<f32>,
}

/// Wald test of `contrast @ beta == 0`, using the normal approximation.
pub fn wald_test(_fit: &GlmFit, _contrast: &[f32]) -> Result<TestStatistics> {
    todo!("feat/hypothesis")
}
