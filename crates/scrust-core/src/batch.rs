//! Removing unwanted variation. Owned by feat/regress-combat.

use candle_core::Device;
use ndarray::Array2;

use crate::error::Result;

/// Regress every gene on the covariates and return the residuals.
///
/// One small least-squares problem per gene, all sharing a design: the batched
/// shape a GPU is built for.
pub fn regress_out(
    _expression: &Array2<f32>,
    _covariates: &Array2<f32>,
    _device: &Device,
) -> Result<Array2<f32>> {
    todo!("feat/regress-combat")
}

/// Empirical Bayes batch correction, as `scanpy.pp.combat`.
pub fn combat(
    _expression: &Array2<f32>,
    _batch: &[u32],
    _n_batches: usize,
    _covariates: Option<&Array2<f32>>,
    _device: &Device,
) -> Result<Array2<f32>> {
    todo!("feat/regress-combat")
}
