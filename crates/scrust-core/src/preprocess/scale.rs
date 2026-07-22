use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Standardise each gene to zero mean and unit variance, as `scanpy.pp.scale`.
///
/// Returns a dense tensor: scaling destroys sparsity, which is why scanpy also
/// densifies here.
pub fn scale(
    _matrix: &CsrMatrix,
    _zero_center: bool,
    _max_value: Option<f32>,
    _device: &Device,
) -> Result<Tensor> {
    todo!("feat/scale")
}
