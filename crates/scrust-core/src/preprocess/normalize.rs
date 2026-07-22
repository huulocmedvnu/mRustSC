use candle_core::Device;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Scale every cell to the same total count, as `scanpy.pp.normalize_total`.
///
/// `target_sum` of `None` uses the median total count across cells.
pub fn normalize_total(
    _matrix: &CsrMatrix,
    _target_sum: Option<f32>,
    _device: &Device,
) -> Result<CsrMatrix> {
    todo!("feat/normalize")
}

/// Natural log of one plus each stored value, as `scanpy.pp.log1p`.
pub fn log1p(_matrix: &CsrMatrix) -> Result<CsrMatrix> {
    todo!("feat/normalize")
}
