//! Spatial autocorrelation over the neighbour graph. Owned by feat/metrics.

use candle_core::Device;

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// Moran's I for every column of `values`, as `scanpy.metrics.morans_i`.
pub fn morans_i(_graph: &CsrMatrix, _values: &CsrMatrix, _device: &Device) -> Result<Vec<f32>> {
    todo!("feat/metrics")
}

/// Geary's C for every column of `values`, as `scanpy.metrics.gearys_c`.
pub fn gearys_c(_graph: &CsrMatrix, _values: &CsrMatrix, _device: &Device) -> Result<Vec<f32>> {
    todo!("feat/metrics")
}
