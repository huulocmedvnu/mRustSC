//! Sparse kernels on the GPU. Owned by feat/sparse-gpu.
//!
//! Densifying a 95%-zero matrix to reach the GPU wastes both the bandwidth and
//! the memory that make the GPU worth using. These kernels consume CSR directly,
//! which is what lets a run scale past what a dense block would allow.

use ndarray::Array2;
use scrust_core::error::Result;
use scrust_core::sparse::CsrMatrix;

use crate::context::MetalContext;

/// Sparse times dense: `(n_rows, n_cols) x (n_cols, k) -> (n_rows, k)`.
///
/// One threadgroup per row, striding its stored entries.
pub fn spmm(
    _context: &MetalContext,
    _sparse: &CsrMatrix,
    _dense: &Array2<f32>,
) -> Result<Array2<f32>> {
    todo!("feat/sparse-gpu")
}

/// Transposed sparse times dense, without materialising the transpose.
pub fn spmm_transposed(
    _context: &MetalContext,
    _sparse: &CsrMatrix,
    _dense: &Array2<f32>,
) -> Result<Array2<f32>> {
    todo!("feat/sparse-gpu")
}

/// Per-column sum and sum of squares in one pass, the reduction every
/// normalisation and variance step needs.
pub fn column_moments(
    _context: &MetalContext,
    _sparse: &CsrMatrix,
) -> Result<(Vec<f32>, Vec<f32>)> {
    todo!("feat/sparse-gpu")
}

/// Scale each row by its factor, in place on the stored values.
pub fn scale_rows(
    _context: &MetalContext,
    _sparse: &mut CsrMatrix,
    _factors: &[f32],
) -> Result<()> {
    todo!("feat/sparse-gpu")
}
