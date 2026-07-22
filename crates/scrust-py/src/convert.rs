//! numpy and scipy conversions shared by the binding modules.
//! Owned by feat/bindings.
#![allow(dead_code)]

use scrust_core::sparse::CsrMatrix;
use scrust_core::Result;

/// Rebuild a CSR matrix from the three arrays scipy hands over.
pub(crate) fn csr_from_parts(
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f32>,
    n_cols: usize,
) -> Result<CsrMatrix> {
    CsrMatrix::new(indptr, indices, values, n_cols)
}
