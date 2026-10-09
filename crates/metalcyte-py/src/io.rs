//! Python entry point for the parallel chunk reader in `metalcyte_core::chunk_read`.
//!
//! h5py reads one dataset slice on one thread under the GIL. For an uncompressed counts
//! file, every chunk sits at a byte offset h5py reports once, so a row block is a set of
//! byte ranges that Rust reads on all cores with the GIL released.

use std::fs::File;

use metalcyte_core::chunk_read;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

/// A block's values and its gene indices.
type Entries<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>);

/// Stored entries `[lo, hi)` of a CSR matrix as `(float32 values, int32 gene indices)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn read_csr_entries<'py>(
    py: Python<'py>,
    path: &str,
    data_offsets: PyReadonlyArray1<'py, u64>,
    data_chunk_len: u64,
    index_offsets: PyReadonlyArray1<'py, u64>,
    index_chunk_len: u64,
    index_itemsize: usize,
    lo: u64,
    hi: u64,
) -> PyResult<Entries<'py>> {
    if hi < lo {
        return Err(PyValueError::new_err("hi must not be below lo"));
    }
    if data_chunk_len == 0 || index_chunk_len == 0 {
        return Err(PyValueError::new_err("chunk lengths must be positive"));
    }
    if index_itemsize != 4 && index_itemsize != 8 {
        return Err(PyValueError::new_err("index_itemsize must be 4 or 8"));
    }
    let data_offsets = data_offsets.as_slice()?;
    let index_offsets = index_offsets.as_slice()?;
    let (values, indices) = py
        .allow_threads(|| {
            let file = File::open(path)?;
            chunk_read::read_csr_entries(
                &file,
                data_offsets,
                data_chunk_len,
                index_offsets,
                index_chunk_len,
                index_itemsize,
                lo,
                (hi - lo) as usize,
            )
        })
        .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    Ok((values.into_pyarray(py), indices.into_pyarray(py)))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(read_csr_entries, module)?)?;
    Ok(())
}
