//! Bindings: sampling. Owned by feat/sampling.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use scrust_core::sampling;

use crate::convert::{csr_from_py, csr_to_py, PyCsr};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (n_cells, n_keep, replace, seed))]
fn subsample(
    py: Python<'_>,
    n_cells: usize,
    n_keep: usize,
    replace: bool,
    seed: u64,
) -> PyResult<Bound<'_, PyArray1<u32>>> {
    let kept = py
        .allow_threads(|| sampling::subsample(n_cells, n_keep, replace, seed))
        .map_err(to_py_error)?;
    Ok(kept.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, counts_per_cell, total_counts, replace, seed))]
#[allow(clippy::too_many_arguments)]
fn downsample_counts<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    counts_per_cell: Option<f32>,
    total_counts: Option<f32>,
    replace: bool,
    seed: u64,
) -> PyResult<PyCsr<'py>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let thinned = py
        .allow_threads(|| {
            sampling::downsample_counts(&matrix, counts_per_cell, total_counts, replace, seed)
        })
        .map_err(to_py_error)?;
    Ok(csr_to_py(py, &thinned))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(subsample, module)?)?;
    module.add_function(wrap_pyfunction!(downsample_counts, module)?)?;
    Ok(())
}
