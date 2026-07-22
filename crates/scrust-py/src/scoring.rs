//! Bindings: scoring. Owned by feat/scoring.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use scrust_core::scoring;

use crate::convert::{csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, gene_set, ctrl_size, n_bins, seed, device))]
#[allow(clippy::too_many_arguments)] // the contract fixes a flat, typed argument list
fn score_genes<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    gene_set: &Bound<'py, PyAny>,
    ctrl_size: usize,
    n_bins: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let gene_set = vec_from_py::<u32>(gene_set, "gene_set")?;
    let device = device_from_py(device)?;
    let scores = py
        .allow_threads(|| {
            scoring::score_genes(&matrix, &gene_set, ctrl_size, n_bins, seed, &device)
        })
        .map_err(to_py_error)?;
    Ok(scores.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(score_genes, module)?)?;
    Ok(())
}
