//! Bindings: dendrograms, force-directed layouts and densities. Owned by feat/layout.
//!
//! A CSR matrix arrives as four arguments, so the layout call is long by design.
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use scrust_core::layout as core_layout;

use crate::convert::{array2_from_py, csr_from_py, device_from_py};
use crate::to_py_error;

/// A merge tree as it crosses back into Python: the `(n_groups - 1, 4)` linkage
/// rows scipy's encoding calls for, and the leaves left to right.
type PyDendrogram<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<u32>>);

#[pyfunction]
#[pyo3(signature = (centroids))]
fn dendrogram<'py>(py: Python<'py>, centroids: &Bound<'py, PyAny>) -> PyResult<PyDendrogram<'py>> {
    let centroids = array2_from_py::<f32>(centroids, "centroids")?;
    let tree = py
        .allow_threads(|| core_layout::dendrogram(&centroids))
        .map_err(to_py_error)?;
    let rows = tree.linkage.len();
    let flat: Vec<f64> = tree.linkage.into_iter().flatten().collect();
    Ok((
        flat.into_pyarray(py).reshape((rows, 4))?,
        tree.leaf_order.to_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_iterations, seed, device))]
fn draw_graph<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    n_iterations: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let layout = py
        .allow_threads(|| core_layout::force_directed_layout(&graph, n_iterations, seed, &device))
        .map_err(to_py_error)?;
    Ok(layout.to_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (embedding, device))]
fn embedding_density<'py>(
    py: Python<'py>,
    embedding: &Bound<'py, PyAny>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let embedding = array2_from_py::<f32>(embedding, "embedding")?;
    let device = device_from_py(device)?;
    let density = py
        .allow_threads(|| core_layout::embedding_density(&embedding, &device))
        .map_err(to_py_error)?;
    Ok(density.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dendrogram, module)?)?;
    module.add_function(wrap_pyfunction!(draw_graph, module)?)?;
    module.add_function(wrap_pyfunction!(embedding_density, module)?)?;
    Ok(())
}
