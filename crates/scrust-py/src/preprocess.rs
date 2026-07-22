//! Bindings: preprocess. Owned by feat/bindings-preprocess.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrust_core::preprocess::hvg::{self, HvgFlavor};
use scrust_core::preprocess::{filter, normalize, scale as core_scale};
use scrust_core::{pca as core_pca, Error};

use crate::convert::{csr_from_py, csr_to_py, device_from_py, tensor_to_array2, PyCsr};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, target_sum, device))]
fn normalize_total<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    target_sum: Option<f32>,
    device: &str,
) -> PyResult<PyCsr<'py>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let normalised = py
        .allow_threads(|| normalize::normalize_total(&matrix, target_sum, &device))
        .map_err(to_py_error)?;
    Ok(csr_to_py(py, &normalised))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols))]
fn log1p<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
) -> PyResult<PyCsr<'py>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let logged = py
        .allow_threads(|| normalize::log1p(&matrix))
        .map_err(to_py_error)?;
    Ok(csr_to_py(py, &logged))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, min_genes, min_counts))]
fn filter_cells<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    min_genes: Option<usize>,
    min_counts: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let keep = py
        .allow_threads(|| filter::filter_cells(&matrix, min_genes, min_counts))
        .map_err(to_py_error)?;
    Ok(keep.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, min_cells, min_counts))]
fn filter_genes<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    min_cells: Option<usize>,
    min_counts: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let keep = py
        .allow_threads(|| filter::filter_genes(&matrix, min_cells, min_counts))
        .map_err(to_py_error)?;
    Ok(keep.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, zero_center, max_value, device))]
fn scale<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f32>,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let scaled = py
        .allow_threads(|| {
            let dense = core_scale::scale(&matrix, zero_center, max_value, &device)?;
            tensor_to_array2(&dense)
        })
        .map_err(to_py_error)?;
    Ok(scaled.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_top_genes, flavor, device))]
fn highly_variable_genes<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    n_top_genes: usize,
    flavor: &str,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let flavor = parse_flavor(flavor)?;
    let device = device_from_py(device)?;
    let genes = py
        .allow_threads(|| hvg::highly_variable_genes(&matrix, n_top_genes, flavor, &device))
        .map_err(to_py_error)?;

    let result = PyDict::new(py);
    result.set_item("means", genes.means.into_pyarray(py))?;
    result.set_item("dispersions", genes.dispersions.into_pyarray(py))?;
    result.set_item(
        "normalised_dispersions",
        genes.normalised_dispersions.into_pyarray(py),
    )?;
    result.set_item("highly_variable", genes.highly_variable.into_pyarray(py))?;
    Ok(result)
}

/// scanpy spells the flavours this way; the enum they name lives in the core.
fn parse_flavor(name: &str) -> PyResult<HvgFlavor> {
    match name {
        "seurat" => Ok(HvgFlavor::Seurat),
        "cell_ranger" => Ok(HvgFlavor::CellRanger),
        other => Err(to_py_error(Error::parameter(
            "flavor",
            "one of seurat, cell_ranger",
            other,
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_components, zero_center, seed, device))]
fn pca<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    n_components: usize,
    zero_center: bool,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let fitted = py
        .allow_threads(|| core_pca::pca(&matrix, n_components, zero_center, seed, &device))
        .map_err(to_py_error)?;

    let result = PyDict::new(py);
    result.set_item("embedding", fitted.embedding.into_pyarray(py))?;
    result.set_item("components", fitted.components.into_pyarray(py))?;
    result.set_item(
        "explained_variance",
        fitted.explained_variance.into_pyarray(py),
    )?;
    result.set_item(
        "explained_variance_ratio",
        fitted.explained_variance_ratio.into_pyarray(py),
    )?;
    Ok(result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_total, module)?)?;
    module.add_function(wrap_pyfunction!(log1p, module)?)?;
    module.add_function(wrap_pyfunction!(filter_cells, module)?)?;
    module.add_function(wrap_pyfunction!(filter_genes, module)?)?;
    module.add_function(wrap_pyfunction!(scale, module)?)?;
    module.add_function(wrap_pyfunction!(highly_variable_genes, module)?)?;
    module.add_function(wrap_pyfunction!(pca, module)?)?;
    Ok(())
}
