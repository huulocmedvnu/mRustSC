//! Bindings: quality control. Owned by feat/qc-metrics.
//!
//! Conversion only: the metric definitions live in `scrust_core::qc`, and the
//! column names and defaults in `scrust.pp._qc`.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrust_core::qc;

use crate::convert::{csr_from_py, csr_to_py, vec_from_py, PyCsr};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, percent_top, gene_subsets))]
fn qc_metrics<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    percent_top: Vec<usize>,
    gene_subsets: Vec<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let subsets = gene_subsets
        .iter()
        .map(|subset| vec_from_py::<bool>(subset, "gene_subsets"))
        .collect::<PyResult<Vec<Vec<bool>>>>()?;

    let (cells, genes) = py
        .allow_threads(|| qc::qc_metrics(&matrix, &percent_top, &subsets))
        .map_err(to_py_error)?;

    let cell_metrics = PyDict::new(py);
    cell_metrics.set_item(
        "n_genes_by_counts",
        cells.n_genes_by_counts.into_pyarray(py),
    )?;
    cell_metrics.set_item("total_counts", cells.total_counts.into_pyarray(py))?;
    cell_metrics.set_item(
        "pct_counts_in_top",
        per_cell_arrays(py, cells.pct_counts_in_top),
    )?;
    cell_metrics.set_item("subset_totals", per_cell_arrays(py, cells.subset_totals))?;

    let gene_metrics = PyDict::new(py);
    gene_metrics.set_item(
        "n_cells_by_counts",
        genes.n_cells_by_counts.into_pyarray(py),
    )?;
    gene_metrics.set_item("mean_counts", genes.mean_counts.into_pyarray(py))?;
    gene_metrics.set_item(
        "pct_dropout_by_counts",
        genes.pct_dropout_by_counts.into_pyarray(py),
    )?;
    gene_metrics.set_item("total_counts", genes.total_counts.into_pyarray(py))?;
    Ok((cell_metrics, gene_metrics))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols))]
fn sqrt<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
) -> PyResult<PyCsr<'py>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let rooted = py
        .allow_threads(|| qc::sqrt(&matrix))
        .map_err(to_py_error)?;
    Ok(csr_to_py(py, &rooted))
}

/// One per-cell array per requested quantity.
///
/// A list of 1-D arrays rather than one 2-D array: the request is often empty,
/// and an empty list is a shape both sides handle without a special case.
fn per_cell_arrays(py: Python<'_>, rows: Vec<Vec<f32>>) -> Vec<Bound<'_, PyArray1<f32>>> {
    rows.into_iter().map(|row| row.into_pyarray(py)).collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(qc_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(sqrt, module)?)?;
    Ok(())
}
