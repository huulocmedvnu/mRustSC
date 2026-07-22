//! Bindings: t-test and logistic-regression differential expression.
//! Owned by feat/de-methods.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrust_core::de::parametric;
use scrust_core::de::wilcoxon::GroupComparison;

use crate::convert::{csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

fn comparison_to_py(py: Python<'_>, comparison: GroupComparison) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("scores", comparison.scores.into_pyarray(py))?;
    result.set_item("p_values", comparison.p_values.into_pyarray(py))?;
    result.set_item(
        "adjusted_p_values",
        comparison.adjusted_p_values.into_pyarray(py),
    )?;
    result.set_item(
        "log2_fold_changes",
        comparison.log2_fold_changes.into_pyarray(py),
    )?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, labels, n_groups, reference, device))]
fn rank_genes_groups_t_test<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    labels: &Bound<'py, PyAny>,
    n_groups: usize,
    reference: Option<u32>,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    let device = device_from_py(device)?;
    let comparison = py
        .allow_threads(|| parametric::t_test(&matrix, &labels, n_groups, reference, &device))
        .map_err(to_py_error)?;
    comparison_to_py(py, comparison)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, labels, n_groups, reference, device))]
fn rank_genes_groups_t_test_overestim_var<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    labels: &Bound<'py, PyAny>,
    n_groups: usize,
    reference: Option<u32>,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    let device = device_from_py(device)?;
    let comparison = py
        .allow_threads(|| {
            parametric::t_test_overestimated_variance(
                &matrix, &labels, n_groups, reference, &device,
            )
        })
        .map_err(to_py_error)?;
    comparison_to_py(py, comparison)
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, labels, n_groups, max_iterations, device))]
fn rank_genes_groups_logreg<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    labels: &Bound<'py, PyAny>,
    n_groups: usize,
    max_iterations: usize,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    let device = device_from_py(device)?;
    let comparison = py
        .allow_threads(|| {
            parametric::logistic_regression(&matrix, &labels, n_groups, max_iterations, &device)
        })
        .map_err(to_py_error)?;
    comparison_to_py(py, comparison)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rank_genes_groups_t_test, module)?)?;
    module.add_function(wrap_pyfunction!(
        rank_genes_groups_t_test_overestim_var,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(rank_genes_groups_logreg, module)?)?;
    Ok(())
}
