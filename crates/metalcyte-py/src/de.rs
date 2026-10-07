//! Bindings: de. Owned by feat/bindings-de.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use metalcyte_core::de::wilcoxon;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, labels, n_groups, reference, tie_correct,
                    device))]
fn rank_genes_groups_wilcoxon<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    labels: &Bound<'py, PyAny>,
    n_groups: usize,
    reference: Option<u32>,
    tie_correct: bool,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    let device = device_from_py(device)?;
    let comparison = py
        .allow_threads(|| {
            wilcoxon::rank_genes_groups_wilcoxon(
                &matrix,
                &labels,
                n_groups,
                reference,
                tie_correct,
                &device,
            )
        })
        .map_err(to_py_error)?;

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

/// `rank_genes_groups_wilcoxon` on a dense row-major `(n_cells, n_genes)` matrix,
/// borrowed from numpy: no CSR conversion, no copy, a gene-block working set.
#[pyfunction]
#[pyo3(signature = (data, labels, n_groups, reference, tie_correct))]
fn rank_genes_groups_wilcoxon_dense<'py>(
    py: Python<'py>,
    data: &Bound<'py, numpy::PyArray2<f32>>,
    labels: &Bound<'py, PyAny>,
    n_groups: usize,
    reference: Option<u32>,
    tie_correct: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};
    let ro = data
        .try_readonly()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("data: {e}")))?;
    let shape = ro.shape().to_vec();
    let slice = ro
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be C-contiguous"))?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    let comparison = py
        .allow_threads(|| {
            wilcoxon::rank_genes_groups_wilcoxon_dense(
                slice,
                shape[0],
                shape[1],
                &labels,
                n_groups,
                reference,
                tie_correct,
            )
        })
        .map_err(to_py_error)?;
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

/// The Wilcoxon test over row blocks of a matrix on disk: `push` each normalised,
/// log-transformed block with its rows' group labels (`u32::MAX` to skip a row), then
/// `finish` once for the statistics.
#[pyclass(unsendable)]
struct WilcoxonStream {
    inner: wilcoxon::StreamedWilcoxon,
}

#[pymethods]
impl WilcoxonStream {
    #[new]
    #[pyo3(signature = (columns, n_genes_total, n_groups))]
    fn new(columns: Vec<usize>, n_genes_total: usize, n_groups: usize) -> PyResult<Self> {
        let inner = wilcoxon::StreamedWilcoxon::new(&columns, n_genes_total, n_groups)
            .map_err(to_py_error)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (indptr, indices, values, n_cols, labels))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        indptr: &Bound<'py, PyAny>,
        indices: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
        n_cols: usize,
        labels: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let matrix = csr_from_py(indptr, indices, values, n_cols)?;
        let labels = vec_from_py::<u32>(labels, "labels")?;
        let inner = &mut self.inner;
        py.allow_threads(|| inner.push(&matrix, &labels))
            .map_err(to_py_error)
    }

    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.n_cells()
    }

    #[pyo3(signature = (reference, tie_correct))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        reference: Option<u32>,
        tie_correct: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inner = &mut self.inner;
        let comparison = py
            .allow_threads(|| inner.finish(reference, tie_correct))
            .map_err(to_py_error)?;
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
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<WilcoxonStream>()?;
    module.add_function(wrap_pyfunction!(rank_genes_groups_wilcoxon_dense, module)?)?;
    module.add_function(wrap_pyfunction!(rank_genes_groups_wilcoxon, module)?)?;
    Ok(())
}
