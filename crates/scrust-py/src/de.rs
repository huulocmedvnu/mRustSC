//! Bindings: de. Owned by feat/bindings-de.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrust_core::de::wilcoxon;

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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rank_genes_groups_wilcoxon, module)?)?;
    Ok(())
}
