//! Bindings: partition-based graph abstraction. Owned by feat/paga.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use scrust_core::paga as core_paga;

use crate::convert::{csr_from_py, vec_from_py};
use crate::to_py_error;

/// The abstracted graph as it crosses back into Python: the two dense
/// `(n_groups, n_groups)` matrices, row-major, plus the side they are square in.
type PyAbstractedGraph<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, usize);

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, groups, n_groups))]
fn paga<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    groups: &Bound<'py, PyAny>,
    n_groups: usize,
) -> PyResult<PyAbstractedGraph<'py>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let groups = vec_from_py::<u32>(groups, "groups")?;
    let abstracted = py
        .allow_threads(|| core_paga::paga(&graph, &groups, n_groups))
        .map_err(to_py_error)?;
    Ok((
        abstracted.connectivities.into_pyarray(py),
        abstracted.connectivities_tree.into_pyarray(py),
        abstracted.n_groups,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(paga, module)?)?;
    Ok(())
}
