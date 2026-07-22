//! Bindings: metrics. Owned by feat/metrics.
//!
//! Not registered in `lib.rs`, which `main` owns — see the branch report.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use scrust_core::autocorrelation;
use scrust_core::cluster;

use crate::convert::{csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

/// The graph and the expression matrix, the pair both statistics take.
///
/// Nine positional arrays are the same nine for `morans_i` and `gearys_c`, so
/// the conversion is written once and each binding differs only in the core
/// function it names.
macro_rules! autocorrelation_binding {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (graph_indptr, graph_indices, graph_values, n_cells,
                            indptr, indices, values, n_cols, device))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            graph_indptr: &Bound<'py, PyAny>,
            graph_indices: &Bound<'py, PyAny>,
            graph_values: &Bound<'py, PyAny>,
            n_cells: usize,
            indptr: &Bound<'py, PyAny>,
            indices: &Bound<'py, PyAny>,
            values: &Bound<'py, PyAny>,
            n_cols: usize,
            device: &str,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let graph = csr_from_py(graph_indptr, graph_indices, graph_values, n_cells)?;
            let matrix = csr_from_py(indptr, indices, values, n_cols)?;
            let device = device_from_py(device)?;
            let statistic = py
                .allow_threads(|| $core(&graph, &matrix, &device))
                .map_err(to_py_error)?;
            Ok(statistic.into_pyarray(py))
        }
    };
}

autocorrelation_binding!(morans_i, autocorrelation::morans_i);
autocorrelation_binding!(gearys_c, autocorrelation::gearys_c);

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cells, labels, resolution))]
fn modularity(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    n_cells: usize,
    labels: &Bound<'_, PyAny>,
    resolution: f64,
) -> PyResult<f64> {
    let graph = csr_from_py(indptr, indices, values, n_cells)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    py.allow_threads(|| cluster::modularity(&graph, &labels, resolution))
        .map_err(to_py_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(morans_i, module)?)?;
    module.add_function(wrap_pyfunction!(gearys_c, module)?)?;
    module.add_function(wrap_pyfunction!(modularity, module)?)?;
    Ok(())
}
