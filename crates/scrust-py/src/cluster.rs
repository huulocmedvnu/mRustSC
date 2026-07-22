//! Bindings: clustering. Owned by feat/leiden.
//!
//! Not registered in `lib.rs`, which `main` owns; `main` has to add
//! `mod cluster;` and `cluster::register(module)?` for these to appear.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use scrust_core::cluster::{self as core_cluster, Partition};

use crate::convert::{csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

/// A partition as it crosses back into Python: labels, the modularity they
/// reach, and how many communities there are.
type PyPartition<'py> = (Bound<'py, PyArray1<u32>>, f64, usize);

fn partition_to_py(py: Python<'_>, partition: Partition) -> PyPartition<'_> {
    (
        partition.labels.into_pyarray(py),
        partition.modularity,
        partition.n_communities,
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, resolution, n_iterations, seed, device))]
fn leiden<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    resolution: f64,
    n_iterations: usize,
    seed: u64,
    device: &str,
) -> PyResult<PyPartition<'py>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let partition = py
        .allow_threads(|| core_cluster::leiden(&graph, resolution, n_iterations, seed, &device))
        .map_err(to_py_error)?;
    Ok(partition_to_py(py, partition))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, resolution, seed, device))]
fn louvain<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    resolution: f64,
    seed: u64,
    device: &str,
) -> PyResult<PyPartition<'py>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let partition = py
        .allow_threads(|| core_cluster::louvain(&graph, resolution, seed, &device))
        .map_err(to_py_error)?;
    Ok(partition_to_py(py, partition))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, labels, resolution))]
fn modularity(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    n_cols: usize,
    labels: &Bound<'_, PyAny>,
    resolution: f64,
) -> PyResult<f64> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(labels, "labels")?;
    py.allow_threads(|| core_cluster::modularity(&graph, &labels, resolution))
        .map_err(to_py_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(leiden, module)?)?;
    module.add_function(wrap_pyfunction!(louvain, module)?)?;
    module.add_function(wrap_pyfunction!(modularity, module)?)?;
    Ok(())
}
