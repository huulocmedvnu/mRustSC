//! Bindings: removing unwanted variation. Owned by feat/regress-combat.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use scrust_core::batch;
use scrust_core::harmony;
use scrust_core::sparse::CsrMatrix;
use scrust_core::{Error, Result};

use crate::convert::{array2_from_py, csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, covariates, device))]
fn regress_out<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    covariates: &Bound<'py, PyAny>,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let covariates = array2_from_py::<f32>(covariates, "covariates")?;
    let device = device_from_py(device)?;
    let residuals = py
        .allow_threads(|| batch::regress_out(&densify(&matrix)?, &covariates, &device))
        .map_err(to_py_error)?;
    Ok(residuals.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, batch, n_batches, covariates, device))]
fn combat<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    batch: &Bound<'py, PyAny>,
    n_batches: usize,
    covariates: Option<&Bound<'py, PyAny>>,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let matrix = csr_from_py(indptr, indices, values, n_cols)?;
    let labels = vec_from_py::<u32>(batch, "batch")?;
    let covariates = covariates
        .map(|array| array2_from_py::<f32>(array, "covariates"))
        .transpose()?;
    let device = device_from_py(device)?;
    let corrected = py
        .allow_threads(|| {
            batch::combat(
                &densify(&matrix)?,
                &labels,
                n_batches,
                covariates.as_ref(),
                &device,
            )
        })
        .map_err(to_py_error)?;
    Ok(corrected.into_pyarray(py))
}

/// Harmony batch-effect correction on a dense PCA embedding. Returns the corrected
/// embedding and the harmony objective at each outer iteration (the convergence curve).
#[pyfunction]
#[pyo3(signature = (embedding, batch, n_batches, theta, sigma, lambda, n_clusters,
                    max_iter_harmony, max_iter_kmeans, seed, device))]
fn harmony_integrate<'py>(
    py: Python<'py>,
    embedding: &Bound<'py, PyAny>,
    batch: &Bound<'py, PyAny>,
    n_batches: usize,
    theta: f32,
    sigma: f32,
    lambda: f32,
    n_clusters: usize,
    max_iter_harmony: usize,
    max_iter_kmeans: usize,
    seed: u64,
    device: &str,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Vec<f32>)> {
    let embedding = array2_from_py::<f32>(embedding, "embedding")?;
    let batch = vec_from_py::<u32>(batch, "batch")?;
    let device = device_from_py(device)?;
    let defaults = harmony::HarmonyParams::defaults(embedding.nrows());
    let params = harmony::HarmonyParams {
        theta,
        sigma,
        lambda,
        n_clusters: if n_clusters == 0 { defaults.n_clusters } else { n_clusters },
        max_iter_harmony,
        max_iter_kmeans,
        epsilon_cluster: defaults.epsilon_cluster,
        epsilon_harmony: defaults.epsilon_harmony,
        block_size: defaults.block_size,
        seed,
    };
    let result = py
        .allow_threads(|| harmony::harmony_integrate(&embedding, &batch, n_batches, &params, &device))
        .map_err(to_py_error)?;
    Ok((result.corrected.into_pyarray(py), result.objective))
}

/// Both algorithms are dense over the whole matrix, so the CSR arrays are
/// expanded once here rather than per call inside the core.
fn densify(matrix: &CsrMatrix) -> Result<Array2<f32>> {
    let shape = (matrix.n_rows(), matrix.n_cols());
    Array2::from_shape_vec(shape, matrix.densify_rows(0, matrix.n_rows()))
        .map_err(|error| Error::shape(format!("{} x {}", shape.0, shape.1), error.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(regress_out, module)?)?;
    module.add_function(wrap_pyfunction!(combat, module)?)?;
    module.add_function(wrap_pyfunction!(harmony_integrate, module)?)?;
    Ok(())
}
