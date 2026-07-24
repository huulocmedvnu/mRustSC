//! Bindings: embedding. Owned by feat/bindings-embedding.
//!
//! The contract fixes a flat, typed function per algorithm, so the argument
//! lists are long by design.
#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;

use candle_core::Device;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use scrust_core::neighbors::{self, KnnGraph};
use scrust_core::tsne::{self as core_tsne, TsneParams};
use scrust_core::umap::{self as core_umap, UmapParams};
use scrust_core::Error;
use scrust_gpu::{kernels::knn::knn_metal, MetalContext};

use crate::convert::{array2_from_py, csr_from_py, csr_to_py, device_from_py, PyCsr, PyKnn};
use crate::to_py_error;

thread_local! {
    /// One `MetalContext` per thread, built on first use and reused after.
    ///
    /// The context owns the compiled-pipeline cache, so recreating it per call would
    /// recompile the MSL shader every time — the dominant cost the spike measured.
    /// Thread-local rather than a global: a `MetalContext` holds `metal` handles that
    /// are not `Sync`, and PyO3 releases the GIL around the call, so two Python threads
    /// could otherwise share one. `None` means we tried and found no usable GPU, so we
    /// do not retry on every call.
    static METAL_CONTEXT: RefCell<Option<Option<MetalContext>>> = const { RefCell::new(None) };
}

/// Run the k-NN search, routing a Metal caller to the hand-written `knn_metal` kernel.
///
/// SPIKE dispatch (`feature/multiagent-gpu-spike`). The candle path in
/// `neighbors::knn` stays the oracle: it runs for `Device::Cpu`, and it also runs on
/// Metal when a `MetalContext` cannot be built (no usable GPU), so the fallback never
/// changes the answer a caller gets. A kernel *computation* error is propagated rather
/// than swallowed — silently falling back on it would hide a real defect. Both paths
/// return the same `KnnGraph`, so the caller above cannot tell which ran.
fn knn_dispatch(embedding: &Array2<f32>, k: usize, device: &Device) -> scrust_core::Result<KnnGraph> {
    if device.is_metal() {
        let kernel_result = METAL_CONTEXT.with(|slot| {
            let mut slot = slot.borrow_mut();
            // Build the context once (outer `None`), caching success or failure.
            let context = slot.get_or_insert_with(|| MetalContext::new().ok());
            context
                .as_ref()
                .map(|context| knn_metal(context, embedding, k))
        });
        if let Some(result) = kernel_result {
            return result;
        }
        // No usable Metal context: fall through to the candle path.
    }
    neighbors::knn(embedding, k, device)
}

#[pyfunction]
#[pyo3(signature = (embedding, k, device))]
fn knn<'py>(
    py: Python<'py>,
    embedding: &Bound<'py, PyAny>,
    k: usize,
    device: &str,
) -> PyResult<PyKnn<'py>> {
    let embedding = array2_from_py::<f32>(embedding, "embedding")?;
    let device = device_from_py(device)?;
    let graph = py
        .allow_threads(|| knn_dispatch(&embedding, k, &device))
        .map_err(to_py_error)?;
    Ok((
        graph.indices.into_pyarray(py),
        graph.distances.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indices, distances))]
fn connectivities<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    distances: &Bound<'py, PyAny>,
) -> PyResult<PyCsr<'py>> {
    let graph = KnnGraph {
        indices: array2_from_py::<u32>(indices, "indices")?,
        distances: array2_from_py::<f32>(distances, "distances")?,
    };
    if graph.indices.dim() != graph.distances.dim() {
        return Err(to_py_error(Error::shape(
            format!("distances shaped {:?}", graph.indices.dim()),
            format!("{:?}", graph.distances.dim()),
        )));
    }
    let weighted = py
        .allow_threads(|| neighbors::connectivities(&graph))
        .map_err(to_py_error)?;
    Ok(csr_to_py(py, &weighted))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_components, n_epochs, min_dist, spread,
                    learning_rate, negative_sample_rate, seed, device))]
fn umap<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    n_components: usize,
    n_epochs: usize,
    min_dist: f32,
    spread: f32,
    learning_rate: f32,
    negative_sample_rate: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let params = UmapParams {
        n_components,
        n_epochs,
        min_dist,
        spread,
        learning_rate,
        negative_sample_rate,
        seed,
    };
    let device = device_from_py(device)?;
    let layout = py
        .allow_threads(|| core_umap::umap(&graph, &params, &device))
        .map_err(to_py_error)?;
    Ok(layout.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (embedding, n_components, perplexity, early_exaggeration, learning_rate,
                    n_iterations, seed, device))]
fn tsne<'py>(
    py: Python<'py>,
    embedding: &Bound<'py, PyAny>,
    n_components: usize,
    perplexity: f32,
    early_exaggeration: f32,
    learning_rate: f32,
    n_iterations: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let embedding = array2_from_py::<f32>(embedding, "embedding")?;
    let params = TsneParams {
        n_components,
        perplexity,
        early_exaggeration,
        learning_rate,
        n_iterations,
        seed,
    };
    let device = device_from_py(device)?;
    let layout = py
        .allow_threads(|| core_tsne::tsne(&embedding, &params, &device))
        .map_err(to_py_error)?;
    Ok(layout.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(knn, module)?)?;
    module.add_function(wrap_pyfunction!(connectivities, module)?)?;
    module.add_function(wrap_pyfunction!(umap, module)?)?;
    module.add_function(wrap_pyfunction!(tsne, module)?)?;
    Ok(())
}
