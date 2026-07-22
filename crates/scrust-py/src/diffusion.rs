//! Bindings: diffusion maps and pseudotime. Owned by feat/diffusion.
//!
//! Not registered in `lib.rs` — that file belongs to `main`. See the branch
//! report.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use scrust_core::diffusion::{self as core_diffusion, DiffusionMap};

use crate::convert::{array2_from_py, csr_from_py, device_from_py, vec_from_py};
use crate::to_py_error;

/// `(X_diffmap, diffmap_evals)`, the two slots `tl.diffmap` writes.
type PyDiffusionMap<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>);

#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_comps, device))]
fn diffmap<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n_cols: usize,
    n_comps: usize,
    device: &str,
) -> PyResult<PyDiffusionMap<'py>> {
    let graph = csr_from_py(indptr, indices, values, n_cols)?;
    let device = device_from_py(device)?;
    let map = py
        .allow_threads(|| core_diffusion::diffmap(&graph, n_comps, &device))
        .map_err(to_py_error)?;
    Ok((
        map.embedding.into_pyarray(py),
        map.eigenvalues.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (embedding, eigenvalues, root, n_dcs))]
fn dpt<'py>(
    py: Python<'py>,
    embedding: &Bound<'py, PyAny>,
    eigenvalues: &Bound<'py, PyAny>,
    root: usize,
    n_dcs: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let map = DiffusionMap {
        embedding: array2_from_py::<f32>(embedding, "embedding")?,
        eigenvalues: vec_from_py::<f32>(eigenvalues, "eigenvalues")?,
    };
    let pseudotime = py
        .allow_threads(|| core_diffusion::dpt(&map, root, n_dcs))
        .map_err(to_py_error)?;
    Ok(pseudotime.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(diffmap, module)?)?;
    module.add_function(wrap_pyfunction!(dpt, module)?)?;
    Ok(())
}
