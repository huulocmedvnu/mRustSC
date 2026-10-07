//! The Python extension module `metalcyte._metalcyte`.
//!
//! This layer only converts: numpy and scipy objects in, Rust types out, errors
//! into exceptions. It holds no algorithm and no defaults — those live in
//! `metalcyte-core` and in the Python `pp`/`tl` wrappers respectively.

use pyo3::prelude::*;

mod batch;
mod cluster;
mod convert;
mod de;
mod diffusion;
mod embedding;
mod fast;
mod layout;
mod metrics;
mod paga;
mod parametric;
mod preprocess;
mod qc;
mod raster;
mod sampling;
mod scoring;

/// Map a core error onto the closest Python exception.
#[allow(dead_code)]
pub(crate) fn to_py_error(error: metalcyte_core::Error) -> PyErr {
    use metalcyte_core::Error::*;
    match error {
        Shape { .. } | InvalidParameter { .. } => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        NoGpu => pyo3::exceptions::PyRuntimeError::new_err(error.to_string()),
        _ => pyo3::exceptions::PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
fn gpu_available() -> bool {
    metalcyte_core::gpu_available()
}

#[pymodule]
fn _metalcyte(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gpu_available, module)?)?;
    cluster::register(module)?;
    preprocess::register(module)?;
    fast::register(module)?;
    embedding::register(module)?;
    raster::register(module)?;
    de::register(module)?;
    diffusion::register(module)?;
    metrics::register(module)?;
    parametric::register(module)?;
    paga::register(module)?;
    qc::register(module)?;
    sampling::register(module)?;
    batch::register(module)?;
    scoring::register(module)?;
    layout::register(module)?;
    Ok(())
}
