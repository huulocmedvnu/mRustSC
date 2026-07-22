//! The Python extension module `scrust._scrust`.
//!
//! This layer only converts: numpy and scipy objects in, Rust types out, errors
//! into exceptions. It holds no algorithm and no defaults — those live in
//! `scrust-core` and in the Python `pp`/`tl` wrappers respectively.

use pyo3::prelude::*;

mod convert;
mod de;
mod diffusion;
mod embedding;
mod preprocess;

/// Map a core error onto the closest Python exception.
#[allow(dead_code)]
pub(crate) fn to_py_error(error: scrust_core::Error) -> PyErr {
    use scrust_core::Error::*;
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
    scrust_core::gpu_available()
}

#[pymodule]
fn _scrust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gpu_available, module)?)?;
    preprocess::register(module)?;
    embedding::register(module)?;
    de::register(module)?;
    diffusion::register(module)?;
    Ok(())
}
