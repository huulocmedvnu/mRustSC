//! Bindings: preprocess. Owned by feat/bindings-preprocess.
#![allow(unused_imports)]

use pyo3::prelude::*;

pub(crate) fn register(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
