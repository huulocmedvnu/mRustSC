//! Bindings: embedding. Owned by feat/bindings-embedding.
#![allow(unused_imports)]

use pyo3::prelude::*;

pub(crate) fn register(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
