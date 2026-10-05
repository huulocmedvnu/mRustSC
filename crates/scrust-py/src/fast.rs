//! Zero-copy bindings for the multi-core preprocessing kernels.
//!
//! These borrow numpy's buffers directly: the index arrays are read in whatever integer
//! dtype scipy holds them (`int32`, `int64` or `uint32`), the value array is updated in
//! place or the dense result is written into an array numpy allocated. The GIL is
//! released while the kernels run, and nothing is copied across the boundary.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scrust_core::preprocess::inplace::{self, Offset};

use crate::to_py_error;

/// Run `$body` with `$name` bound to a read-only slice of an integer index array of any
/// supported dtype.
macro_rules! with_index_slice {
    ($obj:expr, $what:expr, |$name:ident| $body:expr) => {{
        let obj = $obj;
        if let Ok(a) = obj.downcast::<PyArray1<i32>>() {
            let ro = a
                .try_readonly()
                .map_err(|e| PyValueError::new_err(format!("{}: {e}", $what)))?;
            let $name = ro
                .as_slice()
                .map_err(|_| PyValueError::new_err(concat!($what, " must be C-contiguous")))?;
            $body
        } else if let Ok(a) = obj.downcast::<PyArray1<i64>>() {
            let ro = a
                .try_readonly()
                .map_err(|e| PyValueError::new_err(format!("{}: {e}", $what)))?;
            let $name = ro
                .as_slice()
                .map_err(|_| PyValueError::new_err(concat!($what, " must be C-contiguous")))?;
            $body
        } else if let Ok(a) = obj.downcast::<PyArray1<u32>>() {
            let ro = a
                .try_readonly()
                .map_err(|e| PyValueError::new_err(format!("{}: {e}", $what)))?;
            let $name = ro
                .as_slice()
                .map_err(|_| PyValueError::new_err(concat!($what, " must be C-contiguous")))?;
            $body
        } else {
            Err(PyTypeError::new_err(concat!(
                $what,
                " must be an int32, int64 or uint32 numpy array"
            )))
        }
    }};
}

fn values_mut<'a>(
    values: &'a Bound<'_, PyArray1<f32>>,
) -> PyResult<numpy::PyReadwriteArray1<'a, f32>> {
    values
        .try_readwrite()
        .map_err(|e| PyValueError::new_err(format!("values: {e}")))
}

/// Normalise the stored values of a CSR matrix in place; returns the target used.
#[pyfunction]
#[pyo3(signature = (indptr, values, target_sum=None))]
fn normalize_total_inplace(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    values: &Bound<'_, PyArray1<f32>>,
    target_sum: Option<f32>,
) -> PyResult<Option<f32>> {
    let mut rw = values_mut(values)?;
    let vals = rw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    with_index_slice!(indptr, "indptr", |ip| {
        fn run<O: Offset>(
            py: Python<'_>,
            ip: &[O],
            v: &mut [f32],
            t: Option<f32>,
        ) -> PyResult<Option<f32>> {
            py.allow_threads(|| inplace::normalize_total_inplace(ip, v, t))
                .map_err(to_py_error)
        }
        run(py, ip, vals, target_sum)
    })
}

/// `log1p` on the stored values, in place.
#[pyfunction]
fn log1p_inplace(py: Python<'_>, values: &Bound<'_, PyArray1<f32>>) -> PyResult<()> {
    let mut rw = values_mut(values)?;
    let vals = rw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    py.allow_threads(|| inplace::log1p_inplace(vals));
    Ok(())
}

/// Scale a CSR matrix into a new dense `float32` array, in one fused multi-core pass.
#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, zero_center, max_value))]
fn scale_dense<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyArray1<f32>>,
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ro = values
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("values: {e}")))?;
    let vals = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let n_rows = {
        let n = indptr
            .downcast::<numpy::PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("indptr must be a numpy array"))?
            .len();
        n.checked_sub(1)
            .ok_or_else(|| PyValueError::new_err("indptr is empty"))?
    };
    let out = unsafe { PyArray2::<f32>::new(py, [n_rows, n_cols], false) };
    {
        let mut rw = out
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(format!("out: {e}")))?;
        let dest = rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("out not contiguous"))?;
        with_index_slice!(indptr, "indptr", |ip| {
            with_index_slice!(indices, "indices", |ix| {
                #[allow(clippy::too_many_arguments)]
                fn run<O: Offset, I: Offset>(
                    py: Python<'_>,
                    ip: &[O],
                    ix: &[I],
                    v: &[f32],
                    n_cols: usize,
                    zc: bool,
                    mv: Option<f32>,
                    dest: &mut [f32],
                ) -> PyResult<()> {
                    py.allow_threads(|| inplace::scale_into(ip, ix, v, n_cols, zc, mv, dest))
                        .map_err(to_py_error)
                }
                run(py, ip, ix, vals, n_cols, zero_center, max_value, dest)
            })
        })?;
    }
    Ok(out)
}

/// PCA straight from a dense C-contiguous `float32` matrix: one upload, no sparse detour.
#[pyfunction]
#[pyo3(signature = (data, n_components, zero_center, seed, device))]
fn pca_dense<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyArray2<f32>>,
    n_components: usize,
    zero_center: bool,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ro = data
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("data: {e}")))?;
    let shape = ro.shape().to_vec();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let device = crate::convert::device_from_py(device)?;
    let fitted = py
        .allow_threads(|| {
            scrust_core::pca::pca_dense(
                slice,
                shape[0],
                shape[1],
                n_components,
                zero_center,
                seed,
                &device,
            )
        })
        .map_err(to_py_error)?;
    let result = PyDict::new(py);
    result.set_item("embedding", fitted.embedding.into_pyarray(py))?;
    result.set_item("components", fitted.components.into_pyarray(py))?;
    result.set_item(
        "explained_variance",
        fitted.explained_variance.into_pyarray(py),
    )?;
    result.set_item(
        "explained_variance_ratio",
        fitted.explained_variance_ratio.into_pyarray(py),
    )?;
    Ok(result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_total_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(log1p_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(scale_dense, module)?)?;
    module.add_function(wrap_pyfunction!(pca_dense, module)?)?;
    Ok(())
}
