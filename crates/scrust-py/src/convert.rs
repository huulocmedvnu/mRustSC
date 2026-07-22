//! numpy and scipy conversions shared by the binding modules.
//! Owned by feat/bindings.

use candle_core::{Device, Tensor};
use ndarray::Array2;
use numpy::{dtype, Element, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scrust_core::sparse::CsrMatrix;
use scrust_core::{DeviceKind, Error, Result};

/// A sparse result as it crosses back into Python: the three CSR arrays plus
/// the column count, the same shape sparse input arrives in.
pub(crate) type PyCsr<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
);

/// A neighbour graph as it crosses back into Python: ids and distances.
pub(crate) type PyKnn<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>);

/// Rebuild a CSR matrix from the three arrays scipy hands over.
pub(crate) fn csr_from_parts(
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f32>,
    n_cols: usize,
) -> Result<CsrMatrix> {
    CsrMatrix::new(indptr, indices, values, n_cols)
}

/// The one place a CSR matrix enters from Python.
pub(crate) fn csr_from_py(
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    n_cols: usize,
) -> PyResult<CsrMatrix> {
    csr_from_parts(
        vec_from_py(indptr, "indptr")?,
        vec_from_py(indices, "indices")?,
        vec_from_py(values, "values")?,
        n_cols,
    )
    .map_err(crate::to_py_error)
}

/// Copy a 1-D numpy array into a `Vec`.
///
/// The copy is unavoidable: the core owns its buffers, and the GIL is released
/// around the call, so a borrow of numpy's memory could not outlive it.
pub(crate) fn vec_from_py<T: Element + Copy>(
    object: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<T>> {
    let array = object
        .downcast::<PyArray1<T>>()
        .map_err(|_| wrong_array::<T>(object, name, 1))?;
    let readonly = array
        .try_readonly()
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))?;
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?;
    Ok(slice.to_vec())
}

/// Copy a 2-D numpy array into an owned ndarray.
///
/// A strided view is read through its strides, so a non-contiguous argument is
/// copied correctly rather than rejected.
pub(crate) fn array2_from_py<T: Element + Copy>(
    object: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Array2<T>> {
    let array = object
        .downcast::<PyArray2<T>>()
        .map_err(|_| wrong_array::<T>(object, name, 2))?;
    let readonly = array
        .try_readonly()
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))?;
    Ok(readonly.as_array().to_owned())
}

/// Wrong dtype and wrong dimensionality are both silent misreads if let through.
fn wrong_array<T: Element>(object: &Bound<'_, PyAny>, name: &str, n_dims: usize) -> PyErr {
    let expected = dtype::<T>(object.py());
    PyValueError::new_err(format!(
        "{name} must be a {n_dims}-D numpy array of {expected:?}"
    ))
}

/// Resolve the device name Python passed; an unknown name fails here, once.
pub(crate) fn device_from_py(name: &str) -> PyResult<Device> {
    DeviceKind::parse(name)
        .and_then(DeviceKind::resolve)
        .map_err(crate::to_py_error)
}

pub(crate) fn csr_to_py<'py>(py: Python<'py>, matrix: &CsrMatrix) -> PyCsr<'py> {
    (
        matrix.indptr().to_pyarray(py),
        matrix.indices().to_pyarray(py),
        matrix.values().to_pyarray(py),
        matrix.n_cols(),
    )
}

/// Move a device tensor into a host ndarray, ready to become a numpy array.
pub(crate) fn tensor_to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
    let (n_rows, n_cols) = tensor.dims2()?;
    let values = tensor
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Array2::from_shape_vec((n_rows, n_cols), values)
        .map_err(|_| Error::shape(format!("{n_rows}x{n_cols} values"), "a different count"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_round_trips_through_parts() {
        let matrix = csr_from_parts(vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0], 4).unwrap();
        assert_eq!(matrix.indptr(), &[0, 2, 3]);
        assert_eq!(matrix.indices(), &[0, 2, 1]);
        assert_eq!(matrix.values(), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.n_cols(), 4);
        assert_eq!(matrix.n_rows(), 2);
    }

    #[test]
    fn rejects_an_out_of_range_column_index() {
        assert!(csr_from_parts(vec![0, 1], vec![4], vec![1.0], 4).is_err());
    }

    #[test]
    fn rejects_mismatched_array_lengths() {
        assert!(csr_from_parts(vec![0, 2], vec![0, 1], vec![1.0], 4).is_err());
        assert!(csr_from_parts(vec![0, 1], vec![0, 1], vec![1.0, 2.0], 4).is_err());
    }

    #[test]
    fn accepts_a_matrix_with_no_rows() {
        let matrix = csr_from_parts(vec![0], vec![], vec![], 4).unwrap();
        assert_eq!(matrix.n_rows(), 0);
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn tensor_becomes_a_two_dimensional_array() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let array = tensor_to_array2(&tensor).unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array[[1, 0]], 3.0);
    }
}
