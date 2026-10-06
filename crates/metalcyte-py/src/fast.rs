//! Zero-copy bindings for the multi-core preprocessing kernels.
//!
//! These borrow numpy's buffers directly: the index arrays are read in whatever integer
//! dtype scipy holds them (`int32`, `int64` or `uint32`), the value array is updated in
//! place or the dense result is written into an array numpy allocated. The GIL is
//! released while the kernels run, and nothing is copied across the boundary.

use metalcyte_core::preprocess::inplace::{self, Offset};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
            metalcyte_core::pca::pca_dense(
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

/// UMAP with the opt-in multi-core Hogwild optimiser (`tl.umap(..., parallel=True)`).
#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, n_components, n_epochs, min_dist, spread,
                    learning_rate, negative_sample_rate, seed))]
#[allow(clippy::too_many_arguments)]
fn umap_parallel<'py>(
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
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let graph = crate::convert::csr_from_py(indptr, indices, values, n_cols)?;
    let params = metalcyte_core::umap::UmapParams {
        n_components,
        n_epochs,
        min_dist,
        spread,
        learning_rate,
        negative_sample_rate,
        seed,
    };
    let layout = py
        .allow_threads(|| metalcyte_core::umap::umap_parallel(&graph, &params))
        .map_err(to_py_error)?;
    Ok(layout.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Streaming: the pieces a matrix too large for memory is pushed through one
// row block at a time. Each takes a block's CSR arrays (or a dense block) and
// returns its contribution; Python adds the contributions and finishes.
// ---------------------------------------------------------------------------

/// The two `f64` accumulators a block contributes: per-gene `sum(x)` and `sum(x^2)`.
type SumPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn ro_f32<'a>(values: &'a Bound<'_, PyArray1<f32>>) -> PyResult<numpy::PyReadonlyArray1<'a, f32>> {
    values
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("values: {e}")))
}

/// Stored entries per column of a block.
#[pyfunction]
fn column_nnz<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let counts = with_index_slice!(indices, "indices", |ix| {
        fn run<I: Offset>(py: Python<'_>, ix: &[I], n_cols: usize) -> Vec<u32> {
            py.allow_threads(|| inplace::column_nnz(ix, n_cols))
        }
        Ok::<_, PyErr>(run(py, ix, n_cols))
    })?;
    Ok(counts.into_pyarray(py))
}

/// Per-gene `sum(x)` and `sum(x^2)` of a block (`expm1` applied first for the
/// `seurat` flavour), the accumulators `highly_variable_genes_from_sums` finishes.
#[pyfunction]
fn hvg_partial_sums<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyArray1<f32>>,
    n_cols: usize,
    expm1: bool,
) -> PyResult<SumPair<'py>> {
    let ro = ro_f32(values)?;
    let vals = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let (sums, squares) = with_index_slice!(indptr, "indptr", |ip| {
        with_index_slice!(indices, "indices", |ix| {
            fn run<O: Offset, I: Offset>(
                py: Python<'_>,
                ip: &[O],
                ix: &[I],
                v: &[f32],
                n_cols: usize,
                expm1: bool,
            ) -> PyResult<(Vec<f64>, Vec<f64>)> {
                py.allow_threads(|| inplace::hvg_partial_sums(ip, ix, v, n_cols, expm1))
                    .map_err(to_py_error)
            }
            run(py, ip, ix, vals, n_cols, expm1)
        })
    })?;
    Ok((sums.into_pyarray(py), squares.into_pyarray(py)))
}

/// `highly_variable_genes` from accumulated sums (see `hvg_partial_sums`).
#[pyfunction]
fn highly_variable_genes_from_sums<'py>(
    py: Python<'py>,
    sums: &Bound<'py, PyArray1<f64>>,
    squared_sums: &Bound<'py, PyArray1<f64>>,
    n_cells: usize,
    n_top_genes: usize,
    flavor: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use metalcyte_core::preprocess::hvg::{self, HvgFlavor};
    let flavor = match flavor {
        "seurat" => HvgFlavor::Seurat,
        "cell_ranger" => HvgFlavor::CellRanger,
        other => {
            return Err(PyValueError::new_err(format!(
                "flavor must be 'seurat' or 'cell_ranger', got {other:?}"
            )))
        }
    };
    let s = sums
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("sums: {e}")))?;
    let q = squared_sums
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("squared_sums: {e}")))?;
    let (s, q) = (
        s.as_slice()
            .map_err(|_| PyValueError::new_err("sums must be C-contiguous"))?,
        q.as_slice()
            .map_err(|_| PyValueError::new_err("squared_sums must be C-contiguous"))?,
    );
    let genes = py
        .allow_threads(|| hvg::highly_variable_genes_from_sums(s, q, n_cells, n_top_genes, flavor))
        .map_err(to_py_error)?;
    let result = PyDict::new(py);
    result.set_item("means", genes.means.into_pyarray(py))?;
    result.set_item("dispersions", genes.dispersions.into_pyarray(py))?;
    result.set_item(
        "normalised_dispersions",
        genes.normalised_dispersions.into_pyarray(py),
    )?;
    result.set_item("highly_variable", genes.highly_variable.into_pyarray(py))?;
    Ok(result)
}

/// `scale_dense` with the per-gene mean and deviation supplied, for one block of
/// a streamed matrix.
#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, means, deviations, zero_center, max_value=None))]
#[allow(clippy::too_many_arguments)]
fn scale_dense_with<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyArray1<f32>>,
    n_cols: usize,
    means: &Bound<'py, PyArray1<f32>>,
    deviations: &Bound<'py, PyArray1<f32>>,
    zero_center: bool,
    max_value: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ro = ro_f32(values)?;
    let vals = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let m = ro_f32(means)?;
    let d = ro_f32(deviations)?;
    let (m, d) = (
        m.as_slice()
            .map_err(|_| PyValueError::new_err("means must be C-contiguous"))?,
        d.as_slice()
            .map_err(|_| PyValueError::new_err("deviations must be C-contiguous"))?,
    );
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
                    m: &[f32],
                    d: &[f32],
                    zc: bool,
                    mv: Option<f32>,
                    out: &mut [f32],
                ) -> PyResult<()> {
                    py.allow_threads(|| {
                        inplace::scale_into_with(ip, ix, v, n_cols, m, d, zc, mv, out)
                    })
                    .map_err(to_py_error)
                }
                run(py, ip, ix, vals, n_cols, m, d, zero_center, max_value, dest)
            })
        })?;
    }
    Ok(out)
}

/// `block^T block` of a dense `(n_rows, n_cols)` block, in `f64`, on `device`.
#[pyfunction]
fn gram_dense<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyArray2<f32>>,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ro = data
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("data: {e}")))?;
    let shape = ro.shape().to_vec();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let device = crate::convert::device_from_py(device)?;
    let gram = py
        .allow_threads(|| metalcyte_core::pca::gram_dense(slice, shape[0], shape[1], &device))
        .map_err(to_py_error)?;
    let n = shape[1];
    PyArray1::from_vec(py, gram)
        .reshape([n, n])
        .map_err(|e| PyValueError::new_err(format!("gram reshape: {e}")))
}

/// Loadings and explained variance from an accumulated scatter matrix.
#[pyfunction]
fn pca_from_scatter<'py>(
    py: Python<'py>,
    scatter: &Bound<'py, PyArray2<f64>>,
    n_cells: usize,
    n_components: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ro = scatter
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("scatter: {e}")))?;
    let n_genes = ro.shape()[0];
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("scatter must be C-contiguous"))?;
    let device = crate::convert::device_from_py(device)?;
    let (components, variance, ratio) = py
        .allow_threads(|| {
            metalcyte_core::pca::pca_from_scatter(
                slice,
                n_genes,
                n_cells,
                n_components,
                seed,
                &device,
            )
        })
        .map_err(to_py_error)?;
    let result = PyDict::new(py);
    result.set_item("components", components.into_pyarray(py))?;
    result.set_item("explained_variance", variance.into_pyarray(py))?;
    result.set_item("explained_variance_ratio", ratio.into_pyarray(py))?;
    Ok(result)
}

/// Scores of a dense block on `(n_components, n_cols)` loadings, on `device`.
#[pyfunction]
fn project_dense<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyArray2<f32>>,
    components: &Bound<'py, PyArray2<f32>>,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ro = data
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("data: {e}")))?;
    let shape = ro.shape().to_vec();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let loadings = components
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("components: {e}")))?
        .as_array()
        .to_owned();
    let device = crate::convert::device_from_py(device)?;
    let scores = py
        .allow_threads(|| {
            metalcyte_core::pca::project_dense(slice, shape[0], shape[1], &loadings, &device)
        })
        .map_err(to_py_error)?;
    Ok(scores.into_pyarray(py))
}

/// `filter_cells` mask straight off numpy's CSR arrays: no index cast, no copy.
#[pyfunction]
#[pyo3(signature = (indptr, values, min_genes=None, min_counts=None))]
fn filter_cells_mask<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    values: &Bound<'py, PyArray1<f32>>,
    min_genes: Option<usize>,
    min_counts: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ro = ro_f32(values)?;
    let vals = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let mask = with_index_slice!(indptr, "indptr", |ip| {
        fn run<O: Offset>(
            py: Python<'_>,
            ip: &[O],
            v: &[f32],
            min_genes: Option<usize>,
            min_counts: Option<f32>,
        ) -> PyResult<Vec<bool>> {
            py.allow_threads(|| inplace::filter_cells_mask(ip, v, min_genes, min_counts))
                .map_err(to_py_error)
        }
        run(py, ip, vals, min_genes, min_counts)
    })?;
    Ok(mask.into_pyarray(py))
}

/// `filter_genes` mask straight off numpy's CSR arrays: no index cast, no copy.
#[pyfunction]
#[pyo3(signature = (indptr, indices, values, n_cols, min_cells=None, min_counts=None))]
fn filter_genes_mask<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyArray1<f32>>,
    n_cols: usize,
    min_cells: Option<usize>,
    min_counts: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ro = ro_f32(values)?;
    let vals = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let mask = with_index_slice!(indptr, "indptr", |ip| {
        with_index_slice!(indices, "indices", |ix| {
            fn run<O: Offset, I: Offset>(
                py: Python<'_>,
                ip: &[O],
                ix: &[I],
                v: &[f32],
                n_cols: usize,
                min_cells: Option<usize>,
                min_counts: Option<f32>,
            ) -> PyResult<Vec<bool>> {
                py.allow_threads(|| {
                    inplace::filter_genes_mask(ip, ix, v, n_cols, min_cells, min_counts)
                })
                .map_err(to_py_error)
            }
            run(py, ip, ix, vals, n_cols, min_cells, min_counts)
        })
    })?;
    Ok(mask.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(filter_cells_mask, module)?)?;
    module.add_function(wrap_pyfunction!(filter_genes_mask, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_total_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(log1p_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(scale_dense, module)?)?;
    module.add_function(wrap_pyfunction!(pca_dense, module)?)?;
    module.add_function(wrap_pyfunction!(umap_parallel, module)?)?;
    module.add_function(wrap_pyfunction!(column_nnz, module)?)?;
    module.add_function(wrap_pyfunction!(hvg_partial_sums, module)?)?;
    module.add_function(wrap_pyfunction!(highly_variable_genes_from_sums, module)?)?;
    module.add_function(wrap_pyfunction!(scale_dense_with, module)?)?;
    module.add_function(wrap_pyfunction!(gram_dense, module)?)?;
    module.add_function(wrap_pyfunction!(pca_from_scatter, module)?)?;
    module.add_function(wrap_pyfunction!(project_dense, module)?)?;
    Ok(())
}
