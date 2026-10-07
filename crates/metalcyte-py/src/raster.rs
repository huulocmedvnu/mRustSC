//! Bindings: the point-cloud renderer.

use metalcyte_core::raster::{self, RenderSpec, Viewport};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::device_from_py;
use crate::to_py_error;

thread_local! {
    static RENDER_CONTEXT: std::cell::RefCell<Option<Option<metalcyte_gpu::MetalContext>>> =
        const { std::cell::RefCell::new(None) };
}

/// Render points to an RGBA8 image, `(height, width, 4)`, top row first. On a Metal
/// device the GPU renderer runs when a usable GPU exists; otherwise the cores.
#[pyfunction]
#[pyo3(signature = (xy, rgba, width, height, point_size, x_min, x_max, y_min, y_max, background, device))]
#[allow(clippy::too_many_arguments)]
fn render_points<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f32>,
    rgba: PyReadonlyArray1<'py, u32>,
    width: usize,
    height: usize,
    point_size: f32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    background: u32,
    device: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let device = device_from_py(device)?;
    let xy = xy.as_array();
    let xy: Vec<f32> = xy.iter().copied().collect();
    let rgba: Vec<u32> = rgba.as_slice()?.to_vec();
    let spec = RenderSpec {
        width,
        height,
        point_size,
        viewport: Viewport {
            x_min,
            x_max,
            y_min,
            y_max,
        },
        background,
    };
    let image = py
        .allow_threads(|| {
            if device.is_metal() {
                // One context per thread, kept between calls: the device, queue and
                // compiled pipeline then outlive any single plot.
                let rendered = RENDER_CONTEXT.with(|slot| {
                    let mut slot = slot.borrow_mut();
                    let context =
                        slot.get_or_insert_with(|| metalcyte_gpu::MetalContext::new().ok());
                    context.as_ref().map(|context| {
                        metalcyte_gpu::kernels::raster::render_points(context, &xy, &rgba, &spec)
                    })
                });
                if let Some(Ok(image)) = rendered {
                    return Ok(image);
                }
            }
            raster::render_points(&xy, &rgba, &spec)
        })
        .map_err(to_py_error)?;
    let array = ndarray::Array3::from_shape_vec((height, width, 4), image)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(render_points, module)?)?;
    Ok(())
}
