use std::collections::HashMap;
use std::sync::Mutex;

use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions};
use scrust_core::error::{Error, Result};

/// A Metal device with its command queue and a cache of compiled kernels.
///
/// Compiling a kernel takes milliseconds; an algorithm dispatching one per
/// iteration would spend all its time in the compiler, so pipelines are built
/// once per source name and reused.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipelines: Mutex<HashMap<&'static str, ComputePipelineState>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(Error::NoGpu)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Compile `source` once and return the cached pipeline for `function_name`.
    pub fn pipeline(
        &self,
        function_name: &'static str,
        source: &str,
    ) -> Result<ComputePipelineState> {
        let mut pipelines = self.pipelines.lock().expect("pipeline cache poisoned");
        if let Some(pipeline) = pipelines.get(function_name) {
            return Ok(pipeline.clone());
        }
        let kernel = |message: String| Error::Kernel {
            name: function_name,
            message,
        };
        let library = self
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(kernel)?;
        let function = library.get_function(function_name, None).map_err(kernel)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(kernel)?;
        pipelines.insert(function_name, pipeline.clone());
        Ok(pipeline)
    }

    /// A shared-memory buffer holding `data`, readable from both CPU and GPU.
    ///
    /// Apple silicon has unified memory, so this is a view rather than a copy
    /// across a bus.
    pub fn buffer<T: Copy>(&self, data: &[T]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// An uninitialised shared buffer with room for `count` elements of `T`.
    pub fn empty_buffer<T>(&self, count: usize) -> metal::Buffer {
        self.device.new_buffer(
            (count * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Read `count` elements back out of a shared buffer.
    ///
    /// # Safety
    /// The buffer must hold at least `count` initialised elements of `T`.
    pub unsafe fn read<T: Copy>(buffer: &metal::Buffer, count: usize) -> Vec<T> {
        std::slice::from_raw_parts(buffer.contents() as *const T, count).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL_KERNEL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void double_values(device float* values [[buffer(0)]],
                              uint index [[thread_position_in_grid]]) {
        values[index] = values[index] * 2.0f;
    }
    "#;

    #[test]
    fn compiles_and_caches_a_pipeline() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        let first = context.pipeline("double_values", TRIVIAL_KERNEL).unwrap();
        let second = context.pipeline("double_values", TRIVIAL_KERNEL).unwrap();
        assert_eq!(
            first.max_total_threads_per_threadgroup(),
            second.max_total_threads_per_threadgroup()
        );
    }

    #[test]
    fn reports_a_helpful_error_for_a_broken_kernel() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let error = context
            .pipeline("missing", "kernel void other() {}")
            .unwrap_err();
        assert!(matches!(error, Error::Kernel { .. }));
    }
}
