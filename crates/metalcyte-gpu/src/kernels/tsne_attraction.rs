//! The attractive term of FFT-accelerated t-SNE on the GPU.
//!
//! Per iteration the term is one pass over the sparse affinities, about 180 per cell:
//! for every stored `(i, j, p)`, `p * w_ij * (y_i - y_j)` with `w_ij = 1 / (1 + |y_i -
//! y_j|^2)`, summed into row `i`. On the cores at a million cells this is the longest
//! step of an iteration, bound by the gathers of `y_j`. One thread per cell on the GPU
//! reads the same 1.4 GB per iteration from unified memory at the memory's full
//! bandwidth, and the layout, written by the optimiser on the cores, is handed over
//! without a copy: the buffer it lives in is shared.
//!
//! The affinities are uploaded once and the row is summed by one thread in order, so
//! the result is bit-for-bit reproducible across runs.

use std::ffi::c_void;

use metal::{ComputePipelineState, MTLCommandBufferStatus, MTLSize, NSUInteger};
use metalcyte_core::error::Result;
use metalcyte_core::tsne_fft::{Attraction, SparseAffinities};

use crate::context::MetalContext;

const FUNCTION: &str = "tsne_fft_attraction";
/// Lanes of one SIMD group on Apple GPUs.
const SIMD_WIDTH: usize = 32;
/// Cells handled by one threadgroup, one SIMD group each.
const ROWS_PER_GROUP: usize = 8;

/// The attractive term on Metal, keeping the affinities and the working buffers on the
/// device between iterations.
pub struct MetalAttraction {
    context: MetalContext,
    pipeline: ComputePipelineState,
    n_cells: usize,
    indptr: metal::Buffer,
    indices: metal::Buffer,
    values: metal::Buffer,
    layout: metal::Buffer,
    out: metal::Buffer,
    error: metal::Buffer,
    /// Which affinities the device buffers hold, by address and length.
    loaded: Option<(usize, usize)>,
}

impl MetalAttraction {
    pub fn new(context: MetalContext) -> Result<Self> {
        let pipeline = context.pipeline(FUNCTION, SOURCE)?;
        let (indptr, indices, values) = (
            context_empty(&context, 1),
            context_empty(&context, 1),
            context_empty(&context, 1),
        );
        let (layout, out, error) = (
            context_empty(&context, 2),
            context_empty(&context, 2),
            context_empty(&context, 1),
        );
        Ok(Self {
            context,
            pipeline,
            n_cells: 0,
            indptr,
            indices,
            values,
            layout,
            out,
            error,
            loaded: None,
        })
    }

    fn load(&mut self, affinities: &SparseAffinities) {
        let key = (affinities.values.as_ptr() as usize, affinities.values.len());
        if self.loaded == Some(key) {
            return;
        }
        let n_cells = affinities.indptr.len() - 1;
        let indptr: Vec<u32> = affinities.indptr.iter().map(|&v| v as u32).collect();
        self.indptr = self.context.buffer(&indptr);
        self.indices = self.context.buffer(non_empty(&affinities.indices, &[0]));
        self.values = self.context.buffer(non_empty(&affinities.values, &[0.0]));
        self.layout = self.context.empty_buffer::<f32>(2 * n_cells.max(1));
        self.out = self.context.empty_buffer::<f32>(2 * n_cells.max(1));
        self.error = self.context.empty_buffer::<f32>(n_cells.max(1));
        self.n_cells = n_cells;
        self.loaded = Some(key);
    }
}

fn context_empty(context: &MetalContext, n: usize) -> metal::Buffer {
    context.empty_buffer::<f32>(n)
}

impl Attraction for MetalAttraction {
    fn compute(
        &mut self,
        layout: &[f32],
        affinities: &SparseAffinities,
        exaggeration: f32,
        normaliser: f64,
        compute_error: bool,
    ) -> (Vec<f32>, Option<f64>) {
        self.load(affinities);
        let n_cells = self.n_cells;
        debug_assert_eq!(layout.len(), 2 * n_cells);
        // SAFETY: the shared buffer holds 2 * n_cells floats; the device is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(
                layout.as_ptr(),
                self.layout.contents() as *mut f32,
                2 * n_cells,
            );
        }
        let command = self.context.queue().new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.indptr), 0);
        encoder.set_buffer(1, Some(&self.indices), 0);
        encoder.set_buffer(2, Some(&self.values), 0);
        encoder.set_buffer(3, Some(&self.layout), 0);
        encoder.set_buffer(4, Some(&self.out), 0);
        encoder.set_buffer(5, Some(&self.error), 0);
        set_scalar(encoder, 6, &exaggeration);
        set_scalar(encoder, 7, &(normaliser as f32));
        set_scalar(encoder, 8, &(u32::from(compute_error)));
        set_scalar(encoder, 9, &(n_cells as u32));
        // One SIMD group of 32 lanes per cell, ROWS_PER_GROUP cells per threadgroup: the
        // lanes stride the cell's affinities, so the index and value reads coalesce and
        // the gathers of the neighbours' coordinates run 32 at a time.
        let groups = n_cells.div_ceil(ROWS_PER_GROUP).max(1);
        encoder.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new((ROWS_PER_GROUP * SIMD_WIDTH) as u64, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            // A device failure mid-run: fall back to the cores for this iteration.
            let mut cpu = metalcyte_core::tsne_fft::CpuAttraction;
            return cpu.compute(layout, affinities, exaggeration, normaliser, compute_error);
        }
        // SAFETY: the kernel wrote every cell's two components and its error term.
        let attraction = unsafe { MetalContext::read::<f32>(&self.out, 2 * n_cells) };
        let error = compute_error.then(|| {
            let per_cell = unsafe { MetalContext::read::<f32>(&self.error, n_cells) };
            per_cell.iter().map(|&v| f64::from(v)).sum::<f64>()
        });
        (attraction, error)
    }
}

/// Build a [`MetalAttraction`] when a usable GPU exists.
pub fn metal_attraction() -> Result<MetalAttraction> {
    MetalAttraction::new(MetalContext::new()?)
}

fn non_empty<'a, T>(slice: &'a [T], fallback: &'a [T]) -> &'a [T] {
    if slice.is_empty() {
        fallback
    } else {
        slice
    }
}

fn set_scalar<T>(encoder: &metal::ComputeCommandEncoderRef, index: NSUInteger, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as NSUInteger,
        value as *const T as *const c_void,
    );
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One SIMD group per cell. Its 32 lanes stride the cell's stored affinities, each
// summing p * w * (y_i - y_j) and, when asked, the cell's share of sum p log(p / q)
// with q = w / Z; a SIMD reduction folds the lanes and lane 0 writes the row.
kernel void tsne_fft_attraction(device const uint *indptr [[buffer(0)]],
                                device const uint *indices [[buffer(1)]],
                                device const float *values [[buffer(2)]],
                                device const float *layout [[buffer(3)]],
                                device float *out [[buffer(4)]],
                                device float *error [[buffer(5)]],
                                constant float &exaggeration [[buffer(6)]],
                                constant float &normaliser [[buffer(7)]],
                                constant uint &compute_error [[buffer(8)]],
                                constant uint &n_cells [[buffer(9)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group * 8 + simd_id;
    if (i >= n_cells) {
        return;
    }
    const float yi0 = layout[2 * i];
    const float yi1 = layout[2 * i + 1];
    float a0 = 0.0f;
    float a1 = 0.0f;
    float kl = 0.0f;
    const uint end = indptr[i + 1];
    for (uint at = indptr[i] + lane; at < end; at += 32) {
        const uint j = indices[at];
        const float p = values[at] * exaggeration;
        const float d0 = yi0 - layout[2 * j];
        const float d1 = yi1 - layout[2 * j + 1];
        const float w = 1.0f / (1.0f + d0 * d0 + d1 * d1);
        a0 = fma(p * w, d0, a0);
        a1 = fma(p * w, d1, a1);
        if (compute_error != 0 && p > 0.0f) {
            const float q = max(w / normaliser, 2.220446049250313e-16f);
            kl += p * log(max(p, 2.220446049250313e-16f) / q);
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    kl = simd_sum(kl);
    if (lane == 0) {
        out[2 * i] = a0;
        out[2 * i + 1] = a1;
        error[i] = kl;
    }
}
"#;
