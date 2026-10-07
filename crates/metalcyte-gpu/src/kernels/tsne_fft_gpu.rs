//! FFT-accelerated t-SNE with the per-cell work on the GPU.
//!
//! One iteration is five small kernels and two pieces of host work, all over buffers in
//! unified memory that neither side copies:
//!
//! 1. the attractive term, one SIMD group per cell striding its affinities (issued first
//!    and left to run while the grid is prepared);
//! 2. placement: each cell's box and Lagrange weights on both axes;
//! 3. on the host, a counting sort of the cells by box, in parallel and in a fixed order;
//! 4. spreading: one SIMD group per box sums the charges of the box's cells onto the
//!    box's nine nodes, so no two groups write the same node and no atomics are needed;
//! 5. on the host, the convolution of the node charges by FFT (Accelerate);
//! 6. gathering: each cell reads its potentials back from its nine nodes;
//! 7. the update: the gradient from the attraction and the potentials, scikit-learn's
//!    gains and momentum, the step.
//!
//! Every reduction runs in a fixed order (lanes stride the same elements each time, the
//! host sums fixed chunks), so a seed gives the same bytes on the same machine, as the
//! engine on the cores does. The arithmetic is the core's to the operation; the two
//! engines differ only in the order f32 sums are taken.

use std::ffi::c_void;

use metal::{ComputePipelineState, MTLCommandBufferStatus, MTLSize, NSUInteger};
use metalcyte_core::error::{Error, Result};
use metalcyte_core::tsne_fft::{convolve_charges, Engine, Grid, SparseAffinities, NODES_PER_BOX};
use rayon::prelude::*;

use crate::context::MetalContext;

const SIMD_WIDTH: usize = 32;
const ROWS_PER_GROUP: usize = 8;
const MIN_GAIN: f32 = 0.01;

/// The GPU engine: see the module documentation.
pub struct GpuEngine {
    context: MetalContext,
    /// The attractive term's own queue: command buffers on one queue run in order, so
    /// it would otherwise serialise with the grid kernels it is meant to overlap.
    attraction_queue: metal::CommandQueue,
    attraction: ComputePipelineState,
    kl: ComputePipelineState,
    placement: ComputePipelineState,
    spread: ComputePipelineState,
    gather: ComputePipelineState,
    update: ComputePipelineState,
    n_cells: usize,
    layout: metal::Buffer,
    step_buffer: metal::Buffer,
    gains: metal::Buffer,
    attraction_buffer: metal::Buffer,
    kl_buffer: metal::Buffer,
    norm_buffer: metal::Buffer,
    indptr: metal::Buffer,
    indices: metal::Buffer,
    values: metal::Buffer,
    boxes: metal::Buffer,
    weights: metal::Buffer,
    order: metal::Buffer,
    offsets: metal::Buffer,
    charges: metal::Buffer,
    potentials: metal::Buffer,
    phi: metal::Buffer,
    zterm: metal::Buffer,
    grid_nodes: usize,
    grid_boxes: usize,
    /// The FFT work buffers (`size * size` each) and the kernel transform for the
    /// current grid, with the twiddle table of the current size.
    fft_re: metal::Buffer,
    fft_im: metal::Buffer,
    kernel_tr: metal::Buffer,
    twiddle_re: metal::Buffer,
    twiddle_im: metal::Buffer,
    fft_size: usize,
    kernel_key: Option<(usize, u32)>,
    fft_rows_pipeline: ComputePipelineState,
    transpose_pipeline: ComputePipelineState,
    pack_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    unpack_pipeline: ComputePipelineState,
    /// Seconds per phase when `METALCYTE_PROFILE` is set, printed on drop.
    profile: Option<[f64; 8]>,
}

impl Drop for GpuEngine {
    fn drop(&mut self) {
        if let Some(p) = self.profile {
            eprintln!(
                "tsne gpu profile: placement {:.2} s, sort {:.2} s, spread {:.2} s, convolve {:.2} s, gather {:.2} s, attraction wait {:.2} s, kl {:.2} s, update {:.2} s",
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            );
        }
    }
}

/// Build a [`GpuEngine`] when a usable GPU exists.
pub fn gpu_engine(initial: Vec<f32>, affinities: SparseAffinities) -> Result<Box<dyn Engine>> {
    let context = MetalContext::new()?;
    Ok(Box::new(GpuEngine::new(context, initial, affinities)?))
}

impl GpuEngine {
    pub fn new(
        context: MetalContext,
        initial: Vec<f32>,
        affinities: SparseAffinities,
    ) -> Result<Self> {
        let n_cells = initial.len() / 2;
        if affinities.indptr.len() != n_cells + 1 {
            return Err(Error::shape(
                format!("affinities for {n_cells} cells"),
                format!("{} rows", affinities.indptr.len().saturating_sub(1)),
            ));
        }
        let pipeline = |name| context.pipeline(name, SOURCE);
        let attraction = pipeline("tsne_fft_attraction")?;
        let kl = pipeline("tsne_fft_kl")?;
        let placement = pipeline("tsne_fft_placement")?;
        let spread = pipeline("tsne_fft_spread")?;
        let gather = pipeline("tsne_fft_gather")?;
        let update = pipeline("tsne_fft_update")?;
        let fft_rows_pipeline = pipeline("fft_rows")?;
        let transpose_pipeline = pipeline("transpose_square")?;
        let pack_pipeline = pipeline("fft_pack")?;
        let multiply_pipeline = pipeline("fft_multiply")?;
        let unpack_pipeline = pipeline("fft_unpack")?;
        let indptr: Vec<u32> = affinities.indptr.iter().map(|&v| v as u32).collect();
        let n = n_cells.max(1);
        let attraction_queue = context.device().new_command_queue();
        let engine = Self {
            attraction_queue,
            layout: context.buffer(&initial),
            step_buffer: context.buffer(&vec![0f32; 2 * n]),
            gains: context.buffer(&vec![1f32; 2 * n]),
            attraction_buffer: context.empty_buffer::<f32>(2 * n),
            kl_buffer: context.empty_buffer::<f32>(n),
            norm_buffer: context.empty_buffer::<f32>(n),
            indptr: context.buffer(&indptr),
            indices: context.buffer(non_empty(&affinities.indices, &[0])),
            values: context.buffer(non_empty(&affinities.values, &[0.0])),
            boxes: context.empty_buffer::<u32>(2 * n),
            weights: context.empty_buffer::<f32>(2 * NODES_PER_BOX * n),
            order: context.empty_buffer::<u32>(n),
            offsets: context.empty_buffer::<u32>(1),
            charges: context.empty_buffer::<f32>(1),
            potentials: context.empty_buffer::<f32>(1),
            phi: context.empty_buffer::<f32>(4 * n),
            zterm: context.empty_buffer::<f32>(n),
            grid_nodes: 0,
            grid_boxes: 0,
            fft_re: context.empty_buffer::<f32>(1),
            fft_im: context.empty_buffer::<f32>(1),
            kernel_tr: context.empty_buffer::<f32>(1),
            twiddle_re: context.empty_buffer::<f32>(1),
            twiddle_im: context.empty_buffer::<f32>(1),
            fft_size: 0,
            kernel_key: None,
            fft_rows_pipeline,
            transpose_pipeline,
            pack_pipeline,
            multiply_pipeline,
            unpack_pipeline,
            profile: std::env::var_os("METALCYTE_PROFILE").map(|_| [0.0; 8]),
            n_cells,
            context,
            attraction,
            kl,
            placement,
            spread,
            gather,
            update,
        };
        Ok(engine)
    }

    fn slice<T>(buffer: &metal::Buffer, len: usize) -> &[T] {
        // SAFETY: every buffer is shared and at least `len` elements long; callers only
        // read while no command buffer is in flight on it.
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) }
    }

    /// A shared buffer is host-writable memory the handle merely names, so a mutable
    /// view through a shared handle is what the API is for.
    #[allow(clippy::mut_from_ref)]
    fn slice_mut<T>(buffer: &metal::Buffer, len: usize) -> &mut [T] {
        // SAFETY: as above, for a write while the device is idle on the buffer.
        unsafe { std::slice::from_raw_parts_mut(buffer.contents() as *mut T, len) }
    }

    fn ensure_grid(&mut self, grid: &Grid) {
        if grid.n_nodes != self.grid_nodes {
            let grid_len = grid.n_nodes * grid.n_nodes;
            self.charges = self.context.empty_buffer::<f32>(4 * grid_len);
            self.potentials = self.context.empty_buffer::<f32>(4 * grid_len);
            self.grid_nodes = grid.n_nodes;
        }
        if grid.n_boxes != self.grid_boxes {
            self.offsets = self
                .context
                .empty_buffer::<u32>(grid.n_boxes * grid.n_boxes + 1);
            self.grid_boxes = grid.n_boxes;
        }
    }

    /// Cells sorted by box (`b0 * n_boxes + b1`), stable within each box, in parallel:
    /// per-chunk histograms give every chunk its own range inside each box.
    fn sort_by_box(&self, n_boxes: usize) {
        let n_cells = self.n_cells;
        let n_bins = n_boxes * n_boxes;
        let boxes: &[u32] = Self::slice(&self.boxes, 2 * n_cells);
        let n_chunks = (rayon::current_num_threads() * 2).max(1);
        let chunk = n_cells.div_ceil(n_chunks).max(1);
        let bin = |i: usize| boxes[2 * i] as usize * n_boxes + boxes[2 * i + 1] as usize;
        let histograms: Vec<Vec<u32>> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let mut h = vec![0u32; n_bins];
                for i in c * chunk..((c + 1) * chunk).min(n_cells) {
                    h[bin(i)] += 1;
                }
                h
            })
            .collect();
        let offsets: &mut [u32] = Self::slice_mut(&self.offsets, n_bins + 1);
        offsets[0] = 0;
        let mut running = 0u32;
        for (b, slot) in offsets.iter_mut().enumerate().skip(1) {
            running += histograms.iter().map(|h| h[b - 1]).sum::<u32>();
            *slot = running;
        }
        // Start of each chunk's range inside each bin.
        let mut starts: Vec<Vec<u32>> = vec![vec![0u32; n_bins]; n_chunks];
        for (b, &offset) in offsets.iter().enumerate().take(n_bins) {
            let mut at = offset;
            for c in 0..n_chunks {
                starts[c][b] = at;
                at += histograms[c][b];
            }
        }
        let order_ptr = self.order.contents() as usize;
        starts
            .into_par_iter()
            .enumerate()
            .for_each(|(c, mut cursor)| {
                // SAFETY: each chunk writes only the ranges it was given, which are disjoint.
                let order =
                    unsafe { std::slice::from_raw_parts_mut(order_ptr as *mut u32, n_cells) };
                for i in c * chunk..((c + 1) * chunk).min(n_cells) {
                    let b = bin(i);
                    order[cursor[b] as usize] = i as u32;
                    cursor[b] += 1;
                }
            });
    }

    fn run(
        &self,
        name: &'static str,
        encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
    ) -> Result<()> {
        let command = self.context.queue().new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Kernel {
                name,
                message: format!("dispatch ended in state {:?}", command.status()),
            });
        }
        Ok(())
    }

    fn per_cell_grid(&self, pipeline: &ComputePipelineState) -> (MTLSize, MTLSize) {
        let width = (pipeline.max_total_threads_per_threadgroup() as usize).min(256);
        (
            MTLSize::new(self.n_cells.div_ceil(width).max(1) as u64, 1, 1),
            MTLSize::new(width as u64, 1, 1),
        )
    }

    /// Largest FFT side the row kernel serves: its threadgroup memory holds one row of
    /// split complex values.
    const MAX_GPU_FFT: usize = 2048;

    /// The convolution of the node charges by the squared Cauchy kernel on the GPU:
    /// pack two charge grids per complex field, forward FFT (rows, transpose, rows,
    /// transpose), multiply by the kernel's real transform, inverse FFT, unpack. Falls
    /// back to the host when the padded size exceeds [`Self::MAX_GPU_FFT`].
    fn convolve_on_device(&mut self, grid: &Grid) -> Result<bool> {
        let n_nodes = grid.n_nodes;
        let size = (2 * n_nodes).next_power_of_two();
        if size > Self::MAX_GPU_FFT {
            return Ok(false);
        }
        if size != self.fft_size {
            self.fft_re = self.context.empty_buffer::<f32>(size * size);
            self.fft_im = self.context.empty_buffer::<f32>(size * size);
            // Twiddles w_k = exp(-2 pi i k / size), k < size / 2, from f64.
            let (re, im): (Vec<f32>, Vec<f32>) = (0..size / 2)
                .map(|k| {
                    let angle = -2.0 * std::f64::consts::PI * k as f64 / size as f64;
                    (angle.cos() as f32, angle.sin() as f32)
                })
                .unzip();
            self.twiddle_re = self.context.buffer(&re);
            self.twiddle_im = self.context.buffer(&im);
            self.fft_size = size;
            self.kernel_key = None;
        }
        let key = (n_nodes, grid.node_spacing.to_bits());
        if self.kernel_key != Some(key) {
            let fft = metalcyte_core::tsne_fft::Fft2d::new(size);
            let transform =
                metalcyte_core::tsne_fft::kernel_transform(&fft, size, n_nodes, grid.node_spacing);
            self.kernel_tr = self.context.buffer(&transform);
            self.kernel_key = Some(key);
        }
        let log2n = size.trailing_zeros();
        let command = self.context.queue().new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let fft_pass = |encoder: &metal::ComputeCommandEncoderRef, inverse: u32| {
            for _ in 0..2 {
                encoder.set_compute_pipeline_state(&self.fft_rows_pipeline);
                encoder.set_buffer(0, Some(&self.fft_re), 0);
                encoder.set_buffer(1, Some(&self.fft_im), 0);
                encoder.set_buffer(2, Some(&self.twiddle_re), 0);
                encoder.set_buffer(3, Some(&self.twiddle_im), 0);
                set_scalar(encoder, 4, &(size as u32));
                set_scalar(encoder, 5, &log2n);
                set_scalar(encoder, 6, &inverse);
                encoder
                    .set_threadgroup_memory_length(0, (2 * size * size_of::<f32>()) as NSUInteger);
                encoder.dispatch_thread_groups(
                    MTLSize::new(size as u64, 1, 1),
                    MTLSize::new((size / 2).min(1024) as u64, 1, 1),
                );
                encoder.set_compute_pipeline_state(&self.transpose_pipeline);
                encoder.set_buffer(0, Some(&self.fft_re), 0);
                encoder.set_buffer(1, Some(&self.fft_im), 0);
                set_scalar(encoder, 2, &(size as u32));
                encoder.set_threadgroup_memory_length(
                    0,
                    (4 * 33 * 32 * size_of::<f32>()) as NSUInteger,
                );
                let tiles = size.div_ceil(32) as u64;
                encoder
                    .dispatch_thread_groups(MTLSize::new(tiles, tiles, 1), MTLSize::new(32, 8, 1));
            }
        };
        for pair in 0..2u32 {
            encoder.set_compute_pipeline_state(&self.pack_pipeline);
            encoder.set_buffer(0, Some(&self.charges), 0);
            encoder.set_buffer(1, Some(&self.fft_re), 0);
            encoder.set_buffer(2, Some(&self.fft_im), 0);
            set_scalar(encoder, 3, &(n_nodes as u32));
            set_scalar(encoder, 4, &(size as u32));
            set_scalar(encoder, 5, &pair);
            encoder.dispatch_thread_groups(
                MTLSize::new(size.div_ceil(16) as u64, size.div_ceil(16) as u64, 1),
                MTLSize::new(16, 16, 1),
            );
            fft_pass(encoder, 0);
            encoder.set_compute_pipeline_state(&self.multiply_pipeline);
            encoder.set_buffer(0, Some(&self.fft_re), 0);
            encoder.set_buffer(1, Some(&self.fft_im), 0);
            encoder.set_buffer(2, Some(&self.kernel_tr), 0);
            set_scalar(encoder, 3, &((size * size) as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new((size * size).div_ceil(256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            fft_pass(encoder, 1);
            encoder.set_compute_pipeline_state(&self.unpack_pipeline);
            encoder.set_buffer(0, Some(&self.fft_re), 0);
            encoder.set_buffer(1, Some(&self.fft_im), 0);
            encoder.set_buffer(2, Some(&self.potentials), 0);
            set_scalar(encoder, 3, &(n_nodes as u32));
            set_scalar(encoder, 4, &(size as u32));
            set_scalar(encoder, 5, &pair);
            set_scalar(encoder, 6, &(1.0f32 / (size * size) as f32));
            encoder.dispatch_thread_groups(
                MTLSize::new(n_nodes.div_ceil(16) as u64, n_nodes.div_ceil(16) as u64, 1),
                MTLSize::new(16, 16, 1),
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Kernel {
                name: "fft_rows",
                message: format!("dispatch ended in state {:?}", command.status()),
            });
        }
        Ok(true)
    }

    /// A deterministic sum of a buffer of per-cell f32 terms: fixed chunks, summed in order.
    fn chunked_sum(buffer: &metal::Buffer, n: usize) -> f64 {
        let values: &[f32] = Self::slice(buffer, n);
        let partials: Vec<f64> = values
            .par_chunks(8192)
            .map(|chunk| chunk.iter().map(|&v| f64::from(v)).sum::<f64>())
            .collect();
        partials.iter().sum()
    }

    fn step_inner(
        &mut self,
        exaggeration: f32,
        momentum: f32,
        learning_rate: f32,
        reset: bool,
        compute_error: bool,
    ) -> Result<(f64, Option<f64>)> {
        let n_cells = self.n_cells;
        if reset {
            Self::slice_mut::<f32>(&self.step_buffer, 2 * n_cells).fill(0.0);
            Self::slice_mut::<f32>(&self.gains, 2 * n_cells).fill(1.0);
        }
        // 1. The attractive term, left running.
        let attraction_command = self.attraction_queue.new_command_buffer().to_owned();
        {
            let encoder = attraction_command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.attraction);
            encoder.set_buffer(0, Some(&self.indptr), 0);
            encoder.set_buffer(1, Some(&self.indices), 0);
            encoder.set_buffer(2, Some(&self.values), 0);
            encoder.set_buffer(3, Some(&self.layout), 0);
            encoder.set_buffer(4, Some(&self.attraction_buffer), 0);
            set_scalar(encoder, 5, &exaggeration);
            set_scalar(encoder, 6, &(n_cells as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new(n_cells.div_ceil(ROWS_PER_GROUP).max(1) as u64, 1, 1),
                MTLSize::new((ROWS_PER_GROUP * SIMD_WIDTH) as u64, 1, 1),
            );
            encoder.end_encoding();
            attraction_command.commit();
        }
        let mut mark = std::time::Instant::now();
        let mut lap = |slot: usize, profile: &mut Option<[f64; 8]>| {
            if let Some(p) = profile {
                p[slot] += mark.elapsed().as_secs_f64();
            }
            mark = std::time::Instant::now();
        };
        // 2. The grid and each cell's place on it.
        let grid = Grid::for_layout(Self::slice::<f32>(&self.layout, 2 * n_cells));
        self.ensure_grid(&grid);
        let nodes = Grid::nodes_within_box();
        let denominators = Grid::lagrange_denominators();
        let (groups, threads) = self.per_cell_grid(&self.placement);
        self.run("tsne_fft_placement", |encoder| {
            encoder.set_compute_pipeline_state(&self.placement);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.boxes), 0);
            encoder.set_buffer(2, Some(&self.weights), 0);
            set_scalar(encoder, 3, &grid.lo);
            set_scalar(encoder, 4, &grid.box_width);
            set_scalar(encoder, 5, &(grid.n_boxes as u32));
            set_scalar(encoder, 6, &nodes);
            set_scalar(encoder, 7, &denominators);
            set_scalar(encoder, 8, &(n_cells as u32));
            encoder.dispatch_thread_groups(groups, threads);
        })?;
        lap(0, &mut self.profile);
        // 3. Cells by box, on the host.
        self.sort_by_box(grid.n_boxes);
        lap(1, &mut self.profile);
        // 4. Charges onto the nodes, one SIMD group per box.
        let n_bins = grid.n_boxes * grid.n_boxes;
        self.run("tsne_fft_spread", |encoder| {
            encoder.set_compute_pipeline_state(&self.spread);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.weights), 0);
            encoder.set_buffer(2, Some(&self.order), 0);
            encoder.set_buffer(3, Some(&self.offsets), 0);
            encoder.set_buffer(4, Some(&self.charges), 0);
            set_scalar(encoder, 5, &(grid.n_boxes as u32));
            encoder.set_threadgroup_memory_length(
                0,
                (ROWS_PER_GROUP * NODES_PER_BOX * NODES_PER_BOX * 4 * size_of::<f32>()) as NSUInteger,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(n_bins as u64, 1, 1),
                MTLSize::new((ROWS_PER_GROUP * SIMD_WIDTH) as u64, 1, 1),
            );
        })?;
        lap(2, &mut self.profile);
        // 5. The convolution, on the device, or on the host through Accelerate when the
        // padded grid is too large for the row kernel.
        let grid_len = grid.n_nodes * grid.n_nodes;
        if !self.convolve_on_device(&grid)? {
            let potentials =
                convolve_charges(&grid, Self::slice::<f32>(&self.charges, 4 * grid_len));
            Self::slice_mut::<f32>(&self.potentials, 4 * grid_len).copy_from_slice(&potentials);
        }
        lap(3, &mut self.profile);
        // 6. Potentials back at the cells, and the terms of Z.
        let (groups, threads) = self.per_cell_grid(&self.gather);
        self.run("tsne_fft_gather", |encoder| {
            encoder.set_compute_pipeline_state(&self.gather);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.boxes), 0);
            encoder.set_buffer(2, Some(&self.weights), 0);
            encoder.set_buffer(3, Some(&self.potentials), 0);
            encoder.set_buffer(4, Some(&self.phi), 0);
            encoder.set_buffer(5, Some(&self.zterm), 0);
            set_scalar(encoder, 6, &(grid.n_nodes as u32));
            set_scalar(encoder, 7, &(n_cells as u32));
            encoder.dispatch_thread_groups(groups, threads);
        })?;
        let normaliser =
            (Self::chunked_sum(&self.zterm, n_cells) - n_cells as f64).max(f64::EPSILON);
        // 7. The attraction must be in by now; the objective when asked; the step.
        attraction_command.wait_until_completed();
        if attraction_command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Kernel {
                name: "tsne_fft_attraction",
                message: format!("dispatch ended in state {:?}", attraction_command.status()),
            });
        }
        let error = if compute_error {
            self.run("tsne_fft_kl", |encoder| {
                encoder.set_compute_pipeline_state(&self.kl);
                encoder.set_buffer(0, Some(&self.indptr), 0);
                encoder.set_buffer(1, Some(&self.indices), 0);
                encoder.set_buffer(2, Some(&self.values), 0);
                encoder.set_buffer(3, Some(&self.layout), 0);
                encoder.set_buffer(4, Some(&self.kl_buffer), 0);
                set_scalar(encoder, 5, &exaggeration);
                set_scalar(encoder, 6, &(normaliser as f32));
                set_scalar(encoder, 7, &(n_cells as u32));
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_cells.div_ceil(ROWS_PER_GROUP).max(1) as u64, 1, 1),
                    MTLSize::new((ROWS_PER_GROUP * SIMD_WIDTH) as u64, 1, 1),
                );
            })?;
            Some(Self::chunked_sum(&self.kl_buffer, n_cells))
        } else {
            None
        };
        lap(6, &mut self.profile);
        let (groups, threads) = self.per_cell_grid(&self.update);
        self.run("tsne_fft_update", |encoder| {
            encoder.set_compute_pipeline_state(&self.update);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.step_buffer), 0);
            encoder.set_buffer(2, Some(&self.gains), 0);
            encoder.set_buffer(3, Some(&self.attraction_buffer), 0);
            encoder.set_buffer(4, Some(&self.phi), 0);
            encoder.set_buffer(5, Some(&self.norm_buffer), 0);
            set_scalar(encoder, 6, &(normaliser as f32));
            set_scalar(encoder, 7, &momentum);
            set_scalar(encoder, 8, &learning_rate);
            set_scalar(encoder, 9, &MIN_GAIN);
            set_scalar(encoder, 10, &(n_cells as u32));
            encoder.dispatch_thread_groups(groups, threads);
        })?;
        let norm_square = if compute_error {
            Self::chunked_sum(&self.norm_buffer, n_cells)
        } else {
            f64::INFINITY
        };
        lap(7, &mut self.profile);
        Ok((norm_square, error))
    }

    /// The device convolution against the host one, for tests: both from the same random
    /// charges on the current grid. Returns the relative RMS difference.
    pub fn convolution_difference_for_test(&mut self, grid: &Grid, charges: &[f32]) -> Result<f64> {
        let grid_len = grid.n_nodes * grid.n_nodes;
        self.ensure_grid(grid);
        Self::slice_mut::<f32>(&self.charges, 4 * grid_len).copy_from_slice(charges);
        let host = convolve_charges(grid, charges);
        if !self.convolve_on_device(grid)? {
            return Err(Error::parameter(
                "grid",
                "small enough for the device FFT",
                grid.n_nodes,
            ));
        }
        let device: &[f32] = Self::slice(&self.potentials, 4 * grid_len);
        let (mut diff, mut norm) = (0.0f64, 0.0f64);
        for (a, b) in host.iter().zip(device) {
            diff += (f64::from(*a) - f64::from(*b)).powi(2);
            norm += f64::from(*a).powi(2);
        }
        Ok((diff / norm.max(1e-30)).sqrt())
    }

    /// The repulsive forces and `Z` for the current layout, for tests against the engine
    /// on the cores.
    pub fn repulsion_for_test(&mut self) -> Result<(Vec<f64>, f64)> {
        let n_cells = self.n_cells;
        let grid = Grid::for_layout(Self::slice::<f32>(&self.layout, 2 * n_cells));
        self.ensure_grid(&grid);
        let nodes = Grid::nodes_within_box();
        let denominators = Grid::lagrange_denominators();
        let (groups, threads) = self.per_cell_grid(&self.placement);
        self.run("tsne_fft_placement", |encoder| {
            encoder.set_compute_pipeline_state(&self.placement);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.boxes), 0);
            encoder.set_buffer(2, Some(&self.weights), 0);
            set_scalar(encoder, 3, &grid.lo);
            set_scalar(encoder, 4, &grid.box_width);
            set_scalar(encoder, 5, &(grid.n_boxes as u32));
            set_scalar(encoder, 6, &nodes);
            set_scalar(encoder, 7, &denominators);
            set_scalar(encoder, 8, &(n_cells as u32));
            encoder.dispatch_thread_groups(groups, threads);
        })?;
        self.sort_by_box(grid.n_boxes);
        let n_bins = grid.n_boxes * grid.n_boxes;
        self.run("tsne_fft_spread", |encoder| {
            encoder.set_compute_pipeline_state(&self.spread);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.weights), 0);
            encoder.set_buffer(2, Some(&self.order), 0);
            encoder.set_buffer(3, Some(&self.offsets), 0);
            encoder.set_buffer(4, Some(&self.charges), 0);
            set_scalar(encoder, 5, &(grid.n_boxes as u32));
            encoder.set_threadgroup_memory_length(
                0,
                (ROWS_PER_GROUP * NODES_PER_BOX * NODES_PER_BOX * 4 * size_of::<f32>()) as NSUInteger,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(n_bins as u64, 1, 1),
                MTLSize::new((ROWS_PER_GROUP * SIMD_WIDTH) as u64, 1, 1),
            );
        })?;
        let grid_len = grid.n_nodes * grid.n_nodes;
        let potentials = convolve_charges(&grid, Self::slice::<f32>(&self.charges, 4 * grid_len));
        Self::slice_mut::<f32>(&self.potentials, 4 * grid_len).copy_from_slice(&potentials);
        let (groups, threads) = self.per_cell_grid(&self.gather);
        self.run("tsne_fft_gather", |encoder| {
            encoder.set_compute_pipeline_state(&self.gather);
            encoder.set_buffer(0, Some(&self.layout), 0);
            encoder.set_buffer(1, Some(&self.boxes), 0);
            encoder.set_buffer(2, Some(&self.weights), 0);
            encoder.set_buffer(3, Some(&self.potentials), 0);
            encoder.set_buffer(4, Some(&self.phi), 0);
            encoder.set_buffer(5, Some(&self.zterm), 0);
            set_scalar(encoder, 6, &(grid.n_nodes as u32));
            set_scalar(encoder, 7, &(n_cells as u32));
            encoder.dispatch_thread_groups(groups, threads);
        })?;
        let normaliser =
            (Self::chunked_sum(&self.zterm, n_cells) - n_cells as f64).max(f64::EPSILON);
        let layout: &[f32] = Self::slice(&self.layout, 2 * n_cells);
        let phi: &[f32] = Self::slice(&self.phi, 4 * n_cells);
        let mut forces = vec![0f64; 2 * n_cells];
        for i in 0..n_cells {
            let (y0, y1) = (layout[2 * i] as f64, layout[2 * i + 1] as f64);
            let p = &phi[4 * i..4 * i + 4];
            forces[2 * i] = (y0 * p[0] as f64 - p[1] as f64) / normaliser;
            forces[2 * i + 1] = (y1 * p[0] as f64 - p[2] as f64) / normaliser;
        }
        Ok((forces, normaliser))
    }
}

impl Engine for GpuEngine {
    fn step(
        &mut self,
        exaggeration: f32,
        momentum: f32,
        learning_rate: f32,
        reset: bool,
        compute_error: bool,
    ) -> (f64, Option<f64>) {
        self.step_inner(exaggeration, momentum, learning_rate, reset, compute_error)
            .expect(
                "the GPU t-SNE step failed after its kernels compiled and its buffers were built",
            )
    }

    fn layout(&self) -> Vec<f32> {
        Self::slice::<f32>(&self.layout, 2 * self.n_cells).to_vec()
    }
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

constexpr constant uint NODES = 3;
constexpr constant uint ROWS_PER_GROUP = 8;

// sum_j p w (y_i - y_j): one SIMD group per cell, lanes striding its affinities.
kernel void tsne_fft_attraction(device const uint *indptr [[buffer(0)]],
                                device const uint *indices [[buffer(1)]],
                                device const float *values [[buffer(2)]],
                                device const float *layout [[buffer(3)]],
                                device float *out [[buffer(4)]],
                                constant float &exaggeration [[buffer(5)]],
                                constant uint &n_cells [[buffer(6)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group * ROWS_PER_GROUP + simd_id;
    if (i >= n_cells) {
        return;
    }
    const float yi0 = layout[2 * i];
    const float yi1 = layout[2 * i + 1];
    float a0 = 0.0f;
    float a1 = 0.0f;
    const uint end = indptr[i + 1];
    for (uint at = indptr[i] + lane; at < end; at += 32) {
        const uint j = indices[at];
        const float p = values[at] * exaggeration;
        const float d0 = yi0 - layout[2 * j];
        const float d1 = yi1 - layout[2 * j + 1];
        const float w = 1.0f / (1.0f + d0 * d0 + d1 * d1);
        a0 = fma(p * w, d0, a0);
        a1 = fma(p * w, d1, a1);
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    if (lane == 0) {
        out[2 * i] = a0;
        out[2 * i + 1] = a1;
    }
}

// Each cell's share of sum p log(p / q), q = w / Z, for the convergence checks.
kernel void tsne_fft_kl(device const uint *indptr [[buffer(0)]],
                        device const uint *indices [[buffer(1)]],
                        device const float *values [[buffer(2)]],
                        device const float *layout [[buffer(3)]],
                        device float *kl_out [[buffer(4)]],
                        constant float &exaggeration [[buffer(5)]],
                        constant float &normaliser [[buffer(6)]],
                        constant uint &n_cells [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint simd_id [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group * ROWS_PER_GROUP + simd_id;
    if (i >= n_cells) {
        return;
    }
    const float yi0 = layout[2 * i];
    const float yi1 = layout[2 * i + 1];
    float kl = 0.0f;
    const uint end = indptr[i + 1];
    for (uint at = indptr[i] + lane; at < end; at += 32) {
        const uint j = indices[at];
        const float p = values[at] * exaggeration;
        if (p > 0.0f) {
            const float d0 = yi0 - layout[2 * j];
            const float d1 = yi1 - layout[2 * j + 1];
            const float w = 1.0f / (1.0f + d0 * d0 + d1 * d1);
            const float q = max(w / normaliser, 2.220446049250313e-16f);
            kl += p * log(max(p, 2.220446049250313e-16f) / q);
        }
    }
    kl = simd_sum(kl);
    if (lane == 0) {
        kl_out[i] = kl;
    }
}

// Box index and Lagrange weights of each cell on both axes.
kernel void tsne_fft_placement(device const float *layout [[buffer(0)]],
                               device uint *boxes [[buffer(1)]],
                               device float *weights [[buffer(2)]],
                               constant float &lo [[buffer(3)]],
                               constant float &box_width [[buffer(4)]],
                               constant uint &n_boxes [[buffer(5)]],
                               constant float *nodes [[buffer(6)]],
                               constant float *denominators [[buffer(7)]],
                               constant uint &n_cells [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    for (uint axis = 0; axis < 2; ++axis) {
        const float y = layout[2 * i + axis];
        const uint b = min(uint(max((y - lo) / box_width, 0.0f)), n_boxes - 1);
        const float fraction = (y - lo - float(b) * box_width) / box_width;
        boxes[2 * i + axis] = b;
        for (uint k = 0; k < NODES; ++k) {
            float w = 1.0f;
            for (uint m = 0; m < NODES; ++m) {
                if (m != k) {
                    w *= fraction - nodes[m];
                }
            }
            weights[(2 * i + axis) * NODES + k] = w / denominators[k];
        }
    }
}

// Charges of the cells of one box onto its nine nodes: one threadgroup per box, its
// 256 lanes striding the box's cells in their sorted order, each SIMD group folded by a
// fixed reduction and the eight partials summed in order by one thread. No atomics, so
// a box with thousands of cells is as reproducible as one with twenty.
kernel void tsne_fft_spread(device const float *layout [[buffer(0)]],
                            device const float *weights [[buffer(1)]],
                            device const uint *order [[buffer(2)]],
                            device const uint *offsets [[buffer(3)]],
                            device float *charges [[buffer(4)]],
                            constant uint &n_boxes [[buffer(5)]],
                            threadgroup float *partials [[threadgroup(0)]],
                            uint box [[threadgroup_position_in_grid]],
                            uint simd_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint t [[thread_position_in_threadgroup]],
                            uint width [[threads_per_threadgroup]]) {
    const uint n_nodes = n_boxes * NODES;
    const uint bx = box / n_boxes;
    const uint by = box % n_boxes;
    float acc[NODES * NODES * 4];
    for (uint s = 0; s < NODES * NODES * 4; ++s) {
        acc[s] = 0.0f;
    }
    const uint end = offsets[box + 1];
    for (uint at = offsets[box] + t; at < end; at += width) {
        const uint i = order[at];
        const float y0 = layout[2 * i];
        const float y1 = layout[2 * i + 1];
        const float q3 = y0 * y0 + y1 * y1;
        for (uint a = 0; a < NODES; ++a) {
            const float wa = weights[(2 * i) * NODES + a];
            for (uint b = 0; b < NODES; ++b) {
                const float w = wa * weights[(2 * i + 1) * NODES + b];
                const uint s = (a * NODES + b) * 4;
                acc[s] += w;
                acc[s + 1] = fma(w, y0, acc[s + 1]);
                acc[s + 2] = fma(w, y1, acc[s + 2]);
                acc[s + 3] = fma(w, q3, acc[s + 3]);
            }
        }
    }
    for (uint s = 0; s < NODES * NODES * 4; ++s) {
        const float total = simd_sum(acc[s]);
        if (lane == 0) {
            partials[simd_id * (NODES * NODES * 4) + s] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        const uint n_simd = width / 32;
        for (uint a = 0; a < NODES; ++a) {
            const uint row = bx * NODES + a;
            for (uint b = 0; b < NODES; ++b) {
                const uint col = by * NODES + b;
                const uint node = (row * n_nodes + col) * 4;
                const uint s = (a * NODES + b) * 4;
                for (uint c = 0; c < 4; ++c) {
                    float sum = 0.0f;
                    for (uint g = 0; g < n_simd; ++g) {
                        sum += partials[g * (NODES * NODES * 4) + s + c];
                    }
                    charges[node + c] = sum;
                }
            }
        }
    }
}

// Potentials at each cell from its nine nodes, and the cell's term of Z.
kernel void tsne_fft_gather(device const float *layout [[buffer(0)]],
                            device const uint *boxes [[buffer(1)]],
                            device const float *weights [[buffer(2)]],
                            device const float *potentials [[buffer(3)]],
                            device float *phi_out [[buffer(4)]],
                            device float *zterm [[buffer(5)]],
                            constant uint &n_nodes [[buffer(6)]],
                            constant uint &n_cells [[buffer(7)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    float phi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint bx = boxes[2 * i];
    const uint by = boxes[2 * i + 1];
    for (uint a = 0; a < NODES; ++a) {
        const uint row = bx * NODES + a;
        const float wa = weights[(2 * i) * NODES + a];
        for (uint b = 0; b < NODES; ++b) {
            const uint col = by * NODES + b;
            const float w = wa * weights[(2 * i + 1) * NODES + b];
            const uint node = (row * n_nodes + col) * 4;
            phi[0] = fma(w, potentials[node], phi[0]);
            phi[1] = fma(w, potentials[node + 1], phi[1]);
            phi[2] = fma(w, potentials[node + 2], phi[2]);
            phi[3] = fma(w, potentials[node + 3], phi[3]);
        }
    }
    const float y0 = layout[2 * i];
    const float y1 = layout[2 * i + 1];
    phi_out[4 * i] = phi[0];
    phi_out[4 * i + 1] = phi[1];
    phi_out[4 * i + 2] = phi[2];
    phi_out[4 * i + 3] = phi[3];
    zterm[i] = (1.0f + y0 * y0 + y1 * y1) * phi[0] - 2.0f * (y0 * phi[1] + y1 * phi[2]) + phi[3];
}

// ---- the convolution: a radix-2 FFT of every row in threadgroup memory, a tiled
// transpose, and the packing of two charge grids into one complex field.

// One threadgroup per row: bit-reversed load into threadgroup memory, log2n butterfly
// stages with the twiddles w_k = exp(-2 pi i k / n) (conjugated for the inverse), store.
kernel void fft_rows(device float *re [[buffer(0)]],
                     device float *im [[buffer(1)]],
                     device const float *twiddle_re [[buffer(2)]],
                     device const float *twiddle_im [[buffer(3)]],
                     constant uint &n [[buffer(4)]],
                     constant uint &log2n [[buffer(5)]],
                     constant uint &inverse [[buffer(6)]],
                     threadgroup float *shared [[threadgroup(0)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint width [[threads_per_threadgroup]]) {
    threadgroup float *sre = shared;
    threadgroup float *sim = shared + n;
    device float *r = re + (ulong)row * n;
    device float *i = im + (ulong)row * n;
    for (uint k = t; k < n; k += width) {
        const uint j = reverse_bits(k) >> (32 - log2n);
        sre[j] = r[k];
        sim[j] = i[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sign = inverse != 0 ? -1.0f : 1.0f;
    for (uint len = 2; len <= n; len <<= 1) {
        const uint span = len >> 1;
        const uint stride = n / len;
        for (uint b = t; b < n / 2; b += width) {
            const uint group = b / span;
            const uint k = b - group * span;
            const uint idx = group * len + k;
            const float wr = twiddle_re[k * stride];
            const float wi = sign * twiddle_im[k * stride];
            const float ur = sre[idx];
            const float ui = sim[idx];
            const float xr = sre[idx + span];
            const float xi = sim[idx + span];
            const float vr = xr * wr - xi * wi;
            const float vi = xr * wi + xi * wr;
            sre[idx] = ur + vr;
            sim[idx] = ui + vi;
            sre[idx + span] = ur - vr;
            sim[idx + span] = ui - vi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = t; k < n; k += width) {
        r[k] = sre[k];
        i[k] = sim[k];
    }
}

// In-place transpose of two square arrays by 32 x 32 tiles: each threadgroup swaps a
// tile with its mirror (or transposes a diagonal tile), so no element is touched twice.
kernel void transpose_square(device float *re [[buffer(0)]],
                             device float *im [[buffer(1)]],
                             constant uint &n [[buffer(2)]],
                             threadgroup float *shared [[threadgroup(0)]],
                             uint2 tile [[threadgroup_position_in_grid]],
                             uint2 t [[thread_position_in_threadgroup]]) {
    if (tile.y < tile.x) {
        return;
    }
    threadgroup float *a_re = shared;
    threadgroup float *a_im = shared + 33 * 32;
    threadgroup float *b_re = shared + 2 * 33 * 32;
    threadgroup float *b_im = shared + 3 * 33 * 32;
    const uint x0 = tile.x * 32;
    const uint y0 = tile.y * 32;
    for (uint j = t.y; j < 32; j += 8) {
        const uint ra = y0 + j;
        const uint ca = x0 + t.x;
        a_re[j * 33 + t.x] = (ra < n && ca < n) ? re[(ulong)ra * n + ca] : 0.0f;
        a_im[j * 33 + t.x] = (ra < n && ca < n) ? im[(ulong)ra * n + ca] : 0.0f;
        const uint rb = x0 + j;
        const uint cb = y0 + t.x;
        b_re[j * 33 + t.x] = (rb < n && cb < n) ? re[(ulong)rb * n + cb] : 0.0f;
        b_im[j * 33 + t.x] = (rb < n && cb < n) ? im[(ulong)rb * n + cb] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t.y; j < 32; j += 8) {
        const uint rb = x0 + j;
        const uint cb = y0 + t.x;
        if (rb < n && cb < n) {
            re[(ulong)rb * n + cb] = a_re[t.x * 33 + j];
            im[(ulong)rb * n + cb] = a_im[t.x * 33 + j];
        }
        if (tile.y != tile.x) {
            const uint ra = y0 + j;
            const uint ca = x0 + t.x;
            if (ra < n && ca < n) {
                re[(ulong)ra * n + ca] = b_re[t.x * 33 + j];
                im[(ulong)ra * n + ca] = b_im[t.x * 33 + j];
            }
        }
    }
}

// Charges c = 2 * pair and 2 * pair + 1 into the real and imaginary parts of the
// zero-padded field.
kernel void fft_pack(device const float *charges [[buffer(0)]],
                     device float *re [[buffer(1)]],
                     device float *im [[buffer(2)]],
                     constant uint &n_nodes [[buffer(3)]],
                     constant uint &size [[buffer(4)]],
                     constant uint &pair [[buffer(5)]],
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x >= size || p.y >= size) {
        return;
    }
    const ulong at = (ulong)p.y * size + p.x;
    if (p.x < n_nodes && p.y < n_nodes) {
        const ulong node = ((ulong)p.y * n_nodes + p.x) * 4;
        re[at] = charges[node + 2 * pair];
        im[at] = charges[node + 2 * pair + 1];
    } else {
        re[at] = 0.0f;
        im[at] = 0.0f;
    }
}

kernel void fft_multiply(device float *re [[buffer(0)]],
                         device float *im [[buffer(1)]],
                         device const float *kernel_re [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const float k = kernel_re[i];
    re[i] *= k;
    im[i] *= k;
}

// The top-left n_nodes x n_nodes block of the field back into the potentials, scaled by
// 1 / size^2 to complete the unnormalised inverse.
kernel void fft_unpack(device const float *re [[buffer(0)]],
                       device const float *im [[buffer(1)]],
                       device float *potentials [[buffer(2)]],
                       constant uint &n_nodes [[buffer(3)]],
                       constant uint &size [[buffer(4)]],
                       constant uint &pair [[buffer(5)]],
                       constant float &scale [[buffer(6)]],
                       uint2 p [[thread_position_in_grid]]) {
    if (p.x >= n_nodes || p.y >= n_nodes) {
        return;
    }
    const ulong at = (ulong)p.y * size + p.x;
    const ulong node = ((ulong)p.y * n_nodes + p.x) * 4;
    potentials[node + 2 * pair] = re[at] * scale;
    potentials[node + 2 * pair + 1] = im[at] * scale;
}

// The gradient 4 (attraction - repulsion), scikit-learn's gains and momentum, the step,
// and the cell's share of the squared norm of the scaled gradient.
kernel void tsne_fft_update(device float *layout [[buffer(0)]],
                            device float *step [[buffer(1)]],
                            device float *gains [[buffer(2)]],
                            device const float *attraction [[buffer(3)]],
                            device const float *phi [[buffer(4)]],
                            device float *norm_out [[buffer(5)]],
                            constant float &normaliser [[buffer(6)]],
                            constant float &momentum [[buffer(7)]],
                            constant float &learning_rate [[buffer(8)]],
                            constant float &min_gain [[buffer(9)]],
                            constant uint &n_cells [[buffer(10)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    const float y0 = layout[2 * i];
    const float y1 = layout[2 * i + 1];
    const float p0 = phi[4 * i];
    const float p1 = phi[4 * i + 1];
    const float p2 = phi[4 * i + 2];
    const float rep0 = (y0 * p0 - p1) / normaliser;
    const float rep1 = (y1 * p0 - p2) / normaliser;
    float norm = 0.0f;
    const float grad[2] = {4.0f * (attraction[2 * i] - rep0), 4.0f * (attraction[2 * i + 1] - rep1)};
    for (uint axis = 0; axis < 2; ++axis) {
        const uint at = 2 * i + axis;
        const float g = grad[axis];
        const bool overshooting = step[at] * g < 0.0f;
        float gain = overshooting ? gains[at] + 0.2f : gains[at] * 0.8f;
        gain = max(gain, min_gain);
        gains[at] = gain;
        const float scaled = g * gain;
        const float u = momentum * step[at] - learning_rate * scaled;
        step[at] = u;
        layout[at] += u;
        norm = fma(scaled, scaled, norm);
    }
    norm_out[i] = norm;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use metalcyte_core::neighbors::KnnGraph;
    use metalcyte_core::tsne_fft::{repulsive_forces, symmetric_affinities, CpuEngine};
    use ndarray::Array2;

    fn blobs(n: usize, seed: u64) -> (Vec<f32>, Array2<f32>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let centres: Vec<[f32; 2]> = (0..8)
            .map(|_| [(next() * 40.0 - 20.0) as f32, (next() * 40.0 - 20.0) as f32])
            .collect();
        let mut layout = Vec::with_capacity(2 * n);
        let mut high = Array2::<f32>::zeros((n, 6));
        for i in 0..n {
            let c = centres[i % 8];
            layout.push(c[0] + (next() as f32 - 0.5) * 3.0);
            layout.push(c[1] + (next() as f32 - 0.5) * 3.0);
            for f in 0..6 {
                high[[i, f]] = (i % 8) as f32 * 5.0 + (next() as f32 - 0.5) * 2.0 + f as f32;
            }
        }
        (layout, high)
    }

    fn affinities(high: &Array2<f32>, k: usize) -> SparseAffinities {
        let graph: KnnGraph =
            metalcyte_core::neighbors::knn(high, k, &candle_core::Device::Cpu).unwrap();
        symmetric_affinities(&graph, 30.0)
    }

    #[test]
    fn gpu_repulsion_matches_the_cores() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let n = 3000;
        let (layout, high) = blobs(n, 3);
        let (cpu_forces, cpu_z) = repulsive_forces(&layout, n);
        let mut engine = GpuEngine::new(context, layout.clone(), affinities(&high, 60)).unwrap();
        let (gpu_forces, gpu_z) = engine.repulsion_for_test().unwrap();
        let z_error = (gpu_z - cpu_z).abs() / cpu_z;
        assert!(z_error < 1e-4, "Z: gpu {gpu_z} cpu {cpu_z}");
        let rms: f64 = (cpu_forces.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let worst = cpu_forces
            .iter()
            .zip(&gpu_forces)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-3 * rms,
            "worst force difference {worst} against rms {rms}"
        );
    }

    #[test]
    fn device_convolution_matches_the_host() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let n = 500;
        let (layout, high) = blobs(n, 7);
        let mut engine = GpuEngine::new(context, layout.clone(), affinities(&high, 60)).unwrap();
        for extent in [30.0f32, 120.0, 300.0] {
            let grid = Grid::for_extent(-extent / 2.0, extent / 2.0);
            let mut state = 11u64;
            let charges: Vec<f32> = (0..4 * grid.n_nodes * grid.n_nodes)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
                })
                .collect();
            let difference = engine
                .convolution_difference_for_test(&grid, &charges)
                .unwrap();
            assert!(
                difference < 1e-4,
                "extent {extent}: relative RMS difference {difference}"
            );
        }
    }

    #[test]
    fn gpu_and_cpu_engines_take_the_same_step() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let n = 3000;
        let (layout, high) = blobs(n, 5);
        let aff = affinities(&high, 60);
        let aff_copy = SparseAffinities {
            indptr: aff.indptr.clone(),
            indices: aff.indices.clone(),
            values: aff.values.clone(),
        };
        let mut cpu = CpuEngine::new(layout.clone(), aff);
        let mut gpu = GpuEngine::new(context, layout.clone(), aff_copy).unwrap();
        // Twenty steps of early exaggeration: the objective and the gradient norm of the
        // two engines agree step by step. The layouts themselves are not compared, since
        // the sign-switched gains amplify any rounding difference over the iterations.
        for it in 0..20 {
            let (cpu_norm, cpu_err) = cpu.step(12.0, 0.5, 200.0, it == 0, true);
            let (gpu_norm, gpu_err) = gpu.step(12.0, 0.5, 200.0, it == 0, true);
            let (cpu_err, gpu_err) = (cpu_err.unwrap(), gpu_err.unwrap());
            assert!(
                (cpu_err - gpu_err).abs() < 1e-3 * cpu_err.abs().max(1e-6),
                "objective at {it}: cpu {cpu_err} gpu {gpu_err}"
            );
            assert!(
                (cpu_norm - gpu_norm).abs() < 1e-2 * cpu_norm.max(1e-12),
                "gradient norm at {it}: cpu {cpu_norm} gpu {gpu_norm}"
            );
        }
        // The full schedule from the same start reaches the same objective.
        let params = metalcyte_core::tsne::TsneParams {
            n_iterations: 500,
            ..Default::default()
        };
        let (layout, high) = blobs(n, 6);
        let aff = affinities(&high, 60);
        let aff_copy = SparseAffinities {
            indptr: aff.indptr.clone(),
            indices: aff.indices.clone(),
            values: aff.values.clone(),
        };
        let mut cpu = CpuEngine::new(layout.clone(), aff);
        let mut gpu = GpuEngine::new(MetalContext::new().unwrap(), layout, aff_copy).unwrap();
        metalcyte_core::tsne_fft::optimise(&mut cpu, &params);
        metalcyte_core::tsne_fft::optimise(&mut gpu, &params);
        let (_, cpu_final) = cpu.step(1.0, 0.8, 200.0, false, true);
        let (_, gpu_final) = gpu.step(1.0, 0.8, 200.0, false, true);
        let (cpu_final, gpu_final) = (cpu_final.unwrap(), gpu_final.unwrap());
        println!("final objective: cpu {cpu_final:.5} gpu {gpu_final:.5}");
        assert!(
            (cpu_final - gpu_final).abs() < 0.02 * cpu_final.abs(),
            "final objective: cpu {cpu_final} gpu {gpu_final}"
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_bytes() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let Ok(context2) = MetalContext::new() else {
            return;
        };
        let n = 2000;
        let (layout, high) = blobs(n, 9);
        let make = || affinities(&high, 60);
        let mut a = GpuEngine::new(context, layout.clone(), make()).unwrap();
        let mut b = GpuEngine::new(context2, layout.clone(), make()).unwrap();
        for it in 0..5 {
            a.step(12.0, 0.5, 200.0, it == 0, it == 4);
            b.step(12.0, 0.5, 200.0, it == 0, it == 4);
        }
        assert_eq!(a.layout(), b.layout());
    }
}
