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
                (ROWS_PER_GROUP * NODES_PER_BOX * NODES_PER_BOX * 4 * size_of::<f32>())
                    as NSUInteger,
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
                (ROWS_PER_GROUP * NODES_PER_BOX * NODES_PER_BOX * 4 * size_of::<f32>())
                    as NSUInteger,
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

const SOURCE: &str = include_str!("../shaders/tsne_fft.metal");

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
