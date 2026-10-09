use std::ffi::c_void;

use metal::{MTLCommandBufferStatus, MTLSize};
use metalcyte_core::error::{Error, Result};
use metalcyte_core::neighbors::KnnGraph;
use ndarray::{Array2, ArrayView2};

use crate::context::MetalContext;

const KERNEL_NAME: &str = "knn_select";

/// Threadgroup memory one candidate slot costs: an `f32` distance next to a
/// `u32` cell index, held in two parallel arrays.
const SLOT_BYTES: usize = 8;

/// The smallest threadgroup we launch. A threadgroup narrower than one SIMD
/// group cannot fill an execution unit, so we spend the threadgroup memory
/// budget on `k` only down to this point and reject larger `k` instead.
const MIN_THREADS: usize = 32;

/// The largest `k` this kernel can serve on `context`.
///
/// Every thread of a query's threadgroup owns a private top-k list in
/// threadgroup memory, so the whole group needs `threads * k * SLOT_BYTES`
/// bytes. Holding the group at [`MIN_THREADS`] turns that into a ceiling on
/// `k`: 32 KiB of threadgroup memory on Apple silicon gives k <= 128.
pub fn max_supported_k(context: &MetalContext) -> usize {
    context.device().max_threadgroup_memory_length() as usize / (SLOT_BYTES * MIN_THREADS)
}

/// Exact k nearest neighbours in one pass over tiles of the distance matrix.
///
/// Selection is what needs a kernel: candle can produce the distances, but
/// keeping the k smallest per row without materialising an `(n, n)` matrix
/// cannot be expressed as tensor algebra.
pub fn knn_metal(context: &MetalContext, embedding: &Array2<f32>, k: usize) -> Result<KnnGraph> {
    let (n_cells, n_dims) = embedding.dim();
    if k == 0 {
        return Err(Error::parameter("k", "at least 1", k));
    }
    if n_dims == 0 {
        return Err(Error::parameter(
            "embedding",
            "at least one dimension",
            n_dims,
        ));
    }
    // A cell is never its own neighbour, so k neighbours need k + 1 cells.
    if n_cells < k + 1 {
        return Err(Error::parameter("k", "smaller than the cell count", k));
    }
    let k_limit = max_supported_k(context);
    if k > k_limit {
        return Err(Error::parameter(
            "k",
            "within the threadgroup memory budget",
            k,
        ));
    }

    if k <= TILED_MAX_K && n_dims <= TILED_MAX_DIMS {
        return knn_metal_tiled(context, embedding, k);
    }
    let pipeline = context.pipeline(KERNEL_NAME, KNN_SOURCE)?;
    let threads = threads_per_query(
        pipeline.max_total_threads_per_threadgroup() as usize,
        context.device().max_threadgroup_memory_length() as usize,
        k,
    );

    // Centre column-wise and carry each row's squared norm, so the shader can
    // reproduce `neighbors::knn`'s zero-snapping. Both are the CPU path's numerical
    // safeguards: without them a tight cluster orders by sub-ulp noise on the GPU
    // where the CPU has snapped it to zero, and device parity breaks.
    let (centred, norm_sq) = centre_and_norms(embedding);

    let input = context.buffer(&centred);
    let norms = context.buffer(&norm_sq);
    let out_indices = context.empty_buffer::<u32>(n_cells * k);
    let out_distances = context.empty_buffer::<f32>(n_cells * k);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&out_indices), 0);
    encoder.set_buffer(2, Some(&out_distances), 0);
    encoder.set_buffer(6, Some(&norms), 0);
    set_u32(encoder, 3, n_cells as u32);
    set_u32(encoder, 4, n_dims as u32);
    set_u32(encoder, 5, k as u32);
    let list_bytes = (threads * k * 4) as u64;
    encoder.set_threadgroup_memory_length(0, list_bytes);
    encoder.set_threadgroup_memory_length(1, list_bytes);
    // One threadgroup per query cell; its threads stripe the candidate cells.
    encoder.dispatch_thread_groups(
        MTLSize::new(n_cells as u64, 1, 1),
        MTLSize::new(threads as u64, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Error::Kernel {
            name: KERNEL_NAME,
            message: format!("dispatch ended in state {:?}", command.status()),
        });
    }

    let indices = unsafe { MetalContext::read::<u32>(&out_indices, n_cells * k) };
    let distances = unsafe { MetalContext::read::<f32>(&out_distances, n_cells * k) };
    let shape = || Error::shape(format!("({n_cells}, {k})"), "a buffer of another size");
    Ok(KnnGraph {
        indices: Array2::from_shape_vec((n_cells, k), indices).map_err(|_| shape())?,
        distances: Array2::from_shape_vec((n_cells, k), distances).map_err(|_| shape())?,
    })
}

/// Largest `k` the tiled kernel keeps in per-thread registers.
pub const TILED_MAX_K: usize = 64;
/// Largest embedding width whose query and candidate tiles fit threadgroup memory.
pub const TILED_MAX_DIMS: usize = 128;
const TILED_KERNEL_NAME: &str = "knn_tiled";

/// Exact k nearest neighbours with query and candidate tiling.
///
/// Each threadgroup serves `T` query cells, one per thread, and streams the candidates
/// through threadgroup memory `T` rows at a time, so every candidate row is read from
/// device memory once per `T` queries instead of once per query. That turns the
/// search from bandwidth-bound into compute-bound on Apple silicon. Distances, the
/// zero-snapping threshold and the (distance, index) tie break are those of
/// `knn_select`, and each thread sees every candidate, so the result is identical.
pub fn knn_metal_tiled(
    context: &MetalContext,
    embedding: &Array2<f32>,
    k: usize,
) -> Result<KnnGraph> {
    knn_metal_tiled_view(context, embedding.view(), k)
}

/// [`knn_metal_tiled`] on a borrowed view (e.g. numpy's own buffer): no owned copy of
/// the embedding is made. The centred coordinates are written straight into the shared
/// Metal buffer the GPU reads, which unified memory makes a plain CPU write.
pub fn knn_metal_tiled_view(
    context: &MetalContext,
    embedding: ArrayView2<'_, f32>,
    k: usize,
) -> Result<KnnGraph> {
    let (n_cells, n_dims) = embedding.dim();
    if k == 0 || k > TILED_MAX_K {
        return Err(Error::parameter(
            "k",
            "between 1 and 64 for the tiled kernel",
            k,
        ));
    }
    if n_dims == 0 || n_dims > TILED_MAX_DIMS {
        return Err(Error::parameter("embedding", "1 to 128 dimensions", n_dims));
    }
    if n_cells < k + 1 {
        return Err(Error::parameter("k", "smaller than the cell count", k));
    }
    if std::env::var("METALCYTE_KNN_KERNEL").as_deref() == Ok("simd") && n_dims <= 56 {
        return knn_metal_simd_view(context, embedding, k);
    }
    // Specialise the shader for this (n_dims, k): with both known at compile time the
    // query row and the top-k list live in registers. One pipeline per shape, cached.
    let name: &'static str = specialised_name(n_dims, k);
    let source = KNN_TILED_SOURCE
        .replace("__NDIMS__", &n_dims.to_string())
        .replace("__K__", &k.to_string())
        .replace("knn_tiled(", &format!("{name}("));
    let pipeline = context.pipeline(name, &source)?;
    let memory = context.device().max_threadgroup_memory_length() as usize;
    // The candidate tile (rows x dims) plus one norm per row must fit.
    let mut tile = 64usize;
    while tile > 8 && tile * (n_dims + 1) * 4 > memory {
        tile /= 2;
    }
    let tile = tile.min(pipeline.max_total_threads_per_threadgroup() as usize);

    let input = context.empty_buffer::<f32>(n_cells * n_dims);
    // SAFETY: a fresh shared buffer of exactly n_cells * n_dims f32, written before use.
    let dest =
        unsafe { std::slice::from_raw_parts_mut(input.contents() as *mut f32, n_cells * n_dims) };
    let norm_sq = centre_into(embedding, dest);
    let norms = context.buffer(&norm_sq);
    let out_indices = context.empty_buffer::<u32>(n_cells * k);
    let out_distances = context.empty_buffer::<f32>(n_cells * k);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&out_indices), 0);
    encoder.set_buffer(2, Some(&out_distances), 0);
    encoder.set_buffer(6, Some(&norms), 0);
    set_u32(encoder, 3, n_cells as u32);
    set_u32(encoder, 4, n_dims as u32);
    set_u32(encoder, 5, k as u32);
    encoder.set_threadgroup_memory_length(0, (tile * n_dims * 4) as u64);
    encoder.set_threadgroup_memory_length(1, (tile * 4) as u64);
    let groups = n_cells.div_ceil(tile);
    encoder.dispatch_thread_groups(
        MTLSize::new(groups as u64, 1, 1),
        MTLSize::new(tile as u64, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Error::Kernel {
            name: TILED_KERNEL_NAME,
            message: format!("dispatch ended in state {:?}", command.status()),
        });
    }
    let indices = unsafe { MetalContext::read::<u32>(&out_indices, n_cells * k) };
    let distances = unsafe { MetalContext::read::<f32>(&out_distances, n_cells * k) };
    let shape = || Error::shape(format!("({n_cells}, {k})"), "a buffer of another size");
    Ok(KnnGraph {
        indices: Array2::from_shape_vec((n_cells, k), indices).map_err(|_| shape())?,
        distances: Array2::from_shape_vec((n_cells, k), distances).map_err(|_| shape())?,
    })
}

/// Query cells per threadgroup in the simdgroup-matrix kernel: two SIMD groups of 32
/// threads, one thread per query for the selection, two 32-row halves for the products.
const SIMD_QUERY_TILE: usize = 64;
/// Candidate rows per step: 64 when the padded width allows (`64 x 64 x 4 B` of
/// candidates plus the `64 x 64` product block is 32 KiB at 64 dims), else 32.
fn simd_candidate_tile(_padded_dims: usize) -> usize {
    // Query tile (64 x NDP) + candidate tile (NDP x 32) + product block (32 x 64), all f32:
    // 96 x NDP x 4 + 8 KiB, which is under 32 KiB up to NDP = 56 and must be checked.
    32
}
const SIMD_KERNEL_NAME: &str = "knn_simd";

/// Exact k nearest neighbours with the distance products on `simdgroup_float8x8`
/// matrices. **Experimental, opt-in** (`METALCYTE_KNN_KERNEL=simd`), and as written it is
/// slower than the tiled kernel: 3.1 s against 2.2 s on 115 868 cells x 50 dims.
///
/// The tiled kernel forms every squared distance as a scalar loop of `n_dims` fused
/// multiply-adds reading the candidate row out of threadgroup memory. Here the `64 x 32`
/// block of query-candidate dot products is built from 8 x 8 `simdgroup_load` /
/// `simdgroup_multiply_accumulate` steps (queries and dimension-major candidates both
/// staged in threadgroup memory), stored candidate-major, and the per-query thread does
/// the expansion, zero-snap and top-k insertion on its own column. Removing the
/// products alone takes the run from 3.25 s to 0.76 s, so the 8 x 8 steps are the cost:
/// with only 16 accumulators per SIMD group the kernel issues two 8 x 8 loads per
/// multiply-accumulate, and on an M3 Pro that is no faster than the FMA loop. A
/// gemm-grade blocking (MLX's 64 x 64 output tiles, loads amortised over 8 accumulators
/// per load) is what it would take, and is left as the next step in
/// `docs/PLAN_APPLE_SILICON.md`. Same expansion, resolution floor and tie rule as the
/// CPU search in `metalcyte_core::neighbors`.
///
/// The embedding is laid out with its width padded to a multiple of 8 and its height to
/// a multiple of 64 so every simdgroup load is a full tile; padded rows are zero and are
/// never selected, padded columns contribute nothing to a product.
pub fn knn_metal_simd_view(
    context: &MetalContext,
    embedding: ArrayView2<'_, f32>,
    k: usize,
) -> Result<KnnGraph> {
    let (n_cells, n_dims) = embedding.dim();
    if k == 0 || k > TILED_MAX_K {
        return Err(Error::parameter("k", "between 1 and 64", k));
    }
    if n_dims == 0 || n_dims > TILED_MAX_DIMS {
        return Err(Error::parameter("embedding", "1 to 128 dimensions", n_dims));
    }
    if n_cells < k + 1 {
        return Err(Error::parameter("k", "smaller than the cell count", k));
    }
    let padded_dims = n_dims.div_ceil(8) * 8;
    let padded_cells = n_cells.div_ceil(SIMD_QUERY_TILE) * SIMD_QUERY_TILE;

    let candidate_tile = simd_candidate_tile(padded_dims);
    let name: &'static str = simd_name(padded_dims, k);
    let source = KNN_SIMD_SOURCE
        .replace("__NDP__", &padded_dims.to_string())
        .replace("__CT__", &format!("{candidate_tile}u"))
        .replace("__K__", &k.to_string())
        .replace("knn_simd(", &format!("{name}("));
    let pipeline = context.pipeline(name, &source)?;

    // Centred coordinates straight into the padded shared buffer: the padding columns
    // and rows are zeroed first, the real rows are written in place by `centre_into`
    // through a strided view of the same buffer.
    let input = context.empty_buffer::<f32>(padded_cells * padded_dims);
    let dest = unsafe {
        std::slice::from_raw_parts_mut(input.contents() as *mut f32, padded_cells * padded_dims)
    };
    dest.fill(0.0);
    let norm_sq = {
        let mut tight = vec![0.0f32; n_cells * n_dims];
        let norms = centre_into(embedding, &mut tight);
        for (row, src) in tight.chunks_exact(n_dims).enumerate() {
            dest[row * padded_dims..row * padded_dims + n_dims].copy_from_slice(src);
        }
        norms
    };
    let mut norms_padded = norm_sq;
    norms_padded.resize(padded_cells, 0.0);
    let norms = context.buffer(&norms_padded);
    let out_indices = context.empty_buffer::<u32>(n_cells * k);
    let out_distances = context.empty_buffer::<f32>(n_cells * k);

    let command = context.queue().new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&out_indices), 0);
    encoder.set_buffer(2, Some(&out_distances), 0);
    set_u32(encoder, 3, n_cells as u32);
    set_u32(encoder, 4, padded_cells as u32);
    set_u32(encoder, 5, n_dims as u32);
    encoder.set_buffer(6, Some(&norms), 0);
    let threadgroup_bytes =
        (candidate_tile + SIMD_QUERY_TILE) * padded_dims * 4 + SIMD_QUERY_TILE * candidate_tile * 4;
    if threadgroup_bytes > context.device().max_threadgroup_memory_length() as usize {
        return Err(Error::parameter(
            "embedding",
            "a width whose query and candidate tiles fit threadgroup memory (about 56 dimensions)",
            n_dims,
        ));
    }
    encoder.set_threadgroup_memory_length(0, (candidate_tile * padded_dims * 4) as u64);
    encoder.set_threadgroup_memory_length(1, (SIMD_QUERY_TILE * candidate_tile * 4) as u64);
    encoder.set_threadgroup_memory_length(2, (SIMD_QUERY_TILE * padded_dims * 4) as u64);
    let groups = padded_cells / SIMD_QUERY_TILE;
    encoder.dispatch_thread_groups(
        MTLSize::new(groups as u64, 1, 1),
        MTLSize::new(SIMD_QUERY_TILE as u64, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Error::Kernel {
            name: SIMD_KERNEL_NAME,
            message: format!("dispatch ended in state {:?}", command.status()),
        });
    }
    let indices = unsafe { MetalContext::read::<u32>(&out_indices, n_cells * k) };
    let distances = unsafe { MetalContext::read::<f32>(&out_distances, n_cells * k) };
    let shape = || Error::shape(format!("({n_cells}, {k})"), "a buffer of another size");
    Ok(KnnGraph {
        indices: Array2::from_shape_vec((n_cells, k), indices).map_err(|_| shape())?,
        distances: Array2::from_shape_vec((n_cells, k), distances).map_err(|_| shape())?,
    })
}

fn simd_name(padded_dims: usize, k: usize) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<(usize, usize), &'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(Default::default)
        .lock()
        .expect("name cache poisoned");
    names
        .entry((padded_dims, k))
        .or_insert_with(|| Box::leak(format!("knn_simd_d{padded_dims}_k{k}").into_boxed_str()))
}

const KNN_SIMD_SOURCE: &str = include_str!("../shaders/knn_simd.metal");

/// A stable, leaked kernel name per (n_dims, k), so the pipeline cache can key on it.
fn specialised_name(n_dims: usize, k: usize) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<(usize, usize), &'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(Default::default)
        .lock()
        .expect("name cache poisoned");
    names
        .entry((n_dims, k))
        .or_insert_with(|| Box::leak(format!("knn_tiled_d{n_dims}_k{k}").into_boxed_str()))
}

/// Threads per query cell: as wide as the hardware and the top-k lists allow.
///
/// The merge tree halves the number of active threads each round, so the count
/// must be a power of two. `k <= max_supported_k` keeps the result at or above
/// [`MIN_THREADS`].
fn threads_per_query(pipeline_limit: usize, threadgroup_memory: usize, k: usize) -> usize {
    let memory_limit = threadgroup_memory / (SLOT_BYTES * k);
    let threads = pipeline_limit.min(memory_limit).max(1);
    1 << (usize::BITS - 1 - threads.leading_zeros())
}

/// The embedding centred column-wise, plus each centred row's squared norm.
///
/// Mirrors `metalcyte_core::neighbors`' `centred`: the means accumulate in `f64` and
/// round once, so the centred coordinates are correct to one `f32` rounding and the
/// snapping threshold the shader forms from these norms matches the CPU path. The
/// distance itself is translation-invariant, but the expansion's *resolution* is
/// not -- centring makes it the radius of the cloud rather than the distance to the
/// origin, which is what keeps the snapping floor from swallowing a real neighbour.
/// [`centre_and_norms`] writing the centred rows into `dest` (row-major) and returning
/// only the norms; identical arithmetic, so the snapping threshold is unchanged.
fn centre_into(embedding: ArrayView2<'_, f32>, dest: &mut [f32]) -> Vec<f32> {
    let (n_cells, n_dims) = embedding.dim();
    let mut means = vec![0.0f64; n_dims];
    for row in embedding.rows() {
        for (mean, &value) in means.iter_mut().zip(row) {
            *mean += value as f64;
        }
    }
    for mean in means.iter_mut() {
        *mean /= n_cells as f64;
    }
    let mut norm_sq = Vec::with_capacity(n_cells);
    for (row, out) in embedding
        .rows()
        .into_iter()
        .zip(dest.chunks_exact_mut(n_dims))
    {
        let mut norm = 0.0f32;
        for ((&value, &mean), slot) in row.iter().zip(means.iter()).zip(out.iter_mut()) {
            let coord = (value as f64 - mean) as f32;
            *slot = coord;
            norm = coord.mul_add(coord, norm);
        }
        norm_sq.push(norm);
    }
    norm_sq
}

fn centre_and_norms(embedding: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let (n_cells, n_dims) = embedding.dim();
    let mut means = vec![0.0f64; n_dims];
    for row in embedding.rows() {
        for (mean, &value) in means.iter_mut().zip(row) {
            *mean += value as f64;
        }
    }
    for mean in means.iter_mut() {
        *mean /= n_cells as f64;
    }
    let mut centred = Vec::with_capacity(n_cells * n_dims);
    let mut norm_sq = Vec::with_capacity(n_cells);
    for row in embedding.rows() {
        let mut norm = 0.0f32;
        for (&value, &mean) in row.iter().zip(means.iter()) {
            let coord = (value as f64 - mean) as f32;
            centred.push(coord);
            norm = coord.mul_add(coord, norm);
        }
        norm_sq.push(norm);
    }
    (centred, norm_sq)
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const c_void,
    );
}

const KNN_SOURCE: &str = include_str!("../shaders/knn.metal");

const KNN_TILED_SOURCE: &str = include_str!("../shaders/knn_tiled.metal");

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force reference: every distance, sorted by `(distance, cell)`.
    ///
    /// `mul_add` mirrors the kernel's `fma`. Without it the two accumulations
    /// differ by an ulp, which is enough to swap a pair of neighbours that are
    /// equidistant to `f32` and break the exact index comparison.
    fn cpu_knn(embedding: &Array2<f32>, k: usize) -> KnnGraph {
        let (n_cells, n_dims) = embedding.dim();
        // Mirror the kernel: centre, then snap sub-resolution squared distances to
        // zero using the same `(n_dims + 2) * EPSILON * (|a|^2 + |b|^2)` threshold.
        let (centred, norm_sq) = centre_and_norms(embedding);
        let mut indices = Array2::zeros((n_cells, k));
        let mut distances = Array2::zeros((n_cells, k));
        for query in 0..n_cells {
            let mut candidates: Vec<(f32, u32)> = (0..n_cells)
                .filter(|&cell| cell != query)
                .map(|cell| {
                    let mut squared = 0.0f32;
                    for dim in 0..n_dims {
                        let delta = centred[query * n_dims + dim] - centred[cell * n_dims + dim];
                        squared = delta.mul_add(delta, squared);
                    }
                    let threshold =
                        (n_dims as f32 + 2.0) * f32::EPSILON * (norm_sq[query] + norm_sq[cell]);
                    if squared < threshold {
                        squared = 0.0;
                    }
                    (squared, cell as u32)
                })
                .collect();
            candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (slot, &(squared, cell)) in candidates.iter().take(k).enumerate() {
                indices[[query, slot]] = cell;
                distances[[query, slot]] = squared.sqrt();
            }
        }
        KnnGraph { indices, distances }
    }

    /// A tiny LCG: `rand` is not a dependency of this crate and the tests only
    /// need spread out points, not statistical quality.
    fn random_embedding(n_cells: usize, n_dims: usize, seed: u64) -> Array2<f32> {
        let mut state = seed.wrapping_mul(2) + 1;
        Array2::from_shape_fn((n_cells, n_dims), |_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
    }

    /// The two graphs name the same neighbours, except where two candidates sit within
    /// `rel` of each other and the `|a|^2 + |b|^2 - 2 a.b` expansion (which the matrix
    /// kernels and the CPU core path both use) ranks them the other way round from the
    /// direct `|a - b|^2` the reference takes. Every distance must agree to `rel`.
    fn assert_graphs_agree(got: &KnnGraph, want: &KnnGraph, rel: f32) {
        assert_eq!(got.indices.dim(), want.indices.dim());
        for row in 0..got.indices.nrows() {
            let g: Vec<(u32, f32)> = got
                .indices
                .row(row)
                .iter()
                .copied()
                .zip(got.distances.row(row).iter().copied())
                .collect();
            let w: Vec<(u32, f32)> = want
                .indices
                .row(row)
                .iter()
                .copied()
                .zip(want.distances.row(row).iter().copied())
                .collect();
            for ((gi, gd), (wi, wd)) in g.iter().zip(&w) {
                assert!(
                    (gd - wd).abs() <= rel * wd.abs().max(1e-6),
                    "row {row}: distance {gd} != {wd}"
                );
                if gi != wi {
                    // A swap is only acceptable among near-ties: the stranger must be at a
                    // distance the reference also has in this row, to `rel`.
                    let near = w
                        .iter()
                        .any(|(_, d)| (d - gd).abs() <= rel * d.abs().max(1e-6));
                    assert!(
                        near,
                        "row {row}: neighbour {gi} at {gd} is not a near-tie of the reference"
                    );
                }
            }
        }
    }

    fn assert_matches_reference(embedding: &Array2<f32>, k: usize, context: &MetalContext) {
        let gpu = knn_metal(context, embedding, k).unwrap();
        let cpu = cpu_knn(embedding, k);
        assert_graphs_agree(&gpu, &cpu, 1e-4);
    }

    #[test]
    fn matches_the_cpu_reference_on_random_embeddings() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        // 257 and 300 cells are not multiples of the 256 thread threadgroup.
        for (n_cells, n_dims, k) in [(64, 8, 1), (100, 5, 7), (257, 3, 15), (300, 32, 16)] {
            let embedding = random_embedding(n_cells, n_dims, n_cells as u64);
            assert_matches_reference(&embedding, k, &context);
        }
    }

    #[test]
    fn matches_the_cpu_reference_at_the_largest_supported_k() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let k = max_supported_k(&context);
        let embedding = random_embedding(k + 71, 4, 7);
        assert_matches_reference(&embedding, k, &context);
    }

    #[test]
    fn finds_the_obvious_neighbours_of_points_on_a_line() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding = Array2::from_shape_fn((50, 1), |(cell, _)| cell as f32);
        let graph = knn_metal(&context, &embedding, 2).unwrap();
        // The interior point 10 sits between 9 and 11, both one unit away, and
        // the tie goes to the smaller index.
        assert_eq!(graph.indices.row(10).to_vec(), vec![9, 11]);
        assert_eq!(graph.distances.row(10).to_vec(), vec![1.0, 1.0]);
        // The endpoint's neighbours are its two successors.
        assert_eq!(graph.indices.row(0).to_vec(), vec![1, 2]);
        assert_eq!(graph.distances.row(0).to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn duplicated_points_break_ties_by_the_smaller_index() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // Forty copies of the origin, then a far away point.
        let mut embedding = Array2::<f32>::zeros((41, 2));
        embedding[[40, 0]] = 100.0;
        let graph = knn_metal(&context, &embedding, 3).unwrap();
        assert_eq!(graph.indices.row(5).to_vec(), vec![0, 1, 2]);
        assert_eq!(graph.indices.row(0).to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn two_runs_produce_identical_output() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding = random_embedding(500, 6, 11);
        let first = knn_metal(&context, &embedding, 9).unwrap();
        let second = knn_metal(&context, &embedding, 9).unwrap();
        assert_eq!(first.indices, second.indices);
        assert_eq!(first.distances, second.distances);
    }

    #[test]
    fn rejects_parameters_it_cannot_serve() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding = random_embedding(8, 3, 3);
        for k in [0, 8, 9] {
            let error = knn_metal(&context, &embedding, k).unwrap_err();
            assert!(matches!(error, Error::InvalidParameter { .. }), "k = {k}");
        }
        let wide = random_embedding(4096, 2, 4);
        let error = knn_metal(&context, &wide, max_supported_k(&context) + 1).unwrap_err();
        assert!(matches!(error, Error::InvalidParameter { .. }));
    }

    #[test]
    fn threadgroup_width_is_a_power_of_two_within_both_limits() {
        // 32 KiB of threadgroup memory, 1024 threads: k = 15 leaves room for
        // 273 lists, so the tree runs 256 threads wide.
        assert_eq!(threads_per_query(1024, 32768, 15), 256);
        assert_eq!(threads_per_query(1024, 32768, 1), 1024);
        assert_eq!(threads_per_query(1024, 32768, 128), 32);
    }

    /// Run with `cargo test --release -- --ignored --nocapture` to measure.
    #[test]
    #[ignore = "takes minutes: the CPU reference is O(n^2 d)"]
    fn reports_the_speedup_over_the_cpu_reference() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding = random_embedding(20_000, 50, 1);
        let started = std::time::Instant::now();
        let gpu = knn_metal(&context, &embedding, 15).unwrap();
        let gpu_elapsed = started.elapsed();
        let started = std::time::Instant::now();
        let cpu = cpu_knn(&embedding, 15);
        let cpu_elapsed = started.elapsed();
        assert_eq!(gpu.indices, cpu.indices);
        println!(
            "20000 x 50, k = 15: gpu {gpu_elapsed:?}, cpu {cpu_elapsed:?}, speedup {:.1}x",
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn tiled_kernel_matches_the_brute_force_reference_with_duplicates_and_ragged_tiles() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        // 1 037 cells is not a multiple of any tile; every 7th cell duplicates another.
        for (n_cells, n_dims, k) in [(1037usize, 50usize, 15usize), (300, 128, 64), (97, 3, 1)] {
            let mut data: Vec<f32> = (0..n_cells * n_dims)
                .map(|_| rng.gen_range(-2.0..2.0))
                .collect();
            for row in (7..n_cells).step_by(7) {
                let src = (row / 2) * n_dims;
                let (a, b) = data.split_at_mut(row * n_dims);
                b[..n_dims].copy_from_slice(&a[src..src + n_dims]);
            }
            let embedding = Array2::from_shape_vec((n_cells, n_dims), data).unwrap();
            let got = knn_metal_tiled(&context, &embedding, k).unwrap();
            let want = cpu_knn(&embedding, k);
            assert_graphs_agree(&got, &want, 1e-4);
        }
    }

    #[test]
    fn simdgroup_kernel_matches_the_brute_force_reference() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        // Widths that need padding (30 -> 32, 50 -> 56) and one that does not (128); cell
        // counts that are not multiples of the 64-query or 32-candidate tiles; duplicates.
        for (n_cells, n_dims, k) in [
            (1037usize, 50usize, 15usize),
            (300, 56, 64),
            (97, 3, 1),
            (130, 30, 7),
        ] {
            let mut data: Vec<f32> = (0..n_cells * n_dims)
                .map(|_| rng.gen_range(-2.0..2.0))
                .collect();
            for row in (7..n_cells).step_by(7) {
                let src = (row / 2) * n_dims;
                let (a, b) = data.split_at_mut(row * n_dims);
                b[..n_dims].copy_from_slice(&a[src..src + n_dims]);
            }
            let embedding = Array2::from_shape_vec((n_cells, n_dims), data).unwrap();
            let got = knn_metal_simd_view(&context, embedding.view(), k).unwrap();
            let want = cpu_knn(&embedding, k);
            assert_graphs_agree(&got, &want, 1e-4);
        }
    }
}
