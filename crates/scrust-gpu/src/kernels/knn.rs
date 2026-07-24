use std::ffi::c_void;

use metal::{MTLCommandBufferStatus, MTLSize};
use ndarray::Array2;
use scrust_core::error::{Error, Result};
use scrust_core::neighbors::KnnGraph;

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
/// Mirrors `scrust_core::neighbors`' `centred`: the means accumulate in `f64` and
/// round once, so the centred coordinates are correct to one `f32` rounding and the
/// snapping threshold the shader forms from these norms matches the CPU path. The
/// distance itself is translation-invariant, but the expansion's *resolution* is
/// not -- centring makes it the radius of the cloud rather than the distance to the
/// origin, which is what keeps the snapping floor from swallowing a real neighbour.
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

const KNN_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// f32::EPSILON (2^-23) as an exact literal, so the snapping threshold below is the
// same number the CPU path forms with `f32::EPSILON`.
constant float F32_EPSILON = 1.1920928955078125e-07f;

// Nearer wins; equal distances are broken by the smaller cell index. Without
// the tie break the answer would depend on which lane happened to see a
// duplicate point first, and would not match the CPU implementation.
inline bool closer(float lhs_distance, uint lhs_cell, float rhs_distance, uint rhs_cell) {
    return lhs_distance < rhs_distance
        || (lhs_distance == rhs_distance && lhs_cell < rhs_cell);
}

// Insert one candidate into an ascending top-k list, dropping its worst entry.
inline void insert(threadgroup float* distances,
                   threadgroup uint* cells,
                   uint k,
                   float new_distance,
                   uint cell) {
    if (!closer(new_distance, cell, distances[k - 1], cells[k - 1])) {
        return;
    }
    uint slot = k - 1;
    while (slot > 0 && closer(new_distance, cell, distances[slot - 1], cells[slot - 1])) {
        distances[slot] = distances[slot - 1];
        cells[slot] = cells[slot - 1];
        slot--;
    }
    distances[slot] = new_distance;
    cells[slot] = cell;
}

kernel void knn_select(device const float* embedding [[buffer(0)]],
                       device uint* out_cells [[buffer(1)]],
                       device float* out_distances [[buffer(2)]],
                       device const float* norm_sq [[buffer(6)]],
                       constant uint& n_cells [[buffer(3)]],
                       constant uint& n_dims [[buffer(4)]],
                       constant uint& k [[buffer(5)]],
                       threadgroup float* list_distances [[threadgroup(0)]],
                       threadgroup uint* list_cells [[threadgroup(1)]],
                       uint query [[threadgroup_position_in_grid]],
                       uint lane [[thread_position_in_threadgroup]],
                       uint lane_count [[threads_per_threadgroup]]) {
    threadgroup float* mine_distances = list_distances + lane * k;
    threadgroup uint* mine_cells = list_cells + lane * k;
    for (uint slot = 0; slot < k; slot++) {
        mine_distances[slot] = INFINITY;
        mine_cells[slot] = 0xFFFFFFFFu;
    }

    device const float* query_row = embedding + (ulong)query * n_dims;
    for (uint cell = lane; cell < n_cells; cell += lane_count) {
        if (cell == query) {
            continue;  // a cell is not its own neighbour
        }
        device const float* candidate_row = embedding + (ulong)cell * n_dims;
        float squared = 0.0f;
        for (uint dim = 0; dim < n_dims; dim++) {
            float delta = query_row[dim] - candidate_row[dim];
            squared = fma(delta, delta, squared);
        }
        // Below the expansion's own resolution the result is rounding noise, not a
        // distance, so it is snapped to zero -- exactly as `neighbors::knn` does on
        // the CPU -- and both devices then treat a knot tighter than f32 can resolve
        // as a set of coincident points, ordered by index alone.
        float threshold = (float(n_dims) + 2.0f) * F32_EPSILON * (norm_sq[query] + norm_sq[cell]);
        if (squared < threshold) {
            squared = 0.0f;
        }
        // Squared distances rank identically to Euclidean ones, so the n per
        // row square roots are deferred to the k survivors below.
        insert(mine_distances, mine_cells, k, squared, cell);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge the private lists pairwise up a tree, halving the active threads
    // each round until list 0 holds the k nearest of the whole row.
    for (uint stride = lane_count / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            threadgroup float* other_distances = list_distances + (lane + stride) * k;
            threadgroup uint* other_cells = list_cells + (lane + stride) * k;
            for (uint slot = 0; slot < k; slot++) {
                // The other list ascends, so once it stops beating our worst
                // entry nothing behind it can either.
                if (!closer(other_distances[slot], other_cells[slot],
                            mine_distances[k - 1], mine_cells[k - 1])) {
                    break;
                }
                insert(mine_distances, mine_cells, k, other_distances[slot], other_cells[slot]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = lane; slot < k; slot += lane_count) {
        out_cells[(ulong)query * k + slot] = list_cells[slot];
        out_distances[(ulong)query * k + slot] = sqrt(list_distances[slot]);
    }
}
"#;

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

    fn assert_matches_reference(embedding: &Array2<f32>, k: usize, context: &MetalContext) {
        let gpu = knn_metal(context, embedding, k).unwrap();
        let cpu = cpu_knn(embedding, k);
        assert_eq!(gpu.indices, cpu.indices);
        for (got, want) in gpu.distances.iter().zip(cpu.distances.iter()) {
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1e-6),
                "distance {got} != {want}"
            );
        }
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
}
