use ndarray::Array2;
use scrust_core::error::{Error, Result};
use scrust_core::umap::{fit_ab_params, UmapParams};

use crate::context::MetalContext;

/// Largest embedding dimension the kernel keeps in thread-private registers.
///
/// UMAP layouts are two or three dimensional in practice; the bound exists so
/// the kernel can hold a vertex in registers instead of spilling to memory.
const MAX_EMBEDDING_DIM: usize = 16;

/// `repulsion_strength` in umap-learn. `UmapParams` does not expose it, and
/// scanpy never changes it from its default.
const REPULSION_STRENGTH: f32 = 1.0;

const KERNEL_NAME: &str = "umap_sgd_epoch";

/// One epoch of UMAP's attractive and repulsive updates.
///
/// `head`/`tail` are the graph edges and `epochs_per_sample` the schedule that
/// decides which edges fire this epoch.
///
/// # Concurrency
///
/// One thread per edge, and edges share endpoints, so threads race on the same
/// coordinates. That race is **accepted**, as umap-learn accepts it in its
/// `parallel=True` mode: edge colouring would leave most of the GPU idle on a
/// realistic k-NN graph, and a per-thread accumulate-then-reduce pass would cost
/// a full extra `edges x dim` of traffic to remove noise the algorithm is known
/// to tolerate.
///
/// The race is narrowed where it is cheap to do so. Writes are atomic adds, so
/// no thread's contribution is ever lost, and a thread accumulates its own
/// attraction and repulsions in registers so each vertex takes one atomic add
/// per dimension. What stays unordered is the *reads*, most visibly the negative
/// samples, which observe other vertices mid-flight.
///
/// Two consequences a caller must know:
///
/// - The result is **not bit-for-bit reproducible** on a graph whose edges share
///   endpoints, even for a fixed seed. It is reproducible in structure — which
///   vertices end up together — not in coordinates.
/// - Because the whole edge list is in flight at once, a high-degree vertex
///   receives most of its attractions computed from the same starting position.
///   The update is therefore closer to a simultaneous (Jacobi) step than to
///   umap-learn's sequential sweep. Still a descent step, but not the same one.
///
/// Edges that share no endpoint, with negative sampling off, race with nothing
/// and are bit-for-bit reproducible; the tests assert exactly that split.
pub fn umap_epoch(
    context: &MetalContext,
    embedding: &mut Array2<f32>,
    head: &[u32],
    tail: &[u32],
    epochs_per_sample: &[f32],
    epoch: usize,
    params: &UmapParams,
) -> Result<()> {
    let n_vertices = embedding.nrows();
    let dim = embedding.ncols();
    validate(embedding, head, tail, epochs_per_sample, epoch, params)?;
    if head.is_empty() {
        return Ok(());
    }

    let (a, b) = fit_ab_params(params.min_dist, params.spread)?;
    let uniforms = EpochUniforms {
        a,
        b,
        gamma: REPULSION_STRENGTH,
        // umap-learn decays the learning rate linearly to zero over the run.
        alpha: params.learning_rate * (1.0 - epoch as f32 / params.n_epochs as f32),
        dim: dim as u32,
        n_vertices: n_vertices as u32,
        n_edges: head.len() as u32,
        negative_sample_rate: params.negative_sample_rate as u32,
        epoch: epoch as u32,
        seed_lo: params.seed as u32,
        seed_hi: (params.seed >> 32) as u32,
        padding: 0,
    };

    // The signature owns the embedding on the host, so the buffers are rebuilt
    // and the coordinates copied back every epoch. At 500k edges that is about
    // half the wall time of an epoch; see the benchmark test. Removing it needs
    // a caller-held handle to the device-side embedding, which the contract for
    // this function does not have.
    let coordinates = embedding
        .as_slice_mut()
        .ok_or_else(|| Error::shape("contiguous row-major embedding", "a strided view"))?;
    let embedding_buffer = context.buffer(coordinates);
    let head_buffer = context.buffer(head);
    let tail_buffer = context.buffer(tail);
    let schedule_buffer = context.buffer(epochs_per_sample);

    let pipeline = context.pipeline(KERNEL_NAME, KERNEL_SOURCE)?;
    let command_buffer = context.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&embedding_buffer), 0);
    encoder.set_buffer(1, Some(&head_buffer), 0);
    encoder.set_buffer(2, Some(&tail_buffer), 0);
    encoder.set_buffer(3, Some(&schedule_buffer), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<EpochUniforms>() as u64,
        &uniforms as *const EpochUniforms as *const std::ffi::c_void,
    );
    let threads_per_group = pipeline
        .max_total_threads_per_threadgroup()
        .min(head.len() as u64);
    encoder.dispatch_threads(
        metal::MTLSize::new(head.len() as u64, 1, 1),
        metal::MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // SAFETY: the buffer was created from, and has the same length as, `coordinates`.
    let updated = unsafe { MetalContext::read::<f32>(&embedding_buffer, coordinates.len()) };
    coordinates.copy_from_slice(&updated);
    Ok(())
}

fn validate(
    embedding: &Array2<f32>,
    head: &[u32],
    tail: &[u32],
    epochs_per_sample: &[f32],
    epoch: usize,
    params: &UmapParams,
) -> Result<()> {
    if tail.len() != head.len() || epochs_per_sample.len() != head.len() {
        return Err(Error::shape(
            format!("head, tail and epochs_per_sample of length {}", head.len()),
            format!(
                "tail of length {} and epochs_per_sample of length {}",
                tail.len(),
                epochs_per_sample.len()
            ),
        ));
    }
    let dim = embedding.ncols();
    if dim != params.n_components {
        return Err(Error::shape(
            format!("an embedding with {} columns", params.n_components),
            format!("{dim} columns"),
        ));
    }
    if dim == 0 || dim > MAX_EMBEDDING_DIM {
        return Err(Error::parameter(
            "n_components",
            "between 1 and 16",
            params.n_components,
        ));
    }
    if params.n_epochs == 0 {
        return Err(Error::parameter("n_epochs", "at least 1", params.n_epochs));
    }
    if epoch >= params.n_epochs {
        return Err(Error::parameter(
            "epoch",
            "less than n_epochs",
            format!("{epoch} of {}", params.n_epochs),
        ));
    }
    let n_vertices = embedding.nrows() as u32;
    if head.iter().chain(tail).any(|&vertex| vertex >= n_vertices) {
        return Err(Error::shape(
            format!("edge endpoints below {n_vertices}"),
            "an endpoint outside the embedding",
        ));
    }
    Ok(())
}

/// Mirrors `EpochUniforms` in the Metal source; both are 12 tightly packed
/// 4-byte scalars.
#[repr(C)]
#[derive(Clone, Copy)]
struct EpochUniforms {
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    dim: u32,
    n_vertices: u32,
    n_edges: u32,
    negative_sample_rate: u32,
    epoch: u32,
    seed_lo: u32,
    seed_hi: u32,
    padding: u32,
}

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define MAX_EMBEDDING_DIM 16

struct EpochUniforms {
    float a;
    float b;
    float gamma;
    float alpha;
    uint dim;
    uint n_vertices;
    uint n_edges;
    uint negative_sample_rate;
    uint epoch;
    uint seed_lo;
    uint seed_hi;
    uint padding;
};

// umap-learn clamps every per-dimension gradient to this range, which is what
// keeps a near-coincident pair from launching itself out of the layout.
static inline float clip(float value) {
    return clamp(value, -4.0f, 4.0f);
}

static inline ulong scramble(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

// splitmix64: the whole per-thread state is derived from the seed, the epoch and
// the edge index, so a thread draws the same negative samples on every run.
static inline uint next_random(thread ulong &state) {
    state += 0x9E3779B97F4A7C15ul;
    return (uint)(scramble(state) >> 32);
}

kernel void umap_sgd_epoch(device atomic_float *embedding [[buffer(0)]],
                           device const uint *head [[buffer(1)]],
                           device const uint *tail [[buffer(2)]],
                           device const float *epochs_per_sample [[buffer(3)]],
                           constant EpochUniforms &uniforms [[buffer(4)]],
                           uint edge [[thread_position_in_grid]]) {
    if (edge >= uniforms.n_edges) {
        return;
    }
    const float schedule = epochs_per_sample[edge];
    if (!(schedule > 0.0f)) {
        return;
    }
    // umap-learn fires an edge when a per-edge counter, stepped by `schedule`,
    // reaches the epoch number. Since `schedule >= 1` an edge fires at most once
    // per epoch, so that counter is exactly floor(epoch / schedule) and the
    // schedule needs no state carried between epochs.
    const uint fired_through_now = (uint)floor((float)uniforms.epoch / schedule);
    const uint fired_before = uniforms.epoch == 0u
        ? 0u
        : (uint)floor((float)(uniforms.epoch - 1u) / schedule);
    if (fired_through_now == fired_before) {
        return;
    }

    const uint dim = uniforms.dim;
    const uint j = head[edge];
    const uint k = tail[edge];
    const float a = uniforms.a;
    const float b = uniforms.b;
    const float alpha = uniforms.alpha;

    float current[MAX_EMBEDDING_DIM];
    float other[MAX_EMBEDDING_DIM];
    float accumulated[MAX_EMBEDDING_DIM]; // this thread's total motion of the head vertex

    float distance_squared = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        current[d] = atomic_load_explicit(&embedding[j * dim + d], memory_order_relaxed);
        other[d] = atomic_load_explicit(&embedding[k * dim + d], memory_order_relaxed);
        const float difference = current[d] - other[d];
        distance_squared += difference * difference;
    }

    float coefficient = 0.0f;
    if (distance_squared > 0.0f) {
        coefficient = -2.0f * a * b * pow(distance_squared, b - 1.0f);
        coefficient /= a * pow(distance_squared, b) + 1.0f;
    }
    for (uint d = 0; d < dim; ++d) {
        const float step = clip(coefficient * (current[d] - other[d])) * alpha;
        accumulated[d] = step;
        current[d] += step;
        // The tail vertex only ever feels the attractive half, so it is written
        // straight back; the head keeps accumulating through the repulsions.
        atomic_fetch_add_explicit(&embedding[k * dim + d], -step, memory_order_relaxed);
    }

    ulong random_state = scramble(((ulong)uniforms.seed_hi << 32) | (ulong)uniforms.seed_lo)
                       ^ scramble(((ulong)uniforms.epoch << 32) | (ulong)edge);
    for (uint sample = 0; sample < uniforms.negative_sample_rate; ++sample) {
        const uint c = next_random(random_state) % uniforms.n_vertices;
        if (c == j) {
            continue; // a vertex cannot repel itself
        }
        float negative_distance_squared = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            other[d] = atomic_load_explicit(&embedding[c * dim + d], memory_order_relaxed);
            const float difference = current[d] - other[d];
            negative_distance_squared += difference * difference;
        }
        if (!(negative_distance_squared > 0.0f)) {
            continue; // coincident points give a zero gradient in umap-learn
        }
        float repulsion = 2.0f * uniforms.gamma * b;
        repulsion /= (0.001f + negative_distance_squared)
                   * (a * pow(negative_distance_squared, b) + 1.0f);
        for (uint d = 0; d < dim; ++d) {
            const float step = clip(repulsion * (current[d] - other[d])) * alpha;
            accumulated[d] += step;
            current[d] += step;
        }
    }

    for (uint d = 0; d < dim; ++d) {
        atomic_fetch_add_explicit(&embedding[j * dim + d], accumulated[d], memory_order_relaxed);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Sequential CPU reference for one epoch, transcribed from umap-learn's
    /// `_optimize_layout_euclidean_single_epoch`. It has no races by
    /// construction and is the oracle the kernel is measured against.
    fn cpu_epoch(
        embedding: &mut Array2<f32>,
        head: &[u32],
        tail: &[u32],
        epochs_per_sample: &[f32],
        epoch: usize,
        params: &UmapParams,
    ) {
        let (a, b) = fit_ab_params(params.min_dist, params.spread).unwrap();
        let alpha = params.learning_rate * (1.0 - epoch as f32 / params.n_epochs as f32);
        let dim = embedding.ncols();
        let clip = |value: f32| value.clamp(-4.0, 4.0);

        for edge in 0..head.len() {
            let schedule = epochs_per_sample[edge];
            let fired = (epoch as f32 / schedule).floor() as u32;
            let fired_before = if epoch == 0 {
                0
            } else {
                ((epoch - 1) as f32 / schedule).floor() as u32
            };
            if fired == fired_before {
                continue;
            }
            let (j, k) = (head[edge] as usize, tail[edge] as usize);
            let distance_squared: f32 = (0..dim)
                .map(|d| (embedding[[j, d]] - embedding[[k, d]]).powi(2))
                .sum();
            let coefficient = if distance_squared > 0.0 {
                -2.0 * a * b * distance_squared.powf(b - 1.0) / (a * distance_squared.powf(b) + 1.0)
            } else {
                0.0
            };
            for d in 0..dim {
                let step = clip(coefficient * (embedding[[j, d]] - embedding[[k, d]])) * alpha;
                embedding[[j, d]] += step;
                embedding[[k, d]] -= step;
            }
            assert_eq!(
                params.negative_sample_rate, 0,
                "the reference only covers the deterministic, repulsion-free update"
            );
        }
    }

    const CLUSTER_SIZE: usize = 40;

    /// Two cliques with no edge between them, laid out from an interleaved start
    /// so the optimiser has to pull them apart rather than start apart.
    fn two_cliques() -> (Vec<u32>, Vec<u32>, Vec<f32>, Array2<f32>) {
        let (mut head, mut tail) = (Vec::new(), Vec::new());
        for clique in 0..2 {
            let offset = clique * CLUSTER_SIZE;
            for i in 0..CLUSTER_SIZE {
                for j in (i + 1)..CLUSTER_SIZE {
                    head.push((offset + i) as u32);
                    tail.push((offset + j) as u32);
                }
            }
        }
        let schedule = vec![1.0f32; head.len()];
        let start = Array2::from_shape_fn((2 * CLUSTER_SIZE, 2), |(row, col)| {
            ((row * 31 + col * 17) % 41) as f32 * 0.2 - 4.0
        });
        (head, tail, schedule, start)
    }

    fn centroid(embedding: &Array2<f32>, clique: usize) -> (f32, f32) {
        let rows = clique * CLUSTER_SIZE..(clique + 1) * CLUSTER_SIZE;
        let n = CLUSTER_SIZE as f32;
        (
            rows.clone().map(|row| embedding[[row, 0]]).sum::<f32>() / n,
            rows.map(|row| embedding[[row, 1]]).sum::<f32>() / n,
        )
    }

    /// Whether every vertex ended up nearer its own clique's centroid than the
    /// other's. This is the structure of the layout rather than its coordinates,
    /// which is the level at which a Hogwild run is reproducible.
    fn cliques_stay_separated(embedding: &Array2<f32>) -> bool {
        let centres = [centroid(embedding, 0), centroid(embedding, 1)];
        let distance_to = |row: usize, centre: (f32, f32)| {
            (embedding[[row, 0]] - centre.0).powi(2) + (embedding[[row, 1]] - centre.1).powi(2)
        };
        (0..2 * CLUSTER_SIZE).all(|row| {
            let own = row / CLUSTER_SIZE;
            distance_to(row, centres[own]) < distance_to(row, centres[1 - own])
        })
    }

    fn optimise(context: &MetalContext, params: &UmapParams) -> Array2<f32> {
        let (head, tail, schedule, mut embedding) = two_cliques();
        for epoch in 0..params.n_epochs {
            umap_epoch(
                context,
                &mut embedding,
                &head,
                &tail,
                &schedule,
                epoch,
                params,
            )
            .unwrap();
        }
        embedding
    }

    #[test]
    fn fits_the_same_curve_parameters_as_umap_learn() {
        // scipy's curve_fit on umap-learn's own sample points.
        let (a, b) = fit_ab_params(0.5, 1.0).unwrap();
        assert!((a - 0.583_030).abs() < 1e-3, "a was {a}");
        assert!((b - 1.334_167).abs() < 1e-3, "b was {b}");
        let (a, b) = fit_ab_params(0.1, 1.0).unwrap();
        assert!((a - 1.576_943).abs() < 1e-3, "a was {a}");
        assert!((b - 0.895_061).abs() < 1e-3, "b was {b}");
    }

    #[test]
    fn rejects_mismatched_edge_arrays() {
        let embedding = arr2(&[[0.0f32, 0.0], [1.0, 0.0]]);
        let error = validate(
            &embedding,
            &[0, 1],
            &[1],
            &[1.0, 1.0],
            0,
            &UmapParams::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Shape { .. }));
    }

    #[test]
    fn rejects_an_endpoint_outside_the_embedding() {
        let embedding = arr2(&[[0.0f32, 0.0], [1.0, 0.0]]);
        let error =
            validate(&embedding, &[0], &[7], &[1.0], 0, &UmapParams::default()).unwrap_err();
        assert!(matches!(error, Error::Shape { .. }));
    }

    #[test]
    fn a_single_edge_pulls_its_endpoints_together_by_the_predicted_amount() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        // alpha = learning_rate * (1 - epoch / n_epochs) = 2 * (1 - 1/2) = 1.
        // Epoch 1 is the first one an edge with epochs_per_sample = 1 fires on,
        // exactly as in umap-learn, whose counter starts at epochs_per_sample.
        let params = UmapParams {
            n_epochs: 2,
            negative_sample_rate: 0,
            learning_rate: 2.0,
            ..UmapParams::default()
        };
        let mut embedding = arr2(&[[0.0f32, 0.0], [1.0, 0.0]]);
        umap_epoch(&context, &mut embedding, &[0], &[1], &[1.0], 1, &params).unwrap();

        // Worked through by hand: d^2 = 1, so d^(2(b-1)) = d^(2b) = 1 and the
        // attractive coefficient is -2ab / (1 + a) ~ -0.98, under the +-4 clip.
        // The x separation head - tail is -1, so the head moves by +0.98 and the
        // tail by the negative of that.
        let (a, b) = fit_ab_params(params.min_dist, params.spread).unwrap();
        let step = 2.0 * a * b / (1.0 + a);
        assert!((embedding[[0, 0]] - step).abs() < 1e-6, "{embedding:?}");
        assert!(
            (embedding[[1, 0]] - (1.0 - step)).abs() < 1e-6,
            "{embedding:?}"
        );
        assert_eq!(embedding[[0, 1]], 0.0);
        assert_eq!(embedding[[1, 1]], 0.0);
        assert!(
            (embedding[[1, 0]] - embedding[[0, 0]]).abs() < 1.0,
            "the endpoints must have moved towards each other"
        );
    }

    #[test]
    fn matches_the_cpu_reference_without_negative_sampling() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let params = UmapParams {
            n_epochs: 20,
            negative_sample_rate: 0,
            ..UmapParams::default()
        };
        // A perfect matching: no two edges share an endpoint, so the kernel's
        // Hogwild reads race with nothing and the run is exactly deterministic.
        let head: Vec<u32> = (0..12).step_by(2).collect();
        let tail: Vec<u32> = (1..12).step_by(2).collect();
        let schedule: Vec<f32> = (0..head.len()).map(|i| 1.0 + i as f32).collect();

        let start = Array2::from_shape_fn((12, 2), |(row, col)| {
            ((row * 7 + col * 3) % 11) as f32 * 0.25 - 1.0
        });
        let mut gpu = start.clone();
        let mut cpu = start;
        for epoch in 0..params.n_epochs {
            umap_epoch(&context, &mut gpu, &head, &tail, &schedule, epoch, &params).unwrap();
            cpu_epoch(&mut cpu, &head, &tail, &schedule, epoch, &params);
        }
        for (gpu_value, cpu_value) in gpu.iter().zip(cpu.iter()) {
            assert!(
                (gpu_value - cpu_value).abs() <= 1e-5 * cpu_value.abs().max(1e-3),
                "gpu {gpu:?} cpu {cpu:?}"
            );
        }
    }

    /// Race-free case: no two edges share an endpoint and there is no negative
    /// sampling, so the kernel is *bit-for-bit* reproducible. This is the
    /// strongest determinism the Hogwild design allows; see `umap_epoch`.
    #[test]
    fn is_bit_for_bit_deterministic_when_no_two_threads_share_a_vertex() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let params = UmapParams {
            n_epochs: 10,
            negative_sample_rate: 0,
            ..UmapParams::default()
        };
        let head: Vec<u32> = (0..40).step_by(2).collect();
        let tail: Vec<u32> = (1..40).step_by(2).collect();
        let schedule = vec![1.0f32; head.len()];
        let start = Array2::from_shape_fn((40, 2), |(row, col)| (row + 2 * col) as f32 * 0.1);

        let run = || {
            let mut embedding = start.clone();
            for epoch in 0..params.n_epochs {
                umap_epoch(
                    &context,
                    &mut embedding,
                    &head,
                    &tail,
                    &schedule,
                    epoch,
                    &params,
                )
                .unwrap();
            }
            embedding
        };
        assert_eq!(run(), run());
    }

    /// Racing case: threads share vertices and each draws negative samples from
    /// wherever other threads have got to, so two runs do **not** agree
    /// coordinate by coordinate, nor even in the scale of the layout. What
    /// survives the race is which vertices end up together, and that is all this
    /// asserts; see `umap_epoch`.
    #[test]
    fn is_reproducible_only_in_structure_when_threads_share_vertices() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let params = UmapParams {
            n_epochs: 100,
            ..UmapParams::default()
        };
        let (first, second) = (optimise(&context, &params), optimise(&context, &params));
        assert!(cliques_stay_separated(&first));
        assert!(cliques_stay_separated(&second));
        assert!(
            first != second,
            "identical coordinates would mean the dispatch is not actually concurrent"
        );
    }

    #[test]
    fn two_hundred_epochs_keep_two_clusters_apart_and_finite() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding = optimise(
            &context,
            &UmapParams {
                n_epochs: 200,
                ..UmapParams::default()
            },
        );
        assert!(embedding.iter().all(|value| value.is_finite()));
        assert!(cliques_stay_separated(&embedding), "clusters merged");
    }

    #[test]
    #[ignore = "benchmark: cargo test -- --ignored --nocapture"]
    fn reports_time_per_epoch_on_half_a_million_edges() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let n_vertices = 50_000;
        let n_edges = 500_000;
        let head: Vec<u32> = (0..n_edges as u32)
            .map(|edge| edge % n_vertices as u32)
            .collect();
        let tail: Vec<u32> = (0..n_edges as u32)
            .map(|edge| (edge * 7919 + 13) % n_vertices as u32)
            .collect();
        let schedule = vec![1.0f32; n_edges];
        let params = UmapParams::default();
        let mut embedding = Array2::from_shape_fn((n_vertices, 2), |(row, col)| {
            ((row * 37 + col * 11) % 97) as f32 * 0.1 - 5.0
        });

        umap_epoch(
            &context,
            &mut embedding,
            &head,
            &tail,
            &schedule,
            0,
            &params,
        )
        .unwrap();
        let epochs = 20;
        let started = std::time::Instant::now();
        for epoch in 1..=epochs {
            umap_epoch(
                &context,
                &mut embedding,
                &head,
                &tail,
                &schedule,
                epoch,
                &params,
            )
            .unwrap();
        }
        let gpu = started.elapsed() / epochs as u32;

        // The signature hands the embedding in and out by value, so every epoch
        // rebuilds its buffers and copies the coordinates back. A schedule no
        // edge ever meets isolates that fixed cost from the arithmetic.
        let idle_schedule = vec![f32::MAX; n_edges];
        let started = std::time::Instant::now();
        for epoch in 1..=epochs {
            umap_epoch(
                &context,
                &mut embedding.clone(),
                &head,
                &tail,
                &idle_schedule,
                epoch,
                &params,
            )
            .unwrap();
        }
        let overhead = started.elapsed() / epochs as u32;

        let mut cpu_embedding = embedding.clone();
        let cpu_params = UmapParams {
            negative_sample_rate: 0,
            ..params
        };
        let started = std::time::Instant::now();
        cpu_epoch(&mut cpu_embedding, &head, &tail, &schedule, 1, &cpu_params);
        let cpu = started.elapsed();
        println!(
            "{n_edges} edges: GPU {gpu:?} per epoch (5 negative samples per edge), \
             of which {overhead:?} is buffer setup and copy back; \
             single-threaded CPU reference {cpu:?} per epoch (attraction only)"
        );
    }
}
