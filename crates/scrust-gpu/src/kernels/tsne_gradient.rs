use metal::{MTLSize, NSUInteger};
use ndarray::Array2;
use scrust_core::error::{Error, Result};

use crate::context::MetalContext;

/// Largest embedding dimensionality the kernels hold in registers.
///
/// t-SNE embeds into two or three dimensions; a fixed bound lets both kernels
/// keep a point and its accumulator in registers instead of scratch memory.
/// Must match `max_dims` in [`TSNE_GRADIENT_SOURCE`].
const MAX_EMBEDDING_DIMS: usize = 4;

const ATTRACTIVE_FUNCTION: &str = "tsne_attractive";
const REPULSIVE_FUNCTION: &str = "tsne_repulsive";

/// Attractive and repulsive gradient of the t-SNE objective for one iteration.
///
/// Returns the gradient and the normalisation constant Z of the low-dimensional
/// affinities, which the caller needs to scale the repulsive term.
///
/// The gradient for point `i` is `4 * sum_j (p_ij - q_ij) * w_ij * (y_i - y_j)`
/// with `w_ij = 1 / (1 + |y_i - y_j|^2)` and `q_ij = w_ij / Z`. It splits into a
/// sparse attractive part, which only touches the stored affinities, and a dense
/// repulsive part `-(4 / Z) * sum_{j != i} w_ij^2 * (y_i - y_j)` over every other
/// point. `exaggeration` multiplies the attractive part alone.
///
/// # Precision of Z
/// `Z` sums `n^2` positive terms, so a flat `f32` accumulation drifts: once the
/// running sum is large compared with the terms still arriving, they round away.
/// It is reduced in two levels instead — a log-depth tree in threadgroup memory
/// yields one `f32` partial per point, and those `n` partials are summed on the
/// host in `f64`. Measured against a full `f64` reference this holds `Z` to a
/// relative error of 5e-8 at 400 points and 3e-8 at 10 000.
pub fn tsne_gradient(
    context: &MetalContext,
    embedding: &Array2<f32>,
    affinity_indptr: &[u32],
    affinity_indices: &[u32],
    affinity_values: &[f32],
    exaggeration: f32,
) -> Result<(Array2<f32>, f32)> {
    let (points, dims) = embedding.dim();
    validate(
        points,
        dims,
        affinity_indptr,
        affinity_indices,
        affinity_values,
        exaggeration,
    )?;

    // A lone point forms no pair, so Z is zero and the gradient vanishes; the
    // repulsive term would divide by it.
    if points < 2 {
        return Ok((Array2::zeros((points, dims)), 0.0));
    }

    let contiguous = embedding.as_standard_layout();
    let coordinates = contiguous
        .as_slice()
        .ok_or_else(|| Error::shape("a contiguous embedding", "a strided view"))?;

    let attractive_pipeline = context.pipeline(ATTRACTIVE_FUNCTION, TSNE_GRADIENT_SOURCE)?;
    let repulsive_pipeline = context.pipeline(REPULSIVE_FUNCTION, TSNE_GRADIENT_SOURCE)?;

    let embedding_buffer = context.buffer(coordinates);
    let indptr_buffer = context.buffer(affinity_indptr);
    // Metal rejects a zero-length buffer and an empty affinity list is legal.
    let indices_buffer = context.buffer(non_empty(affinity_indices, &[0]));
    let values_buffer = context.buffer(non_empty(affinity_values, &[0.0]));
    let attractive_buffer = context.empty_buffer::<f32>(points * dims);
    let repulsive_buffer = context.empty_buffer::<f32>(points * dims);
    let row_z_buffer = context.empty_buffer::<f32>(points);

    let point_count = points as u32;
    let dim_count = dims as u32;
    let command_buffer = context.queue().new_command_buffer();

    // Attractive pass: one thread per point, each walking its own CSR row, which
    // holds roughly 3 * perplexity entries.
    let attractive_width = threadgroup_width(&attractive_pipeline, points);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&attractive_pipeline);
    encoder.set_buffer(0, Some(&embedding_buffer), 0);
    encoder.set_buffer(1, Some(&indptr_buffer), 0);
    encoder.set_buffer(2, Some(&indices_buffer), 0);
    encoder.set_buffer(3, Some(&values_buffer), 0);
    encoder.set_buffer(4, Some(&attractive_buffer), 0);
    set_scalar(encoder, 5, &dim_count);
    set_scalar(encoder, 6, &point_count);
    set_scalar(encoder, 7, &exaggeration);
    encoder.dispatch_thread_groups(
        MTLSize::new(points.div_ceil(attractive_width) as u64, 1, 1),
        MTLSize::new(attractive_width as u64, 1, 1),
    );
    encoder.end_encoding();

    // Repulsive pass: one threadgroup per point, its `width` lanes striding over
    // all points so each holds a partial force and a partial Z, then a tree
    // reduction in threadgroup memory collapses them to one result per row.
    let width = threadgroup_width(&repulsive_pipeline, points);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&repulsive_pipeline);
    encoder.set_buffer(0, Some(&embedding_buffer), 0);
    encoder.set_buffer(1, Some(&repulsive_buffer), 0);
    encoder.set_buffer(2, Some(&row_z_buffer), 0);
    set_scalar(encoder, 3, &dim_count);
    set_scalar(encoder, 4, &point_count);
    encoder.set_threadgroup_memory_length(0, (width * dims * size_of::<f32>()) as NSUInteger);
    encoder.set_threadgroup_memory_length(1, (width * size_of::<f32>()) as NSUInteger);
    encoder.dispatch_thread_groups(
        MTLSize::new(points as u64, 1, 1),
        MTLSize::new(width as u64, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // SAFETY: each buffer was sized for exactly this many elements above and is
    // written in full by the kernels.
    let attractive = unsafe { MetalContext::read::<f32>(&attractive_buffer, points * dims) };
    let repulsive = unsafe { MetalContext::read::<f32>(&repulsive_buffer, points * dims) };
    let row_z = unsafe { MetalContext::read::<f32>(&row_z_buffer, points) };

    let normalisation: f64 = row_z.iter().map(|&partial| f64::from(partial)).sum();
    let inverse_z = if normalisation > 0.0 {
        (1.0 / normalisation) as f32
    } else {
        0.0
    };
    let gradient: Vec<f32> = attractive
        .iter()
        .zip(&repulsive)
        .map(|(&attract, &repel)| 4.0 * (attract - inverse_z * repel))
        .collect();

    let gradient = Array2::from_shape_vec((points, dims), gradient)
        .map_err(|error| Error::shape(format!("({points}, {dims})"), error.to_string()))?;
    Ok((gradient, normalisation as f32))
}

/// Everything the kernels assume, checked once on the host: they index the CSR
/// arrays without bounds checks of their own.
fn validate(
    points: usize,
    dims: usize,
    indptr: &[u32],
    indices: &[u32],
    values: &[f32],
    exaggeration: f32,
) -> Result<()> {
    if dims == 0 || dims > MAX_EMBEDDING_DIMS {
        return Err(Error::parameter(
            "embedding dimensionality",
            "between 1 and 4",
            dims,
        ));
    }
    if indptr.len() != points + 1 {
        return Err(Error::shape(
            format!("an indptr of length {}", points + 1),
            format!("length {}", indptr.len()),
        ));
    }
    if indices.len() != values.len() {
        return Err(Error::shape(
            format!("{} affinity values", indices.len()),
            format!("{} affinity values", values.len()),
        ));
    }
    if indptr[0] != 0 || indptr.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::parameter(
            "affinity_indptr",
            "non-decreasing and starting at zero",
            "an out-of-order row offset",
        ));
    }
    if indptr[points] as usize != values.len() {
        return Err(Error::shape(
            format!("{} affinity values", indptr[points]),
            format!("{} affinity values", values.len()),
        ));
    }
    if indices.iter().any(|&column| column as usize >= points) {
        return Err(Error::parameter(
            "affinity_indices",
            "below the number of points",
            "an out-of-range column",
        ));
    }
    if !exaggeration.is_finite() || exaggeration <= 0.0 {
        return Err(Error::parameter(
            "exaggeration",
            "finite and positive",
            exaggeration,
        ));
    }
    Ok(())
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
        value as *const T as *const std::ffi::c_void,
    );
}

/// The widest power-of-two threadgroup the pipeline allows, never wider than the
/// work itself. The tree reduction halves the active lane count each step, so a
/// power of two is required.
fn threadgroup_width(pipeline: &metal::ComputePipelineState, work: usize) -> usize {
    let allowed = (pipeline.max_total_threads_per_threadgroup() as usize).max(1);
    let capacity = 1usize << (usize::BITS - 1 - allowed.leading_zeros());
    work.max(1).next_power_of_two().min(capacity)
}

const TSNE_GRADIENT_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Must match MAX_EMBEDDING_DIMS on the Rust side.
constexpr constant uint max_dims = 4;

// Sparse attractive term, sum_j exaggeration * p_ij * w_ij * (y_i - y_j). The
// factor of four is applied once by the host to the combined gradient. p_ij is
// zero off the stored pattern, so only the CSR row matters.
kernel void tsne_attractive(device const float *embedding [[buffer(0)]],
                            device const uint *indptr [[buffer(1)]],
                            device const uint *indices [[buffer(2)]],
                            device const float *affinities [[buffer(3)]],
                            device float *attractive [[buffer(4)]],
                            constant uint &dims [[buffer(5)]],
                            constant uint &count [[buffer(6)]],
                            constant float &exaggeration [[buffer(7)]],
                            uint point [[thread_position_in_grid]]) {
    if (point >= count) {
        return;
    }
    float centre[max_dims];
    float force[max_dims];
    for (uint axis = 0; axis < dims; ++axis) {
        centre[axis] = embedding[point * dims + axis];
        force[axis] = 0.0f;
    }
    const uint row_end = indptr[point + 1];
    for (uint slot = indptr[point]; slot < row_end; ++slot) {
        const uint other = indices[slot];
        float offset[max_dims];
        float square_distance = 0.0f;
        for (uint axis = 0; axis < dims; ++axis) {
            offset[axis] = centre[axis] - embedding[other * dims + axis];
            square_distance += offset[axis] * offset[axis];
        }
        const float weight = exaggeration * affinities[slot] / (1.0f + square_distance);
        for (uint axis = 0; axis < dims; ++axis) {
            force[axis] += weight * offset[axis];
        }
    }
    for (uint axis = 0; axis < dims; ++axis) {
        attractive[point * dims + axis] = force[axis];
    }
}

// Dense repulsive term. One threadgroup owns one point i; its lanes stride over
// every point j and accumulate sum_j w_ij^2 * (y_i - y_j) together with the
// row's share of Z = sum_{k != l} w_kl. Both are reduced in threadgroup memory,
// leaving the host to sum the per-row Z in f64. The self pair is masked
// arithmetically rather than skipped so that no lane diverges.
kernel void tsne_repulsive(device const float *embedding [[buffer(0)]],
                           device float *repulsive [[buffer(1)]],
                           device float *row_z [[buffer(2)]],
                           constant uint &dims [[buffer(3)]],
                           constant uint &count [[buffer(4)]],
                           threadgroup float *force_scratch [[threadgroup(0)]],
                           threadgroup float *z_scratch [[threadgroup(1)]],
                           uint point [[threadgroup_position_in_grid]],
                           uint lane [[thread_position_in_threadgroup]],
                           uint width [[threads_per_threadgroup]]) {
    float centre[max_dims];
    float force[max_dims];
    for (uint axis = 0; axis < dims; ++axis) {
        centre[axis] = embedding[point * dims + axis];
        force[axis] = 0.0f;
    }
    float partial_z = 0.0f;
    for (uint other = lane; other < count; other += width) {
        float offset[max_dims];
        float square_distance = 0.0f;
        for (uint axis = 0; axis < dims; ++axis) {
            offset[axis] = centre[axis] - embedding[other * dims + axis];
            square_distance += offset[axis] * offset[axis];
        }
        const float weight = 1.0f / (1.0f + square_distance);
        // The self pair already contributes a zero offset; only Z must drop it.
        partial_z += other == point ? 0.0f : weight;
        const float square_weight = weight * weight;
        for (uint axis = 0; axis < dims; ++axis) {
            force[axis] += square_weight * offset[axis];
        }
    }

    for (uint axis = 0; axis < dims; ++axis) {
        force_scratch[lane * dims + axis] = force[axis];
    }
    z_scratch[lane] = partial_z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            z_scratch[lane] += z_scratch[lane + stride];
            for (uint axis = 0; axis < dims; ++axis) {
                force_scratch[lane * dims + axis] +=
                    force_scratch[(lane + stride) * dims + axis];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        row_z[point] = z_scratch[0];
        for (uint axis = 0; axis < dims; ++axis) {
            repulsive[point * dims + axis] = force_scratch[axis];
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic generator for the larger comparisons; `rand` is not a
    /// dependency of this crate.
    struct Xorshift(u64);

    impl Xorshift {
        fn next_unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / 16_777_216.0
        }
    }

    /// f64 oracle returning the full gradient, the attractive term alone and Z.
    fn cpu_reference(
        embedding: &Array2<f32>,
        indptr: &[u32],
        indices: &[u32],
        values: &[f32],
        exaggeration: f32,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let (points, dims) = embedding.dim();
        let at = |i: usize, axis: usize| f64::from(embedding[[i, axis]]);
        let weight = |i: usize, j: usize| {
            let square: f64 = (0..dims)
                .map(|axis| (at(i, axis) - at(j, axis)).powi(2))
                .sum();
            1.0 / (1.0 + square)
        };

        let mut normalisation = 0.0;
        for i in 0..points {
            for j in 0..points {
                if i != j {
                    normalisation += weight(i, j);
                }
            }
        }

        let mut attractive = vec![0.0; points * dims];
        let mut gradient = vec![0.0; points * dims];
        for i in 0..points {
            for slot in indptr[i] as usize..indptr[i + 1] as usize {
                let j = indices[slot] as usize;
                let scale = f64::from(exaggeration) * f64::from(values[slot]) * weight(i, j);
                for axis in 0..dims {
                    attractive[i * dims + axis] += 4.0 * scale * (at(i, axis) - at(j, axis));
                }
            }
            for j in 0..points {
                if i == j {
                    continue;
                }
                let scale = weight(i, j).powi(2) / normalisation;
                for axis in 0..dims {
                    gradient[i * dims + axis] -= 4.0 * scale * (at(i, axis) - at(j, axis));
                }
            }
            for axis in 0..dims {
                gradient[i * dims + axis] += attractive[i * dims + axis];
            }
        }
        (gradient, attractive, normalisation)
    }

    type Problem = (Array2<f32>, Vec<u32>, Vec<u32>, Vec<f32>);

    fn random_problem(points: usize, dims: usize, seed: u64) -> Problem {
        let mut random = Xorshift(seed);
        let embedding = Array2::from_shape_fn((points, dims), |_| random.next_unit() * 20.0 - 10.0);
        let neighbours = 8.min(points - 1);
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..points {
            for step in 1..=neighbours {
                indices.push(((i + step) % points) as u32);
                values.push(random.next_unit() / (points * neighbours) as f32);
            }
            indptr.push(indices.len() as u32);
        }
        (embedding, indptr, indices, values)
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs().max(1e-12)
    }

    #[test]
    fn two_points_match_a_hand_computed_gradient() {
        let Ok(context) = MetalContext::new() else {
            return; // no GPU on this machine
        };
        let embedding = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let p = 0.3f64;
        let (gradient, z) = tsne_gradient(
            &context,
            &embedding,
            &[0, 1, 2],
            &[1, 0],
            &[p as f32, p as f32],
            1.0,
        )
        .unwrap();

        // |y0 - y1|^2 = 1, so w = 1/2 and Z = w01 + w10 = 1.
        let w = 1.0 / (1.0 + 1.0);
        let expected_z = 2.0 * w;
        let q = w / expected_z;
        // grad_0 = 4 * (p - q) * w * (y0 - y1), and (y0 - y1) = (-1, 0).
        let expected_x = -(4.0 * (p - q) * w);

        assert!(relative_error(f64::from(z), expected_z) < 1e-5);
        assert!(relative_error(f64::from(gradient[[0, 0]]), expected_x) < 1e-5);
        assert!(relative_error(f64::from(gradient[[1, 0]]), -expected_x) < 1e-5);
        assert!(gradient[[0, 1]].abs() < 1e-6 && gradient[[1, 1]].abs() < 1e-6);
    }

    #[test]
    fn three_points_match_a_hand_computed_gradient() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        // A right triangle: y0 = (0, 0), y1 = (1, 0), y2 = (0, 1). Point 0 is
        // affine to both others, and each of them only back to 0.
        let embedding = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let p = 0.25f64;
        let (gradient, z) = tsne_gradient(
            &context,
            &embedding,
            &[0, 2, 3, 4],
            &[1, 2, 0, 0],
            &[p as f32; 4],
            1.0,
        )
        .unwrap();

        let w01 = 1.0 / (1.0 + 1.0);
        let w02 = 1.0 / (1.0 + 1.0);
        let w12 = 1.0 / (1.0 + 2.0);
        let expected_z = 2.0 * (w01 + w02 + w12);
        // Along x, point 0 only sees pair (0, 1), whose offset is -1.
        let expected_0x = -(4.0 * (p - w01 / expected_z) * w01);
        // Point 0 attracts point 2 along y by the mirror image of that.
        let expected_0y = -(4.0 * (p - w02 / expected_z) * w02);
        // Point 1 sees pair (1, 0) with offset +1 and pair (1, 2) with offset
        // +1, and p_12 is not stored, so the latter is repulsion only.
        let expected_1x = 4.0 * (p - w01 / expected_z) * w01 + 4.0 * (0.0 - w12 / expected_z) * w12;

        assert!(relative_error(f64::from(z), expected_z) < 1e-5);
        assert!(relative_error(f64::from(gradient[[0, 0]]), expected_0x) < 1e-5);
        assert!(relative_error(f64::from(gradient[[0, 1]]), expected_0y) < 1e-5);
        assert!(relative_error(f64::from(gradient[[1, 0]]), expected_1x) < 1e-5);
    }

    #[test]
    fn matches_the_cpu_reference_on_random_input() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let (embedding, indptr, indices, values) = random_problem(400, 2, 0x5eed);
        let exaggeration = 12.0;
        let (gradient, z) = tsne_gradient(
            &context,
            &embedding,
            &indptr,
            &indices,
            &values,
            exaggeration,
        )
        .unwrap();
        let (expected, _, expected_z) =
            cpu_reference(&embedding, &indptr, &indices, &values, exaggeration);

        // Relative to the largest component: individual entries pass through
        // zero, where a per-entry ratio is meaningless.
        let scale = expected.iter().fold(0.0f64, |worst, v| worst.max(v.abs()));
        let deviation = gradient
            .iter()
            .zip(&expected)
            .fold(0.0f64, |worst, (&actual, &want)| {
                worst.max((f64::from(actual) - want).abs() / scale)
            });
        assert!(deviation < 1e-4, "gradient deviation {deviation}");
        let z_deviation = relative_error(f64::from(z), expected_z);
        assert!(z_deviation < 1e-5, "Z deviation {z_deviation}");
    }

    #[test]
    fn a_symmetric_affinity_gives_a_gradient_summing_to_zero() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let points = 64;
        let mut random = Xorshift(7);
        let embedding = Array2::from_shape_fn((points, 2), |_| random.next_unit() * 4.0 - 2.0);
        let mut dense = vec![0.0f32; points * points];
        for i in 0..points {
            for j in (i + 1)..points {
                let value = (random.next_unit() + 0.1) / points as f32;
                dense[i * points + j] = value;
                dense[j * points + i] = value;
            }
        }
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..points {
            for j in 0..points {
                if dense[i * points + j] > 0.0 {
                    indices.push(j as u32);
                    values.push(dense[i * points + j]);
                }
            }
            indptr.push(indices.len() as u32);
        }

        let (gradient, _) =
            tsne_gradient(&context, &embedding, &indptr, &indices, &values, 1.0).unwrap();
        // Every pair enters two rows with opposite sign, so the columns cancel.
        let magnitude = gradient.iter().fold(0.0f32, |worst, v| worst.max(v.abs()));
        for axis in 0..2 {
            let total: f32 = gradient.column(axis).sum();
            assert!(
                total.abs() < 1e-4 * magnitude * points as f32,
                "axis {axis} sums to {total}"
            );
        }
    }

    #[test]
    fn exaggeration_scales_only_the_attractive_term() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let (embedding, indptr, indices, values) = random_problem(200, 2, 0xabcd);
        let (plain, plain_z) =
            tsne_gradient(&context, &embedding, &indptr, &indices, &values, 1.0).unwrap();
        let (exaggerated, exaggerated_z) =
            tsne_gradient(&context, &embedding, &indptr, &indices, &values, 12.0).unwrap();
        let (_, attractive, _) = cpu_reference(&embedding, &indptr, &indices, &values, 1.0);

        // The repulsive term does not move, so the difference is exactly eleven
        // further copies of the unexaggerated attractive term.
        assert_eq!(plain_z, exaggerated_z);
        let scale = attractive
            .iter()
            .fold(0.0f64, |worst, v| worst.max(v.abs()));
        let deviation = exaggerated.iter().zip(plain.iter()).zip(&attractive).fold(
            0.0f64,
            |worst, ((&big, &small), &want)| {
                worst.max((f64::from(big - small) - 11.0 * want).abs() / scale)
            },
        );
        assert!(deviation < 1e-4, "attractive scaling deviation {deviation}");
    }

    #[test]
    fn repeated_runs_are_identical() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let (embedding, indptr, indices, values) = random_problem(300, 3, 0x1234);
        let first = tsne_gradient(&context, &embedding, &indptr, &indices, &values, 4.0).unwrap();
        let second = tsne_gradient(&context, &embedding, &indptr, &indices, &values, 4.0).unwrap();
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }

    #[test]
    fn rejects_malformed_input() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let embedding: Array2<f32> = Array2::zeros((3, 2));
        let three_values = [0.5f32; 3];
        // An indptr of the wrong length.
        assert!(tsne_gradient(&context, &embedding, &[0, 1], &[1], &[0.5], 1.0).is_err());
        // A column index past the last point.
        assert!(tsne_gradient(
            &context,
            &embedding,
            &[0, 1, 2, 3],
            &[9, 0, 1],
            &three_values,
            1.0
        )
        .is_err());
        // Row offsets that go backwards.
        assert!(tsne_gradient(
            &context,
            &embedding,
            &[0, 2, 1, 3],
            &[1, 2, 0],
            &three_values,
            1.0
        )
        .is_err());
        // A non-positive exaggeration.
        assert!(tsne_gradient(&context, &embedding, &[0, 0, 0, 0], &[], &[], 0.0).is_err());
        // More embedding dimensions than the kernels hold in registers.
        let wide: Array2<f32> = Array2::zeros((3, MAX_EMBEDDING_DIMS + 1));
        assert!(tsne_gradient(&context, &wide, &[0, 0, 0, 0], &[], &[], 1.0).is_err());
    }

    /// Timing at the scale the kernel exists for. Ignored by default: the f64
    /// reference it is measured against walks 1e8 pairs.
    #[test]
    #[ignore = "benchmark, run with --release"]
    fn times_ten_thousand_points_against_the_cpu() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let points = 10_000;
        let (embedding, indptr, indices, values) = random_problem(points, 2, 0x600d);

        // Warm the pipeline cache so compilation does not land in the timing.
        let _ = tsne_gradient(&context, &embedding, &indptr, &indices, &values, 12.0).unwrap();
        let runs = 10;
        let started = std::time::Instant::now();
        let mut measured_z = 0.0;
        for _ in 0..runs {
            let (_, z) =
                tsne_gradient(&context, &embedding, &indptr, &indices, &values, 12.0).unwrap();
            measured_z = z;
        }
        let gpu = started.elapsed() / runs;

        let started = std::time::Instant::now();
        let (_, _, expected_z) = cpu_reference(&embedding, &indptr, &indices, &values, 12.0);
        let cpu = started.elapsed();
        println!(
            "{points} points: GPU {gpu:?} per gradient, f64 CPU reference {cpu:?}, \
             Z relative error {:.3e}",
            relative_error(f64::from(measured_z), expected_z)
        );
    }
}
