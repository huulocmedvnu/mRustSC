
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
