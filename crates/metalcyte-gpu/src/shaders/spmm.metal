
#include <metal_stdlib>
using namespace metal;

// C = A * B with A in CSR. One threadgroup owns one row of A: its lanes stride
// the row's stored entries, each accumulating a private length-k row of C in
// threadgroup memory, and a pairwise tree sums the lanes. A row with no stored
// entries writes zeros, which is the right answer for it.
//
// Only the k columns of B named by the row's column indices are ever read, so
// the traffic is proportional to the stored entries and not to n_cols * k.
kernel void csr_spmm(device const uint *indptr [[buffer(0)]],
                     device const uint *indices [[buffer(1)]],
                     device const float *values [[buffer(2)]],
                     device const float *dense [[buffer(3)]],
                     device float *out [[buffer(4)]],
                     constant uint &k [[buffer(5)]],
                     threadgroup float *scratch [[threadgroup(0)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_position_in_threadgroup]],
                     uint width [[threads_per_threadgroup]]) {
    threadgroup float *mine = scratch + lane * k;
    for (uint column = 0; column < k; ++column) {
        mine[column] = 0.0f;
    }

    const uint row_end = indptr[row + 1];
    for (uint entry = indptr[row] + lane; entry < row_end; entry += width) {
        const float value = values[entry];
        device const float *dense_row = dense + (ulong)indices[entry] * k;
        for (uint column = 0; column < k; ++column) {
            mine[column] = fma(value, dense_row[column], mine[column]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            threadgroup const float *other = scratch + (lane + stride) * k;
            for (uint column = 0; column < k; ++column) {
                mine[column] += other[column];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint column = lane; column < k; column += width) {
        out[(ulong)row * k + column] = scratch[column];
    }
}
