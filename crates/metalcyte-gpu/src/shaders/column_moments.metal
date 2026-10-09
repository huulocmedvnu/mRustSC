
#include <metal_stdlib>
using namespace metal;

// Sum and sum of squares of one column, given the entries already grouped by
// column. One threadgroup per column, lanes striding its entries, one tree
// reducing both moments at once so the values are read exactly once. A column
// with no stored entries has both moments zero.
kernel void csr_column_moments(device const uint *indptr [[buffer(0)]],
                               device const float *values [[buffer(1)]],
                               device float *sums [[buffer(2)]],
                               device float *squares [[buffer(3)]],
                               threadgroup float *scratch [[threadgroup(0)]],
                               uint column [[threadgroup_position_in_grid]],
                               uint lane [[thread_position_in_threadgroup]],
                               uint width [[threads_per_threadgroup]]) {
    float sum = 0.0f;
    float square_sum = 0.0f;
    const uint column_end = indptr[column + 1];
    for (uint entry = indptr[column] + lane; entry < column_end; entry += width) {
        const float value = values[entry];
        sum += value;
        square_sum = fma(value, value, square_sum);
    }
    scratch[lane] = sum;
    scratch[width + lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
            scratch[width + lane] += scratch[width + lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        sums[column] = scratch[0];
        squares[column] = scratch[width];
    }
}
