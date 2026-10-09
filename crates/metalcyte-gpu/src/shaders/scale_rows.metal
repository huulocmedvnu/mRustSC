
#include <metal_stdlib>
using namespace metal;

// values[entry] *= factors[row]. One threadgroup per row so the row a stored
// entry belongs to is known without searching indptr for it.
kernel void csr_scale_rows(device const uint *indptr [[buffer(0)]],
                           device float *values [[buffer(1)]],
                           device const float *factors [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lane [[thread_position_in_threadgroup]],
                           uint width [[threads_per_threadgroup]]) {
    const float factor = factors[row];
    const uint row_end = indptr[row + 1];
    for (uint entry = indptr[row] + lane; entry < row_end; entry += width) {
        values[entry] *= factor;
    }
}
