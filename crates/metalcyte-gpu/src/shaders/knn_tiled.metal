
#include <metal_stdlib>
using namespace metal;

constant float F32_EPSILON = 1.1920928955078125e-07f;
#define NDIMS __NDIMS__
#define KK __K__

inline bool closer(float lhs_distance, uint lhs_cell, float rhs_distance, uint rhs_cell) {
    return lhs_distance < rhs_distance
        || (lhs_distance == rhs_distance && lhs_cell < rhs_cell);
}

kernel void knn_tiled(device const float* embedding [[buffer(0)]],
                      device uint* out_cells [[buffer(1)]],
                      device float* out_distances [[buffer(2)]],
                      device const float* norm_sq [[buffer(6)]],
                      constant uint& n_cells [[buffer(3)]],
                      constant uint& n_dims_unused [[buffer(4)]],
                      constant uint& k_unused [[buffer(5)]],
                      threadgroup float* candidates [[threadgroup(0)]],
                      threadgroup float* cand_norms [[threadgroup(1)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint lane [[thread_position_in_threadgroup]],
                      uint tile [[threads_per_threadgroup]]) {
    uint query = group * tile + lane;
    bool active = query < n_cells;

    float qv[NDIMS];
    for (uint d = 0; d < NDIMS; d++) {
        qv[d] = active ? embedding[(ulong)query * NDIMS + d] : 0.0f;
    }
    float best_d[KK];
    uint best_c[KK];
    for (uint s = 0; s < KK; s++) { best_d[s] = INFINITY; best_c[s] = 0xFFFFFFFFu; }
    float worst_d = INFINITY;
    uint worst_c = 0xFFFFFFFFu;
    float own_norm = active ? norm_sq[query] : 0.0f;
    const float scale = (float(NDIMS) + 2.0f) * F32_EPSILON;

    for (uint base = 0; base < n_cells; base += tile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < tile * NDIMS; i += tile) {
            uint row = base + i / NDIMS;
            candidates[i] = row < n_cells ? embedding[(ulong)row * NDIMS + i % NDIMS] : 0.0f;
        }
        {
            uint row = base + lane;
            cand_norms[lane] = row < n_cells ? norm_sq[row] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) { continue; }
        uint limit = min(tile, n_cells - base);
        for (uint c = 0; c < limit; c++) {
            uint cell = base + c;
            threadgroup const float* row = candidates + c * NDIMS;
            float squared = 0.0f;
            for (uint d = 0; d < NDIMS; d++) {
                float delta = qv[d] - row[d];
                squared = fma(delta, delta, squared);
            }
            if (squared < scale * (own_norm + cand_norms[c])) { squared = 0.0f; }
            if (cell == query || !closer(squared, cell, worst_d, worst_c)) { continue; }
            uint slot = KK - 1;
            while (slot > 0 && closer(squared, cell, best_d[slot - 1], best_c[slot - 1])) {
                best_d[slot] = best_d[slot - 1];
                best_c[slot] = best_c[slot - 1];
                slot--;
            }
            best_d[slot] = squared;
            best_c[slot] = cell;
            worst_d = best_d[KK - 1];
            worst_c = best_c[KK - 1];
        }
    }
    if (active) {
        for (uint s = 0; s < KK; s++) {
            out_cells[(ulong)query * KK + s] = best_c[s];
            out_distances[(ulong)query * KK + s] = sqrt(best_d[s]);
        }
    }
}
