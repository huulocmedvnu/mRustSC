
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float F32_EPSILON = 1.1920928955078125e-07f;
#define NDP __NDP__
#define KK __K__
#define QT 64u
#define CT __CT__

inline bool closer(float lhs_distance, uint lhs_cell, float rhs_distance, uint rhs_cell) {
    return lhs_distance < rhs_distance
        || (lhs_distance == rhs_distance && lhs_cell < rhs_cell);
}

kernel void knn_simd(device const float* embedding [[buffer(0)]],
                     device uint* out_cells [[buffer(1)]],
                     device float* out_distances [[buffer(2)]],
                     constant uint& n_cells [[buffer(3)]],
                     constant uint& n_pad [[buffer(4)]],
                     constant uint& n_dims [[buffer(5)]],
                     device const float* norm_sq [[buffer(6)]],
                     threadgroup float* cand_t [[threadgroup(0)]],
                     threadgroup float* dots [[threadgroup(1)]],
                     threadgroup float* qtile [[threadgroup(2)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint lane [[thread_position_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    uint qbase = group * QT;
    uint query = qbase + lane;
    bool active = query < n_cells;
    // The query block, read from device memory once and kept for every candidate step.
    for (uint i = lane; i < QT * NDP; i += QT) {
        qtile[i] = embedding[(ulong)qbase * NDP + i];
    }

    float best_d[KK];
    uint best_c[KK];
    for (uint s = 0; s < KK; s++) { best_d[s] = INFINITY; best_c[s] = 0xFFFFFFFFu; }
    float worst_d = INFINITY;
    uint worst_c = 0xFFFFFFFFu;
    float own_norm = norm_sq[query];
    const float scale = (float(n_dims) + 2.0f) * F32_EPSILON;
    threadgroup const float* qblock = qtile + sg * 32u * NDP;

    for (uint base = 0; base < n_cells; base += CT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Candidates land dimension-major, cand_t[d * CT + c], so the B tiles below are
        // plain row-major 8 x 8 loads with no transpose.
        for (uint i = lane; i < CT * NDP; i += QT) {
            uint c = i / NDP;
            uint d = i % NDP;
            uint row = base + c;
            cand_t[d * CT + c] = row < n_pad ? embedding[(ulong)row * NDP + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // This SIMD group's 32 queries x CT candidates, in blocks of 8 x 8, CT/8 column
        // blocks at a time so the accumulators stay in registers.
        for (uint cbase = 0; cbase < CT; cbase += 32) {
            simdgroup_float8x8 acc[4][4];
            for (uint qb = 0; qb < 4; qb++) {
                for (uint cb = 0; cb < 4; cb++) {
                    acc[qb][cb] = simdgroup_float8x8(0.0f);
                }
            }
            for (uint kk = 0; kk < NDP; kk += 8) {
                simdgroup_float8x8 b[4];
                for (uint cb = 0; cb < 4; cb++) {
                    simdgroup_load(b[cb], cand_t + kk * CT + cbase + cb * 8, CT);
                }
                for (uint qb = 0; qb < 4; qb++) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, qblock + (ulong)(qb * 8) * NDP + kk, NDP);
                    for (uint cb = 0; cb < 4; cb++) {
                        simdgroup_multiply_accumulate(acc[qb][cb], a, b[cb], acc[qb][cb]);
                    }
                }
            }
            // Stored transposed, candidate-major: the selection below has thread `lane`
            // read `dots[c * QT + lane]`, so the 32 lanes of a SIMD group touch 32
            // consecutive words and no two share a bank.
            for (uint qb = 0; qb < 4; qb++) {
                for (uint cb = 0; cb < 4; cb++) {
                    simdgroup_store(acc[qb][cb], dots + (cbase + cb * 8) * QT + sg * 32u + qb * 8, QT, ulong2(0, 0), true);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!active) { continue; }
        uint limit = min(CT, n_cells - base);
        for (uint c = 0; c < limit; c++) {
            uint cell = base + c;
            float norm_sum = own_norm + norm_sq[cell];
            float squared = fma(-2.0f, dots[c * QT + lane], norm_sum);
            if (squared < scale * norm_sum) { squared = 0.0f; }
            squared = max(squared, 0.0f);
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
