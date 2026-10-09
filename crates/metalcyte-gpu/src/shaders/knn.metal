
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
