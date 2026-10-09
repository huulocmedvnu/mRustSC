
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
    // reaches the epoch number: that is, on every epoch containing a multiple of
    // `schedule`, which is exactly where floor(epoch / schedule) increments. So
    // the schedule needs no state carried between epochs. In exact arithmetic
    // the two agree for every schedule; in f32 the multiples near an integer can
    // land on the other side of it, which shifts about one firing in 30000 by a
    // single epoch.
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
