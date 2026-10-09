
#include <metal_stdlib>
using namespace metal;

constexpr constant uint NODES = 3;
constexpr constant uint ROWS_PER_GROUP = 8;

// sum_j p w (y_i - y_j): one SIMD group per cell, lanes striding its affinities.
kernel void tsne_fft_attraction(device const uint *indptr [[buffer(0)]],
                                device const uint *indices [[buffer(1)]],
                                device const float *values [[buffer(2)]],
                                device const float *layout [[buffer(3)]],
                                device float *out [[buffer(4)]],
                                constant float &exaggeration [[buffer(5)]],
                                constant uint &n_cells [[buffer(6)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group * ROWS_PER_GROUP + simd_id;
    if (i >= n_cells) {
        return;
    }
    const float yi0 = layout[2 * i];
    const float yi1 = layout[2 * i + 1];
    float a0 = 0.0f;
    float a1 = 0.0f;
    const uint end = indptr[i + 1];
    for (uint at = indptr[i] + lane; at < end; at += 32) {
        const uint j = indices[at];
        const float p = values[at] * exaggeration;
        const float d0 = yi0 - layout[2 * j];
        const float d1 = yi1 - layout[2 * j + 1];
        const float w = 1.0f / (1.0f + d0 * d0 + d1 * d1);
        a0 = fma(p * w, d0, a0);
        a1 = fma(p * w, d1, a1);
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    if (lane == 0) {
        out[2 * i] = a0;
        out[2 * i + 1] = a1;
    }
}

// Each cell's share of sum p log(p / q), q = w / Z, for the convergence checks.
kernel void tsne_fft_kl(device const uint *indptr [[buffer(0)]],
                        device const uint *indices [[buffer(1)]],
                        device const float *values [[buffer(2)]],
                        device const float *layout [[buffer(3)]],
                        device float *kl_out [[buffer(4)]],
                        constant float &exaggeration [[buffer(5)]],
                        constant float &normaliser [[buffer(6)]],
                        constant uint &n_cells [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint simd_id [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group * ROWS_PER_GROUP + simd_id;
    if (i >= n_cells) {
        return;
    }
    const float yi0 = layout[2 * i];
    const float yi1 = layout[2 * i + 1];
    float kl = 0.0f;
    const uint end = indptr[i + 1];
    for (uint at = indptr[i] + lane; at < end; at += 32) {
        const uint j = indices[at];
        const float p = values[at] * exaggeration;
        if (p > 0.0f) {
            const float d0 = yi0 - layout[2 * j];
            const float d1 = yi1 - layout[2 * j + 1];
            const float w = 1.0f / (1.0f + d0 * d0 + d1 * d1);
            const float q = max(w / normaliser, 2.220446049250313e-16f);
            kl += p * log(max(p, 2.220446049250313e-16f) / q);
        }
    }
    kl = simd_sum(kl);
    if (lane == 0) {
        kl_out[i] = kl;
    }
}

// Box index and Lagrange weights of each cell on both axes.
kernel void tsne_fft_placement(device const float *layout [[buffer(0)]],
                               device uint *boxes [[buffer(1)]],
                               device float *weights [[buffer(2)]],
                               constant float &lo [[buffer(3)]],
                               constant float &box_width [[buffer(4)]],
                               constant uint &n_boxes [[buffer(5)]],
                               constant float *nodes [[buffer(6)]],
                               constant float *denominators [[buffer(7)]],
                               constant uint &n_cells [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    for (uint axis = 0; axis < 2; ++axis) {
        const float y = layout[2 * i + axis];
        const uint b = min(uint(max((y - lo) / box_width, 0.0f)), n_boxes - 1);
        const float fraction = (y - lo - float(b) * box_width) / box_width;
        boxes[2 * i + axis] = b;
        for (uint k = 0; k < NODES; ++k) {
            float w = 1.0f;
            for (uint m = 0; m < NODES; ++m) {
                if (m != k) {
                    w *= fraction - nodes[m];
                }
            }
            weights[(2 * i + axis) * NODES + k] = w / denominators[k];
        }
    }
}

// Charges of the cells of one box onto its nine nodes: one threadgroup per box, its
// 256 lanes striding the box's cells in their sorted order, each SIMD group folded by a
// fixed reduction and the eight partials summed in order by one thread. No atomics, so
// a box with thousands of cells is as reproducible as one with twenty.
kernel void tsne_fft_spread(device const float *layout [[buffer(0)]],
                            device const float *weights [[buffer(1)]],
                            device const uint *order [[buffer(2)]],
                            device const uint *offsets [[buffer(3)]],
                            device float *charges [[buffer(4)]],
                            constant uint &n_boxes [[buffer(5)]],
                            threadgroup float *partials [[threadgroup(0)]],
                            uint box [[threadgroup_position_in_grid]],
                            uint simd_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint t [[thread_position_in_threadgroup]],
                            uint width [[threads_per_threadgroup]]) {
    const uint n_nodes = n_boxes * NODES;
    const uint bx = box / n_boxes;
    const uint by = box % n_boxes;
    float acc[NODES * NODES * 4];
    for (uint s = 0; s < NODES * NODES * 4; ++s) {
        acc[s] = 0.0f;
    }
    const uint end = offsets[box + 1];
    for (uint at = offsets[box] + t; at < end; at += width) {
        const uint i = order[at];
        const float y0 = layout[2 * i];
        const float y1 = layout[2 * i + 1];
        const float q3 = y0 * y0 + y1 * y1;
        for (uint a = 0; a < NODES; ++a) {
            const float wa = weights[(2 * i) * NODES + a];
            for (uint b = 0; b < NODES; ++b) {
                const float w = wa * weights[(2 * i + 1) * NODES + b];
                const uint s = (a * NODES + b) * 4;
                acc[s] += w;
                acc[s + 1] = fma(w, y0, acc[s + 1]);
                acc[s + 2] = fma(w, y1, acc[s + 2]);
                acc[s + 3] = fma(w, q3, acc[s + 3]);
            }
        }
    }
    for (uint s = 0; s < NODES * NODES * 4; ++s) {
        const float total = simd_sum(acc[s]);
        if (lane == 0) {
            partials[simd_id * (NODES * NODES * 4) + s] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        const uint n_simd = width / 32;
        for (uint a = 0; a < NODES; ++a) {
            const uint row = bx * NODES + a;
            for (uint b = 0; b < NODES; ++b) {
                const uint col = by * NODES + b;
                const uint node = (row * n_nodes + col) * 4;
                const uint s = (a * NODES + b) * 4;
                for (uint c = 0; c < 4; ++c) {
                    float sum = 0.0f;
                    for (uint g = 0; g < n_simd; ++g) {
                        sum += partials[g * (NODES * NODES * 4) + s + c];
                    }
                    charges[node + c] = sum;
                }
            }
        }
    }
}

// Potentials at each cell from its nine nodes, and the cell's term of Z.
kernel void tsne_fft_gather(device const float *layout [[buffer(0)]],
                            device const uint *boxes [[buffer(1)]],
                            device const float *weights [[buffer(2)]],
                            device const float *potentials [[buffer(3)]],
                            device float *phi_out [[buffer(4)]],
                            device float *zterm [[buffer(5)]],
                            constant uint &n_nodes [[buffer(6)]],
                            constant uint &n_cells [[buffer(7)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    float phi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint bx = boxes[2 * i];
    const uint by = boxes[2 * i + 1];
    for (uint a = 0; a < NODES; ++a) {
        const uint row = bx * NODES + a;
        const float wa = weights[(2 * i) * NODES + a];
        for (uint b = 0; b < NODES; ++b) {
            const uint col = by * NODES + b;
            const float w = wa * weights[(2 * i + 1) * NODES + b];
            const uint node = (row * n_nodes + col) * 4;
            phi[0] = fma(w, potentials[node], phi[0]);
            phi[1] = fma(w, potentials[node + 1], phi[1]);
            phi[2] = fma(w, potentials[node + 2], phi[2]);
            phi[3] = fma(w, potentials[node + 3], phi[3]);
        }
    }
    const float y0 = layout[2 * i];
    const float y1 = layout[2 * i + 1];
    phi_out[4 * i] = phi[0];
    phi_out[4 * i + 1] = phi[1];
    phi_out[4 * i + 2] = phi[2];
    phi_out[4 * i + 3] = phi[3];
    zterm[i] = (1.0f + y0 * y0 + y1 * y1) * phi[0] - 2.0f * (y0 * phi[1] + y1 * phi[2]) + phi[3];
}

// ---- the convolution: a radix-2 FFT of every row in threadgroup memory, a tiled
// transpose, and the packing of two charge grids into one complex field.

// One threadgroup per row: bit-reversed load into threadgroup memory, log2n butterfly
// stages with the twiddles w_k = exp(-2 pi i k / n) (conjugated for the inverse), store.
kernel void fft_rows(device float *re [[buffer(0)]],
                     device float *im [[buffer(1)]],
                     device const float *twiddle_re [[buffer(2)]],
                     device const float *twiddle_im [[buffer(3)]],
                     constant uint &n [[buffer(4)]],
                     constant uint &log2n [[buffer(5)]],
                     constant uint &inverse [[buffer(6)]],
                     threadgroup float *shared [[threadgroup(0)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint width [[threads_per_threadgroup]]) {
    threadgroup float *sre = shared;
    threadgroup float *sim = shared + n;
    device float *r = re + (ulong)row * n;
    device float *i = im + (ulong)row * n;
    for (uint k = t; k < n; k += width) {
        const uint j = reverse_bits(k) >> (32 - log2n);
        sre[j] = r[k];
        sim[j] = i[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sign = inverse != 0 ? -1.0f : 1.0f;
    for (uint len = 2; len <= n; len <<= 1) {
        const uint span = len >> 1;
        const uint stride = n / len;
        for (uint b = t; b < n / 2; b += width) {
            const uint group = b / span;
            const uint k = b - group * span;
            const uint idx = group * len + k;
            const float wr = twiddle_re[k * stride];
            const float wi = sign * twiddle_im[k * stride];
            const float ur = sre[idx];
            const float ui = sim[idx];
            const float xr = sre[idx + span];
            const float xi = sim[idx + span];
            const float vr = xr * wr - xi * wi;
            const float vi = xr * wi + xi * wr;
            sre[idx] = ur + vr;
            sim[idx] = ui + vi;
            sre[idx + span] = ur - vr;
            sim[idx + span] = ui - vi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = t; k < n; k += width) {
        r[k] = sre[k];
        i[k] = sim[k];
    }
}

// In-place transpose of two square arrays by 32 x 32 tiles: each threadgroup swaps a
// tile with its mirror (or transposes a diagonal tile), so no element is touched twice.
kernel void transpose_square(device float *re [[buffer(0)]],
                             device float *im [[buffer(1)]],
                             constant uint &n [[buffer(2)]],
                             threadgroup float *shared [[threadgroup(0)]],
                             uint2 tile [[threadgroup_position_in_grid]],
                             uint2 t [[thread_position_in_threadgroup]]) {
    if (tile.y < tile.x) {
        return;
    }
    threadgroup float *a_re = shared;
    threadgroup float *a_im = shared + 33 * 32;
    threadgroup float *b_re = shared + 2 * 33 * 32;
    threadgroup float *b_im = shared + 3 * 33 * 32;
    const uint x0 = tile.x * 32;
    const uint y0 = tile.y * 32;
    for (uint j = t.y; j < 32; j += 8) {
        const uint ra = y0 + j;
        const uint ca = x0 + t.x;
        a_re[j * 33 + t.x] = (ra < n && ca < n) ? re[(ulong)ra * n + ca] : 0.0f;
        a_im[j * 33 + t.x] = (ra < n && ca < n) ? im[(ulong)ra * n + ca] : 0.0f;
        const uint rb = x0 + j;
        const uint cb = y0 + t.x;
        b_re[j * 33 + t.x] = (rb < n && cb < n) ? re[(ulong)rb * n + cb] : 0.0f;
        b_im[j * 33 + t.x] = (rb < n && cb < n) ? im[(ulong)rb * n + cb] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t.y; j < 32; j += 8) {
        const uint rb = x0 + j;
        const uint cb = y0 + t.x;
        if (rb < n && cb < n) {
            re[(ulong)rb * n + cb] = a_re[t.x * 33 + j];
            im[(ulong)rb * n + cb] = a_im[t.x * 33 + j];
        }
        if (tile.y != tile.x) {
            const uint ra = y0 + j;
            const uint ca = x0 + t.x;
            if (ra < n && ca < n) {
                re[(ulong)ra * n + ca] = b_re[t.x * 33 + j];
                im[(ulong)ra * n + ca] = b_im[t.x * 33 + j];
            }
        }
    }
}

// Charges c = 2 * pair and 2 * pair + 1 into the real and imaginary parts of the
// zero-padded field.
kernel void fft_pack(device const float *charges [[buffer(0)]],
                     device float *re [[buffer(1)]],
                     device float *im [[buffer(2)]],
                     constant uint &n_nodes [[buffer(3)]],
                     constant uint &size [[buffer(4)]],
                     constant uint &pair [[buffer(5)]],
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x >= size || p.y >= size) {
        return;
    }
    const ulong at = (ulong)p.y * size + p.x;
    if (p.x < n_nodes && p.y < n_nodes) {
        const ulong node = ((ulong)p.y * n_nodes + p.x) * 4;
        re[at] = charges[node + 2 * pair];
        im[at] = charges[node + 2 * pair + 1];
    } else {
        re[at] = 0.0f;
        im[at] = 0.0f;
    }
}

kernel void fft_multiply(device float *re [[buffer(0)]],
                         device float *im [[buffer(1)]],
                         device const float *kernel_re [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const float k = kernel_re[i];
    re[i] *= k;
    im[i] *= k;
}

// The top-left n_nodes x n_nodes block of the field back into the potentials, scaled by
// 1 / size^2 to complete the unnormalised inverse.
kernel void fft_unpack(device const float *re [[buffer(0)]],
                       device const float *im [[buffer(1)]],
                       device float *potentials [[buffer(2)]],
                       constant uint &n_nodes [[buffer(3)]],
                       constant uint &size [[buffer(4)]],
                       constant uint &pair [[buffer(5)]],
                       constant float &scale [[buffer(6)]],
                       uint2 p [[thread_position_in_grid]]) {
    if (p.x >= n_nodes || p.y >= n_nodes) {
        return;
    }
    const ulong at = (ulong)p.y * size + p.x;
    const ulong node = ((ulong)p.y * n_nodes + p.x) * 4;
    potentials[node + 2 * pair] = re[at] * scale;
    potentials[node + 2 * pair + 1] = im[at] * scale;
}

// The gradient 4 (attraction - repulsion), scikit-learn's gains and momentum, the step,
// and the cell's share of the squared norm of the scaled gradient.
kernel void tsne_fft_update(device float *layout [[buffer(0)]],
                            device float *step [[buffer(1)]],
                            device float *gains [[buffer(2)]],
                            device const float *attraction [[buffer(3)]],
                            device const float *phi [[buffer(4)]],
                            device float *norm_out [[buffer(5)]],
                            constant float &normaliser [[buffer(6)]],
                            constant float &momentum [[buffer(7)]],
                            constant float &learning_rate [[buffer(8)]],
                            constant float &min_gain [[buffer(9)]],
                            constant uint &n_cells [[buffer(10)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n_cells) {
        return;
    }
    const float y0 = layout[2 * i];
    const float y1 = layout[2 * i + 1];
    const float p0 = phi[4 * i];
    const float p1 = phi[4 * i + 1];
    const float p2 = phi[4 * i + 2];
    const float rep0 = (y0 * p0 - p1) / normaliser;
    const float rep1 = (y1 * p0 - p2) / normaliser;
    float norm = 0.0f;
    const float grad[2] = {4.0f * (attraction[2 * i] - rep0), 4.0f * (attraction[2 * i + 1] - rep1)};
    for (uint axis = 0; axis < 2; ++axis) {
        const uint at = 2 * i + axis;
        const float g = grad[axis];
        const bool overshooting = step[at] * g < 0.0f;
        float gain = overshooting ? gains[at] + 0.2f : gains[at] * 0.8f;
        gain = max(gain, min_gain);
        gains[at] = gain;
        const float scaled = g * gain;
        const float u = momentum * step[at] - learning_rate * scaled;
        step[at] = u;
        layout[at] += u;
        norm = fma(scaled, scaled, norm);
    }
    norm_out[i] = norm;
}
