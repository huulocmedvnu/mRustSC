//! t-SNE for large inputs: sparse affinities over nearest neighbours, and the repulsive
//! term by polynomial interpolation on a grid with an FFT convolution (FIt-SNE,
//! Linderman et al. 2019).
//!
//! The exact formulation in [`crate::tsne`] holds an `(n, n)` affinity matrix and stops
//! at 20 000 cells. Here the attractive term runs over the `3 * perplexity` nearest
//! neighbours of each cell, as in Barnes-Hut t-SNE, and the repulsive term, which
//! couples every pair, is evaluated at the nodes of a regular grid: each cell's
//! "charge" is spread onto the nodes of its box by Lagrange interpolation, the node
//! potentials are one convolution with the squared Cauchy kernel, done by FFT, and
//! the potential at each cell is read back by the same interpolation. The cost per
//! iteration is linear in the number of cells plus the FFT of a grid a few hundred
//! nodes wide.
//!
//! The optimiser schedule (momentum, gains, early exaggeration, convergence checks)
//! is the one the exact path uses, so the two agree on everything but the
//! approximation of the gradient. Two-dimensional layouts only.

use ndarray::Array2;
use rayon::prelude::*;

use candle_core::Device;

use crate::error::{Error, Result};
use crate::neighbors;
use crate::nndescent::{knn_approximate, NnDescentParams};
use crate::tsne::{
    principal_component_initialisation, TsneParams, CONVERGENCE_CHECK_INTERVAL,
    EXPLORATION_ITERATIONS, EXPLORATION_MOMENTUM, FINAL_MOMENTUM, ITERATIONS_WITHOUT_PROGRESS,
    MACHINE_EPSILON, MIN_GAIN, MIN_GRADIENT_NORM, PERPLEXITY_SEARCH_STEPS, PERPLEXITY_TOLERANCE,
};

/// Cells above which the neighbour search behind the affinities is approximate.
const APPROXIMATE_NEIGHBOURS_FROM: usize = 200_000;
/// Interpolation nodes per box and side, as FIt-SNE's `n_interpolation_points`.
const NODES_PER_BOX: usize = 3;
/// Fewest boxes per side, as FIt-SNE's `min_num_intervals`.
const MIN_BOXES: usize = 50;
/// Boxes per unit of layout extent, as FIt-SNE's `intervals_per_integer`.
const BOXES_PER_UNIT: f32 = 1.0;

/// Symmetric affinities in compressed sparse row form, summing to one.
struct SparseAffinities {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

/// t-SNE by FFT-accelerated interpolation.
///
/// `embedding` is `(n_cells, n_features)`; the output is `(n_cells, 2)`. `device`
/// selects the neighbour search only; the layout itself runs on the cores.
pub fn tsne_fft(
    embedding: &Array2<f32>,
    params: &TsneParams,
    device: &Device,
) -> Result<Array2<f32>> {
    let (n_cells, _) = embedding.dim();
    if params.n_components != 2 {
        return Err(Error::parameter(
            "n_components",
            "2 for the FFT-accelerated method",
            params.n_components,
        ));
    }
    let k = ((3.0 * params.perplexity).floor() as usize).clamp(1, n_cells - 1);
    let graph = if n_cells > APPROXIMATE_NEIGHBOURS_FROM {
        let nn = NnDescentParams {
            seed: params.seed,
            ..NnDescentParams::default()
        };
        knn_approximate(embedding, k, &nn)?
    } else {
        neighbors::knn(embedding, k, device)?
    };
    let affinities = symmetric_affinities(&graph, params.perplexity);
    drop(graph);

    let initial = principal_component_initialisation(embedding, params, &Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let layout = optimise(initial, n_cells, &affinities, params);
    Array2::from_shape_vec((n_cells, 2), layout)
        .map_err(|_| Error::shape(format!("{n_cells} by 2"), "a mismatch"))
}

/// Conditional affinities over each cell's neighbours at the requested perplexity,
/// symmetrised and normalised to sum to one, as scikit-learn's `_joint_probabilities_nn`.
fn symmetric_affinities(graph: &neighbors::KnnGraph, perplexity: f32) -> SparseAffinities {
    let (n_cells, k) = graph.indices.dim();
    let target_entropy = (perplexity as f64).ln();
    // Row-wise bandwidth search, in parallel: the binary search of
    // `sklearn.manifold._utils._binary_search_perplexity`.
    let conditional: Vec<f32> = (0..n_cells)
        .into_par_iter()
        .flat_map_iter(|i| {
            let squares: Vec<f64> = graph
                .distances
                .row(i)
                .iter()
                .map(|&d| (d as f64) * (d as f64))
                .collect();
            let mut beta = 1.0f64;
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;
            let mut probabilities = vec![0.0f64; k];
            for _ in 0..PERPLEXITY_SEARCH_STEPS {
                let mut sum = 0.0f64;
                for (p, &d) in probabilities.iter_mut().zip(&squares) {
                    *p = (-d * beta).exp();
                    sum += *p;
                }
                let sum = sum.max(f64::EPSILON);
                let mut entropy = 0.0f64;
                for (p, &d) in probabilities.iter_mut().zip(&squares) {
                    *p /= sum;
                    entropy += d * *p;
                }
                entropy = sum.ln() + beta * entropy;
                let difference = entropy - target_entropy;
                if difference.abs() <= PERPLEXITY_TOLERANCE as f64 {
                    break;
                }
                if difference > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) / 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + beta_min) / 2.0
                    };
                }
            }
            probabilities
                .into_iter()
                .map(|p| p as f32)
                .collect::<Vec<f32>>()
        })
        .collect();

    // Symmetrise: every directed entry (i, j, p) contributes to row i and to row j.
    let mut counts = vec![0usize; n_cells + 1];
    for i in 0..n_cells {
        counts[i + 1] += k;
        for &j in graph.indices.row(i) {
            counts[j as usize + 1] += 1;
        }
    }
    for i in 0..n_cells {
        counts[i + 1] += counts[i];
    }
    let mut cursor = counts.clone();
    let total = counts[n_cells];
    let mut indices = vec![0u32; total];
    let mut values = vec![0f32; total];
    for i in 0..n_cells {
        for (slot, &j) in graph.indices.row(i).iter().enumerate() {
            let p = conditional[i * k + slot];
            let at = cursor[i];
            indices[at] = j;
            values[at] = p;
            cursor[i] += 1;
            let at = cursor[j as usize];
            indices[at] = i as u32;
            values[at] = p;
            cursor[j as usize] += 1;
        }
    }
    // Merge duplicates within each row and scale by 1 / (2n).
    let scale = 0.5 / n_cells as f32;
    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let (start, end) = (counts[i], counts[i + 1]);
            let mut entries: Vec<(u32, f32)> = indices[start..end]
                .iter()
                .copied()
                .zip(values[start..end].iter().copied())
                .collect();
            entries.sort_unstable_by_key(|e| e.0);
            let mut cols = Vec::with_capacity(entries.len());
            let mut vals = Vec::with_capacity(entries.len());
            for (j, p) in entries {
                if cols.last() == Some(&j) {
                    *vals.last_mut().unwrap() += p * scale;
                } else {
                    cols.push(j);
                    vals.push(p * scale);
                }
            }
            (cols, vals)
        })
        .collect();
    let mut indptr = Vec::with_capacity(n_cells + 1);
    indptr.push(0);
    let mut indices = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    for (cols, vals) in rows {
        indices.extend_from_slice(&cols);
        values.extend_from_slice(&vals);
        indptr.push(indices.len());
    }
    SparseAffinities {
        indptr,
        indices,
        values,
    }
}

/// Gradient descent with the schedule of the exact path, on plain vectors.
fn optimise(
    initial: Vec<f32>,
    n_cells: usize,
    affinities: &SparseAffinities,
    params: &TsneParams,
) -> Vec<f32> {
    let mut layout = initial;
    let mut update = vec![0.0f32; 2 * n_cells];
    let mut gains = vec![1.0f32; 2 * n_cells];
    let mut best_error = f64::MAX;
    let mut best_iteration = 0usize;
    let mut exploration_end = EXPLORATION_ITERATIONS;
    let learning_rate = params.learning_rate;

    let mut iteration = 0usize;
    while iteration < params.n_iterations {
        let (momentum, exaggeration) = if iteration < exploration_end {
            (EXPLORATION_MOMENTUM as f32, params.early_exaggeration)
        } else {
            (FINAL_MOMENTUM as f32, 1.0)
        };
        if iteration == exploration_end {
            update.iter_mut().for_each(|u| *u = 0.0);
            gains.iter_mut().for_each(|g| *g = 1.0);
            best_error = f64::MAX;
            best_iteration = iteration;
        }
        let checking = (iteration + 1).is_multiple_of(CONVERGENCE_CHECK_INTERVAL);
        let (gradient, error) = gradient(&layout, n_cells, affinities, exaggeration, checking);

        let norm_square: f64 = layout
            .par_iter_mut()
            .zip(update.par_iter_mut())
            .zip(gains.par_iter_mut())
            .zip(gradient.par_iter())
            .map(|(((y, u), g), &grad)| {
                let overshooting = *u * grad < 0.0;
                *g = if overshooting { *g + 0.2 } else { *g * 0.8 }.max(MIN_GAIN);
                let scaled = grad * *g;
                *u = momentum * *u - learning_rate * scaled;
                *y += *u;
                (scaled as f64) * (scaled as f64)
            })
            .sum();

        let exploring = iteration < exploration_end;
        iteration += 1;
        if !checking {
            continue;
        }
        let patience = if exploring {
            EXPLORATION_ITERATIONS
        } else {
            ITERATIONS_WITHOUT_PROGRESS
        };
        let error = error.unwrap_or(f64::MAX);
        let mut stop = false;
        if error < best_error {
            best_error = error;
            best_iteration = iteration - 1;
        } else if iteration - 1 - best_iteration > patience {
            stop = true;
        }
        if !stop {
            stop = norm_square.sqrt() <= MIN_GRADIENT_NORM as f64;
        }
        if stop {
            if !exploring {
                break;
            }
            exploration_end = iteration;
        }
    }
    layout
}

/// The gradient of the (exaggerated) objective, `4 (F_attr - F_rep)`, and the
/// objective itself when asked for.
fn gradient(
    layout: &[f32],
    n_cells: usize,
    affinities: &SparseAffinities,
    exaggeration: f32,
    compute_error: bool,
) -> (Vec<f32>, Option<f64>) {
    let (repulsion, normaliser) = repulsive_forces(layout, n_cells);
    let mut gradient = vec![0.0f32; 2 * n_cells];
    let error: f64 = gradient
        .par_chunks_mut(2)
        .enumerate()
        .map(|(i, out)| {
            let (yi0, yi1) = (layout[2 * i], layout[2 * i + 1]);
            let (mut a0, mut a1) = (0.0f32, 0.0f32);
            let mut kl = 0.0f64;
            for at in affinities.indptr[i]..affinities.indptr[i + 1] {
                let j = affinities.indices[at] as usize;
                let p = affinities.values[at] * exaggeration;
                let (d0, d1) = (yi0 - layout[2 * j], yi1 - layout[2 * j + 1]);
                let w = 1.0 / (1.0 + d0 * d0 + d1 * d1);
                a0 += p * w * d0;
                a1 += p * w * d1;
                if compute_error && p > 0.0 {
                    let q = ((w as f64) / normaliser).max(MACHINE_EPSILON);
                    kl += (p as f64) * ((p as f64).max(MACHINE_EPSILON) / q).ln();
                }
            }
            out[0] = 4.0 * (a0 - repulsion[2 * i] as f32);
            out[1] = 4.0 * (a1 - repulsion[2 * i + 1] as f32);
            kl
        })
        .sum();
    (gradient, compute_error.then_some(error))
}

/// `sum_j W_ij^2 (y_i - y_j) / Z` for every cell, and `Z = sum_{i != j} W_ij`, with
/// `W_ij = 1 / (1 + |y_i - y_j|^2)`, by interpolation onto a grid and one FFT
/// convolution with the squared Cauchy kernel.
///
/// Four charges per cell, `[1, y_0, y_1, |y|^2]`, give the four node potentials from
/// which both quantities follow, because `(1 + |y_i - y_j|^2) W_ij^2 = W_ij`.
fn repulsive_forces(layout: &[f32], n_cells: usize) -> (Vec<f64>, f64) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in layout {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let extent = (hi - lo).max(1e-6);
    let n_boxes = ((extent * BOXES_PER_UNIT).ceil() as usize).max(MIN_BOXES);
    let box_width = extent / n_boxes as f32;
    let n_nodes = n_boxes * NODES_PER_BOX;
    let node_spacing = box_width / NODES_PER_BOX as f32;
    // Node m sits at lo + (m + 0.5) * node_spacing.
    let nodes_within_box: Vec<f32> = (0..NODES_PER_BOX)
        .map(|k| (k as f32 + 0.5) / NODES_PER_BOX as f32)
        .collect();
    let denominators: Vec<f32> = nodes_within_box
        .iter()
        .enumerate()
        .map(|(k, &node)| {
            nodes_within_box
                .iter()
                .enumerate()
                .filter(|&(m, _)| m != k)
                .map(|(_, &other)| node - other)
                .product()
        })
        .collect();

    // Each cell's box and Lagrange weights along both axes.
    let placement: Vec<([usize; 2], [[f32; NODES_PER_BOX]; 2])> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let mut boxes = [0usize; 2];
            let mut weights = [[0.0f32; NODES_PER_BOX]; 2];
            for axis in 0..2 {
                let y = layout[2 * i + axis];
                let b = (((y - lo) / box_width) as usize).min(n_boxes - 1);
                let fraction = (y - lo - b as f32 * box_width) / box_width;
                boxes[axis] = b;
                for (k, denominator) in denominators.iter().enumerate() {
                    let mut w = 1.0f32;
                    for (m, &node) in nodes_within_box.iter().enumerate() {
                        if m != k {
                            w *= fraction - node;
                        }
                    }
                    weights[axis][k] = w / denominator;
                }
            }
            (boxes, weights)
        })
        .collect();

    // Spread the four charges onto the nodes: per-thread grids, then a sum.
    let grid_len = n_nodes * n_nodes;
    let chunk = (n_cells / (rayon::current_num_threads() * 4)).max(4096);
    let charges: Vec<f64> = placement
        .par_chunks(chunk)
        .enumerate()
        .map(|(c, cells)| {
            let mut grid = vec![0.0f64; 4 * grid_len];
            for (offset, (boxes, weights)) in cells.iter().enumerate() {
                let i = c * chunk + offset;
                let (y0, y1) = (layout[2 * i] as f64, layout[2 * i + 1] as f64);
                let q = [1.0, y0, y1, y0 * y0 + y1 * y1];
                for a in 0..NODES_PER_BOX {
                    let row = boxes[0] * NODES_PER_BOX + a;
                    for b in 0..NODES_PER_BOX {
                        let col = boxes[1] * NODES_PER_BOX + b;
                        let w = (weights[0][a] * weights[1][b]) as f64;
                        let at = row * n_nodes + col;
                        for (charge, &value) in q.iter().enumerate() {
                            grid[charge * grid_len + at] += w * value;
                        }
                    }
                }
            }
            grid
        })
        .reduce(
            || vec![0.0f64; 4 * grid_len],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );

    // Convolve with the squared Cauchy kernel on the embedded circulant.
    let size = (2 * n_nodes).next_power_of_two();
    let mut kernel = vec![Complex::ZERO; size * size];
    for di in 0..n_nodes {
        for dj in 0..n_nodes {
            let r2 = {
                let x = di as f64 * node_spacing as f64;
                let y = dj as f64 * node_spacing as f64;
                x * x + y * y
            };
            let value = 1.0 / ((1.0 + r2) * (1.0 + r2));
            for ii in [di, (size - di) % size] {
                for jj in [dj, (size - dj) % size] {
                    kernel[ii * size + jj] = Complex::real(value);
                }
            }
        }
    }
    fft_2d(&mut kernel, size, false);
    // The kernel is even in both axes, so its transform is real, and two real charge
    // grids packed as the real and imaginary parts of one field convolve independently.
    let potentials: Vec<Vec<f64>> = (0..2)
        .into_par_iter()
        .flat_map_iter(|pair| {
            let (c_re, c_im) = (2 * pair, 2 * pair + 1);
            let mut field = vec![Complex::ZERO; size * size];
            for r in 0..n_nodes {
                for col in 0..n_nodes {
                    let at = r * n_nodes + col;
                    field[r * size + col] = Complex {
                        re: charges[c_re * grid_len + at],
                        im: charges[c_im * grid_len + at],
                    };
                }
            }
            fft_2d(&mut field, size, false);
            for (f, k) in field.iter_mut().zip(&kernel) {
                f.re *= k.re;
                f.im *= k.re;
            }
            fft_2d(&mut field, size, true);
            let mut out_re = vec![0.0f64; grid_len];
            let mut out_im = vec![0.0f64; grid_len];
            for r in 0..n_nodes {
                for col in 0..n_nodes {
                    out_re[r * n_nodes + col] = field[r * size + col].re;
                    out_im[r * n_nodes + col] = field[r * size + col].im;
                }
            }
            [out_re, out_im]
        })
        .collect();

    // Read the potentials back at the cells.
    let per_cell: Vec<[f64; 4]> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let (boxes, weights) = &placement[i];
            let mut phi = [0.0f64; 4];
            for a in 0..NODES_PER_BOX {
                let row = boxes[0] * NODES_PER_BOX + a;
                for b in 0..NODES_PER_BOX {
                    let col = boxes[1] * NODES_PER_BOX + b;
                    let w = (weights[0][a] * weights[1][b]) as f64;
                    let at = row * n_nodes + col;
                    for c in 0..4 {
                        phi[c] += w * potentials[c][at];
                    }
                }
            }
            phi
        })
        .collect();

    // Z = sum_i [(1 + |y_i|^2) phi_1 - 2 y_i . phi_23 + phi_4], minus the n self terms.
    let normaliser: f64 = per_cell
        .par_iter()
        .enumerate()
        .map(|(i, phi)| {
            let (y0, y1) = (layout[2 * i] as f64, layout[2 * i + 1] as f64);
            (1.0 + y0 * y0 + y1 * y1) * phi[0] - 2.0 * (y0 * phi[1] + y1 * phi[2]) + phi[3]
        })
        .sum::<f64>()
        - n_cells as f64;
    let normaliser = normaliser.max(MACHINE_EPSILON);
    let mut forces = vec![0.0f64; 2 * n_cells];
    forces.par_chunks_mut(2).enumerate().for_each(|(i, out)| {
        let (y0, y1) = (layout[2 * i] as f64, layout[2 * i + 1] as f64);
        let phi = &per_cell[i];
        out[0] = (y0 * phi[0] - phi[1]) / normaliser;
        out[1] = (y1 * phi[0] - phi[2]) / normaliser;
    });
    (forces, normaliser)
}

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn mul(self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// In-place radix-2 FFT of one row of `size` complex values.
fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let step = Complex {
            re: angle.cos(),
            im: angle.sin(),
        };
        for start in (0..n).step_by(len) {
            let mut w = Complex::real(1.0);
            for k in 0..len / 2 {
                let u = data[start + k];
                let v = data[start + k + len / 2].mul(&w);
                data[start + k] = Complex {
                    re: u.re + v.re,
                    im: u.im + v.im,
                };
                data[start + k + len / 2] = Complex {
                    re: u.re - v.re,
                    im: u.im - v.im,
                };
                w = w.mul(&step);
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for v in data.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

/// In-place 2-D FFT of a `size x size` row-major array: rows, transpose, rows, transpose.
fn fft_2d(data: &mut [Complex], size: usize, inverse: bool) {
    data.par_chunks_mut(size)
        .for_each(|row| fft_1d(row, inverse));
    transpose(data, size);
    data.par_chunks_mut(size)
        .for_each(|row| fft_1d(row, inverse));
    transpose(data, size);
}

/// Square transpose in place, by 32 x 32 tiles so each tile stays in cache.
fn transpose(data: &mut [Complex], size: usize) {
    const TILE: usize = 32;
    let ptr = data.as_mut_ptr() as usize;
    let tiles = size.div_ceil(TILE);
    (0..tiles).into_par_iter().for_each(|ti| {
        for tj in ti..tiles {
            for i in ti * TILE..((ti + 1) * TILE).min(size) {
                let j0 = if tj == ti { i + 1 } else { tj * TILE };
                for j in j0..((tj + 1) * TILE).min(size) {
                    // Safety: tile (ti, tj) and its mirror are touched by this task only,
                    // and tasks cover disjoint tile pairs.
                    unsafe {
                        let p = ptr as *mut Complex;
                        std::ptr::swap(p.add(i * size + j), p.add(j * size + i));
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..2 * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 40.0 - 20.0) as f32
            })
            .collect()
    }

    #[test]
    fn interpolated_repulsion_matches_brute_force() {
        let n = 1500;
        let layout = points(n, 5);
        let (forces, z) = repulsive_forces(&layout, n);
        let mut z_exact = 0.0f64;
        let mut error_square = 0.0f64;
        let mut force_square = 0.0f64;
        let mut worst = 0.0f64;
        for i in 0..n {
            let (yi0, yi1) = (layout[2 * i] as f64, layout[2 * i + 1] as f64);
            let (mut f0, mut f1) = (0.0f64, 0.0f64);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let (d0, d1) = (yi0 - layout[2 * j] as f64, yi1 - layout[2 * j + 1] as f64);
                let w = 1.0 / (1.0 + d0 * d0 + d1 * d1);
                z_exact += w;
                f0 += w * w * d0;
                f1 += w * w * d1;
            }
            let e0 = forces[2 * i] * z - f0;
            let e1 = forces[2 * i + 1] * z - f1;
            error_square += e0 * e0 + e1 * e1;
            force_square += f0 * f0 + f1 * f1;
            worst = worst.max(e0 * e0 + e1 * e1);
        }
        let z_error = (z - z_exact).abs() / z_exact;
        let rms_force = (force_square / n as f64).sqrt();
        let rms_error = (error_square / n as f64).sqrt() / rms_force;
        let worst = worst.sqrt() / rms_force;
        println!("Z error {z_error:.2e}, rms force error {rms_error:.2e}, worst {worst:.2e}");
        assert!(z_error < 5e-3, "Z relative error {z_error}");
        assert!(
            rms_error < 3e-2,
            "rms force error {rms_error} of the rms force"
        );
        assert!(worst < 2e-1, "worst force error {worst} of the rms force");
    }

    /// Ten separated Gaussian blobs in 10 dimensions.
    fn blobs(n: usize) -> Array2<f32> {
        let mut state = 11u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let centres: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..10).map(|_| (next() * 60.0) as f32).collect())
            .collect();
        let mut data = Array2::<f32>::zeros((n, 10));
        for i in 0..n {
            for d in 0..10 {
                data[[i, d]] = centres[i % 10][d] + ((next() - 0.5) * 2.0) as f32;
            }
        }
        data
    }

    #[test]
    fn blobs_stay_separated_and_the_objective_matches_the_exact_path() {
        let input = blobs(3000);
        let fft = TsneParams {
            method: crate::tsne::TsneMethod::Fft,
            ..Default::default()
        };
        let exact = TsneParams {
            method: crate::tsne::TsneMethod::Exact,
            ..Default::default()
        };
        let ours = crate::tsne::tsne(&input, &fft, &Device::Cpu).unwrap();
        let reference = crate::tsne::tsne(&input, &exact, &Device::Cpu).unwrap();
        assert_eq!(ours.dim(), (3000, 2));
        // Every blob's cells sit closer to their own centroid than to any other.
        let mut centroids = [[0.0f64; 2]; 10];
        for i in 0..3000 {
            centroids[i % 10][0] += ours[[i, 0]] as f64 / 300.0;
            centroids[i % 10][1] += ours[[i, 1]] as f64 / 300.0;
        }
        let mut misplaced = 0usize;
        for i in 0..3000 {
            let own = i % 10;
            let d = |c: &[f64; 2]| {
                (ours[[i, 0]] as f64 - c[0]).powi(2) + (ours[[i, 1]] as f64 - c[1]).powi(2)
            };
            let mine = d(&centroids[own]);
            if (0..10).any(|c| c != own && d(&centroids[c]) < mine) {
                misplaced += 1;
            }
        }
        assert!(
            misplaced < 30,
            "{misplaced} of 3000 cells nearer another blob"
        );
        let kl_ours = kl_divergence(&input, &ours, 30.0);
        let kl_exact = kl_divergence(&input, &reference, 30.0);
        println!("KL fft {kl_ours:.4}, exact {kl_exact:.4}");
        assert!(
            kl_ours <= kl_exact * 1.15,
            "KL {kl_ours} against exact {kl_exact}"
        );
    }

    /// The exact objective of a layout, from the exact path's affinities.
    fn kl_divergence(input: &Array2<f32>, layout: &Array2<f32>, perplexity: f32) -> f64 {
        use candle_core::Tensor;
        let n = input.nrows();
        let points = Tensor::from_vec(
            input.iter().copied().collect::<Vec<f32>>(),
            input.dim(),
            &Device::Cpu,
        )
        .unwrap();
        let distances = crate::tsne::squared_distances(&points)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let joint = crate::tsne::joint_probabilities(
            &crate::tsne::conditional_affinities(&distances, n, perplexity),
            n,
        );
        let mut weights = vec![0.0f64; n * n];
        let mut total = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d0 = (layout[[i, 0]] - layout[[j, 0]]) as f64;
                    let d1 = (layout[[i, 1]] - layout[[j, 1]]) as f64;
                    let w = 1.0 / (1.0 + d0 * d0 + d1 * d1);
                    weights[i * n + j] = w;
                    total += w;
                }
            }
        }
        joint
            .iter()
            .zip(&weights)
            .filter(|(&p, _)| p > 0.0)
            .map(|(&p, &w)| p as f64 * ((p as f64) / (w / total).max(1e-12)).ln())
            .sum()
    }

    #[test]
    fn fft_round_trip_is_identity() {
        let size = 16;
        let original: Vec<Complex> = (0..size * size)
            .map(|i| Complex::real((i % 7) as f64 - 3.0))
            .collect();
        let mut data = original.clone();
        fft_2d(&mut data, size, false);
        fft_2d(&mut data, size, true);
        for (a, b) in data.iter().zip(&original) {
            assert!((a.re - b.re).abs() < 1e-9 && a.im.abs() < 1e-9);
        }
    }
}
