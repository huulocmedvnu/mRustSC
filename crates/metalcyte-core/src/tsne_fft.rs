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
use crate::neighbors::KnnGraph;
use crate::nndescent::{knn_approximate, NnDescentParams};
use crate::tsne::{
    principal_component_initialisation, TsneParams, CONVERGENCE_CHECK_INTERVAL,
    EXPLORATION_ITERATIONS, EXPLORATION_MOMENTUM, FINAL_MOMENTUM, ITERATIONS_WITHOUT_PROGRESS,
    MACHINE_EPSILON, MIN_GAIN, MIN_GRADIENT_NORM, PERPLEXITY_SEARCH_STEPS, PERPLEXITY_TOLERANCE,
};

/// Neighbours found by NN-descent before the list is widened by expansion.
const SEED_NEIGHBOURS: usize = 15;
/// Interpolation nodes per box and side, as FIt-SNE's `n_interpolation_points`.
const NODES_PER_BOX: usize = 3;
/// Fewest boxes per side, as FIt-SNE's `min_num_intervals`.
const MIN_BOXES: usize = 50;
/// Boxes per unit of layout extent, as FIt-SNE's `intervals_per_integer`.
const BOXES_PER_UNIT: f32 = 1.0;

/// Cell indices sorted along a Z-order (Morton) curve of a two-dimensional layout.
fn z_order(layout: &[f32], n_cells: usize) -> Vec<usize> {
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for i in 0..n_cells {
        for axis in 0..2 {
            lo[axis] = lo[axis].min(layout[2 * i + axis]);
            hi[axis] = hi[axis].max(layout[2 * i + axis]);
        }
    }
    let spread = |v: f32, axis: usize| {
        let range = (hi[axis] - lo[axis]).max(1e-12);
        (((v - lo[axis]) / range) * 65535.0) as u32
    };
    let interleave = |mut v: u32| -> u64 {
        let mut out = 0u64;
        for bit in 0..16 {
            out |= ((v & 1) as u64) << (2 * bit);
            v >>= 1;
        }
        out
    };
    let mut keys: Vec<(u64, usize)> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let x = interleave(spread(layout[2 * i], 0));
            let y = interleave(spread(layout[2 * i + 1], 1));
            (x | (y << 1), i)
        })
        .collect();
    keys.par_sort_unstable();
    keys.into_iter().map(|(_, i)| i).collect()
}

static PROFILE: std::sync::Mutex<[f64; 9]> = std::sync::Mutex::new([0.0; 9]);

fn tick(slot: usize, since: std::time::Instant) {
    PROFILE.lock().unwrap()[slot] += since.elapsed().as_secs_f64();
}

/// Symmetric affinities in compressed sparse row form, summing to one.
pub struct SparseAffinities {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// The attractive term of the gradient, `sum_j P_ij W_ij (y_i - y_j)` per cell, with
/// the exaggerated affinities, and the objective's `sum P log(P / Q)` when asked.
///
/// The CPU implementation is [`CpuAttraction`]; `metalcyte-gpu` provides one on Metal.
pub trait Attraction: Send {
    /// `layout` is `(n_cells, 2)` flat. Returns the `(n_cells, 2)` flat attraction and
    /// the objective (`None` unless `compute_error`); `normaliser` is `Z`.
    fn compute(
        &mut self,
        layout: &[f32],
        affinities: &SparseAffinities,
        exaggeration: f32,
        normaliser: f64,
        compute_error: bool,
    ) -> (Vec<f32>, Option<f64>);
}

/// The attractive term on every core, one row of affinities per task.
pub struct CpuAttraction;

impl Attraction for CpuAttraction {
    fn compute(
        &mut self,
        layout: &[f32],
        affinities: &SparseAffinities,
        exaggeration: f32,
        normaliser: f64,
        compute_error: bool,
    ) -> (Vec<f32>, Option<f64>) {
        let n_cells = layout.len() / 2;
        let mut attraction = vec![0.0f32; 2 * n_cells];
        let error: f64 = attraction
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
                out[0] = a0;
                out[1] = a1;
                kl
            })
            .sum();
        (attraction, compute_error.then_some(error))
    }
}

/// t-SNE by FFT-accelerated interpolation.
///
/// `embedding` is `(n_cells, n_features)`; the output is `(n_cells, 2)`. `device`
/// selects the neighbour search only; the layout itself runs on the cores.
pub fn tsne_fft(
    embedding: &Array2<f32>,
    params: &TsneParams,
    _device: &Device,
) -> Result<Array2<f32>> {
    tsne_fft_with(embedding, params, Box::new(CpuAttraction))
}

/// [`tsne_fft`] with the attractive term computed by `attraction`.
pub fn tsne_fft_with(
    embedding: &Array2<f32>,
    params: &TsneParams,
    mut attraction: Box<dyn Attraction>,
) -> Result<Array2<f32>> {
    let (n_cells, _) = embedding.dim();
    *PROFILE.lock().unwrap() = [0.0; 9];
    if params.n_components != 2 {
        return Err(Error::parameter(
            "n_components",
            "2 for the FFT-accelerated method",
            params.n_components,
        ));
    }
    let k = ((3.0 * params.perplexity).floor() as usize).clamp(1, n_cells - 1);
    let profile = std::env::var_os("METALCYTE_PROFILE").is_some();

    // Cells are renumbered along a Z-order curve of their PCA initialisation, so that
    // cells that will be neighbours in the layout sit near each other in memory: the
    // attractive term and the neighbour search then read the layout mostly in order.
    let initial = principal_component_initialisation(embedding, params, &Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let order = z_order(&initial, n_cells);
    let mut inverse = vec![0usize; n_cells];
    for (new, &old) in order.iter().enumerate() {
        inverse[old] = new;
    }
    let embedding = {
        let d = embedding.ncols();
        let source = embedding.as_standard_layout();
        let source = source.as_slice().expect("contiguous");
        let mut rows = vec![0f32; n_cells * d];
        rows.par_chunks_mut(d)
            .zip(order.par_iter())
            .for_each(|(row, &old)| row.copy_from_slice(&source[old * d..(old + 1) * d]));
        Array2::from_shape_vec((n_cells, d), rows).expect("sized above")
    };
    let initial: Vec<f32> = order
        .iter()
        .flat_map(|&old| [initial[2 * old], initial[2 * old + 1]])
        .collect();

    let t0 = std::time::Instant::now();
    // FIt-SNE uses an approximate index for the 3 * perplexity neighbours. NN-descent
    // is run for a short list and the list is widened to k from the neighbours of those
    // neighbours, which costs one exact distance per candidate and no further search.
    let graph: KnnGraph = {
        let base = k.min(SEED_NEIGHBOURS);
        let seed_graph = knn_approximate(
            &embedding,
            base,
            &NnDescentParams {
                seed: params.seed,
                ..NnDescentParams::default()
            },
        )?;
        if base == k {
            seed_graph
        } else {
            expand_neighbours(&embedding, &seed_graph, k)
        }
    };
    if profile {
        eprintln!(
            "tsne profile: neighbours {:.2} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = std::time::Instant::now();
    let affinities = symmetric_affinities(&graph, params.perplexity);
    drop(graph);
    if profile {
        eprintln!(
            "tsne profile: affinities {:.2} s",
            t0.elapsed().as_secs_f64()
        );
    }

    let t0 = std::time::Instant::now();
    let layout = optimise(initial, n_cells, &affinities, params, attraction.as_mut());
    if profile {
        eprintln!(
            "tsne profile: optimisation {:.2} s",
            t0.elapsed().as_secs_f64()
        );
        let s = PROFILE.lock().unwrap();
        eprintln!(
            "tsne profile: repulsion placement {:.2} s, spread {:.2} s, kernel fft {:.2} s, charge ffts {:.2} s, gather {:.2} s, attraction {:.2} s, update {:.2} s, last grid {} nodes / fft {}",
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7] as usize, s[8] as usize
        );
    }
    // Back to the caller's cell order.
    let mut out = vec![0f32; 2 * n_cells];
    for (new, &old) in order.iter().enumerate() {
        out[2 * old] = layout[2 * new];
        out[2 * old + 1] = layout[2 * new + 1];
    }
    let layout = out;
    Array2::from_shape_vec((n_cells, 2), layout)
        .map_err(|_| Error::shape(format!("{n_cells} by 2"), "a mismatch"))
}

/// Widen a neighbour list to `k` entries from the neighbours of each cell's neighbours:
/// the candidates are the union of the second-order neighbourhood, scored by exact
/// distance, and the `k` nearest are kept, nearest first.
fn expand_neighbours(embedding: &Array2<f32>, seed: &KnnGraph, k: usize) -> KnnGraph {
    let (n_cells, base) = seed.indices.dim();
    let d = embedding.ncols();
    let data = embedding.as_slice().expect("contiguous");
    let row = |i: usize| &data[i * d..(i + 1) * d];
    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let mut candidates: Vec<u32> = Vec::with_capacity(base * base + base);
            for &j in seed.indices.row(i) {
                candidates.push(j);
                candidates.extend(seed.indices.row(j as usize).iter().copied());
            }
            candidates.sort_unstable();
            candidates.dedup();
            let pi = row(i);
            let mut scored: Vec<(f32, u32)> = candidates
                .into_iter()
                .filter(|&j| j as usize != i)
                .map(|j| {
                    let s: f32 = pi
                        .iter()
                        .zip(row(j as usize))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (s, j)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            scored.truncate(k);
            // A cell whose second-order neighbourhood is too small keeps what it has,
            // padded with its own last neighbour so the rows stay rectangular.
            while scored.len() < k {
                let last = *scored.last().expect("at least the seed neighbours");
                scored.push(last);
            }
            scored.iter().map(|&(s, j)| (j, s.max(0.0).sqrt())).unzip()
        })
        .collect();
    let mut indices = Array2::<u32>::zeros((n_cells, k));
    let mut distances = Array2::<f32>::zeros((n_cells, k));
    for (i, (cols, dists)) in rows.into_iter().enumerate() {
        for (j, (c, dd)) in cols.into_iter().zip(dists).enumerate() {
            indices[(i, j)] = c;
            distances[(i, j)] = dd;
        }
    }
    KnnGraph { indices, distances }
}

/// Conditional affinities over each cell's neighbours at the requested perplexity,
/// symmetrised and normalised to sum to one, as scikit-learn's `_joint_probabilities_nn`.
fn symmetric_affinities(graph: &KnnGraph, perplexity: f32) -> SparseAffinities {
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
    attraction: &mut dyn Attraction,
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
        let (gradient, error) = gradient(
            &layout,
            n_cells,
            affinities,
            exaggeration,
            checking,
            attraction,
        );

        let t_upd = std::time::Instant::now();
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

        tick(6, t_upd);
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
///
/// The repulsive term needs the grid and the FFTs, the attractive one the affinities;
/// they share nothing but the layout, so they run at the same time. The objective
/// needs `Z` from the repulsion, so the attraction evaluates it once that is known.
fn gradient(
    layout: &[f32],
    n_cells: usize,
    affinities: &SparseAffinities,
    exaggeration: f32,
    compute_error: bool,
    attraction: &mut dyn Attraction,
) -> (Vec<f32>, Option<f64>) {
    let t_attr = std::time::Instant::now();
    let ((repulsion, normaliser), (mut attractive, _)) = rayon::join(
        || repulsive_forces(layout, n_cells),
        || attraction.compute(layout, affinities, exaggeration, 1.0, false),
    );
    let error = if compute_error {
        let (again, error) = attraction.compute(layout, affinities, exaggeration, normaliser, true);
        attractive = again;
        error
    } else {
        None
    };
    tick(5, t_attr);
    let mut gradient = vec![0.0f32; 2 * n_cells];
    gradient
        .par_iter_mut()
        .zip(attractive.par_iter())
        .zip(repulsion.par_iter())
        .for_each(|((g, &a), &r)| *g = 4.0 * (a - r as f32));
    (gradient, error)
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
    // Boxes are exactly 1 / BOXES_PER_UNIT wide once the layout spans MIN_BOXES of them,
    // so the node spacing, and with it the kernel transform, stays fixed between
    // iterations; below that the grid stretches to the layout.
    let box_width = if extent * BOXES_PER_UNIT >= MIN_BOXES as f32 {
        1.0 / BOXES_PER_UNIT
    } else {
        extent / n_boxes as f32
    };
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

    let t_ph = std::time::Instant::now();
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
    // Cells bucketed by box row, so a band of box rows is one thread's and the grid
    // rows it writes are its own: no per-thread grid and no reduction.
    let mut offsets = vec![0usize; n_boxes + 1];
    for (boxes, _) in &placement {
        offsets[boxes[0] + 1] += 1;
    }
    for b in 0..n_boxes {
        offsets[b + 1] += offsets[b];
    }
    let mut cursor = offsets.clone();
    let mut order = vec![0u32; n_cells];
    for (i, (boxes, _)) in placement.iter().enumerate() {
        order[cursor[boxes[0]]] = i as u32;
        cursor[boxes[0]] += 1;
    }
    tick(0, t_ph);

    let t_ph = std::time::Instant::now();
    let grid_len = n_nodes * n_nodes;
    let mut charges = vec![0f32; 4 * grid_len];
    let charges_ptr = charges.as_mut_ptr() as usize;
    let n_bands = (rayon::current_num_threads() * 4).min(n_boxes).max(1);
    let rows_per_band = n_boxes.div_ceil(n_bands);
    (0..n_bands).into_par_iter().for_each(|band| {
        let (r0, r1) = (
            band * rows_per_band,
            ((band + 1) * rows_per_band).min(n_boxes),
        );
        if r0 >= r1 {
            return;
        }
        // Safety: this band writes grid rows r0 * 3 .. r1 * 3 only, and bands are disjoint.
        let grid = unsafe { std::slice::from_raw_parts_mut(charges_ptr as *mut f32, 4 * grid_len) };
        for &i in &order[offsets[r0]..offsets[r1]] {
            let i = i as usize;
            let (y0, y1) = (layout[2 * i], layout[2 * i + 1]);
            let q = [1.0f32, y0, y1, y0 * y0 + y1 * y1];
            let (boxes, weights) = &placement[i];
            for a in 0..NODES_PER_BOX {
                let row = boxes[0] * NODES_PER_BOX + a;
                for b in 0..NODES_PER_BOX {
                    let col = boxes[1] * NODES_PER_BOX + b;
                    let w = weights[0][a] * weights[1][b];
                    let at = (row * n_nodes + col) * 4;
                    for (slot, &value) in grid[at..at + 4].iter_mut().zip(&q) {
                        *slot += w * value;
                    }
                }
            }
        }
    });
    tick(1, t_ph);

    let t_ph = std::time::Instant::now();
    // Convolve with the squared Cauchy kernel on the embedded circulant. The kernel is
    // even in both axes, so its transform is real; two real charge grids packed as the
    // real and imaginary parts of one field convolve independently.
    let size = (2 * n_nodes).next_power_of_two();
    let fft = Fft2d::new(size);
    let kernel_re = kernel_transform(&fft, size, n_nodes, node_spacing);
    tick(2, t_ph);
    let t_ph = std::time::Instant::now();
    let potentials: Vec<Vec<f32>> = (0..2)
        .into_par_iter()
        .flat_map_iter(|pair| {
            let (c_re, c_im) = (2 * pair, 2 * pair + 1);
            let mut re = vec![0f32; size * size];
            let mut im = vec![0f32; size * size];
            for r in 0..n_nodes {
                let src = r * n_nodes;
                let dst = r * size;
                for col in 0..n_nodes {
                    re[dst + col] = charges[(src + col) * 4 + c_re];
                    im[dst + col] = charges[(src + col) * 4 + c_im];
                }

            }
            fft.forward(&mut re, &mut im);
            for ((a, b), k) in re.iter_mut().zip(im.iter_mut()).zip(kernel_re.iter()) {
                *a *= k;
                *b *= k;
            }
            fft.inverse(&mut re, &mut im);
            let mut out_re = vec![0f32; grid_len];
            let mut out_im = vec![0f32; grid_len];
            for r in 0..n_nodes {
                out_re[r * n_nodes..(r + 1) * n_nodes]
                    .copy_from_slice(&re[r * size..r * size + n_nodes]);
                out_im[r * n_nodes..(r + 1) * n_nodes]
                    .copy_from_slice(&im[r * size..r * size + n_nodes]);
            }
            [out_re, out_im]
        })
        .collect();
    tick(3, t_ph);
    {
        let mut s = PROFILE.lock().unwrap();
        s[7] = n_nodes as f64;
        s[8] = size as f64;
    }

    let t_ph = std::time::Instant::now();
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
                        phi[c] += w * potentials[c][at] as f64;
                    }
                }
            }
            phi
        })
        .collect();
    tick(4, t_ph);
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

/// The transform of the squared Cauchy kernel on the embedded circulant of a grid,
/// cached: the grid changes only when the layout's extent crosses a box boundary, so
/// most iterations reuse the previous transform.
fn kernel_transform(
    fft: &Fft2d,
    size: usize,
    n_nodes: usize,
    node_spacing: f32,
) -> std::sync::Arc<Vec<f32>> {
    use std::sync::{Arc, Mutex};
    type Cached = Option<((usize, u32), Arc<Vec<f32>>)>;
    static CACHE: Mutex<Cached> = Mutex::new(None);
    let key = (n_nodes, node_spacing.to_bits());
    if let Some((cached_key, transform)) = CACHE.lock().unwrap().as_ref() {
        if *cached_key == key {
            return Arc::clone(transform);
        }
    }
    let mut kernel_re = vec![0f32; size * size];
    let mut kernel_im = vec![0f32; size * size];
    for di in 0..n_nodes {
        for dj in 0..n_nodes {
            let x = di as f32 * node_spacing;
            let y = dj as f32 * node_spacing;
            let r2 = x * x + y * y;
            let value = 1.0 / ((1.0 + r2) * (1.0 + r2));
            for ii in [di, (size - di) % size] {
                for jj in [dj, (size - dj) % size] {
                    kernel_re[ii * size + jj] = value;
                }
            }
        }
    }
    fft.forward(&mut kernel_re, &mut kernel_im);
    let transform = Arc::new(kernel_re);
    *CACHE.lock().unwrap() = Some((key, Arc::clone(&transform)));
    transform
}

/// A square power-of-two 2-D FFT on split complex data: Apple's vDSP on macOS, a
/// radix-2 implementation elsewhere.
struct Fft2d {
    size: usize,
    #[cfg(target_os = "macos")]
    setup: std::sync::Arc<vdsp::FftSetup>,
}

impl Fft2d {
    fn new(size: usize) -> Self {
        debug_assert!(size.is_power_of_two());
        Self {
            size,
            #[cfg(target_os = "macos")]
            setup: vdsp::setup_for(size.trailing_zeros() as usize),
        }
    }

    fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        #[cfg(target_os = "macos")]
        {
            self.setup.run(re, im, self.size, false);
        }
        #[cfg(not(target_os = "macos"))]
        {
            fft_2d_scalar(re, im, self.size, false);
        }
    }

    /// The unnormalised inverse scaled by `1 / size^2`, so it undoes `forward`.
    fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        #[cfg(target_os = "macos")]
        {
            self.setup.run(re, im, self.size, true);
        }
        #[cfg(not(target_os = "macos"))]
        {
            fft_2d_scalar(re, im, self.size, true);
        }
        let scale = 1.0 / (self.size * self.size) as f32;
        re.iter_mut().for_each(|v| *v *= scale);
        im.iter_mut().for_each(|v| *v *= scale);
    }
}

/// Apple Accelerate's vDSP FFT, through its C interface.
#[cfg(target_os = "macos")]
mod vdsp {
    use std::ffi::c_void;

    #[repr(C)]
    struct DspSplitComplex {
        realp: *mut f32,
        imagp: *mut f32,
    }

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn vDSP_create_fftsetup(log2n: usize, radix: i32) -> *mut c_void;
        fn vDSP_destroy_fftsetup(setup: *mut c_void);
        fn vDSP_fft2d_zip(
            setup: *mut c_void,
            data: *const DspSplitComplex,
            stride_column: isize,
            stride_row: isize,
            log2n_columns: usize,
            log2n_rows: usize,
            direction: i32,
        );
    }

    const FFT_RADIX2: i32 = 0;
    const FFT_FORWARD: i32 = 1;
    const FFT_INVERSE: i32 = -1;

    /// A setup for one power-of-two size. vDSP setups are read-only after creation and
    /// are used from several threads at once here.
    pub struct FftSetup(*mut c_void);

    /// Setups cost a twiddle table each, so one per size is kept for the process.
    pub fn setup_for(log2n: usize) -> std::sync::Arc<FftSetup> {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
        static SETUPS: Mutex<Option<HashMap<usize, Arc<FftSetup>>>> = Mutex::new(None);
        let mut guard = SETUPS.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        Arc::clone(
            map.entry(log2n)
                .or_insert_with(|| Arc::new(FftSetup::new(log2n))),
        )
    }

    unsafe impl Send for FftSetup {}
    unsafe impl Sync for FftSetup {}

    impl FftSetup {
        pub fn new(log2n: usize) -> Self {
            // Safety: plain C call; a null result is checked.
            let setup = unsafe { vDSP_create_fftsetup(log2n, FFT_RADIX2) };
            assert!(
                !setup.is_null(),
                "vDSP_create_fftsetup failed for 2^{log2n}"
            );
            Self(setup)
        }

        pub fn run(&self, re: &mut [f32], im: &mut [f32], size: usize, inverse: bool) {
            debug_assert_eq!(re.len(), size * size);
            debug_assert_eq!(im.len(), size * size);
            let split = DspSplitComplex {
                realp: re.as_mut_ptr(),
                imagp: im.as_mut_ptr(),
            };
            let log2n = size.trailing_zeros() as usize;
            // Safety: the split arrays hold size * size values each, as the setup expects
            // for a log2n x log2n transform with unit strides.
            unsafe {
                vDSP_fft2d_zip(
                    self.0,
                    &split,
                    1,
                    0,
                    log2n,
                    log2n,
                    if inverse { FFT_INVERSE } else { FFT_FORWARD },
                );
            }
        }
    }

    impl Drop for FftSetup {
        fn drop(&mut self) {
            // Safety: created by vDSP_create_fftsetup and not used after this.
            unsafe { vDSP_destroy_fftsetup(self.0) };
        }
    }
}

/// In-place radix-2 FFT of one row of split complex values.
#[allow(dead_code)]
fn fft_1d_scalar(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (step_re, step_im) = (angle.cos() as f32, angle.sin() as f32);
        for start in (0..n).step_by(len) {
            let (mut w_re, mut w_im) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (u_re, u_im) = (re[start + k], im[start + k]);
                let (x_re, x_im) = (re[start + k + len / 2], im[start + k + len / 2]);
                let v_re = x_re * w_re - x_im * w_im;
                let v_im = x_re * w_im + x_im * w_re;
                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + len / 2] = u_re - v_re;
                im[start + k + len / 2] = u_im - v_im;
                let next_re = w_re * step_re - w_im * step_im;
                w_im = w_re * step_im + w_im * step_re;
                w_re = next_re;
            }
        }
        len <<= 1;
    }
}

/// In-place 2-D FFT of a `size x size` row-major split array: rows, transpose, rows,
/// transpose. Unnormalised in both directions.
#[allow(dead_code)]
fn fft_2d_scalar(re: &mut [f32], im: &mut [f32], size: usize, inverse: bool) {
    re.par_chunks_mut(size)
        .zip(im.par_chunks_mut(size))
        .for_each(|(r, i)| fft_1d_scalar(r, i, inverse));
    transpose(re, size);
    transpose(im, size);
    re.par_chunks_mut(size)
        .zip(im.par_chunks_mut(size))
        .for_each(|(r, i)| fft_1d_scalar(r, i, inverse));
    transpose(re, size);
    transpose(im, size);
}

/// Square transpose in place, by 32 x 32 tiles so each tile stays in cache.
#[allow(dead_code)]
fn transpose(data: &mut [f32], size: usize) {
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
                        let p = ptr as *mut f32;
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
    fn fft_round_trip_is_identity_and_matches_the_scalar_transform() {
        let size = 16;
        let original: Vec<f32> = (0..size * size).map(|i| (i % 7) as f32 - 3.0).collect();
        let fft = Fft2d::new(size);
        let (mut re, mut im) = (original.clone(), vec![0f32; size * size]);
        fft.forward(&mut re, &mut im);
        let (mut re_s, mut im_s) = (original.clone(), vec![0f32; size * size]);
        fft_2d_scalar(&mut re_s, &mut im_s, size, false);
        for ((a, b), (c, d)) in re.iter().zip(&im).zip(re_s.iter().zip(&im_s)) {
            assert!(
                (a - c).abs() < 1e-3 && (b - d).abs() < 1e-3,
                "{a} {b} vs {c} {d}"
            );
        }
        fft.inverse(&mut re, &mut im);
        for (a, b) in re.iter().zip(&original) {
            assert!((a - b).abs() < 1e-4);
        }
        assert!(im.iter().all(|v| v.abs() < 1e-4));
    }
}
