//! Harmony batch-effect correction (Korsunsky et al. 2019).
//!
//! Harmony integrates batches in a PCA embedding by alternating a soft k-means E-step
//! that clusters cells while penalising batch-imbalanced clusters (diversity term
//! `theta`), and a ridge-regression M-step that removes the batch-specific shift within
//! each cluster. It is iterative and initialised from k-means, so it does not reproduce
//! `harmonypy` (a compiled C++ backend seeded from scikit-learn k-means) bit for bit;
//! correctness is judged by batch mixing (iLISI) and correlation with `harmonypy`, not
//! equality. See `tests/test_harmony_audit.py`.
//!
//! The two matmul-heavy steps -- the E-step `Yᵀ Z` and the M-step correction `Wᵀ (Φ∘R)`
//! -- run through candle, so they use the Metal backend when `device` is `Device::Metal`.
//! The per-cluster ridge solve is a tiny `(B+1)×(B+1)` system done on the CPU.

use candle_core::{Device, Tensor};
use ndarray::{s, Array1, Array2, Axis};

use crate::error::{Error, Result};

/// Tunables, mirroring `harmonypy.run_harmony`'s defaults where they have one.
#[derive(Debug, Clone)]
pub struct HarmonyParams {
    pub theta: f32,
    pub sigma: f32,
    pub lambda: f32,
    pub n_clusters: usize,
    pub max_iter_harmony: usize,
    pub max_iter_kmeans: usize,
    pub epsilon_cluster: f32,
    pub epsilon_harmony: f32,
    pub block_size: f32,
    pub seed: u64,
}

impl HarmonyParams {
    /// `harmonypy`'s defaults, with `n_clusters = min(round(N/30), 100)` filled in later.
    pub fn defaults(n_cells: usize) -> Self {
        Self {
            theta: 2.0,
            sigma: 0.1,
            lambda: 1.0,
            n_clusters: ((n_cells as f64 / 30.0).round() as usize).clamp(1, 100),
            max_iter_harmony: 10,
            max_iter_kmeans: 20,
            epsilon_cluster: 1e-5,
            epsilon_harmony: 1e-4,
            block_size: 0.05,
            seed: 0,
        }
    }
}

/// The corrected embedding and the harmony objective at each outer iteration.
pub struct HarmonyResult {
    /// Corrected PCA embedding, `(n_cells, n_pcs)` -- the same shape as the input.
    pub corrected: Array2<f32>,
    /// Harmony objective after each outer iteration; a decreasing convergence curve.
    pub objective: Vec<f32>,
}

/// Correct `z_pca` (cells by PCs) for the batch of each cell.
///
/// `batch[i]` is the batch code of cell `i`, in `0..n_batches`.
pub fn harmony_integrate(
    z_pca: &Array2<f32>,
    batch: &[u32],
    n_batches: usize,
    params: &HarmonyParams,
    device: &Device,
) -> Result<HarmonyResult> {
    let (n_cells, _n_pcs) = z_pca.dim();
    if n_cells != batch.len() {
        return Err(Error::shape(
            format!("a batch label per cell ({n_cells})"),
            format!("{}", batch.len()),
        ));
    }
    if n_batches < 2 {
        // Nothing to integrate; hand the embedding back unchanged.
        return Ok(HarmonyResult { corrected: z_pca.clone(), objective: Vec::new() });
    }
    let k = params.n_clusters.max(1);
    let bp1 = n_batches + 1;

    // Work in harmonypy's orientation: features (PCs) as rows, cells as columns.
    let z_orig = z_pca.t().to_owned(); // (d, N)
    let mut z_corr = z_orig.clone();
    let mut z_cos = l2_normalise_columns(&z_corr);

    // One-hot batch design phi (B, N) and phi_moe (B+1, N) with an intercept row of ones.
    let mut phi = Array2::<f32>::zeros((n_batches, n_cells));
    for (cell, &b) in batch.iter().enumerate() {
        phi[[b as usize, cell]] = 1.0;
    }
    let mut phi_moe = Array2::<f32>::ones((bp1, n_cells));
    phi_moe.slice_mut(s![1.., ..]).assign(&phi);

    let n_b = phi.sum_axis(Axis(1)); // (B,)
    let pr_b = &n_b / n_cells as f32; // (B,)
    let theta = Array1::from_elem(n_batches, params.theta);
    // Ridge diagonal: no penalty on the intercept, `lambda` on each batch coefficient.
    let mut lamb = Array1::<f32>::from_elem(bp1, params.lambda);
    lamb[0] = 0.0;

    // ---- init cluster: k-means centroids, then the soft assignment R ----
    let mut y = kmeans_centroids(&z_cos, k, params.seed); // (d, K)
    normalise_columns_inplace(&mut y);
    let mut dist = distance_matrix(&y, &z_cos, device)?; // (K, N) = 2(1 - YᵀZ)
    let mut r = softmax_over_clusters(&dist, params.sigma); // (K, N)
    // E = outer(R.sum(1), Pr_b); O = R @ phiᵀ   both (K, B)
    let mut e_mat = outer(&r.sum_axis(Axis(1)), &pr_b);
    let mut o_mat = r.dot(&phi.t());

    let mut objective = Vec::new();
    let mut rng = SplitMix64::new(params.seed);

    for _outer in 0..params.max_iter_harmony {
        // ---- cluster (soft k-means E-step) ----
        let mut prev_kmeans = f32::INFINITY;
        for _ in 0..params.max_iter_kmeans {
            y = gemm(&z_cos, &r.t().to_owned(), device)?; // (d, K) = Z R'
            normalise_columns_inplace(&mut y);
            dist = distance_matrix(&y, &z_cos, device)?;
            update_r(
                &mut r, &dist, &phi, &pr_b, &theta, &mut e_mat, &mut o_mat, params, &mut rng,
            );
            let obj = kmeans_objective(&r, &dist, &e_mat, &o_mat, &phi, &theta, params.sigma);
            let converged =
                (prev_kmeans - obj).abs() / prev_kmeans.abs().max(1e-9) < params.epsilon_cluster;
            prev_kmeans = obj;
            if converged {
                break;
            }
        }

        // ---- moe_correct_ridge (M-step) ----
        z_corr = z_orig.clone();
        for cluster in 0..k {
            let r_k = r.row(cluster); // (N,)
            // phi_rk = phi_moe * R_k, each column scaled by that cell's assignment (B+1, N)
            let mut phi_rk = phi_moe.clone();
            for j in 0..n_cells {
                let scale = r_k[j];
                for row in 0..bp1 {
                    phi_rk[[row, j]] *= scale;
                }
            }
            // x = phi_rk @ phi_moeᵀ + diag(lamb)   (B+1, B+1)
            let mut x = phi_rk.dot(&phi_moe.t());
            for d in 0..bp1 {
                x[[d, d]] += lamb[d];
            }
            let x_inv = invert(&x)?; // (B+1, B+1)
            // W = x_inv @ phi_rk @ z_origᵀ    (B+1, d)
            let w = x_inv.dot(&phi_rk.dot(&z_orig.t()));
            let mut w = w;
            w.row_mut(0).fill(0.0); // never remove the intercept
            // z_corr -= Wᵀ @ phi_rk   (d, N)
            let correction = gemm(&w.t().to_owned(), &phi_rk, device)?;
            z_corr = &z_corr - &correction;
        }
        z_cos = l2_normalise_columns(&z_corr);

        let harmony_obj = kmeans_objective(&r, &dist, &e_mat, &o_mat, &phi, &theta, params.sigma);
        objective.push(harmony_obj);
        if objective.len() >= 2 {
            let prev = objective[objective.len() - 2];
            if (prev - harmony_obj).abs() / prev.abs().max(1e-9) < params.epsilon_harmony {
                break;
            }
        }
    }

    Ok(HarmonyResult { corrected: z_corr.t().to_owned(), objective })
}

/// `2 (1 - Yᵀ Z)` via candle, so the matmul runs on the caller's device.
fn distance_matrix(y: &Array2<f32>, z: &Array2<f32>, device: &Device) -> Result<Array2<f32>> {
    let yt_z = gemm(&y.t().to_owned(), z, device)?; // (K, N)
    Ok(yt_z.mapv(|v| 2.0 * (1.0 - v)))
}

/// `R = softmax_over_clusters(exp(-dist / sigma))`.
fn softmax_over_clusters(dist: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let mut r = dist.mapv(|v| (-v / sigma).exp());
    let sums = r.sum_axis(Axis(0));
    for (j, mut col) in r.axis_iter_mut(Axis(1)).enumerate() {
        let s = sums[j].max(1e-30);
        col.mapv_inplace(|v| v / s);
    }
    r
}

/// The block-wise R update with the diversity penalty (harmonypy `update_R`).
#[allow(clippy::too_many_arguments)]
fn update_r(
    r: &mut Array2<f32>,
    dist: &Array2<f32>,
    phi: &Array2<f32>,
    pr_b: &Array1<f32>,
    theta: &Array1<f32>,
    e_mat: &mut Array2<f32>,
    o_mat: &mut Array2<f32>,
    params: &HarmonyParams,
    rng: &mut SplitMix64,
) {
    let scale = dist.mapv(|v| (-v / params.sigma).exp()); // (K, N)
    let n_cells = r.ncols();
    let n_clusters = r.nrows();
    let batch_of: Vec<usize> = (0..n_cells)
        .map(|cell| phi.column(cell).iter().position(|&v| v > 0.0).unwrap_or(0))
        .collect();
    let mut order: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut order);
    let n_blocks = (1.0 / params.block_size).ceil() as usize;
    let block_len = n_cells.div_ceil(n_blocks.max(1));

    for chunk in order.chunks(block_len.max(1)) {
        // STEP 1: remove this block's cells from the observed (O) and expected (E) counts.
        for &cell in chunk {
            let batch = batch_of[cell];
            for kk in 0..n_clusters {
                let contribution = r[[kk, cell]];
                o_mat[[kk, batch]] -= contribution;
                for b in 0..pr_b.len() {
                    e_mat[[kk, b]] -= contribution * pr_b[b];
                }
            }
        }
        // STEP 2: recompute R for the block with the diversity penalty ((E+1)/(O+1))^theta.
        let penalty = penalty_matrix(e_mat, o_mat, theta);
        for &cell in chunk {
            let batch = batch_of[cell];
            let mut col_sum = 0.0f32;
            for kk in 0..n_clusters {
                let val = scale[[kk, cell]] * penalty[[kk, batch]];
                r[[kk, cell]] = val;
                col_sum += val;
            }
            let col_sum = col_sum.max(1e-30);
            for kk in 0..n_clusters {
                r[[kk, cell]] /= col_sum;
            }
        }
        // STEP 3: add the block's new assignments back into O and E.
        for &cell in chunk {
            let batch = batch_of[cell];
            for kk in 0..n_clusters {
                let contribution = r[[kk, cell]];
                o_mat[[kk, batch]] += contribution;
                for b in 0..pr_b.len() {
                    e_mat[[kk, b]] += contribution * pr_b[b];
                }
            }
        }
    }
}

fn penalty_matrix(e: &Array2<f32>, o: &Array2<f32>, theta: &Array1<f32>) -> Array2<f32> {
    let mut p = Array2::<f32>::zeros(e.raw_dim());
    for k in 0..e.nrows() {
        for b in 0..e.ncols() {
            p[[k, b]] = (((e[[k, b]] + 1.0) / (o[[k, b]] + 1.0)).max(1e-30)).powf(theta[b]);
        }
    }
    p
}

/// Harmony's objective: k-means error + entropy + a batch-diversity cross entropy.
fn kmeans_objective(
    r: &Array2<f32>,
    dist: &Array2<f32>,
    e: &Array2<f32>,
    o: &Array2<f32>,
    phi: &Array2<f32>,
    theta: &Array1<f32>,
    sigma: f32,
) -> f32 {
    let kmeans_error: f32 = (r * dist).sum();
    let entropy: f32 = r.mapv(|v| if v > 0.0 { -v * v.ln() } else { 0.0 }).sum() * sigma;
    // cross entropy: sum over cells of sigma * R[:,cell] . ( theta * log((O+1)/(E+1)) )[:, batch]
    let mut log_ratio = Array2::<f32>::zeros(e.raw_dim());
    for k in 0..e.nrows() {
        for b in 0..e.ncols() {
            log_ratio[[k, b]] = theta[b] * ((o[[k, b]] + 1.0) / (e[[k, b]] + 1.0)).ln();
        }
    }
    let projected = log_ratio.dot(phi); // (K, N)
    let cross: f32 = (r * &projected).sum() * sigma;
    kmeans_error + entropy + cross
}

// ---------------------------------------------------------------- linear algebra helpers

/// One candle matmul `(m,k) x (k,n) -> (m,n)`, on the caller's device.
fn gemm(a: &Array2<f32>, b: &Array2<f32>, device: &Device) -> Result<Array2<f32>> {
    let (m, ka) = a.dim();
    let (kb, n) = b.dim();
    if ka != kb {
        return Err(Error::shape(format!("({m}, {ka}) x ({kb}, {n})"), "a matmul"));
    }
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let ta = Tensor::from_slice(a.as_slice().unwrap(), (m, ka), device)?;
    let tb = Tensor::from_slice(b.as_slice().unwrap(), (kb, n), device)?;
    let tc = ta.matmul(&tb)?.contiguous()?;
    let data = tc.flatten_all()?.to_vec1::<f32>()?;
    Array2::from_shape_vec((m, n), data).map_err(|_| Error::shape("a matmul result", "wrong length"))
}

fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            out[[i, j]] = a[i] * b[j];
        }
    }
    out
}

fn l2_normalise_columns(m: &Array2<f32>) -> Array2<f32> {
    let mut out = m.clone();
    normalise_columns_inplace(&mut out);
    out
}

fn normalise_columns_inplace(m: &mut Array2<f32>) {
    for mut col in m.axis_iter_mut(Axis(1)) {
        let norm = col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        col.mapv_inplace(|v| v / norm);
    }
}

/// Gauss-Jordan inverse of a small square matrix (the `(B+1)×(B+1)` ridge system).
fn invert(a: &Array2<f32>) -> Result<Array2<f32>> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut inv = Array2::<f32>::eye(n);
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[[row, col]].abs() > m[[pivot, col]].abs() {
                pivot = row;
            }
        }
        if m[[pivot, col]].abs() < 1e-12 {
            return Err(Error::parameter("harmony ridge", "a solvable system", col as f32));
        }
        if pivot != col {
            swap_rows(&mut m, col, pivot);
            swap_rows(&mut inv, col, pivot);
        }
        let d = m[[col, col]];
        for j in 0..n {
            m[[col, j]] /= d;
            inv[[col, j]] /= d;
        }
        for row in 0..n {
            if row != col {
                let factor = m[[row, col]];
                for j in 0..n {
                    m[[row, j]] -= factor * m[[col, j]];
                    inv[[row, j]] -= factor * inv[[col, j]];
                }
            }
        }
    }
    Ok(inv)
}

fn swap_rows(m: &mut Array2<f32>, a: usize, b: usize) {
    for j in 0..m.ncols() {
        m.swap([a, j], [b, j]);
    }
}

/// k-means (Lloyd) on the columns of `z` (d, N), returning `(d, K)` centroids.
fn kmeans_centroids(z: &Array2<f32>, k: usize, seed: u64) -> Array2<f32> {
    let (d, n) = z.dim();
    let mut rng = SplitMix64::new(seed ^ 0x9e37_79b9);
    // k-means++-ish seeding: first centre random, the rest far from chosen ones.
    let mut centres: Vec<usize> = Vec::with_capacity(k);
    centres.push((rng.next_u64() % n as u64) as usize);
    let mut min_d = vec![f32::INFINITY; n];
    while centres.len() < k {
        let last = *centres.last().unwrap();
        for (i, dist) in min_d.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for f in 0..d {
                let diff = z[[f, i]] - z[[f, last]];
                s += diff * diff;
            }
            *dist = dist.min(s);
        }
        let total: f32 = min_d.iter().sum();
        let mut target = rng.next_f32() * total.max(1e-30);
        let mut chosen = 0;
        for (i, &dd) in min_d.iter().enumerate() {
            target -= dd;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.push(chosen);
    }
    let mut y = Array2::<f32>::zeros((d, k));
    for (c, &cell) in centres.iter().enumerate() {
        for f in 0..d {
            y[[f, c]] = z[[f, cell]];
        }
    }
    // A few Lloyd iterations to settle the centroids.
    for _ in 0..10 {
        let mut sums = Array2::<f32>::zeros((d, k));
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let mut s = 0.0f32;
                for f in 0..d {
                    let diff = z[[f, i]] - y[[f, c]];
                    s += diff * diff;
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            counts[best] += 1;
            for f in 0..d {
                sums[[f, best]] += z[[f, i]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for f in 0..d {
                    y[[f, c]] = sums[[f, c]] / counts[c] as f32;
                }
            }
        }
    }
    y
}

/// A small deterministic RNG for the block shuffle and k-means seeding.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}
