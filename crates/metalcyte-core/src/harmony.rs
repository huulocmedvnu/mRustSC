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
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Tunables, mirroring `harmonypy.run_harmony`'s defaults where they have one.
#[derive(Debug, Clone)]
pub struct HarmonyParams {
    pub theta: f32,
    pub sigma: f32,
    /// Ridge penalty on every batch coefficient. `None` estimates it per cluster and
    /// batch as `alpha` times the cluster's soft count of that batch, as Harmony 1.2 and
    /// harmonypy 2 do by default.
    pub lambda: Option<f32>,
    /// Scale of the estimated ridge penalty when `lambda` is `None`.
    pub alpha: f32,
    /// A batch whose soft share of a cluster is below this is not corrected in it.
    pub batch_prop_cutoff: f32,
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
            lambda: None,
            alpha: 0.2,
            batch_prop_cutoff: 1e-5,
            n_clusters: ((n_cells as f64 / 30.0).round() as usize).clamp(1, 100),
            max_iter_harmony: 10,
            max_iter_kmeans: 20,
            // harmonypy's tolerances: looser than the earlier 1e-5/1e-4, which spent many
            // outer/inner iterations chasing digits that do not change the partition.
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
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
///
/// Soft assignments `R` and the cluster distances are held cells by clusters, so every
/// per-cell step reads one contiguous row and runs across all cores. The block-wise
/// `R` update keeps harmonypy's semantics: within a block the diversity penalty is fixed,
/// so the block's cells are independent and are updated in parallel; the observed and
/// expected counts move once per block, by a parallel reduction over the block.
///
/// The M-step uses the structure of the design: with one-hot batches plus an intercept,
/// `Φ R_k Φᵀ` is made of the soft batch counts of cluster `k` and `Φ R_k Zᵀ` of the soft
/// per-batch sums of the embedding, so every cluster's ridge system comes from one pass
/// over the cells, and the correction of each cell is a sum over clusters of its
/// batch's row of `W_k`. Nothing of size `(cells, clusters, features)` is formed.
pub fn harmony_integrate(
    z_pca: &Array2<f32>,
    batch: &[u32],
    n_batches: usize,
    params: &HarmonyParams,
    device: &Device,
) -> Result<HarmonyResult> {
    let (n_cells, n_pcs) = z_pca.dim();
    if n_cells != batch.len() {
        return Err(Error::shape(
            format!("a batch label per cell ({n_cells})"),
            format!("{}", batch.len()),
        ));
    }
    if n_batches < 2 {
        // Nothing to integrate; hand the embedding back unchanged.
        return Ok(HarmonyResult {
            corrected: z_pca.clone(),
            objective: Vec::new(),
        });
    }
    if let Some(&bad) = batch.iter().find(|&&b| b as usize >= n_batches) {
        return Err(Error::parameter(
            "batch",
            "a label below n_batches",
            format!("{bad} with n_batches = {n_batches}"),
        ));
    }
    let k = params.n_clusters.max(1);
    let batch_of: Vec<usize> = batch.iter().map(|&b| b as usize).collect();

    let z_orig = z_pca.as_standard_layout().into_owned(); // (N, d)
    let mut z_corr = z_orig.clone();
    let mut z_cos = l2_normalise_rows(&z_corr); // (N, d)

    let mut n_b = Array1::<f32>::zeros(n_batches);
    for &b in &batch_of {
        n_b[b] += 1.0;
    }
    let pr_b = &n_b / n_cells as f32; // (B,)
    let theta = Array1::from_elem(n_batches, params.theta);

    // ---- init cluster: k-means centroids, then the soft assignment R ----
    let mut y = kmeans_centroids(&z_cos, k, params.seed); // (d, K)
    normalise_columns_inplace(&mut y);
    let mut dist = distance_matrix(&z_cos, &y, device)?; // (N, K) = 2(1 - Z Y)
    let mut r = softmax_over_clusters(&dist, params.sigma); // (N, K)
                                                            // E = outer(R.sum(0), Pr_b); O = Rᵀ Φᵀ   both (K, B)
    let mut e_mat = outer(&r.sum_axis(Axis(0)), &pr_b);
    let mut o_mat = batch_counts(&r, &batch_of, n_batches);

    let mut objective = Vec::new();
    let mut rng = SplitMix64::new(params.seed);

    for _outer in 0..params.max_iter_harmony {
        // ---- cluster (soft k-means E-step) ----
        let mut prev_kmeans = f32::INFINITY;
        for _ in 0..params.max_iter_kmeans {
            y = matmul(z_cos.t(), r.view(), device)?; // (d, K) = Zᵀ R
            normalise_columns_inplace(&mut y);
            dist = distance_matrix(&z_cos, &y, device)?;
            update_r(
                &mut r, &dist, &batch_of, &pr_b, &theta, &mut e_mat, &mut o_mat, params, &mut rng,
            );
            let obj = kmeans_objective(&r, &dist, &e_mat, &o_mat, &batch_of, &theta, params.sigma);
            let converged =
                (prev_kmeans - obj).abs() / prev_kmeans.abs().max(1e-9) < params.epsilon_cluster;
            prev_kmeans = obj;
            if converged {
                break;
            }
        }

        // ---- moe_correct_ridge (M-step) ----
        z_corr = ridge_correction(&z_orig, &r, &batch_of, n_batches, params)?;
        z_cos = l2_normalise_rows(&z_corr);

        let harmony_obj =
            kmeans_objective(&r, &dist, &e_mat, &o_mat, &batch_of, &theta, params.sigma);
        objective.push(harmony_obj);
        if objective.len() >= 2 {
            let prev = objective[objective.len() - 2];
            if (prev - harmony_obj).abs() / prev.abs().max(1e-9) < params.epsilon_harmony {
                break;
            }
        }
    }
    debug_assert_eq!(z_corr.dim(), (n_cells, n_pcs));

    Ok(HarmonyResult {
        corrected: z_corr,
        objective,
    })
}

/// The M-step: `Z - sum_k R_k ∘ (W_kᵀ Φ_moe)`, cells by features.
///
/// For cluster `k`, `W_k = (Φ R_k Φᵀ + Λ)^-1 Φ R_k Zᵀ` with `Φ` the one-hot batches under
/// an intercept row. `Φ R_k Φᵀ` holds the soft batch counts `n_kb` of the cluster (the
/// intercept row and column hold them too, the corner their sum) and `Φ R_k Zᵀ` the soft
/// per-batch sums `S_kb` of the embedding (the intercept row their sum). Both come from one
/// parallel pass over the cells. The intercept row of `W_k` is zeroed, as harmonypy does,
/// so a cell in batch `b` is corrected by `sum_k R[cell, k] W_k[b + 1, :]`.
fn ridge_correction(
    z: &Array2<f32>,
    r: &Array2<f32>,
    batch_of: &[usize],
    n_batches: usize,
    params: &HarmonyParams,
) -> Result<Array2<f32>> {
    let (n_cells, d) = z.dim();
    let k = r.ncols();
    let z_flat = z.as_slice().expect("contiguous");
    let r_flat = r.as_slice().expect("contiguous");
    let stride = n_batches * (d + 1);
    // Per cluster and batch: d sums of the embedding, then the soft count.
    let stats: Vec<f64> = (0..n_cells)
        .into_par_iter()
        .fold(
            || vec![0f64; k * stride],
            |mut acc, cell| {
                let b = batch_of[cell];
                let zc = &z_flat[cell * d..(cell + 1) * d];
                let rc = &r_flat[cell * k..(cell + 1) * k];
                for (kk, &weight) in rc.iter().enumerate() {
                    let slot = &mut acc[kk * stride + b * (d + 1)..kk * stride + (b + 1) * (d + 1)];
                    let w = f64::from(weight);
                    for (s, &v) in slot[..d].iter_mut().zip(zc) {
                        *s += w * f64::from(v);
                    }
                    slot[d] += w;
                }
                acc
            },
        )
        .reduce(
            || vec![0f64; k * stride],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );

    // Solve every cluster's (B+1)-sized ridge system; keep the batch rows of W_k.
    let bp1 = n_batches + 1;
    let w: Vec<Vec<f32>> = (0..k)
        .into_par_iter()
        .map(|kk| {
            let block = &stats[kk * stride..(kk + 1) * stride];
            let count = |b: usize| block[b * (d + 1) + d];
            let mut x = Array2::<f64>::zeros((bp1, bp1));
            let mut rhs = Array2::<f64>::zeros((bp1, d));
            let n_k: f64 = (0..n_batches).map(count).sum();
            for b in 0..n_batches {
                let n_kb = count(b);
                let lambda = match params.lambda {
                    Some(fixed) => f64::from(fixed),
                    None => f64::from(params.alpha) * n_kb,
                };
                x[[0, 0]] += n_kb;
                x[[0, b + 1]] = n_kb;
                x[[b + 1, 0]] = n_kb;
                x[[b + 1, b + 1]] = n_kb + lambda;
                for f in 0..d {
                    let s = block[b * (d + 1) + f];
                    rhs[[0, f]] += s;
                    rhs[[b + 1, f]] = s;
                }
            }
            // A batch with no soft mass in the cluster has a zero row: regularise it so the
            // system stays invertible; its correction is dropped below anyway.
            for b in 0..n_batches {
                if x[[b + 1, b + 1]] <= 0.0 {
                    x[[b + 1, b + 1]] = 1.0;
                }
            }
            let x_inv = invert_f64(&x).expect("a regularised ridge system is invertible");
            let w_k = x_inv.dot(&rhs); // (B+1, d), row 0 is the intercept and is not applied
            let mut rows = vec![0f32; n_batches * d];
            for b in 0..n_batches {
                // A batch almost absent from the cluster is left alone in it.
                if count(b) < f64::from(params.batch_prop_cutoff) * n_k {
                    continue;
                }
                for f in 0..d {
                    rows[b * d + f] = w_k[[b + 1, f]] as f32;
                }
            }
            rows
        })
        .collect();

    let mut corrected = vec![0f32; n_cells * d];
    corrected
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(cell, out)| {
            let b = batch_of[cell];
            let zc = &z_flat[cell * d..(cell + 1) * d];
            let rc = &r_flat[cell * k..(cell + 1) * k];
            out.copy_from_slice(zc);
            for (kk, &weight) in rc.iter().enumerate() {
                let w_kb = &w[kk][b * d..(b + 1) * d];
                for (o, &v) in out.iter_mut().zip(w_kb) {
                    *o -= weight * v;
                }
            }
        });
    Array2::from_shape_vec((n_cells, d), corrected)
        .map_err(|_| Error::shape("a corrected embedding", "wrong length"))
}

/// `2 (1 - Z Y)`, cells by clusters, via candle so the matmul runs on the caller's device.
fn distance_matrix(z: &Array2<f32>, y: &Array2<f32>, device: &Device) -> Result<Array2<f32>> {
    let mut z_y = matmul(z.view(), y.view(), device)?; // (N, K)
    z_y.as_slice_mut()
        .expect("contiguous")
        .par_iter_mut()
        .for_each(|v| *v = 2.0 * (1.0 - *v));
    Ok(z_y)
}

/// `R = softmax_over_clusters(exp(-dist / sigma))`, one row per cell.
fn softmax_over_clusters(dist: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let k = dist.ncols();
    let mut r = dist.clone();
    r.as_slice_mut()
        .expect("contiguous")
        .par_chunks_mut(k)
        .for_each(|row| {
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (-*v / sigma).exp();
                sum += *v;
            }
            let sum = sum.max(1e-30);
            row.iter_mut().for_each(|v| *v /= sum);
        });
    r
}

/// `O = Rᵀ Φᵀ`, the soft count of each batch in each cluster, `(K, B)`.
fn batch_counts(r: &Array2<f32>, batch_of: &[usize], n_batches: usize) -> Array2<f32> {
    let k = r.ncols();
    let flat = r.as_slice().expect("contiguous");
    let sums = flat
        .par_chunks(k)
        .zip(batch_of.par_iter())
        .fold(
            || vec![0f32; k * n_batches],
            |mut acc, (row, &b)| {
                for (kk, &v) in row.iter().enumerate() {
                    acc[kk * n_batches + b] += v;
                }
                acc
            },
        )
        .reduce(
            || vec![0f32; k * n_batches],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );
    Array2::from_shape_vec((k, n_batches), sums).expect("sized above")
}

/// The block-wise R update with the diversity penalty (harmonypy `update_R`).
#[allow(clippy::too_many_arguments)]
fn update_r(
    r: &mut Array2<f32>,
    dist: &Array2<f32>,
    batch_of: &[usize],
    pr_b: &Array1<f32>,
    theta: &Array1<f32>,
    e_mat: &mut Array2<f32>,
    o_mat: &mut Array2<f32>,
    params: &HarmonyParams,
    rng: &mut SplitMix64,
) {
    let (n_cells, n_clusters) = r.dim();
    let n_batches = pr_b.len();
    let sigma = params.sigma;
    let mut order: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut order);
    let n_blocks = (1.0 / params.block_size).ceil() as usize;
    let block_len = n_cells.div_ceil(n_blocks.max(1));
    let dist_flat = dist.as_slice().expect("contiguous");
    let pr: Vec<f32> = pr_b.to_vec();

    // Soft counts of a set of cells per cluster and batch, `(K, B)`, as a flat vector.
    let counts = |r_flat: &[f32], cells: &[usize]| -> Vec<f32> {
        cells
            .par_iter()
            .fold(
                || vec![0f32; n_clusters * n_batches],
                |mut acc, &cell| {
                    let b = batch_of[cell];
                    let row = &r_flat[cell * n_clusters..(cell + 1) * n_clusters];
                    for (kk, &v) in row.iter().enumerate() {
                        acc[kk * n_batches + b] += v;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0f32; n_clusters * n_batches],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    };
    let apply = |e: &mut Array2<f32>, o: &mut Array2<f32>, c: &[f32], sign: f32| {
        for kk in 0..n_clusters {
            let row_sum: f32 = (0..n_batches).map(|b| c[kk * n_batches + b]).sum();
            for b in 0..n_batches {
                o[[kk, b]] += sign * c[kk * n_batches + b];
                e[[kk, b]] += sign * row_sum * pr[b];
            }
        }
    };

    for chunk in order.chunks(block_len.max(1)) {
        // STEP 1: remove this block's cells from the observed (O) and expected (E) counts.
        let before = counts(r.as_slice().expect("contiguous"), chunk);
        apply(e_mat, o_mat, &before, -1.0);
        // STEP 2: recompute R for the block with the diversity penalty ((E+1)/(O+1))^theta,
        // every cell of the block independently.
        let penalty = penalty_matrix(e_mat, o_mat, theta);
        let penalty_flat = penalty.as_slice().expect("contiguous");
        let r_ptr = r.as_mut_ptr() as usize;
        chunk.par_iter().for_each(|&cell| {
            let b = batch_of[cell];
            // Safety: the cells of one block are distinct, so each row is written by one
            // task only, and nothing else reads `r` during the block.
            let row = unsafe {
                std::slice::from_raw_parts_mut(
                    (r_ptr as *mut f32).add(cell * n_clusters),
                    n_clusters,
                )
            };
            let d = &dist_flat[cell * n_clusters..(cell + 1) * n_clusters];
            let mut sum = 0.0f32;
            for kk in 0..n_clusters {
                let v = (-d[kk] / sigma).exp() * penalty_flat[kk * n_batches + b];
                row[kk] = v;
                sum += v;
            }
            let sum = sum.max(1e-30);
            row.iter_mut().for_each(|v| *v /= sum);
        });
        // STEP 3: add the block's new assignments back into O and E.
        let after = counts(r.as_slice().expect("contiguous"), chunk);
        apply(e_mat, o_mat, &after, 1.0);
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
    batch_of: &[usize],
    theta: &Array1<f32>,
    sigma: f32,
) -> f32 {
    let k = r.ncols();
    let n_batches = e.ncols();
    let r_flat = r.as_slice().expect("R is contiguous");
    let dist_flat = dist.as_slice().expect("dist is contiguous");
    // cross entropy: sum over cells of sigma * R[cell, :] . ( theta * log((O+1)/(E+1)) )[:, batch]
    let mut log_ratio = vec![0f32; k * n_batches];
    for kk in 0..k {
        for b in 0..n_batches {
            log_ratio[kk * n_batches + b] =
                theta[b] * ((o[[kk, b]] + 1.0) / (e[[kk, b]] + 1.0)).ln();
        }
    }
    let (kmeans_error, entropy, cross) = r_flat
        .par_chunks(k)
        .zip(dist_flat.par_chunks(k))
        .zip(batch_of.par_iter())
        .map(|((row, d), &b)| {
            let mut err = 0.0f32;
            let mut ent = 0.0f32;
            let mut cr = 0.0f32;
            for kk in 0..k {
                let v = row[kk];
                err += v * d[kk];
                if v > 0.0 {
                    ent -= v * v.ln();
                }
                cr += v * log_ratio[kk * n_batches + b];
            }
            (err, ent, cr)
        })
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    kmeans_error + sigma * entropy + sigma * cross
}

// ---------------------------------------------------------------- linear algebra helpers

/// Above this many output elements a Metal matmul would repay the host<->device
/// round-trip. Harmony's products are `(d, K)` and `(N, K)` with `K` at most 100 and
/// `d` about 50: at a million cells the `(N, K)` product is 1e8 elements, and measured
/// on an M3 Pro the copy to and from the device still costs more than Accelerate's
/// product on the cores (76 s against 32 s for the whole run at 953 436 cells). The
/// threshold therefore sits above any single-cell size, and `device` is accepted for
/// interface symmetry.
const GPU_MATMUL_THRESHOLD: usize = 1 << 40;

/// Matmul `(m,k) x (k,n) -> (m,n)` that takes views, so a transpose costs nothing.
///
/// Small products go straight through ndarray (SIMD `matrixmultiply`, no allocation or
/// device transfer). Only a large product on a Metal device pays for candle, which is
/// where the GPU actually helps.
fn matmul(a: ArrayView2<f32>, b: ArrayView2<f32>, device: &Device) -> Result<Array2<f32>> {
    let (m, ka) = a.dim();
    let (kb, n) = b.dim();
    if ka != kb {
        return Err(Error::shape(
            format!("({m}, {ka}) x ({kb}, {n})"),
            "a matmul",
        ));
    }
    if !device.is_metal() || m * n < GPU_MATMUL_THRESHOLD {
        // `.dot()` of a transposed view can come back non-standard; force C-contiguous so
        // downstream `as_slice()` (the rayon reductions) always succeeds.
        return Ok(a.dot(&b).as_standard_layout().into_owned());
    }
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let ta = Tensor::from_slice(a.as_slice().unwrap(), (m, ka), device)?;
    let tb = Tensor::from_slice(b.as_slice().unwrap(), (kb, n), device)?;
    let tc = ta.matmul(&tb)?.contiguous()?;
    let data = tc.flatten_all()?.to_vec1::<f32>()?;
    Array2::from_shape_vec((m, n), data)
        .map_err(|_| Error::shape("a matmul result", "wrong length"))
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

/// Each row scaled to unit length: the cosine-normalised cells of a `(N, d)` array.
fn l2_normalise_rows(m: &Array2<f32>) -> Array2<f32> {
    let d = m.ncols();
    let mut out = m.as_standard_layout().into_owned();
    out.as_slice_mut()
        .expect("contiguous")
        .par_chunks_mut(d)
        .for_each(|row| {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
            row.iter_mut().for_each(|v| *v /= norm);
        });
    out
}

fn normalise_columns_inplace(m: &mut Array2<f32>) {
    for mut col in m.axis_iter_mut(Axis(1)) {
        let norm = col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        col.mapv_inplace(|v| v / norm);
    }
}

/// Gauss-Jordan inverse in f64 for the ridge systems, whose counts reach the cell count.
fn invert_f64(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut inv = Array2::<f64>::eye(n);
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[[row, col]].abs() > m[[pivot, col]].abs() {
                pivot = row;
            }
        }
        if m[[pivot, col]].abs() < 1e-12 {
            return Err(Error::parameter(
                "harmony ridge",
                "a solvable system",
                col as f32,
            ));
        }
        if pivot != col {
            for j in 0..n {
                m.swap([col, j], [pivot, j]);
                inv.swap([col, j], [pivot, j]);
            }
        }
        let dd = m[[col, col]];
        for j in 0..n {
            m[[col, j]] /= dd;
            inv[[col, j]] /= dd;
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

/// k-means (Lloyd) on the rows of `z` (N, d), returning `(d, K)` centroids.
fn kmeans_centroids(z: &Array2<f32>, k: usize, seed: u64) -> Array2<f32> {
    let (n, d) = z.dim();
    let flat = z.as_slice().expect("contiguous");
    let row = |i: usize| &flat[i * d..(i + 1) * d];
    let mut rng = SplitMix64::new(seed ^ 0x9e37_79b9);
    // k-means++-ish seeding: first centre random, the rest far from chosen ones.
    let mut centres: Vec<usize> = Vec::with_capacity(k);
    centres.push((rng.next_u64() % n as u64) as usize);
    let mut min_d = vec![f32::INFINITY; n];
    while centres.len() < k {
        let last = row(*centres.last().unwrap());
        min_d.par_iter_mut().enumerate().for_each(|(i, dist)| {
            let s: f32 = row(i)
                .iter()
                .zip(last)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            *dist = dist.min(s);
        });
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
    // Centroids as K rows of d while iterating; transposed to (d, K) at the end.
    let mut y: Vec<f32> = centres.iter().flat_map(|&c| row(c).to_vec()).collect();
    // A few Lloyd iterations to settle the centroids.
    for _ in 0..10 {
        let (sums, counts) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![0f32; k * d], vec![0u32; k]),
                |(mut sums, mut counts), i| {
                    let p = row(i);
                    let mut best = 0;
                    let mut best_d = f32::INFINITY;
                    for c in 0..k {
                        let centre = &y[c * d..(c + 1) * d];
                        let s: f32 = p.iter().zip(centre).map(|(a, b)| (a - b) * (a - b)).sum();
                        if s < best_d {
                            best_d = s;
                            best = c;
                        }
                    }
                    counts[best] += 1;
                    for (acc, &v) in sums[best * d..(best + 1) * d].iter_mut().zip(p) {
                        *acc += v;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![0f32; k * d], vec![0u32; k]),
                |(mut a, mut b), (c, e)| {
                    a.iter_mut().zip(&c).for_each(|(x, y)| *x += y);
                    b.iter_mut().zip(&e).for_each(|(x, y)| *x += y);
                    (a, b)
                },
            );
        for c in 0..k {
            if counts[c] > 0 {
                for f in 0..d {
                    y[c * d + f] = sums[c * d + f] / counts[c] as f32;
                }
            }
        }
    }
    Array2::from_shape_fn((d, k), |(f, c)| y[c * d + f])
}

/// A small deterministic RNG for the block shuffle and k-means seeding.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
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
