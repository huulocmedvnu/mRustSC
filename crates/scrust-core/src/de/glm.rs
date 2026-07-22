use candle_core::{DType, Device, Tensor};
use ndarray::{Array2, Array3};

use crate::error::{Error, Result};

/// A per-gene negative binomial GLM fit on the natural-log scale.
#[derive(Debug, Clone)]
pub struct GlmFit {
    /// `(n_genes, n_coefficients)`.
    pub coefficients: Array2<f32>,
    /// `(n_genes, n_coefficients, n_coefficients)`.
    pub covariance: Array3<f32>,
    pub dispersions: Vec<f32>,
    /// `(n_genes, n_samples)`.
    pub fitted_means: Array2<f32>,
    pub converged: Vec<bool>,
    pub n_iterations: usize,
}

/// Fitted means are kept away from zero: the IRLS weights and the working
/// response both divide by `mu`.
const MINIMUM_MEAN: f64 = 1e-8;

/// `exp(30)` is ~1e13: far above any realistic count, far below f32 overflow.
const MAXIMUM_LINEAR_PREDICTOR: f64 = 30.0;

/// Bounds `(y - mu) / mu` while an early iterate is still far from the mode, so
/// one badly scaled gene cannot produce a non-finite normal equation.
const MAXIMUM_WORKING_RESIDUAL: f64 = 1e4;

/// Tikhonov term added to the information matrix. It only matters for genes
/// whose weights collapse to zero (all-zero counts), where it turns a singular
/// solve into a finite step; against a realistic information matrix it is
/// numerically invisible.
const RIDGE: f64 = 1e-8;

/// Smallest relative coefficient change f32 is trusted to resolve, ~1.5e-5.
/// Without this floor a stricter tolerance keeps every gene iterating to
/// `max_iterations` and reports convergence that did happen as failure.
const PRECISION_FLOOR_ULPS: f32 = 128.0;

/// Keeps `log(y / size_factor)` finite for the least-squares starting values.
const STARTING_PSEUDOCOUNT: f64 = 0.1;

/// Fit one GLM per gene by iteratively reweighted least squares.
///
/// All genes advance through the same iteration as one batch: tens of thousands
/// of tiny identical problems is exactly the shape a GPU is good at.
pub fn fit_negative_binomial(
    counts: &Array2<f32>,
    design: &Array2<f32>,
    size_factors: &[f32],
    dispersions: &[f32],
    max_iterations: usize,
    tolerance: f32,
    device: &Device,
) -> Result<GlmFit> {
    validate(counts, design, size_factors, dispersions, max_iterations)?;
    let (n_genes, n_samples) = counts.dim();
    let n_coefficients = design.ncols();
    if n_genes == 0 {
        return Ok(empty_fit(n_coefficients, n_samples));
    }

    let observed = to_tensor(counts, device)?;
    // A leading axis of one lets the size factors broadcast over genes, and a
    // trailing axis lets the dispersions broadcast over samples.
    let offsets = Tensor::from_slice(size_factors, (1, n_samples), device)?;
    let alpha = Tensor::from_slice(dispersions, (n_genes, 1), device)?.maximum(0f64)?;
    let design_batch = to_tensor(design, device)?.reshape((1, n_samples, n_coefficients))?;
    let ridge = (Tensor::eye(n_coefficients, DType::F32, device)? * RIDGE)?.reshape((
        1,
        n_coefficients,
        n_coefficients,
    ))?;

    let tolerance = f64::from(tolerance.max(PRECISION_FLOOR_ULPS * f32::EPSILON));
    let mut coefficients =
        starting_coefficients(&observed, &offsets, &design_batch, &ridge, device)?;
    // 0 = still moving, 1 = converged; kept on the device to avoid a per-gene
    // host round trip, shaped (n_genes, 1, 1) to broadcast over coefficients.
    let mut frozen = Tensor::zeros((n_genes, 1, 1), DType::F32, device)?;

    let mut n_iterations = 0;
    for iteration in 1..=max_iterations {
        n_iterations = iteration;
        let (weights, response) =
            working_problem(&observed, &offsets, &alpha, &coefficients, &design_batch)?;
        let candidate = weighted_least_squares(&weights, &response, &design_batch, &ridge)?;

        // The step is judged relative for large coefficients, absolute for small
        // ones, and a gene has converged once no coefficient exceeds it.
        let step = (&candidate - &coefficients)?.abs()?;
        let threshold = (coefficients.abs()?.maximum(1f64)? * tolerance)?;
        let n_moving = step.ge(&threshold)?.to_dtype(DType::F32)?.sum_keepdim(1)?;

        // Converged genes stop being updated, so the reported flags always
        // describe the coefficients actually returned.
        let keep = frozen
            .gt(0f64)?
            .broadcast_as(coefficients.shape())?
            .contiguous()?;
        coefficients = keep.where_cond(&coefficients, &candidate)?;
        frozen = frozen.maximum(&n_moving.le(0f64)?.to_dtype(DType::F32)?)?;
        if frozen.sum_all()?.to_scalar::<f32>()? >= n_genes as f32 {
            break;
        }
    }

    let fitted_means = fitted_means(&offsets, &coefficients, &design_batch)?;
    let (weights, _) = working_problem(&observed, &offsets, &alpha, &coefficients, &design_batch)?;
    let (_, information) = information(&weights, &design_batch, &ridge)?;
    let identity = Tensor::eye(n_coefficients, DType::F32, device)?
        .broadcast_as((n_genes, n_coefficients, n_coefficients))?
        .contiguous()?;
    let covariance = cholesky_solve(&cholesky(&information)?, &identity)?;

    Ok(GlmFit {
        coefficients: to_array2(&coefficients.squeeze(2)?)?,
        covariance: to_array3(&covariance)?,
        dispersions: dispersions.to_vec(),
        fitted_means: to_array2(&fitted_means)?,
        converged: frozen
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|flag| flag > 0.0)
            .collect(),
        n_iterations,
    })
}

fn validate(
    counts: &Array2<f32>,
    design: &Array2<f32>,
    size_factors: &[f32],
    dispersions: &[f32],
    max_iterations: usize,
) -> Result<()> {
    let (n_genes, n_samples) = counts.dim();
    if design.nrows() != n_samples {
        return Err(Error::shape(
            format!("a design with {n_samples} rows"),
            format!("{} rows", design.nrows()),
        ));
    }
    if design.ncols() == 0 {
        return Err(Error::parameter("design", "at least one column", 0));
    }
    if size_factors.len() != n_samples {
        return Err(Error::shape(
            format!("{n_samples} size factors"),
            format!("{}", size_factors.len()),
        ));
    }
    if dispersions.len() != n_genes {
        return Err(Error::shape(
            format!("{n_genes} dispersions"),
            format!("{}", dispersions.len()),
        ));
    }
    if let Some(bad) = size_factors.iter().find(|f| !f.is_finite() || **f <= 0.0) {
        return Err(Error::parameter("size_factors", "strictly positive", bad));
    }
    if max_iterations == 0 {
        return Err(Error::parameter("max_iterations", "at least 1", 0));
    }
    Ok(())
}

fn empty_fit(n_coefficients: usize, n_samples: usize) -> GlmFit {
    GlmFit {
        coefficients: Array2::zeros((0, n_coefficients)),
        covariance: Array3::zeros((0, n_coefficients, n_coefficients)),
        dispersions: Vec::new(),
        fitted_means: Array2::zeros((0, n_samples)),
        converged: Vec::new(),
        n_iterations: 0,
    }
}

/// Ordinary least squares on `log(y / size_factor)`.
///
/// Good starting values buy more accuracy per unit of work than extra IRLS
/// iterations, and keep the first weighted solve well conditioned.
fn starting_coefficients(
    observed: &Tensor,
    offsets: &Tensor,
    design_batch: &Tensor,
    ridge: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let (n_genes, n_samples) = observed.dims2()?;
    let log_response = (observed.broadcast_div(offsets)? + STARTING_PSEUDOCOUNT)?.log()?;
    let unit_weights = Tensor::ones((n_genes, n_samples, 1), DType::F32, device)?;
    weighted_least_squares(&unit_weights, &log_response, design_batch, ridge)
}

/// `mu = size_factor * exp(design @ beta)`, clamped away from zero and overflow.
fn fitted_means(offsets: &Tensor, coefficients: &Tensor, design_batch: &Tensor) -> Result<Tensor> {
    let linear_predictor = design_batch
        .broadcast_matmul(coefficients)?
        .squeeze(2)?
        .clamp(-MAXIMUM_LINEAR_PREDICTOR, MAXIMUM_LINEAR_PREDICTOR)?;
    Ok(linear_predictor
        .exp()?
        .broadcast_mul(offsets)?
        .maximum(MINIMUM_MEAN)?)
}

/// IRLS weights `mu / (1 + alpha * mu)` as `(n_genes, n_samples, 1)` columns and
/// the working response `eta + (y - mu) / mu` as `(n_genes, n_samples)`.
fn working_problem(
    observed: &Tensor,
    offsets: &Tensor,
    alpha: &Tensor,
    coefficients: &Tensor,
    design_batch: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let means = fitted_means(offsets, coefficients, design_batch)?;
    let weights = means.div(&(alpha.broadcast_mul(&means)? + 1.0)?)?;
    let residual = observed
        .sub(&means)?
        .div(&means)?
        .clamp(-MAXIMUM_WORKING_RESIDUAL, MAXIMUM_WORKING_RESIDUAL)?;
    // The offset is not part of eta, so it drops out of the working response.
    let linear_predictor = means.log()?.broadcast_sub(&offsets.log()?)?;
    Ok((weights.unsqueeze(2)?, (linear_predictor + residual)?))
}

/// Weighted design `(W X)'` and information matrix `X' W X + ridge`.
///
/// `weights` is `(n_genes, n_samples, 1)` so that it scales the design rows; the
/// design is `(1, n_samples, p)` and broadcasts over genes.
fn information(
    weights: &Tensor,
    design_batch: &Tensor,
    ridge: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let weighted_design_transposed = weights
        .broadcast_mul(design_batch)?
        .transpose(1, 2)?
        .contiguous()?;
    let information = weighted_design_transposed
        .broadcast_matmul(design_batch)?
        .broadcast_add(ridge)?;
    Ok((weighted_design_transposed, information))
}

/// Solve the batched normal equations `(X' W X) b = X' W z`.
fn weighted_least_squares(
    weights: &Tensor,
    response: &Tensor,
    design_batch: &Tensor,
    ridge: &Tensor,
) -> Result<Tensor> {
    let (weighted_design_transposed, information) = information(weights, design_batch, ridge)?;
    let right_hand_side = weighted_design_transposed.matmul(&response.unsqueeze(2)?)?;
    cholesky_solve(&cholesky(&information)?, &right_hand_side)
}

/// Lower Cholesky factor of a batch of small symmetric positive definite
/// matrices, `(n_genes, p, p)`.
///
/// candle 0.9 has no batched linear solve. `p` is 2 to 8, so the factorisation
/// is unrolled over columns: each step is a single tensor op spanning every
/// gene, which is what keeps the whole batch on the device.
fn cholesky(matrices: &Tensor) -> Result<Tensor> {
    let (n_genes, p, _) = matrices.dims3()?;
    let device = matrices.device();
    // Each entry is a whole column `(n_genes, p, 1)`, zero above the diagonal,
    // so the already-computed part of a row can be sliced straight out.
    let mut columns: Vec<Tensor> = Vec::with_capacity(p);
    for j in 0..p {
        let below = p - j;
        let target = matrices
            .narrow(1, j, below)?
            .narrow(2, j, 1)?
            .contiguous()?;
        let adjusted = if j == 0 {
            target
        } else {
            let prior = Tensor::cat(&columns, 2)?;
            let rows = prior.narrow(1, j, below)?.contiguous()?;
            let pivot_row = prior.narrow(1, j, 1)?.transpose(1, 2)?.contiguous()?;
            (target - rows.matmul(&pivot_row)?)?
        };
        // A pivot no larger than the ridge means the gene carries no
        // information; flooring it yields a finite factor instead of a NaN.
        let diagonal = adjusted.narrow(1, 0, 1)?.maximum(RIDGE)?.sqrt()?;
        let mut parts = Vec::with_capacity(3);
        if j > 0 {
            parts.push(Tensor::zeros((n_genes, j, 1), matrices.dtype(), device)?);
        }
        parts.push(diagonal.clone());
        if below > 1 {
            parts.push(adjusted.narrow(1, 1, below - 1)?.broadcast_div(&diagonal)?);
        }
        columns.push(Tensor::cat(&parts, 1)?);
    }
    Ok(Tensor::cat(&columns, 2)?)
}

/// Solve `L L' X = B` by forward and back substitution, for a whole batch at
/// once. `rhs` is `(n_genes, p, k)`, so one call inverts a batch of matrices as
/// readily as it solves a batch of single systems.
fn cholesky_solve(factor: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (_, p, _) = factor.dims3()?;
    let pivot = |row: usize| factor.narrow(1, row, 1)?.narrow(2, row, 1)?.contiguous();

    let mut forward: Vec<Tensor> = Vec::with_capacity(p);
    for row in 0..p {
        let target = rhs.narrow(1, row, 1)?.contiguous()?;
        let adjusted = if row == 0 {
            target
        } else {
            let known = Tensor::cat(&forward, 1)?;
            let coefficients = factor.narrow(1, row, 1)?.narrow(2, 0, row)?.contiguous()?;
            (target - coefficients.matmul(&known)?)?
        };
        forward.push(adjusted.broadcast_div(&pivot(row)?)?);
    }
    let intermediate = Tensor::cat(&forward, 1)?;

    let mut backward: Vec<Tensor> = Vec::with_capacity(p);
    for row in (0..p).rev() {
        let target = intermediate.narrow(1, row, 1)?.contiguous()?;
        let adjusted = if row + 1 == p {
            target
        } else {
            let known = Tensor::cat(&backward, 1)?;
            // Row `row` of `L'` is column `row` of `L` below the diagonal.
            let coefficients = factor
                .narrow(1, row + 1, p - row - 1)?
                .narrow(2, row, 1)?
                .transpose(1, 2)?
                .contiguous()?;
            (target - coefficients.matmul(&known)?)?
        };
        backward.insert(0, adjusted.broadcast_div(&pivot(row)?)?);
    }
    Ok(Tensor::cat(&backward, 1)?)
}

fn to_tensor(array: &Array2<f32>, device: &Device) -> Result<Tensor> {
    let standard = array.as_standard_layout();
    let values: Vec<f32> = standard.iter().copied().collect();
    Ok(Tensor::from_vec(values, array.dim(), device)?)
}

fn to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
    let (rows, columns) = tensor.dims2()?;
    let values = tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Array2::from_shape_vec((rows, columns), values)
        .map_err(|error| Error::shape(format!("{rows}x{columns}"), error.to_string()))
}

fn to_array3(tensor: &Tensor) -> Result<Array3<f32>> {
    let (batch, rows, columns) = tensor.dims3()?;
    let values = tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Array3::from_shape_vec((batch, rows, columns), values)
        .map_err(|error| Error::shape(format!("{batch}x{rows}x{columns}"), error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{gpu_available, DeviceKind};
    use ndarray::{array, concatenate, Axis};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f32 = 1e-6;

    /// Poisson counts drawn with `numpy.random.default_rng(11)`: six genes over
    /// two groups of five, as in the validated Python test.
    fn reference_counts() -> Array2<f32> {
        array![
            [21., 33., 23., 51., 38., 103., 92., 117., 130., 128.],
            [26., 26., 42., 37., 30., 96., 110., 128., 114., 144.],
            [25., 30., 22., 44., 37., 82., 104., 120., 97., 143.],
            [33., 22., 30., 37., 35., 118., 103., 120., 98., 117.],
            [32., 34., 27., 30., 46., 94., 97., 116., 119., 130.],
            [14., 32., 31., 35., 37., 114., 88., 118., 130., 112.],
        ]
    }

    /// Exact maximum likelihood fits of `reference_counts`, one row per gene,
    /// from `scipy.optimize.minimize` on the negative binomial log likelihood.
    /// Held in f64 so that the reference keeps more digits than the fit it
    /// judges; the deviation is measured after widening our f32 result.
    const MAXIMUM_LIKELIHOOD_POISSON_LIMIT: [[f64; 2]; 6] = [
        [3.6585541784, 0.8589550753],
        [3.6279705455, 0.9274089904],
        [3.6091614481, 0.8653303839],
        [3.6028120012, 0.8898291738],
        [3.6764650971, 0.8161760953],
        [3.5505125367, 0.9528622431],
    ];
    const MAXIMUM_LIKELIHOOD_DISPERSION_025: [[f64; 2]; 6] = [
        [3.6471507974, 0.8686238153],
        [3.6318633672, 0.9208057885],
        [3.6046133772, 0.8646619810],
        [3.6086655057, 0.8925382693],
        [3.6812169011, 0.8087348612],
        [3.5371006730, 0.9698045243],
    ];

    fn two_group_design(n_per_group: usize) -> Array2<f32> {
        Array2::from_shape_fn((2 * n_per_group, 2), |(sample, column)| match column {
            0 => 1.0,
            _ => f32::from(sample >= n_per_group),
        })
    }

    fn linear_space(start: f32, end: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
            .collect()
    }

    fn fit(
        counts: &Array2<f32>,
        design: &Array2<f32>,
        size_factors: &[f32],
        dispersion: f32,
        device: &Device,
    ) -> GlmFit {
        fit_negative_binomial(
            counts,
            design,
            size_factors,
            &vec![dispersion; counts.nrows()],
            MAX_ITERATIONS,
            TOLERANCE,
            device,
        )
        .unwrap()
    }

    fn gpu() -> Device {
        DeviceKind::Gpu.resolve().unwrap()
    }

    /// A standard normal deviate by Box-Muller, from a seeded generator so that
    /// simulated data is identical on every machine.
    fn standard_normal(rng: &mut StdRng) -> f32 {
        let radius: f32 = rng.gen_range(f32::EPSILON..1.0);
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        (-2.0 * radius.ln()).sqrt() * angle.cos()
    }

    /// Overdispersed counts around a random baseline with a planted log2 fold
    /// change in every third gene.
    fn simulated_dataset(
        n_genes: usize,
        n_per_group: usize,
        dispersion: f32,
        seed: u64,
    ) -> (Array2<f32>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = 2 * n_per_group;
        // Lognormal noise of this width has the variance the dispersion implies.
        let spread = (1.0 + dispersion).ln().sqrt();
        let mut counts = Array2::zeros((n_genes, n_samples));
        let mut true_log2_fold_change = Vec::with_capacity(n_genes);
        for gene in 0..n_genes {
            let baseline: f32 = rng.gen_range(20.0..200.0);
            let log2_fold_change = if gene % 3 == 0 {
                rng.gen_range(-3.0..3.0)
            } else {
                0.0
            };
            true_log2_fold_change.push(log2_fold_change);
            for sample in 0..n_samples {
                let treated = f32::from(sample >= n_per_group);
                let mean = baseline * 2.0f32.powf(log2_fold_change * treated);
                let noise = (spread * standard_normal(&mut rng) - 0.5 * spread * spread).exp();
                counts[[gene, sample]] = (mean * noise).round();
            }
        }
        (counts, true_log2_fold_change)
    }

    fn correlation(left: &[f32], right: &[f32]) -> f32 {
        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let (left_mean, right_mean) = (mean(left), mean(right));
        let product: f32 = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - left_mean) * (b - right_mean))
            .sum();
        let spread = |values: &[f32], centre: f32| {
            values
                .iter()
                .map(|value| (value - centre).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        product / (spread(left, left_mean) * spread(right, right_mean))
    }

    fn max_deviation_from(fit: &GlmFit, expected: &[[f64; 2]; 6]) -> f64 {
        fit.coefficients
            .rows()
            .into_iter()
            .zip(expected)
            .flat_map(|(row, truth)| {
                [
                    (f64::from(row[0]) - truth[0]).abs(),
                    (f64::from(row[1]) - truth[1]).abs(),
                ]
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn matches_maximum_likelihood_in_the_poisson_limit() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = linear_space(0.7, 1.4, 10);

        let fit = fit(&counts, &design, &size_factors, 1e-8, &Device::Cpu);

        assert!(fit.converged.iter().all(|converged| *converged));
        let deviation = max_deviation_from(&fit, &MAXIMUM_LIKELIHOOD_POISSON_LIMIT);
        assert!(deviation < 1e-4, "max deviation {deviation}");
    }

    #[test]
    fn matches_maximum_likelihood_at_dispersion_025() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = linear_space(0.7, 1.4, 10);

        let fit = fit(&counts, &design, &size_factors, 0.25, &Device::Cpu);

        assert!(fit.converged.iter().all(|converged| *converged));
        let deviation = max_deviation_from(&fit, &MAXIMUM_LIKELIHOOD_DISPERSION_025);
        assert!(deviation < 1e-4, "max deviation {deviation}");
    }

    #[test]
    fn fitted_means_agree_with_the_coefficients() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = linear_space(0.7, 1.4, 10);

        let fit = fit(&counts, &design, &size_factors, 0.2, &Device::Cpu);

        for gene in 0..counts.nrows() {
            for sample in 0..counts.ncols() {
                let linear_predictor: f32 = (0..design.ncols())
                    .map(|k| fit.coefficients[[gene, k]] * design[[sample, k]])
                    .sum();
                let expected = size_factors[sample] * linear_predictor.exp();
                let relative = (fit.fitted_means[[gene, sample]] - expected).abs() / expected;
                assert!(relative < 1e-5, "gene {gene} sample {sample}: {relative}");
            }
        }
    }

    #[test]
    fn recovers_planted_fold_changes() {
        let dispersion = 0.1;
        let (counts, truth) = simulated_dataset(300, 8, dispersion, 4);
        let design = two_group_design(8);
        let size_factors = vec![1.0; design.nrows()];

        let fit = fit(&counts, &design, &size_factors, dispersion, &Device::Cpu);

        let estimated: Vec<f32> = fit
            .coefficients
            .column(1)
            .iter()
            .map(|beta| beta / std::f32::consts::LN_2)
            .collect();
        let correlation = correlation(&estimated, &truth);
        assert!(fit.converged.iter().all(|converged| *converged));
        assert!(correlation > 0.95, "correlation {correlation}");
    }

    #[test]
    fn size_factors_enter_the_model_as_an_offset() {
        // Counts are set to their exact means, so the maximum likelihood
        // solution is the generating coefficient vector however the samples are
        // scaled; a deviation is the offset mishandled, not noise.
        let design = two_group_design(4);
        let truth = array![[4.0f32, 1.5], [2.0, -0.75], [6.0, 0.0]];
        let size_factors = linear_space(0.6, 1.5, design.nrows());
        let counts = Array2::from_shape_fn((truth.nrows(), design.nrows()), |(gene, sample)| {
            let linear_predictor: f32 = (0..design.ncols())
                .map(|k| truth[[gene, k]] * design[[sample, k]])
                .sum();
            size_factors[sample] * linear_predictor.exp()
        });

        let mut scaled_counts = counts.clone();
        let mut scaled_size_factors = size_factors.clone();
        scaled_counts
            .column_mut(0)
            .map_inplace(|count| *count *= 4.0);
        scaled_size_factors[0] *= 4.0;

        let baseline = fit(&counts, &design, &size_factors, 0.2, &Device::Cpu);
        let rescaled = fit(
            &scaled_counts,
            &design,
            &scaled_size_factors,
            0.2,
            &Device::Cpu,
        );

        for gene in 0..truth.nrows() {
            for k in 0..design.ncols() {
                let from_truth = (baseline.coefficients[[gene, k]] - truth[[gene, k]]).abs();
                let from_baseline =
                    (rescaled.coefficients[[gene, k]] - baseline.coefficients[[gene, k]]).abs();
                assert!(
                    from_truth < 2e-3,
                    "gene {gene} coefficient {k}: {from_truth}"
                );
                assert!(
                    from_baseline < 2e-3,
                    "gene {gene} coefficient {k}: {from_baseline}"
                );
            }
        }
    }

    #[test]
    fn standard_errors_shrink_with_duplicated_samples() {
        let (counts, _) = simulated_dataset(40, 6, 0.2, 7);
        let design = two_group_design(6);
        let size_factors = vec![1.0; design.nrows()];
        let doubled_counts = concatenate(Axis(1), &[counts.view(), counts.view()]).unwrap();
        let doubled_design = concatenate(Axis(0), &[design.view(), design.view()]).unwrap();
        let doubled_size_factors = vec![1.0; 2 * design.nrows()];

        let single = fit(&counts, &design, &size_factors, 0.2, &Device::Cpu);
        let doubled = fit(
            &doubled_counts,
            &doubled_design,
            &doubled_size_factors,
            0.2,
            &Device::Cpu,
        );

        for gene in 0..counts.nrows() {
            for k in 0..design.ncols() {
                let ratio =
                    (doubled.covariance[[gene, k, k]] / single.covariance[[gene, k, k]]).sqrt();
                assert!(
                    (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3,
                    "gene {gene} coefficient {k}: ratio {ratio}"
                );
                assert!(
                    (doubled.coefficients[[gene, k]] - single.coefficients[[gene, k]]).abs() < 1e-3
                );
            }
        }
    }

    #[test]
    fn covariance_inverts_the_information_matrix() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = linear_space(0.7, 1.4, 10);
        let dispersion = 0.15;
        let p = design.ncols();

        let fit = fit(&counts, &design, &size_factors, dispersion, &Device::Cpu);

        for gene in 0..counts.nrows() {
            let mut information = Array2::<f32>::zeros((p, p));
            for sample in 0..design.nrows() {
                let mean = fit.fitted_means[[gene, sample]];
                let weight = mean / (1.0 + dispersion * mean);
                for i in 0..p {
                    for j in 0..p {
                        information[[i, j]] += weight * design[[sample, i]] * design[[sample, j]];
                    }
                }
            }
            for i in 0..p {
                for j in 0..p {
                    let product: f32 = (0..p)
                        .map(|k| fit.covariance[[gene, i, k]] * information[[k, j]])
                        .sum();
                    let expected = f32::from(i == j);
                    assert!((product - expected).abs() < 1e-3, "{product} != {expected}");
                }
            }
        }
    }

    #[test]
    fn an_all_zero_gene_does_not_poison_the_batch() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = vec![1.0; design.nrows()];

        let healthy = fit(&counts, &design, &size_factors, 0.2, &Device::Cpu);
        let mut with_zero_gene = counts.clone();
        with_zero_gene.row_mut(0).fill(0.0);
        let degenerate = fit(&with_zero_gene, &design, &size_factors, 0.2, &Device::Cpu);

        // An all-zero gene carries no information about its mean: the clamps
        // leave it at a finite estimate, or it is flagged. Never NaN, and never
        // at the expense of its neighbours in the batch.
        assert!(degenerate.coefficients.iter().all(|v| v.is_finite()));
        assert!(degenerate.covariance.iter().all(|v| v.is_finite()));
        assert!(degenerate.fitted_means.iter().all(|v| v.is_finite()));
        for gene in 1..counts.nrows() {
            assert!(degenerate.converged[gene]);
            for k in 0..design.ncols() {
                assert!(
                    (degenerate.coefficients[[gene, k]] - healthy.coefficients[[gene, k]]).abs()
                        < 1e-5
                );
            }
        }
    }

    #[test]
    fn reports_non_convergence_instead_of_failing() {
        let counts = reference_counts();
        let design = two_group_design(5);
        let size_factors = vec![1.0; design.nrows()];

        let fit = fit_negative_binomial(
            &counts,
            &design,
            &size_factors,
            &vec![0.2; counts.nrows()],
            1,
            TOLERANCE,
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(fit.n_iterations, 1);
        assert!(!fit.converged.iter().any(|converged| *converged));
        assert!(fit.coefficients.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        let design = two_group_design(3);
        let counts = Array2::<f32>::ones((4, design.nrows()));
        let ones = vec![1.0f32; design.nrows()];
        let call = |size_factors: &[f32], dispersions: &[f32]| {
            fit_negative_binomial(
                &counts,
                &design,
                size_factors,
                dispersions,
                10,
                TOLERANCE,
                &Device::Cpu,
            )
        };

        assert!(call(&ones[..design.nrows() - 1], &[0.2; 4]).is_err());
        assert!(call(&ones, &[0.2; 3]).is_err());
        assert!(call(&vec![0.0; design.nrows()], &[0.2; 4]).is_err());
    }

    #[test]
    fn handles_a_design_with_more_than_two_coefficients() {
        // Four coefficients exercise the unrolled Cholesky beyond the 2x2 case.
        let n_per_group = 6;
        let (counts, _) = simulated_dataset(50, n_per_group, 0.15, 21);
        let n_samples = 2 * n_per_group;
        let design = Array2::from_shape_fn((n_samples, 4), |(sample, column)| match column {
            0 => 1.0,
            1 => f32::from(sample >= n_per_group),
            2 => f32::from(sample % 3 == 1),
            _ => f32::from(sample % 3 == 2),
        });
        let size_factors = linear_space(0.8, 1.2, n_samples);

        let fit = fit(&counts, &design, &size_factors, 0.15, &Device::Cpu);

        assert!(fit.converged.iter().all(|converged| *converged));
        assert!(fit.coefficients.iter().all(|v| v.is_finite()));
        assert!(fit.covariance.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cpu_and_gpu_agree() {
        if !gpu_available() {
            return;
        }
        let (counts, _) = simulated_dataset(120, 6, 0.2, 13);
        let design = two_group_design(6);
        let size_factors = linear_space(0.8, 1.2, design.nrows());

        let on_cpu = fit(&counts, &design, &size_factors, 0.2, &Device::Cpu);
        let on_gpu = fit(&counts, &design, &size_factors, 0.2, &gpu());

        let deviation = on_cpu
            .coefficients
            .iter()
            .zip(on_gpu.coefficients.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(deviation < 2e-3, "max coefficient deviation {deviation}");
        for (cpu, metal) in on_cpu.covariance.iter().zip(on_gpu.covariance.iter()) {
            assert!((cpu - metal).abs() <= 5e-3 * cpu.abs().max(1e-6));
        }
    }

    /// Not an assertion: prints the fit times quoted in the branch summary.
    #[test]
    #[ignore = "benchmark"]
    fn benchmark_twenty_thousand_genes() {
        let (counts, _) = simulated_dataset(20_000, 6, 0.2, 99);
        let design = two_group_design(6);
        let size_factors = vec![1.0; design.nrows()];
        let mut devices = vec![("cpu", Device::Cpu)];
        if gpu_available() {
            devices.push(("gpu", gpu()));
        }
        for (name, device) in devices {
            let warm = fit(&counts, &design, &size_factors, 0.2, &device);
            let started = std::time::Instant::now();
            let fit = fit(&counts, &design, &size_factors, 0.2, &device);
            println!(
                "{name}: 20000 genes in {:?} ({} iterations, {}/{} converged)",
                started.elapsed(),
                fit.n_iterations,
                fit.converged.iter().filter(|c| **c).count(),
                warm.converged.len()
            );
        }
    }
}
