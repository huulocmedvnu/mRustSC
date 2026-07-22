use candle_core::{Device, Tensor};
use ndarray::Array2;

use crate::error::{Error, Result};

/// Upper clamp on a dispersion.
///
/// A gene with alpha = 1000 already has a variance a thousand times its squared
/// mean; larger values are indistinguishable in likelihood terms but make the
/// GLM's weights `1 / (1/mu + alpha)` underflow, so they are capped.
const MAXIMUM_DISPERSION: f32 = 1.0e3;

/// Lower clamp: a dispersion must stay strictly positive for the GLM's weights
/// and log-likelihood to be defined.
const MINIMUM_DISPERSION: f32 = 1.0e-8;

/// Guard against dividing by the mean of an all-zero gene.
const MINIMUM_MEAN: f32 = 1.0e-8;

/// Per-sample scaling factors by the median-of-ratios method, normalised to a
/// geometric mean of one.
///
/// `counts` is genes by samples, the pseudobulk layout the GLM is fitted in.
pub fn size_factors_median_of_ratios(counts: &Array2<f32>) -> Result<Vec<f32>> {
    // One memory-bound pass over the matrix, so a device round trip would cost
    // more than the arithmetic it replaces: this stays on the CPU.
    validate_counts(counts)?;
    let n_samples = counts.dim().1;

    // Zero counts collapse the geometric-mean reference, so the reference is
    // built only from genes seen in every sample.
    let mut log_ratios = vec![Vec::new(); n_samples];
    for gene in counts.rows() {
        if gene.iter().any(|&count| count <= 0.0) {
            continue;
        }
        // Ratios become differences of logarithms, so a deep library cannot
        // overflow the product of its counts.
        let logarithms: Vec<f32> = gene.iter().map(|count| count.ln()).collect();
        let reference = logarithms.iter().sum::<f32>() / n_samples as f32;
        for (sample, logarithm) in logarithms.iter().enumerate() {
            log_ratios[sample].push(logarithm - reference);
        }
    }
    if log_ratios[0].is_empty() {
        return Err(Error::parameter(
            "counts",
            "hold at least one gene with a non-zero count in every sample",
            "no gene qualified",
        ));
    }

    let factors = log_ratios
        .iter_mut()
        .map(|ratios| median(ratios).exp())
        .collect();
    normalise_to_unit_geometric_mean(factors)
}

/// Per-gene negative binomial dispersions by the method of moments, taking the
/// design's residual degrees of freedom into account.
///
/// `counts` is genes by samples and `design` samples by coefficients. The sample
/// variance is taken about the least-squares fit of the design rather than about
/// the grand mean, so a gene that differs between conditions is not charged for
/// that difference.
pub fn dispersions_method_of_moments(
    counts: &Array2<f32>,
    size_factors: &[f32],
    design: &Array2<f32>,
    device: &Device,
) -> Result<Vec<f32>> {
    validate_counts(counts)?;
    let (n_genes, n_samples) = counts.dim();
    let (design_rows, n_coefficients) = design.dim();
    if size_factors.len() != n_samples {
        return Err(Error::shape(
            format!("{n_samples} size factors"),
            format!("{}", size_factors.len()),
        ));
    }
    if design_rows != n_samples {
        return Err(Error::shape(
            format!("a design with {n_samples} rows"),
            format!("{design_rows}"),
        ));
    }
    if size_factors.iter().any(|&f| !f.is_finite() || f <= 0.0) {
        return Err(Error::parameter(
            "size_factors",
            "finite and strictly positive",
            "a non-positive or non-finite factor",
        ));
    }

    let raw = Tensor::from_iter(counts.iter().copied(), device)?.reshape((n_genes, n_samples))?;
    let normalised =
        raw.broadcast_div(&Tensor::from_slice(size_factors, (1, n_samples), device)?)?;

    let projection = Tensor::from_vec(
        least_squares_projection(design)?,
        (n_samples, n_samples),
        device,
    )?;
    let residuals = (&normalised - normalised.matmul(&projection)?)?;

    // A saturated design leaves no residual information; one degree of freedom
    // keeps the estimate finite and, with zero residuals, minimal.
    let degrees_of_freedom = n_samples.saturating_sub(n_coefficients).max(1) as f64;
    let residual_variance = (residuals.sqr()?.sum(1)? / degrees_of_freedom)?;

    let mean = normalised.mean(1)?;
    let safe_mean = mean.maximum(MINIMUM_MEAN)?;
    let dispersion = ((residual_variance - &mean)? / (&safe_mean * &safe_mean)?)?;

    Ok(dispersion
        .clamp(MINIMUM_DISPERSION, MAXIMUM_DISPERSION)?
        .to_vec1::<f32>()?)
}

/// Shrink gene-wise dispersions towards a fitted mean-dispersion trend.
///
/// `shrinkage_weight` of 0 returns the input, 1 returns the trend.
pub fn shrink_towards_trend(
    dispersions: &[f32],
    means: &[f32],
    shrinkage_weight: f32,
) -> Result<Vec<f32>> {
    // A handful of reductions over two vectors: CPU work, like the size factors.
    if dispersions.len() != means.len() {
        return Err(Error::shape(
            format!("{} means", dispersions.len()),
            format!("{}", means.len()),
        ));
    }
    if !(0.0..=1.0).contains(&shrinkage_weight) {
        return Err(Error::parameter(
            "shrinkage_weight",
            "in [0, 1]",
            shrinkage_weight,
        ));
    }
    if dispersions.iter().any(|&d| !d.is_finite() || d <= 0.0) {
        return Err(Error::parameter(
            "dispersions",
            "finite and strictly positive",
            "a non-positive or non-finite dispersion",
        ));
    }
    if shrinkage_weight == 0.0 {
        return Ok(dispersions.to_vec());
    }

    let trend = fit_trend(dispersions, means);
    // Geometric blend: dispersions are positive and roughly log-normal, so
    // interpolating in log space keeps the result positive and unbiased.
    Ok(dispersions
        .iter()
        .zip(&trend)
        .map(|(&dispersion, &trend)| {
            ((1.0 - shrinkage_weight) * dispersion.ln() + shrinkage_weight * trend.ln()).exp()
        })
        .collect())
}

/// Least-squares line of `log(dispersion)` on `log(mean)`, evaluated at every mean.
///
/// Preferred over the DESeq2 asymptotic form `a / mu + b` because it is a single
/// unconstrained linear fit — no positivity constraint, no iteration — and it
/// captures the same monotone decay of dispersion with expression.
fn fit_trend(dispersions: &[f32], means: &[f32]) -> Vec<f32> {
    let usable: Vec<usize> = (0..means.len())
        .filter(|&index| means[index] > MINIMUM_MEAN)
        .collect();
    if usable.len() < 2 {
        return dispersions.to_vec(); // nothing to fit; fall back to the gene-wise values
    }

    let log_means: Vec<f32> = usable.iter().map(|&index| means[index].ln()).collect();
    let log_dispersions: Vec<f32> = usable.iter().map(|&i| dispersions[i].ln()).collect();
    let mean_of_log_means = log_means.iter().sum::<f32>() / log_means.len() as f32;
    let mean_of_log_dispersions =
        log_dispersions.iter().sum::<f32>() / log_dispersions.len() as f32;

    let spread: f32 = log_means
        .iter()
        .map(|x| (x - mean_of_log_means) * (x - mean_of_log_means))
        .sum();
    if spread == 0.0 {
        // Every usable gene sits at the same expression; the fit would be
        // rank-deficient, so the trend is their geometric mean.
        return vec![mean_of_log_dispersions.exp(); dispersions.len()];
    }

    let covariance: f32 = log_means
        .iter()
        .zip(&log_dispersions)
        .map(|(x, y)| (x - mean_of_log_means) * (y - mean_of_log_dispersions))
        .sum();
    let slope = covariance / spread;
    let intercept = mean_of_log_dispersions - slope * mean_of_log_means;

    let floor = dispersions.iter().copied().fold(f32::INFINITY, f32::min);
    means
        .iter()
        .map(|mean| {
            let trend = (intercept + slope * mean.max(MINIMUM_MEAN).ln()).exp();
            trend.clamp(floor, MAXIMUM_DISPERSION)
        })
        .collect()
}

/// The hat matrix `X (X'X)^-1 X'`, in row-major order.
///
/// Forming it once turns the per-gene least-squares solve into a single batched
/// matmul, which is the shape the device is good at. It is small — samples by
/// samples — so it is built on the CPU in `f64` for the inverse's sake.
fn least_squares_projection(design: &Array2<f32>) -> Result<Vec<f32>> {
    let (n_samples, n_coefficients) = design.dim();
    let at = |row: usize, column: usize| f64::from(design[[row, column]]);

    let mut gram = vec![0.0; n_coefficients * n_coefficients];
    for i in 0..n_coefficients {
        for j in 0..n_coefficients {
            gram[i * n_coefficients + j] = (0..n_samples).map(|s| at(s, i) * at(s, j)).sum();
        }
    }
    let inverse = invert(&mut gram, n_coefficients)?;

    let mut projection = vec![0.0; n_samples * n_samples];
    for row in 0..n_samples {
        for column in 0..n_samples {
            let entry: f64 = (0..n_coefficients)
                .map(|i| {
                    at(row, i)
                        * (0..n_coefficients)
                            .map(|j| inverse[i * n_coefficients + j] * at(column, j))
                            .sum::<f64>()
                })
                .sum();
            projection[row * n_samples + column] = entry as f32;
        }
    }
    Ok(projection)
}

/// Gauss-Jordan inverse of a small square matrix, with partial pivoting.
fn invert(matrix: &mut [f64], order: usize) -> Result<Vec<f64>> {
    let mut inverse = vec![0.0; order * order];
    for i in 0..order {
        inverse[i * order + i] = 1.0;
    }

    for column in 0..order {
        let pivot = (column..order)
            .max_by(|&a, &b| {
                matrix[a * order + column]
                    .abs()
                    .total_cmp(&matrix[b * order + column].abs())
            })
            .unwrap_or(column);
        if matrix[pivot * order + column].abs() < 1e-12 {
            return Err(Error::parameter(
                "design",
                "of full column rank",
                "a singular X'X",
            ));
        }
        matrix.swap_rows(column, pivot, order);
        inverse.swap_rows(column, pivot, order);

        let scale = matrix[column * order + column];
        for j in 0..order {
            matrix[column * order + j] /= scale;
            inverse[column * order + j] /= scale;
        }
        for row in 0..order {
            if row == column {
                continue;
            }
            let factor = matrix[row * order + column];
            if factor == 0.0 {
                continue;
            }
            for j in 0..order {
                matrix[row * order + j] -= factor * matrix[column * order + j];
                inverse[row * order + j] -= factor * inverse[column * order + j];
            }
        }
    }
    Ok(inverse)
}

/// Row swapping on a flat row-major matrix, so both matrices swap the same way.
trait SwapRows {
    fn swap_rows(&mut self, a: usize, b: usize, width: usize);
}

impl SwapRows for [f64] {
    fn swap_rows(&mut self, a: usize, b: usize, width: usize) {
        if a == b {
            return;
        }
        for column in 0..width {
            self.swap(a * width + column, b * width + column);
        }
    }
}

fn validate_counts(counts: &Array2<f32>) -> Result<()> {
    if counts.is_empty() {
        return Err(Error::parameter(
            "counts",
            "hold at least one gene and one sample",
            format!("{:?}", counts.dim()),
        ));
    }
    if counts
        .iter()
        .any(|&count| !count.is_finite() || count < 0.0)
    {
        return Err(Error::parameter(
            "counts",
            "finite and non-negative",
            "a negative or non-finite count",
        ));
    }
    Ok(())
}

/// The median of a slice, averaging the two central values on an even length,
/// as numpy does. Reorders `values`.
fn median(values: &mut [f32]) -> f32 {
    let middle = values.len() / 2;
    let (_, upper, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
    let upper = *upper;
    if values.len() % 2 == 1 {
        return upper;
    }
    let lower = values[..middle]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    0.5 * (lower + upper)
}

/// Rescale strictly positive factors so their geometric mean is exactly one.
///
/// The contract fixes the scale of size factors but not how they are derived, so
/// the constraint belongs in one place.
fn normalise_to_unit_geometric_mean(factors: Vec<f32>) -> Result<Vec<f32>> {
    if factors.iter().any(|&f| !f.is_finite() || f <= 0.0) {
        return Err(Error::parameter(
            "size factors",
            "finite and strictly positive",
            "a non-positive or non-finite factor",
        ));
    }
    let log_mean = factors.iter().map(|f| f.ln()).sum::<f32>() / factors.len() as f32;
    let geometric_mean = log_mean.exp();
    Ok(factors.iter().map(|f| f / geometric_mean).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use rand::{Rng, SeedableRng};

    /// Gamma-Poisson draws: the tests need a null and a known-dispersion sample,
    /// not a distribution library.
    struct NegativeBinomial {
        rng: rand::rngs::StdRng,
    }

    impl NegativeBinomial {
        fn new(seed: u64) -> Self {
            Self {
                rng: rand::rngs::StdRng::seed_from_u64(seed),
            }
        }

        fn normal(&mut self) -> f64 {
            let radius = (-2.0 * self.rng.gen::<f64>().max(f64::MIN_POSITIVE).ln()).sqrt();
            radius * (std::f64::consts::TAU * self.rng.gen::<f64>()).cos()
        }

        /// Marsaglia-Tsang, valid for shape >= 1.
        fn gamma(&mut self, shape: f64) -> f64 {
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let x = self.normal();
                let v = (1.0 + c * x).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u: f64 = self.rng.gen();
                if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                    return d * v;
                }
            }
        }

        /// Knuth's product method; the means used here keep `exp(-lambda)` normal.
        fn poisson(&mut self, lambda: f64) -> f64 {
            let limit = (-lambda).exp();
            let mut product: f64 = 1.0;
            let mut count = 0.0;
            loop {
                product *= self.rng.gen::<f64>();
                if product <= limit {
                    return count;
                }
                count += 1.0;
            }
        }

        fn sample(&mut self, mean: f64, dispersion: f64) -> f32 {
            if dispersion <= 0.0 {
                return self.poisson(mean) as f32;
            }
            let shape = 1.0 / dispersion;
            let rate = self.gamma(shape) * mean * dispersion;
            self.poisson(rate) as f32
        }
    }

    fn two_group_design(per_group: usize) -> Array2<f32> {
        Array2::from_shape_fn((2 * per_group, 2), |(row, column)| {
            if column == 0 {
                1.0
            } else if row < per_group {
                0.0
            } else {
                1.0
            }
        })
    }

    fn simulate(n_genes: usize, n_samples: usize, mean: f64, dispersion: f64) -> Array2<f32> {
        let mut generator = NegativeBinomial::new(11);
        Array2::from_shape_fn((n_genes, n_samples), |_| generator.sample(mean, dispersion))
    }

    #[test]
    fn identical_libraries_give_all_ones() {
        let counts = Array2::from_shape_fn((50, 4), |(gene, _)| (gene + 1) as f32);
        for factor in size_factors_median_of_ratios(&counts).unwrap() {
            assert!((factor - 1.0).abs() < 1e-6, "{factor}");
        }
    }

    #[test]
    fn scaling_a_sample_scales_its_factor() {
        let counts =
            Array2::from_shape_fn((60, 3), |(gene, sample)| (10 + gene + 3 * sample) as f32);
        let baseline = size_factors_median_of_ratios(&counts).unwrap();

        let mut scaled = counts.clone();
        scaled.column_mut(1).map_inplace(|count| *count *= 4.0);
        let after = size_factors_median_of_ratios(&scaled).unwrap();

        // Scaling sample 1 by 4 multiplies its raw factor by 4; renormalising to
        // a unit geometric mean divides every factor by 4^(1/3).
        let renormalisation = 4.0_f32.powf(1.0 / 3.0);
        assert!((after[1] - baseline[1] * 4.0 / renormalisation).abs() < 1e-4);
        assert!((after[0] - baseline[0] / renormalisation).abs() < 1e-4);
    }

    #[test]
    fn the_geometric_mean_of_the_factors_is_one() {
        let counts = simulate(200, 5, 40.0, 0.3);
        let factors = size_factors_median_of_ratios(&counts).unwrap();

        let log_mean = factors.iter().map(|f| f.ln()).sum::<f32>() / factors.len() as f32;
        assert!(log_mean.abs() < 1e-5, "{log_mean}");
    }

    #[test]
    fn no_usable_gene_is_an_error() {
        let counts = Array2::from_shape_fn(
            (10, 3),
            |(gene, sample)| {
                if gene % 3 == sample {
                    0.0
                } else {
                    5.0
                }
            },
        );
        assert!(size_factors_median_of_ratios(&counts).is_err());
        assert!(size_factors_median_of_ratios(&Array2::zeros((0, 0))).is_err());
    }

    #[test]
    fn recovers_a_known_dispersion() {
        let truth = 0.2;
        let counts = simulate(400, 8, 50.0, truth);
        let design = two_group_design(4);

        let mut estimates =
            dispersions_method_of_moments(&counts, &[1.0; 8], &design, &Device::Cpu).unwrap();
        let estimate = median(&mut estimates);

        assert!(
            estimate > (truth / 1.5) as f32 && estimate < (truth * 1.5) as f32,
            "median dispersion {estimate}, truth {truth}"
        );
    }

    #[test]
    fn poisson_data_collapses_to_the_floor() {
        let counts = simulate(300, 6, 30.0, 0.0);
        let design = two_group_design(3);

        let estimates =
            dispersions_method_of_moments(&counts, &[1.0; 6], &design, &Device::Cpu).unwrap();

        assert!(estimates.iter().all(|&d| d > 0.0 && d.is_finite()));
        let at_floor = estimates.iter().filter(|&&d| d <= 1e-7).count();
        assert!(
            at_floor > estimates.len() / 2,
            "only {at_floor} at the floor"
        );
    }

    #[test]
    fn the_variance_is_taken_about_the_design_fit() {
        // Two groups with very different means but no within-group variation:
        // about the grand mean this looks overdispersed, about the fit it is not.
        let counts = Array2::from_shape_vec((1, 4), vec![10.0, 10.0, 100.0, 100.0]).unwrap();
        let design = two_group_design(2);

        let with_group =
            dispersions_method_of_moments(&counts, &[1.0; 4], &design, &Device::Cpu).unwrap();
        let intercept_only =
            dispersions_method_of_moments(&counts, &[1.0; 4], &Array2::ones((4, 1)), &Device::Cpu)
                .unwrap();

        assert!((with_group[0] - MINIMUM_DISPERSION).abs() < 1e-9);
        assert!(intercept_only[0] > 0.5);
    }

    #[test]
    fn dispersion_rejects_mismatched_inputs() {
        let counts = Array2::from_elem((5, 4), 3.0);
        let design = two_group_design(2);
        assert!(dispersions_method_of_moments(&counts, &[1.0; 3], &design, &Device::Cpu).is_err());
        assert!(dispersions_method_of_moments(
            &counts,
            &[1.0; 4],
            &two_group_design(3),
            &Device::Cpu
        )
        .is_err());
        assert!(dispersions_method_of_moments(
            &counts,
            &[1.0, 0.0, 1.0, 1.0],
            &design,
            &Device::Cpu
        )
        .is_err());
        // A duplicated column makes X'X singular.
        let rank_deficient =
            Array::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(
            dispersions_method_of_moments(&counts, &[1.0; 4], &rank_deficient, &Device::Cpu)
                .is_err()
        );
    }

    #[test]
    fn size_factors_enter_the_dispersion() {
        // Doubling one sample's counts and its size factor leaves the
        // normalised matrix, and therefore every dispersion, unchanged.
        let counts = simulate(50, 6, 40.0, 0.25);
        let design = two_group_design(3);
        let baseline =
            dispersions_method_of_moments(&counts, &[1.0; 6], &design, &Device::Cpu).unwrap();

        let mut doubled = counts.clone();
        doubled.column_mut(2).map_inplace(|count| *count *= 2.0);
        let factors = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        let after =
            dispersions_method_of_moments(&doubled, &factors, &design, &Device::Cpu).unwrap();

        for (a, b) in baseline.iter().zip(&after) {
            assert!((a - b).abs() <= 1e-4 * a.abs().max(1e-3), "{a} != {b}");
        }
    }

    fn trend_inputs() -> (Vec<f32>, Vec<f32>) {
        let means: Vec<f32> = (1..=40).map(|i| i as f32 * 5.0).collect();
        let dispersions: Vec<f32> = means
            .iter()
            .enumerate()
            .map(|(index, mean)| 2.0 / mean * if index % 2 == 0 { 1.6 } else { 0.6 })
            .collect();
        (dispersions, means)
    }

    #[test]
    fn zero_weight_reproduces_the_input() {
        let (dispersions, means) = trend_inputs();
        assert_eq!(
            shrink_towards_trend(&dispersions, &means, 0.0).unwrap(),
            dispersions
        );
    }

    #[test]
    fn full_weight_gives_the_pure_trend() {
        let (dispersions, means) = trend_inputs();
        let shrunk = shrink_towards_trend(&dispersions, &means, 1.0).unwrap();
        let trend = fit_trend(&dispersions, &means);

        for (actual, expected) in shrunk.iter().zip(&trend) {
            assert!(
                (actual - expected).abs() < 1e-5 * expected,
                "{actual} != {expected}"
            );
        }
        // The trend follows the planted 1/mean decay rather than the noise.
        for (value, mean) in trend.iter().zip(&means) {
            assert!((value * mean / 2.0 - 1.0).abs() < 0.15, "{value} at {mean}");
        }
    }

    #[test]
    fn intermediate_weights_lie_between() {
        let (dispersions, means) = trend_inputs();
        let trend = fit_trend(&dispersions, &means);
        let shrunk = shrink_towards_trend(&dispersions, &means, 0.5).unwrap();

        for ((value, &raw), &target) in shrunk.iter().zip(&dispersions).zip(&trend) {
            let (low, high) = (raw.min(target), raw.max(target));
            assert!(*value >= low - 1e-6 && *value <= high + 1e-6, "{value}");
        }
    }

    #[test]
    fn shrinkage_rejects_bad_arguments() {
        let (dispersions, means) = trend_inputs();
        assert!(shrink_towards_trend(&dispersions, &means[..2], 0.5).is_err());
        assert!(shrink_towards_trend(&dispersions, &means, 1.5).is_err());
        assert!(shrink_towards_trend(&[0.0, 1.0], &[1.0, 2.0], 0.5).is_err());
    }

    #[test]
    fn a_degenerate_trend_falls_back() {
        // One usable gene cannot define a line, and identical means cannot
        // define a slope.
        assert_eq!(fit_trend(&[0.3, 0.4], &[0.0, 5.0]), vec![0.3, 0.4]);
        let constant = fit_trend(&[0.25, 0.64], &[7.0, 7.0]);
        assert!((constant[0] - 0.4).abs() < 1e-5 && constant[0] == constant[1]);
    }
}
