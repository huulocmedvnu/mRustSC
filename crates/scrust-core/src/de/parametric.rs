//! t-test and logistic-regression differential expression. Owned by feat/de-methods.
//!
//! This mirrors scanpy's `rank_genes_groups` for `method="t-test"`,
//! `method="t-test_overestim_var"` and `method="logreg"`: the same Welch
//! statistic from the same unbiased moments, the same substituted sample size,
//! the same `expm1`-based fold change, the same per-group Benjamini-Hochberg
//! correction, and the same regularised multinomial objective sklearn minimises.

use std::collections::VecDeque;

use candle_core::{Device, Tensor};
use ndarray::{Array2, ArrayView1};

use crate::de::multiple_testing::benjamini_hochberg_f64;
use crate::de::wilcoxon::GroupComparison;
use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Welch's t-test of every group against the reference, as scanpy's
/// `method="t-test"`.
///
/// `group_labels` holds one group id per cell; `reference` of `None` compares
/// each group against all other cells. With a reference group, that group's row
/// carries the neutral result (score 0, p-value 1, fold change 0), because
/// scanpy leaves it out entirely and a group tested against itself has no
/// result to report.
///
/// The device is unused. The moments are a sparse column reduction and would be
/// a natural matmul against a one-hot group indicator, but the Apple GPU has no
/// `f64` and `f32` is not enough here: `p = 2 T.sf(|t|)` amplifies a relative
/// error in `t` by roughly `t` itself, so the ~1e-7 an `f32` reduction leaves in
/// a variance becomes ~1e-4 in the p-value of a real marker gene — two orders
/// past the tolerance the contract holds p-values to. See the branch report.
pub fn t_test(
    matrix: &CsrMatrix,
    group_labels: &[u32],
    n_groups: usize,
    reference: Option<u32>,
    _device: &Device,
) -> Result<GroupComparison> {
    welch_t_tests(
        matrix,
        group_labels,
        n_groups,
        reference,
        ReferenceSize::Observed,
    )
}

/// scanpy's `method="t-test_overestim_var"`, which uses the group size in place
/// of the Welch degrees of freedom.
///
/// The device is unused, for the reason given on [`t_test`].
pub fn t_test_overestimated_variance(
    matrix: &CsrMatrix,
    group_labels: &[u32],
    n_groups: usize,
    reference: Option<u32>,
    _device: &Device,
) -> Result<GroupComparison> {
    welch_t_tests(
        matrix,
        group_labels,
        n_groups,
        reference,
        ReferenceSize::TestedGroup,
    )
}

/// The sample size attributed to the reference side of the test.
///
/// This is the *only* thing scanpy's two t-tests differ in. It reaches the
/// result twice over — through `var_reference / n_reference` in the standard
/// error and through the Welch degrees of freedom — so a small group borrowing
/// its own size both inflates the reference variance term and shrinks the
/// degrees of freedom, which is the conservatism the variant is named for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceSize {
    /// The number of cells actually on the reference side.
    Observed,
    /// The tested group's own size.
    TestedGroup,
}

fn welch_t_tests(
    matrix: &CsrMatrix,
    group_labels: &[u32],
    n_groups: usize,
    reference: Option<u32>,
    reference_size: ReferenceSize,
) -> Result<GroupComparison> {
    let n_cells = matrix.n_rows();
    let n_genes = matrix.n_cols();
    validate(group_labels, n_cells, n_groups, reference)?;

    let moments = GroupMoments::compute(matrix, group_labels, n_groups);
    let shape = (n_groups, n_genes);
    let mut scores = Array2::<f32>::zeros(shape);
    let mut p_values = Array2::<f64>::zeros(shape);
    let mut adjusted_p_values = Array2::<f64>::zeros(shape);
    let mut log2_fold_changes = Array2::<f32>::zeros(shape);

    for group in 0..n_groups {
        if reference == Some(group as u32) {
            for gene in 0..n_genes {
                p_values[[group, gene]] = 1.0;
                adjusted_p_values[[group, gene]] = 1.0;
            }
            continue;
        }

        let side = moments.reference_side(group, reference);
        let n_reference = match reference_size {
            ReferenceSize::Observed => side.count,
            ReferenceSize::TestedGroup => moments.sizes[group],
        };

        let mut group_p_values = vec![0f64; n_genes];
        for gene in 0..n_genes {
            let (statistic, p_value) = welch(
                Sample {
                    mean: moments.means[[group, gene]],
                    variance: moments.variances[[group, gene]],
                    count: moments.sizes[group],
                },
                Sample {
                    mean: side.means[gene],
                    variance: side.variances[gene],
                    count: n_reference,
                },
            );
            scores[[group, gene]] = statistic as f32;
            group_p_values[gene] = p_value;
            log2_fold_changes[[group, gene]] =
                log2_fold_change(moments.means[[group, gene]], side.means[gene]) as f32;
        }

        for (gene, adjusted) in benjamini_hochberg_f64(&group_p_values).into_iter().enumerate() {
            p_values[[group, gene]] = group_p_values[gene];
            adjusted_p_values[[group, gene]] = adjusted;
        }
    }

    Ok(GroupComparison {
        scores,
        p_values,
        adjusted_p_values,
        log2_fold_changes,
    })
}

fn validate(
    group_labels: &[u32],
    n_cells: usize,
    n_groups: usize,
    reference: Option<u32>,
) -> Result<()> {
    if group_labels.len() != n_cells {
        return Err(Error::shape(
            format!("{n_cells} group labels"),
            format!("{} group labels", group_labels.len()),
        ));
    }
    if n_groups == 0 {
        return Err(Error::parameter("n_groups", "at least 1", n_groups));
    }
    if let Some(&label) = group_labels.iter().find(|&&l| l as usize >= n_groups) {
        return Err(Error::parameter("group_labels", "below n_groups", label));
    }
    match reference {
        Some(reference) if reference as usize >= n_groups => {
            Err(Error::parameter("reference", "below n_groups", reference))
        }
        _ => Ok(()),
    }
}

/// One side of a two-sample test, for one gene.
#[derive(Debug, Clone, Copy)]
struct Sample {
    mean: f64,
    variance: f64,
    count: usize,
}

/// Welch's two-sample t statistic and its two-sided p-value.
///
/// This is `scipy.stats.ttest_ind_from_stats(equal_var=False)` including its
/// degenerate cases: a degrees-of-freedom expression that evaluates to NaN falls
/// back to 1, and scanpy then maps a NaN statistic to 0 and a NaN p-value to 1.
/// Those are not defensive guards — they decide the answer for every gene that
/// is constant in both groups, of which a single-cell matrix has many.
fn welch(group: Sample, reference: Sample) -> (f64, f64) {
    let (group_count, reference_count) = (group.count as f64, reference.count as f64);
    let group_term = group.variance / group_count;
    let reference_term = reference.variance / reference_count;

    let sum = group_term + reference_term;
    let degrees_of_freedom = {
        let value = sum * sum
            / (group_term * group_term / (group_count - 1.0)
                + reference_term * reference_term / (reference_count - 1.0));
        if value.is_nan() {
            1.0
        } else {
            value
        }
    };

    let statistic = (group.mean - reference.mean) / sum.sqrt();
    let p_value = two_sided_t_p(statistic.abs(), degrees_of_freedom);
    (
        if statistic.is_nan() { 0.0 } else { statistic },
        if p_value.is_nan() { 1.0 } else { p_value },
    )
}

/// scanpy's fold change: a ratio of `expm1`'d means, nudged off zero so an
/// unexpressed gene gives a ratio of 1 instead of a division by zero.
fn log2_fold_change(mean_group: f64, mean_reference: f64) -> f64 {
    ((mean_group.exp_m1() + 1e-9) / (mean_reference.exp_m1() + 1e-9)).log2()
}

// --- moments -----------------------------------------------------------------

/// Per-gene mean and unbiased variance of every group, and of every group's rest.
///
/// Both are accumulated in one pass over the stored entries, in `f64`. A zero
/// contributes nothing to either the sum or the sum of squares, so the implicit
/// zeros of a 90-95% sparse column never have to be visited — only counted, and
/// the count is the group size. The rest of a group is the complement of that
/// group within the labelled cells, so it is the totals minus the group's own
/// sums rather than a second pass.
struct GroupMoments {
    means: Array2<f64>,
    variances: Array2<f64>,
    rest_means: Array2<f64>,
    rest_variances: Array2<f64>,
    sizes: Vec<usize>,
}

/// The reference side of one group's test: the reference group, or the rest.
struct ReferenceStatistics<'a> {
    means: ArrayView1<'a, f64>,
    variances: ArrayView1<'a, f64>,
    count: usize,
}

impl GroupMoments {
    fn compute(matrix: &CsrMatrix, group_labels: &[u32], n_groups: usize) -> Self {
        let n_genes = matrix.n_cols();
        let n_cells = matrix.n_rows();
        let mut sizes = vec![0usize; n_groups];
        let mut sums = Array2::<f64>::zeros((n_groups, n_genes));
        let mut squares = Array2::<f64>::zeros((n_groups, n_genes));

        let indptr = matrix.indptr();
        for cell in 0..n_cells {
            let group = group_labels[cell] as usize;
            sizes[group] += 1;
            for entry in indptr[cell] as usize..indptr[cell + 1] as usize {
                let gene = matrix.indices()[entry] as usize;
                let value = matrix.values()[entry] as f64;
                sums[[group, gene]] += value;
                squares[[group, gene]] += value * value;
            }
        }

        let mut means = Array2::<f64>::zeros((n_groups, n_genes));
        let mut variances = Array2::<f64>::zeros((n_groups, n_genes));
        let mut rest_means = Array2::<f64>::zeros((n_groups, n_genes));
        let mut rest_variances = Array2::<f64>::zeros((n_groups, n_genes));
        for gene in 0..n_genes {
            let total_sum: f64 = (0..n_groups).map(|group| sums[[group, gene]]).sum();
            let total_squares: f64 = (0..n_groups).map(|group| squares[[group, gene]]).sum();
            for group in 0..n_groups {
                let (mean, variance) =
                    moments(sums[[group, gene]], squares[[group, gene]], sizes[group]);
                means[[group, gene]] = mean;
                variances[[group, gene]] = variance;
                let (rest_mean, rest_variance) = moments(
                    total_sum - sums[[group, gene]],
                    total_squares - squares[[group, gene]],
                    n_cells - sizes[group],
                );
                rest_means[[group, gene]] = rest_mean;
                rest_variances[[group, gene]] = rest_variance;
            }
        }

        Self {
            means,
            variances,
            rest_means,
            rest_variances,
            sizes,
        }
    }

    fn reference_side(&self, group: usize, reference: Option<u32>) -> ReferenceStatistics<'_> {
        let (row, count) = match reference {
            Some(reference) => (reference as usize, self.sizes[reference as usize]),
            None => (group, self.sizes.iter().sum::<usize>() - self.sizes[group]),
        };
        let (means, variances) = match reference {
            Some(_) => (&self.means, &self.variances),
            None => (&self.rest_means, &self.rest_variances),
        };
        ReferenceStatistics {
            means: means.row(row),
            variances: variances.row(row),
            count,
        }
    }
}

/// Mean and unbiased variance from a sum and a sum of squares.
///
/// `count == 1` leaves the variance uncorrected, which is what scanpy's
/// `mean_var(..., correction=1)` does — it skips the Bessel factor rather than
/// dividing by zero — and `count == 0` gives the NaN scanpy would get from
/// dividing by an empty group, which the test then maps to a neutral result.
fn moments(sum: f64, squares: f64, count: usize) -> (f64, f64) {
    let n = count as f64;
    let mean = sum / n;
    // Rounding can leave this a hair below zero for a gene that is constant
    // within the group; a negative variance would become a NaN standard error
    // and silently discard a gene that is simply uninformative.
    let centred = (squares - sum * sum / n).max(0.0);
    let variance = match count {
        0 => f64::NAN,
        1 => 0.0,
        _ => centred / (n - 1.0),
    };
    (mean, variance)
}

// --- Student's t distribution ------------------------------------------------

/// `2 * scipy.stats.t.sf(t, degrees_of_freedom)` for `t >= 0`.
///
/// `de::hypothesis` carries the normal tail the rank-sum test needs but no t
/// distribution, so this file adds one. Written as the regularised incomplete
/// beta rather than through a normal approximation because the degrees of
/// freedom are as low as 1 in the degenerate cases scipy falls back to.
fn two_sided_t_p(statistic: f64, degrees_of_freedom: f64) -> f64 {
    if statistic.is_nan() || degrees_of_freedom.is_nan() || degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }
    if statistic.is_infinite() {
        return 0.0;
    }
    let x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic);
    regularised_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)
}

/// The regularised incomplete beta `I_x(a, b)`.
///
/// The continued fraction converges quickly only on the side of the
/// distribution's mode, so the symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)` is used
/// to always evaluate the fast side.
fn regularised_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Lentz's evaluation of the continued fraction for the incomplete beta.
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_TERMS: usize = 300;
    const EPSILON: f64 = 3e-16;
    /// Guards against a zero denominator, as in Numerical Recipes.
    const TINY: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut fraction = d;

    for term in 1..=MAX_TERMS {
        let m = term as f64;
        let even = m * (b - m) * x / ((qam + 2.0 * m) * (a + 2.0 * m));
        let odd = -(a + m) * (qab + m) * x / ((a + 2.0 * m) * (qap + 2.0 * m));
        for numerator in [even, odd] {
            d = 1.0 + numerator * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + numerator / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            fraction *= d * c;
        }
        if (d * c - 1.0).abs() < EPSILON {
            break;
        }
    }
    fraction
}

/// `ln(gamma(x))` for `x > 0`, by the Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    /// The shift `g` the coefficients above were fitted for.
    const G: f64 = 7.0;

    let shifted = x - 1.0;
    let mut series = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (shifted + index as f64);
    }
    let t = shifted + G + 0.5;
    0.5 * (std::f64::consts::TAU).ln() + (shifted + 0.5) * t.ln() - t + series.ln()
}

// --- logistic regression -----------------------------------------------------

/// sklearn's default inverse regularisation strength, which scanpy does not
/// override.
const REGULARISATION: f64 = 1.0;

/// Rows densified at a time. The dense block is the only large temporary:
/// `ROW_BLOCK * n_genes * 4` bytes, about 110 MB for 13k genes.
const ROW_BLOCK: usize = 2048;

/// L-BFGS history length, matching scipy's `maxcor`.
const HISTORY: usize = 10;

/// Stop when no parameter's gradient exceeds this.
///
/// Two orders tighter than sklearn's default `tol`, and deliberately so: at
/// sklearn's tolerance the iterate still sits ~10% away from the optimum in the
/// weakly curved directions, so it is a property of scipy's line search rather
/// than of the model. The optimum is what is determined, so that is what we
/// return. The branch report measures the gap.
const GRADIENT_TOLERANCE: f64 = 1e-7;

/// Multinomial logistic regression coefficients as scores, as scanpy's
/// `method="logreg"`.
///
/// scanpy reports no p-values and no fold changes for this method — its `uns`
/// entry carries only `names` and `scores` — so those fields come back NaN and
/// the Python layer drops them rather than inventing a number.
///
/// This one does run on `device`: every iteration is a pair of matmuls between
/// a densified row block and the dense gene-by-class coefficient matrix, which
/// is exactly the shape the GPU wants, and the coefficients are not sensitive
/// to `f32` the way a p-value is.
pub fn logistic_regression(
    matrix: &CsrMatrix,
    group_labels: &[u32],
    n_groups: usize,
    max_iterations: usize,
    device: &Device,
) -> Result<GroupComparison> {
    let n_cells = matrix.n_rows();
    let n_genes = matrix.n_cols();
    validate(group_labels, n_cells, n_groups, None)?;
    if n_groups < 2 {
        return Err(Error::parameter("n_groups", "at least 2 for logreg", n_groups));
    }
    if max_iterations == 0 {
        return Err(Error::parameter("max_iterations", "at least 1", max_iterations));
    }
    let mut sizes = vec![0usize; n_groups];
    for &label in group_labels {
        sizes[label as usize] += 1;
    }
    if let Some(empty) = sizes.iter().position(|&size| size == 0) {
        return Err(Error::parameter("group_labels", "no empty group", empty));
    }

    // With two groups sklearn fits the binary logistic loss, whose single
    // coefficient vector is not the difference of two multinomial rows: the
    // penalty sees one vector instead of two, so the optimum differs. scanpy
    // then reports that one vector for both groups.
    let n_outputs = if n_groups == 2 { 1 } else { n_groups };
    let objective = MultinomialObjective {
        matrix,
        group_labels,
        n_outputs,
        device,
    };
    let parameters = minimise(&objective, (n_genes + 1) * n_outputs, max_iterations)?;

    let mut scores = Array2::<f32>::zeros((n_groups, n_genes));
    for group in 0..n_groups {
        let output = if n_outputs == 1 { 0 } else { group };
        for gene in 0..n_genes {
            scores[[group, gene]] = parameters[gene * n_outputs + output] as f32;
        }
    }
    let undefined = Array2::from_elem((n_groups, n_genes), f64::NAN);
    Ok(GroupComparison {
        scores,
        p_values: undefined.clone(),
        adjusted_p_values: undefined,
        log2_fold_changes: Array2::from_elem((n_groups, n_genes), f32::NAN),
    })
}

/// sklearn's penalised multinomial (or binary) logistic objective.
///
/// The value is `mean(pointwise loss) + ||W||^2 / (2 C n)`, which is the
/// scaling sklearn's lbfgs solver minimises. Same minimiser as the
/// `C * sum(loss) + penalty` form scanpy's documentation describes, but the
/// gradient is on the same scale as sklearn's, so a gradient tolerance means
/// the same thing in both.
struct MultinomialObjective<'a> {
    matrix: &'a CsrMatrix,
    group_labels: &'a [u32],
    n_outputs: usize,
    device: &'a Device,
}

impl MultinomialObjective<'_> {
    /// `(value, gradient)` at `parameters`, laid out as the row-major
    /// `(n_genes, n_outputs)` coefficients followed by the `n_outputs`
    /// intercepts.
    fn evaluate(&self, parameters: &[f64]) -> Result<(f64, Vec<f64>)> {
        let n_genes = self.matrix.n_cols();
        let n_cells = self.matrix.n_rows();
        let n_outputs = self.n_outputs;
        let (coefficients, intercepts) = parameters.split_at(n_genes * n_outputs);

        let weights = Tensor::from_vec(
            coefficients.iter().map(|&w| w as f32).collect::<Vec<f32>>(),
            (n_genes, n_outputs),
            self.device,
        )?;
        let bias = Tensor::from_vec(
            intercepts.iter().map(|&b| b as f32).collect::<Vec<f32>>(),
            (1, n_outputs),
            self.device,
        )?;

        let mut loss = 0f64;
        let mut coefficient_gradient: Option<Tensor> = None;
        let mut intercept_gradient = vec![0f64; n_outputs];

        for start in (0..n_cells).step_by(ROW_BLOCK) {
            let end = (start + ROW_BLOCK).min(n_cells);
            // Densified once and used for both matmuls of this block: the
            // forward pass and the transposed gradient pass.
            let block = self.matrix.to_tensor_rows(start, end, self.device)?;
            let raw = block.matmul(&weights)?.broadcast_add(&bias)?;

            // The nonlinearity is (rows x n_outputs) — thousands of times
            // smaller than the block — so it is done on the host in f64, which
            // keeps the loss exactly summable and costs nothing.
            let (block_loss, residuals) = self.residuals(
                &raw.flatten_all()?.to_vec1::<f32>()?,
                &self.group_labels[start..end],
            );
            loss += block_loss;
            for (output, gradient) in intercept_gradient.iter_mut().enumerate() {
                *gradient += residuals
                    .iter()
                    .skip(output)
                    .step_by(n_outputs)
                    .sum::<f64>();
            }

            let residual_tensor = Tensor::from_vec(
                residuals.iter().map(|&r| r as f32).collect::<Vec<f32>>(),
                (end - start, n_outputs),
                self.device,
            )?;
            let part = block.t()?.contiguous()?.matmul(&residual_tensor)?;
            coefficient_gradient = Some(match coefficient_gradient {
                Some(total) => total.add(&part)?,
                None => part,
            });
        }

        let accumulated = coefficient_gradient
            .ok_or_else(|| Error::shape("at least one cell", "an empty matrix"))?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let scale = 1.0 / n_cells as f64;
        let penalty: f64 = coefficients.iter().map(|&w| w * w).sum::<f64>() / REGULARISATION;
        let mut gradient = Vec::with_capacity(parameters.len());
        for (index, &partial) in accumulated.iter().enumerate() {
            gradient.push(scale * (f64::from(partial) + coefficients[index] / REGULARISATION));
        }
        gradient.extend(intercept_gradient.iter().map(|&g| scale * g));
        Ok((scale * (loss + 0.5 * penalty), gradient))
    }

    /// Pointwise loss and `predicted - observed` for one block of cells.
    fn residuals(&self, raw: &[f32], labels: &[u32]) -> (f64, Vec<f64>) {
        let n_outputs = self.n_outputs;
        let mut loss = 0f64;
        let mut residuals = vec![0f64; raw.len()];
        for (cell, label) in labels.iter().enumerate() {
            let scores = &raw[cell * n_outputs..(cell + 1) * n_outputs];
            let residual = &mut residuals[cell * n_outputs..(cell + 1) * n_outputs];
            if n_outputs == 1 {
                let observed = if *label == 1 { 1.0 } else { 0.0 };
                let score = f64::from(scores[0]);
                // softplus written through the negative branch so a large
                // positive score cannot overflow the exponential.
                loss += score.max(0.0) + (-score.abs()).exp().ln_1p() - observed * score;
                residual[0] = 1.0 / (1.0 + (-score).exp()) - observed;
            } else {
                let largest = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exponentials: Vec<f64> = scores
                    .iter()
                    .map(|&score| (f64::from(score - largest)).exp())
                    .collect();
                let total: f64 = exponentials.iter().sum();
                loss += total.ln() + f64::from(largest - scores[*label as usize]);
                for (output, value) in exponentials.iter().enumerate() {
                    let observed = if output == *label as usize { 1.0 } else { 0.0 };
                    residual[output] = value / total - observed;
                }
            }
        }
        (loss, residuals)
    }
}

/// One L-BFGS correction pair.
struct Correction {
    step: Vec<f64>,
    gradient_change: Vec<f64>,
    rho: f64,
}

/// Minimise `objective` from zero by L-BFGS, as scipy's `L-BFGS-B` does.
///
/// The objective is strictly convex — the ridge penalty removes the softmax's
/// shift invariance — so the minimiser is unique and reaching it, rather than
/// reproducing any particular solver's path to it, is what makes the scores
/// comparable with sklearn's.
fn minimise(
    objective: &MultinomialObjective,
    n_parameters: usize,
    max_iterations: usize,
) -> Result<Vec<f64>> {
    let mut point = vec![0f64; n_parameters];
    let (mut value, mut gradient) = objective.evaluate(&point)?;
    let mut history: VecDeque<Correction> = VecDeque::with_capacity(HISTORY);

    for _ in 0..max_iterations {
        if largest_magnitude(&gradient) <= GRADIENT_TOLERANCE {
            break;
        }
        let direction = two_loop_direction(&history, &gradient);
        // Before any curvature is known the quasi-Newton scale is meaningless,
        // so the first step is normalised to unit length instead.
        let initial = if history.is_empty() {
            1.0 / largest_magnitude(&gradient).max(1.0)
        } else {
            1.0
        };
        let Some(next) = line_search(objective, &point, value, &gradient, &direction, initial)?
        else {
            break;
        };

        let step: Vec<f64> = next.point.iter().zip(&point).map(|(a, b)| a - b).collect();
        let gradient_change: Vec<f64> = next
            .gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let curvature = dot(&step, &gradient_change);
        // A non-positive curvature would make the implicit Hessian indefinite;
        // skipping the pair is the standard damping.
        if curvature > 0.0 {
            if history.len() == HISTORY {
                history.pop_front();
            }
            history.push_back(Correction {
                step,
                gradient_change,
                rho: 1.0 / curvature,
            });
        }
        point = next.point;
        value = next.value;
        gradient = next.gradient;
    }
    Ok(point)
}

/// The two-loop recursion: the L-BFGS direction without ever forming a Hessian.
fn two_loop_direction(history: &VecDeque<Correction>, gradient: &[f64]) -> Vec<f64> {
    let mut direction: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let mut alphas = Vec::with_capacity(history.len());
    for correction in history.iter().rev() {
        let alpha = correction.rho * dot(&correction.step, &direction);
        for (value, change) in direction.iter_mut().zip(&correction.gradient_change) {
            *value -= alpha * change;
        }
        alphas.push(alpha);
    }
    if let Some(newest) = history.back() {
        let scale = 1.0 / (newest.rho * dot(&newest.gradient_change, &newest.gradient_change));
        for value in &mut direction {
            *value *= scale;
        }
    }
    for (correction, alpha) in history.iter().zip(alphas.iter().rev()) {
        let beta = correction.rho * dot(&correction.gradient_change, &direction);
        for (value, step) in direction.iter_mut().zip(&correction.step) {
            *value += (alpha - beta) * step;
        }
    }
    direction
}

struct Step {
    point: Vec<f64>,
    value: f64,
    gradient: Vec<f64>,
}

/// Bisection line search satisfying the Wolfe conditions.
///
/// The curvature condition is what guarantees a positive `s . y`, so it is not
/// optional here: without it the L-BFGS history would have to be thrown away
/// often enough to lose the superlinear rate.
fn line_search(
    objective: &MultinomialObjective,
    point: &[f64],
    value: f64,
    gradient: &[f64],
    direction: &[f64],
    initial: f64,
) -> Result<Option<Step>> {
    const SUFFICIENT_DECREASE: f64 = 1e-4;
    const CURVATURE: f64 = 0.9;
    const MAX_TRIALS: usize = 40;

    let slope = dot(gradient, direction);
    if !slope.is_finite() || slope >= 0.0 {
        return Ok(None);
    }

    let (mut low, mut high) = (0f64, f64::INFINITY);
    let mut length = initial;
    for _ in 0..MAX_TRIALS {
        let candidate: Vec<f64> = point
            .iter()
            .zip(direction)
            .map(|(x, d)| x + length * d)
            .collect();
        let (candidate_value, candidate_gradient) = objective.evaluate(&candidate)?;

        if candidate_value > value + SUFFICIENT_DECREASE * length * slope {
            high = length;
        } else if dot(&candidate_gradient, direction) < CURVATURE * slope {
            low = length;
        } else {
            return Ok(Some(Step {
                point: candidate,
                value: candidate_value,
                gradient: candidate_gradient,
            }));
        }
        length = if high.is_finite() {
            0.5 * (low + high)
        } else {
            2.0 * low.max(f64::MIN_POSITIVE)
        };
    }
    Ok(None)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn largest_magnitude(values: &[f64]) -> f64 {
    values.iter().fold(0f64, |worst, v| worst.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// `(mean1, std1, n1, mean2, std2, n2, statistic, p_value)` from
    /// `scipy.stats.ttest_ind_from_stats(..., equal_var=False)`.
    ///
    /// The last two rows are the degenerate pairs a single-cell matrix produces
    /// in bulk: equal constants, where scipy returns NaN and scanpy substitutes
    /// a neutral result, and different constants, where the statistic is
    /// genuinely infinite.
    #[allow(clippy::type_complexity)]
    const WELCH_REFERENCE: [(f64, f64, usize, f64, f64, usize, f64, f64); 7] = [
        (1.0, 0.5, 10, 0.4, 0.3, 12, 3.328_201_177_351_374_4, 0.004_905_214_566_072_544),
        (0.0, 1.0, 50, 0.0, 1.0, 50, 0.0, 1.0),
        (2.5, 1.2, 80, 0.1, 0.9, 160, 15.803_673_546_393_432, 1.436_431_132_812_541_3e-31),
        (0.3, 0.05, 3, 0.29, 0.04, 4, 0.284_747_398_725_749_95, 0.790_685_860_360_529),
        (5.0, 2.0, 2, 1.0, 0.1, 1_000, 2.828_420_053_704_894_7, 0.216_345_383_686_832_6),
        (1.0, 0.0, 20, 1.0, 0.0, 30, f64::NAN, f64::NAN),
        (2.0, 0.0, 20, 1.0, 0.0, 30, f64::INFINITY, 0.0),
    ];

    fn sample(mean: f64, deviation: f64, count: usize) -> Sample {
        Sample {
            mean,
            variance: deviation * deviation,
            count,
        }
    }

    #[test]
    fn welch_matches_scipy() {
        for (mean1, std1, n1, mean2, std2, n2, statistic, p_value) in WELCH_REFERENCE {
            let (actual_statistic, actual_p) =
                welch(sample(mean1, std1, n1), sample(mean2, std2, n2));
            // scipy's NaN is scanpy's neutral result, applied by `welch` itself.
            let (statistic, p_value) = if statistic.is_nan() {
                (0.0, 1.0)
            } else {
                (statistic, p_value)
            };
            if statistic.is_infinite() {
                assert_eq!(actual_statistic, statistic);
            } else {
                assert!(
                    (actual_statistic - statistic).abs() <= 1e-12 * statistic.abs().max(1e-6),
                    "statistic {actual_statistic} != {statistic}"
                );
            }
            assert!(
                (actual_p - p_value).abs() <= 1e-11 * p_value.max(1e-300),
                "p {actual_p} != {p_value}"
            );
        }
    }

    /// `(t, df, 2 * scipy.stats.t.sf(t, df))`.
    ///
    /// The far tail is in the list on purpose: a differential expression
    /// p-value routinely lands there, which is why the whole field is `f64`.
    const T_TAIL_REFERENCE: [(f64, f64, f64); 9] = [
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.500_000_000_000_000_1),
        (2.0, 3.0, 0.139_325_968_558_843_05),
        (2.5, 10.0, 0.031_446_844_236_608_82),
        (6.0, 25.0, 2.885_327_658_898_858e-6),
        (12.0, 100.0, 4.395_087_715_604_392e-21),
        (40.0, 2_000.0, 1.427_861_853_356_283e-257),
        (3.0, 1e6, 0.002_699_862_541_421_799_2),
        (150.0, 50.0, 4.969_739_685_288_344e-68),
    ];

    #[test]
    fn t_tail_matches_scipy() {
        for (statistic, degrees_of_freedom, expected) in T_TAIL_REFERENCE {
            let actual = two_sided_t_p(statistic, degrees_of_freedom);
            assert!(
                (actual - expected).abs() <= 1e-11 * expected,
                "t.sf({statistic}, {degrees_of_freedom}): {actual} != {expected}"
            );
        }
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        // ln(gamma(n)) = ln((n-1)!) for integers, and ln(gamma(0.5)) = ln(sqrt(pi)).
        for (x, expected) in [
            (1.0, 0.0),
            (2.0, 0.0),
            (5.0, 24f64.ln()),
            (11.0, 3_628_800f64.ln()),
            (0.5, std::f64::consts::PI.sqrt().ln()),
        ] {
            assert!((ln_gamma(x) - expected).abs() <= 1e-12 * expected.abs().max(1.0));
        }
    }

    /// Two groups of two cells over three genes, all values distinct.
    fn small_matrix() -> CsrMatrix {
        CsrMatrix::from_dense(
            &[
                1.0, 0.0, 3.0, //
                2.0, 0.0, 5.0, //
                7.0, 0.0, 1.0, //
                9.0, 0.0, 2.0,
            ],
            4,
            3,
        )
        .unwrap()
    }

    #[test]
    fn moments_ignore_implicit_zeros_but_count_them() {
        let moments = GroupMoments::compute(&small_matrix(), &[0, 0, 1, 1], 2);
        // Gene 0 in group 0: values 1 and 2.
        assert!((moments.means[[0, 0]] - 1.5).abs() < 1e-12);
        assert!((moments.variances[[0, 0]] - 0.5).abs() < 1e-12);
        // Gene 1 is stored nowhere at all, so its mean and variance are zero.
        assert_eq!(moments.means[[0, 1]], 0.0);
        assert_eq!(moments.variances[[0, 1]], 0.0);
        // The rest of group 0 is group 1: values 7 and 9 in gene 0.
        assert!((moments.rest_means[[0, 0]] - 8.0).abs() < 1e-12);
        assert!((moments.rest_variances[[0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_cell_group_leaves_the_variance_uncorrected() {
        let moments = GroupMoments::compute(&small_matrix(), &[0, 1, 1, 1], 2);
        assert_eq!(moments.sizes[0], 1);
        assert_eq!(moments.variances[[0, 0]], 0.0);
        assert!(moments.means[[0, 0]] == 1.0);
    }

    #[test]
    fn an_empty_group_gives_a_neutral_result_rather_than_a_nan() {
        // Group 1 holds no cells at all: nothing to test, so nothing significant.
        let comparison = t_test(&small_matrix(), &[0, 0, 0, 0], 2, None, &cpu()).unwrap();
        assert_eq!(comparison.scores[[0, 0]], 0.0);
        assert_eq!(comparison.p_values[[0, 0]], 1.0);
    }

    #[test]
    fn a_gene_constant_everywhere_is_never_significant() {
        let matrix = CsrMatrix::from_dense(&[4.0, 1.0, 4.0, 2.0, 4.0, 8.0, 4.0, 9.0], 4, 2).unwrap();
        let comparison = t_test(&matrix, &[0, 0, 1, 1], 2, None, &cpu()).unwrap();
        for group in 0..2 {
            assert_eq!(comparison.scores[[group, 0]], 0.0);
            assert_eq!(comparison.p_values[[group, 0]], 1.0);
            assert_eq!(comparison.adjusted_p_values[[group, 0]], 1.0);
        }
    }

    #[test]
    fn the_reference_group_row_is_neutral() {
        let comparison = t_test(&small_matrix(), &[0, 0, 1, 1], 2, Some(1), &cpu()).unwrap();
        for gene in 0..3 {
            assert_eq!(comparison.scores[[1, gene]], 0.0);
            assert_eq!(comparison.p_values[[1, gene]], 1.0);
            assert_eq!(comparison.log2_fold_changes[[1, gene]], 0.0);
        }
        assert!(comparison.scores[[0, 0]] < 0.0, "group 0 is lower in gene 0");
    }

    #[test]
    fn the_two_variants_differ_only_in_the_substituted_sample_size() {
        let matrix = small_matrix();
        let labels = [0, 0, 0, 1];
        let plain = t_test(&matrix, &labels, 2, None, &cpu()).unwrap();
        let inflated = t_test_overestimated_variance(&matrix, &labels, 2, None, &cpu()).unwrap();

        let moments = GroupMoments::compute(&matrix, &labels, 2);
        for group in 0..2 {
            let rest = 4 - moments.sizes[group];
            for gene in 0..3 {
                let tested = Sample {
                    mean: moments.means[[group, gene]],
                    variance: moments.variances[[group, gene]],
                    count: moments.sizes[group],
                };
                let reference = |count| Sample {
                    mean: moments.rest_means[[group, gene]],
                    variance: moments.rest_variances[[group, gene]],
                    count,
                };
                assert_eq!(plain.scores[[group, gene]], welch(tested, reference(rest)).0 as f32);
                assert_eq!(
                    inflated.scores[[group, gene]],
                    welch(tested, reference(moments.sizes[group])).0 as f32
                );
            }
        }
    }

    #[test]
    fn the_two_variants_coincide_when_the_group_is_half_the_cells() {
        let matrix = small_matrix();
        let labels = [0, 0, 1, 1];
        let plain = t_test(&matrix, &labels, 2, None, &cpu()).unwrap();
        let inflated = t_test_overestimated_variance(&matrix, &labels, 2, None, &cpu()).unwrap();
        assert_eq!(plain.scores, inflated.scores);
        assert_eq!(plain.p_values, inflated.p_values);
    }

    #[test]
    fn fold_changes_follow_scanpys_expm1_ratio() {
        let comparison = t_test(&small_matrix(), &[0, 0, 1, 1], 2, None, &cpu()).unwrap();
        let expected = log2_fold_change(1.5, 8.0) as f32;
        assert!((comparison.log2_fold_changes[[0, 0]] - expected).abs() < 1e-6);
    }

    #[test]
    fn rejects_malformed_input() {
        let matrix = small_matrix();
        assert!(t_test(&matrix, &[0, 0, 1], 2, None, &cpu()).is_err());
        assert!(t_test(&matrix, &[0, 0, 1, 1], 0, None, &cpu()).is_err());
        assert!(t_test(&matrix, &[0, 0, 1, 5], 2, None, &cpu()).is_err());
        assert!(t_test(&matrix, &[0, 0, 1, 1], 2, Some(3), &cpu()).is_err());
        assert!(logistic_regression(&matrix, &[0, 0, 0, 0], 1, 10, &cpu()).is_err());
        assert!(logistic_regression(&matrix, &[0, 0, 0, 0], 2, 10, &cpu()).is_err());
        assert!(logistic_regression(&matrix, &[0, 0, 1, 1], 2, 0, &cpu()).is_err());
    }

    /// Two perfectly separated groups over two genes, the first informative.
    fn separable_matrix() -> CsrMatrix {
        let mut dense = Vec::new();
        for cell in 0..40 {
            let signal = if cell < 20 { 0.0 } else { 4.0 };
            dense.extend_from_slice(&[signal, (cell % 3) as f32]);
        }
        CsrMatrix::from_dense(&dense, 40, 2).unwrap()
    }

    #[test]
    fn logistic_regression_finds_the_separating_gene() {
        let labels: Vec<u32> = (0..40).map(|cell| u32::from(cell >= 20)).collect();
        let comparison = logistic_regression(&separable_matrix(), &labels, 2, 200, &cpu()).unwrap();

        // Two groups means one binary fit, reported for both groups.
        assert_eq!(comparison.scores[[0, 0]], comparison.scores[[1, 0]]);
        assert!(comparison.scores[[0, 0]] > 0.5, "the marker carries the weight");
        assert!(comparison.scores[[0, 0]].abs() > 10.0 * comparison.scores[[0, 1]].abs());
        assert!(comparison.p_values[[0, 0]].is_nan());
        assert!(comparison.log2_fold_changes[[0, 0]].is_nan());
    }

    #[test]
    fn logistic_regression_reaches_a_stationary_point() {
        let labels: Vec<u32> = (0..40).map(|cell| (cell % 3) as u32).collect();
        let matrix = CsrMatrix::from_dense(
            &(0..120)
                .map(|index| ((index % 7) as f32) * 0.3)
                .collect::<Vec<f32>>(),
            40,
            3,
        )
        .unwrap();
        let objective = MultinomialObjective {
            matrix: &matrix,
            group_labels: &labels,
            n_outputs: 3,
            device: &cpu(),
        };
        let parameters = minimise(&objective, (3 + 1) * 3, 500).unwrap();
        let (_, gradient) = objective.evaluate(&parameters).unwrap();
        assert!(
            largest_magnitude(&gradient) <= GRADIENT_TOLERANCE,
            "gradient {}",
            largest_magnitude(&gradient)
        );
    }

    #[test]
    fn the_objective_gradient_matches_finite_differences() {
        let labels: Vec<u32> = (0..40).map(|cell| (cell % 3) as u32).collect();
        let objective = MultinomialObjective {
            matrix: &separable_matrix(),
            group_labels: &labels,
            n_outputs: 3,
            device: &cpu(),
        };
        let parameters: Vec<f64> = (0..9).map(|index| 0.1 * (index as f64) - 0.4).collect();
        let (_, gradient) = objective.evaluate(&parameters).unwrap();

        const DELTA: f64 = 1e-5;
        for index in 0..parameters.len() {
            let mut shifted = parameters.clone();
            shifted[index] += DELTA;
            let (up, _) = objective.evaluate(&shifted).unwrap();
            shifted[index] -= 2.0 * DELTA;
            let (down, _) = objective.evaluate(&shifted).unwrap();
            let numeric = (up - down) / (2.0 * DELTA);
            assert!(
                (gradient[index] - numeric).abs() <= 1e-6 * numeric.abs().max(1e-3),
                "parameter {index}: {} != {numeric}",
                gradient[index]
            );
        }
    }
}
