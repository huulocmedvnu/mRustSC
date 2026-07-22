//! Removing unwanted variation. Owned by feat/regress-combat.
//!
//! Both algorithms here are least squares against **one design shared by every
//! gene**, which is why they belong in the same module. The normal equations are
//! formed and inverted once — the design is a few columns wide, so that is a
//! host-side f64 factorisation of a tiny matrix — and every gene then costs two
//! matrix products against the result. Tens of thousands of right-hand sides
//! against one small operator is the shape the GPU exists for.
//!
//! Both densify. Their arithmetic is over a dense cell-by-gene array and no
//! amount of restructuring changes that: the residual of a sparse matrix on a
//! dense design is dense. `regress_out` therefore works in gene blocks and
//! `combat` holds a handful of whole copies, and both refuse an input whose
//! dense working set would exceed [`MAX_DENSE_ELEMENTS`] instead of exhausting
//! memory.
//!
//! At 50 000 cells by 20 000 genes — 1e9 elements, 4 GiB dense — `regress_out`
//! peaks at 8 GiB (the input and the result) plus a 32 MiB block on the device,
//! and is accepted. `combat` would need about 20 GiB there and is refused; its
//! ceiling is 2.1e8 elements, roughly 50 000 cells by 4 000 genes.
//!
//! Matrices are cells by genes throughout, as everywhere else in the workspace.

use candle_core::{DType, Device, Tensor};
use ndarray::{s, Array2, ArrayView2};

use crate::error::{Error, Result};

/// Dense f32 working set this module will allocate before refusing, in
/// elements: 8 GiB.
///
/// `regress_out` holds the input and the result, so it admits 1.07e9 cells x
/// genes — 50 000 cells by 20 000 genes is exactly 1e9 elements, 4 GB per copy,
/// and passes. `combat` needs five such copies and so stops at 2.1e8, which is
/// the honest limit: the same matrix really would cost it 20 GB.
const MAX_DENSE_ELEMENTS: usize = 8 * (1 << 30) / std::mem::size_of::<f32>();

/// Target size of one gene block, in f32 elements (32 MiB).
///
/// The design is shared, so blocks are independent and the block size trades
/// nothing but kernel launch overhead against peak transient memory.
const GENE_BLOCK_ELEMENTS: usize = 8 * 1024 * 1024;

/// A pivot below this fraction of its own column's scale means the column adds
/// no direction the earlier ones do not already span. Values from f32 data
/// summed in f64 leave an exact dependence at ~1e-16 relative, so this
/// separates rank deficiency from conditioning without flagging either.
const RANK_TOLERANCE: f64 = 1e-9;

/// ComBat's convergence criterion, as `scanpy.pp.combat`'s `conv`.
const COMBAT_TOLERANCE: f32 = 1e-4;

/// scanpy's empirical Bayes loop has no iteration cap. Ours does, so a pathology
/// is reported rather than hung on.
const COMBAT_MAX_ITERATIONS: usize = 1000;

/// Regress every gene on the covariates and return the residuals.
///
/// One small least-squares problem per gene, all sharing a design: the batched
/// shape a GPU is built for.
///
/// `expression` is `(n_cells, n_genes)` and `covariates` is `(n_cells, k)`. An
/// intercept column is prepended here rather than expected from the caller: a
/// regression without one removes the covariate's mean effect from the wrong
/// baseline, and that is an algorithmic decision, not a default.
pub fn regress_out(
    expression: &Array2<f32>,
    covariates: &Array2<f32>,
    device: &Device,
) -> Result<Array2<f32>> {
    let (n_cells, n_genes) = expression.dim();
    if covariates.nrows() != n_cells {
        return Err(Error::shape(
            format!("{n_cells} covariate rows, one per cell"),
            format!("{} rows", covariates.nrows()),
        ));
    }
    if n_cells < 2 {
        return Err(Error::shape("at least 2 cells", format!("{n_cells} cells")));
    }
    check_dense_budget(n_cells, n_genes, 2)?;

    let design = Design::new(&with_intercept(covariates), device)?;
    let mut residuals = Array2::zeros((n_cells, n_genes));
    for (start, end) in gene_blocks(n_cells, n_genes) {
        let block = to_tensor(expression.slice(s![.., start..end]), device)?;
        let block_residuals = design.residuals(&block)?;
        residuals
            .slice_mut(s![.., start..end])
            .assign(&to_array2(&block_residuals)?);
    }
    Ok(residuals)
}

/// Empirical Bayes batch correction, as `scanpy.pp.combat`.
///
/// `expression` is `(n_cells, n_genes)`, `batch` holds one label per cell below
/// `n_batches`, and `covariates` is an optional `(n_cells, k)` of extra design
/// columns that the correction preserves rather than removes.
///
/// The heavy steps — the shared least-squares fit, the standardisation and the
/// per-batch reductions inside the empirical Bayes iteration — are tensor
/// algebra on `device`. The prior hyperparameters are four scalars per batch and
/// the convergence test is one comparison per iteration; those are host
/// arithmetic in f64, because they are cheap and precision there is free.
pub fn combat(
    expression: &Array2<f32>,
    batch: &[u32],
    n_batches: usize,
    covariates: Option<&Array2<f32>>,
    device: &Device,
) -> Result<Array2<f32>> {
    let (n_cells, n_genes) = expression.dim();
    let members = batch_members(batch, n_batches, n_cells)?;
    if let Some(covariates) = covariates {
        if covariates.nrows() != n_cells {
            return Err(Error::shape(
                format!("{n_cells} covariate rows, one per cell"),
                format!("{} rows", covariates.nrows()),
            ));
        }
    }
    check_dense_budget(n_cells, n_genes, 5)?;
    if n_genes == 0 {
        return Ok(Array2::zeros((n_cells, n_genes)));
    }

    let expression = to_tensor(expression.view(), device)?;
    let design = Design::new(&combat_design(batch, n_batches, covariates), device)?;
    let coefficients = design.coefficients(&expression)?;

    // Pooled within-batch variance, and the mean the correction must restore:
    // the batch-weighted grand mean plus whatever the covariates explain.
    let pooled_variance = design
        .residuals_from(&expression, &coefficients)?
        .sqr()?
        .mean_keepdim(0)?;
    let standardisation_mean = standardisation_mean(&design, &members, &coefficients, device)?;
    let standardised = standardise(&expression, &standardisation_mean, &pooled_variance)?;

    let batches = fit_batch_effects(&standardised, &members, device)?;
    let shrunk = shrink_towards_prior(&standardised, &members, &batches, device)?;

    // Undo the standardisation with the batch effect now removed.
    let mut corrected = Tensor::zeros((n_cells, n_genes), DType::F32, device)?;
    for (batch, estimate) in members.iter().zip(&shrunk) {
        let rows = Tensor::from_slice(batch, batch.len(), device)?;
        let block = standardised.index_select(&rows, 0)?;
        let adjusted = block
            .broadcast_sub(&estimate.location)?
            // A gene with no within-batch spread would divide by zero; the
            // floor leaves it at its standardised value instead of a NaN.
            .broadcast_div(&estimate.scale.sqrt()?.maximum(f32::MIN_POSITIVE)?)?;
        corrected = corrected.index_add(&rows, &adjusted, 0)?;
    }
    to_array2(
        &corrected
            .broadcast_mul(&pooled_variance.sqrt()?)?
            .add(&standardisation_mean)?,
    )
}

/// The design every gene is regressed on, with its normal equations solved.
///
/// `projector` is `(D'D)^-1 D'`: forming it once is what turns "one least
/// squares per gene" into one matrix product per gene block. It is built on the
/// host in f64 because `D'D` is `p` by `p` with `p` in single digits, where f64
/// costs nothing and buys a trustworthy rank test.
struct Design {
    matrix: Tensor,
    projector: Tensor,
}

impl Design {
    fn new(matrix: &Array2<f32>, device: &Device) -> Result<Self> {
        let (n_cells, n_columns) = matrix.dim();
        if n_columns == 0 {
            return Err(Error::parameter("covariates", "at least one column", 0));
        }
        if n_cells < n_columns {
            return Err(Error::shape(
                format!("at least {n_columns} cells for a {n_columns}-column design"),
                format!("{n_cells} cells"),
            ));
        }

        let inverse = inverse_normal_equations(&gram(matrix), n_columns)?;
        let mut projector = vec![0.0f32; n_columns * n_cells];
        for row in 0..n_columns {
            for cell in 0..n_cells {
                let value: f64 = (0..n_columns)
                    .map(|k| inverse[row * n_columns + k] * f64::from(matrix[[cell, k]]))
                    .sum();
                projector[row * n_cells + cell] = value as f32;
            }
        }

        Ok(Self {
            matrix: to_tensor(matrix.view(), device)?,
            projector: Tensor::from_vec(projector, (n_columns, n_cells), device)?,
        })
    }

    /// `(p, n_genes)` coefficients for a `(n_cells, n_genes)` block.
    fn coefficients(&self, block: &Tensor) -> Result<Tensor> {
        Ok(self.projector.matmul(block)?)
    }

    fn residuals(&self, block: &Tensor) -> Result<Tensor> {
        self.residuals_from(block, &self.coefficients(block)?)
    }

    fn residuals_from(&self, block: &Tensor, coefficients: &Tensor) -> Result<Tensor> {
        Ok(block.sub(&self.matrix.matmul(coefficients)?)?)
    }
}

/// `D'D` in f64. The cross-products are the one place where f32 accumulation
/// over tens of thousands of cells would show, and they are only `p` by `p`.
fn gram(matrix: &Array2<f32>) -> Vec<f64> {
    let (n_cells, p) = matrix.dim();
    let mut gram = vec![0.0f64; p * p];
    for cell in 0..n_cells {
        for i in 0..p {
            let left = f64::from(matrix[[cell, i]]);
            for j in 0..=i {
                gram[i * p + j] += left * f64::from(matrix[[cell, j]]);
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            gram[j * p + i] = gram[i * p + j];
        }
    }
    gram
}

/// Invert `D'D` by Cholesky, refusing a rank-deficient design.
///
/// The pivot at column `j` is the part of that column's cross-product the
/// earlier columns leave unexplained. Judging it against the column's own
/// cross-product makes the test independent of the units the covariates are
/// measured in, and turns a silently wrong answer into an error the caller can
/// act on.
fn inverse_normal_equations(gram: &[f64], p: usize) -> Result<Vec<f64>> {
    let mut factor = vec![0.0f64; p * p];
    for j in 0..p {
        let mut pivot = gram[j * p + j];
        for k in 0..j {
            pivot -= factor[j * p + k].powi(2);
        }
        // A non-finite pivot means a non-finite covariate, which is as unusable
        // as a dependent column and is refused by the same test.
        if !pivot.is_finite() || pivot <= RANK_TOLERANCE * gram[j * p + j] {
            return Err(Error::parameter(
                "covariates",
                "a design of full column rank",
                format!("column {j} is a linear combination of the columns before it"),
            ));
        }
        let diagonal = pivot.sqrt();
        factor[j * p + j] = diagonal;
        for i in j + 1..p {
            let mut value = gram[i * p + j];
            for k in 0..j {
                value -= factor[i * p + k] * factor[j * p + k];
            }
            factor[i * p + j] = value / diagonal;
        }
    }

    // Solve `L L' X = I` one column at a time; p is single digits.
    let mut inverse = vec![0.0f64; p * p];
    for column in 0..p {
        let mut solution = vec![0.0f64; p];
        for i in 0..p {
            let mut value = f64::from(i == column);
            for k in 0..i {
                value -= factor[i * p + k] * solution[k];
            }
            solution[i] = value / factor[i * p + i];
        }
        for i in (0..p).rev() {
            let mut value = solution[i];
            for k in i + 1..p {
                value -= factor[k * p + i] * solution[k];
            }
            solution[i] = value / factor[i * p + i];
            inverse[i * p + column] = solution[i];
        }
    }
    Ok(inverse)
}

/// A `(n_cells, k + 1)` design whose first column is the intercept.
fn with_intercept(covariates: &Array2<f32>) -> Array2<f32> {
    let (n_cells, k) = covariates.dim();
    Array2::from_shape_fn((n_cells, k + 1), |(cell, column)| match column {
        0 => 1.0,
        _ => covariates[[cell, column - 1]],
    })
}

/// One-hot batch indicators followed by the covariate columns.
///
/// There is no intercept: the batch indicators already span it, and scanpy's
/// design is built the same way so that every batch keeps its own mean.
fn combat_design(batch: &[u32], n_batches: usize, covariates: Option<&Array2<f32>>) -> Array2<f32> {
    let n_cells = batch.len();
    let k = covariates.map_or(0, Array2::ncols);
    Array2::from_shape_fn((n_cells, n_batches + k), |(cell, column)| {
        if column < n_batches {
            f32::from(batch[cell] as usize == column)
        } else {
            covariates.map_or(0.0, |c| c[[cell, column - n_batches]])
        }
    })
}

/// The cell indices belonging to each batch, in label order.
fn batch_members(batch: &[u32], n_batches: usize, n_cells: usize) -> Result<Vec<Vec<u32>>> {
    if batch.len() != n_cells {
        return Err(Error::shape(
            format!("{n_cells} batch labels, one per cell"),
            format!("{} labels", batch.len()),
        ));
    }
    if n_batches == 0 {
        return Err(Error::parameter("n_batches", "at least 1", 0));
    }
    let mut members = vec![Vec::new(); n_batches];
    for (cell, &label) in batch.iter().enumerate() {
        let Some(batch) = members.get_mut(label as usize) else {
            return Err(Error::parameter(
                "batch",
                "a label below n_batches",
                format!("{label} with n_batches = {n_batches}"),
            ));
        };
        batch.push(cell as u32);
    }
    // Combat estimates a within-batch variance, which a single cell cannot
    // supply; scanpy raises here too.
    if let Some(empty) = members.iter().position(|cells| cells.len() < 2) {
        return Err(Error::parameter(
            "batch",
            "at least 2 cells in every batch",
            format!("batch {empty} has {}", members[empty].len()),
        ));
    }
    Ok(members)
}

/// The gene-wise mean the correction restores: the batch-size-weighted mean over
/// batches, plus the part of the fit the covariates explain.
fn standardisation_mean(
    design: &Design,
    members: &[Vec<u32>],
    coefficients: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let n_batches = members.len();
    let (n_cells, n_genes) = (design.matrix.dim(0)?, coefficients.dim(1)?);
    let weights: Vec<f32> = members
        .iter()
        .map(|cells| cells.len() as f32 / n_cells as f32)
        .collect();
    let grand_mean = Tensor::from_vec(weights, (1, n_batches), device)?
        .matmul(&coefficients.narrow(0, 0, n_batches)?)?;

    let n_covariates = coefficients.dim(0)? - n_batches;
    if n_covariates == 0 {
        return Ok(grand_mean.broadcast_as((n_cells, n_genes))?.contiguous()?);
    }
    // Only the covariate block contributes, so the batch columns are dropped
    // rather than multiplied by zero.
    let explained = design
        .matrix
        .narrow(1, n_batches, n_covariates)?
        .contiguous()?
        .matmul(
            &coefficients
                .narrow(0, n_batches, n_covariates)?
                .contiguous()?,
        )?;
    Ok(explained.broadcast_add(&grand_mean)?)
}

/// `(x - stand_mean) / sqrt(var_pooled)`, with zero-variance genes set to zero
/// exactly as scanpy does rather than divided by zero.
fn standardise(
    expression: &Tensor,
    standardisation_mean: &Tensor,
    pooled_variance: &Tensor,
) -> Result<Tensor> {
    let deviation = pooled_variance.sqrt()?;
    let usable = deviation.gt(0.0)?;
    let safe = usable.where_cond(&deviation, &deviation.ones_like()?)?;
    let standardised = expression.sub(standardisation_mean)?.broadcast_div(&safe)?;
    let keep = usable.broadcast_as(standardised.shape())?.contiguous()?;
    Ok(keep.where_cond(&standardised, &standardised.zeros_like()?)?)
}

/// The location and scale of one batch: `gamma` and `delta` in Johnson and Li.
struct BatchEffect {
    /// `(1, n_genes)`.
    location: Tensor,
    /// `(1, n_genes)`.
    scale: Tensor,
    /// Prior mean and variance of the location, over genes.
    location_prior: (f64, f64),
    /// Inverse-gamma prior on the scale, over genes.
    scale_prior: (f64, f64),
}

/// Per-batch location and scale of the standardised data, with the empirical
/// priors the shrinkage borrows across genes.
fn fit_batch_effects(
    standardised: &Tensor,
    members: &[Vec<u32>],
    device: &Device,
) -> Result<Vec<BatchEffect>> {
    let mut effects = Vec::with_capacity(members.len());
    for cells in members {
        let rows = Tensor::from_slice(cells, cells.len(), device)?;
        let block = standardised.index_select(&rows, 0)?;
        let location = block.mean_keepdim(0)?;
        // Bessel's correction: scanpy takes the within-batch variance with
        // pandas' default ddof of 1.
        let scale =
            (block.broadcast_sub(&location)?.sqr()?.sum_keepdim(0)? / (cells.len() as f64 - 1.0))?;

        let location_prior = mean_and_variance(&to_vec(&location)?, 0);
        let scale_values = to_vec(&scale)?;
        let (mean, variance) = mean_and_variance(&scale_values, 1);
        effects.push(BatchEffect {
            location,
            scale,
            location_prior,
            // The inverse-gamma hyperparameters that match the observed mean and
            // variance of the per-gene scales.
            scale_prior: (
                (2.0 * variance + mean * mean) / variance,
                (mean * variance + mean.powi(3)) / variance,
            ),
        });
    }
    Ok(effects)
}

/// Mean and variance over genes, in f64: a handful of scalars per batch that the
/// whole shrinkage hangs on.
fn mean_and_variance(values: &[f32], correction: usize) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / (n - correction as f64);
    (mean, variance)
}

/// The empirical Bayes step: shrink each batch's location and scale towards the
/// prior its genes share.
///
/// The two conditional posterior means depend on each other, so they are
/// iterated to a fixed point. Each iteration is one reduction over the batch's
/// cells for every gene at once — on the device — plus a single scalar pulled
/// back to decide whether to stop.
fn shrink_towards_prior(
    standardised: &Tensor,
    members: &[Vec<u32>],
    effects: &[BatchEffect],
    device: &Device,
) -> Result<Vec<BatchEffect>> {
    let mut shrunk = Vec::with_capacity(effects.len());
    for (cells, effect) in members.iter().zip(effects) {
        let rows = Tensor::from_slice(cells, cells.len(), device)?;
        let block = standardised.index_select(&rows, 0)?;
        let n = cells.len() as f64;
        let (location_mean, location_variance) = effect.location_prior;
        let (shape, rate) = effect.scale_prior;

        let mut location = effect.location.clone();
        let mut scale = effect.scale.clone();
        let mut converged = false;
        for _ in 0..COMBAT_MAX_ITERATIONS {
            let numerator =
                ((&effect.location * (location_variance * n))? + (&scale * location_mean)?)?;
            let next_location = numerator.div(&(&scale + location_variance * n)?)?;
            let deviation = block.broadcast_sub(&next_location)?.sqr()?.sum_keepdim(0)?;
            let next_scale =
                ((deviation * 0.5)? + rate)?.affine(1.0 / (n / 2.0 + shape - 1.0), 0.0)?;

            let change = relative_change(&next_location, &location)?
                .max(relative_change(&next_scale, &scale)?);
            location = next_location;
            scale = next_scale;
            if change <= COMBAT_TOLERANCE {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::NotConverged {
                operation: "combat empirical Bayes",
                iterations: COMBAT_MAX_ITERATIONS,
            });
        }
        shrunk.push(BatchEffect {
            location,
            scale,
            location_prior: effect.location_prior,
            scale_prior: effect.scale_prior,
        });
    }
    Ok(shrunk)
}

/// Largest relative change over genes.
///
/// scanpy divides by the signed previous value, so a negative location can hide
/// a real change from its `max`; taking the magnitude only ever stops later, at
/// the same fixed point.
fn relative_change(next: &Tensor, previous: &Tensor) -> Result<f32> {
    let ratio = next
        .sub(previous)?
        .abs()?
        .div(&previous.abs()?.maximum(f32::MIN_POSITIVE)?)?;
    Ok(ratio.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

/// Refuse an input whose dense working set would not fit the budget.
fn check_dense_budget(n_cells: usize, n_genes: usize, copies: usize) -> Result<()> {
    let elements = n_cells.saturating_mul(n_genes).saturating_mul(copies);
    if elements > MAX_DENSE_ELEMENTS {
        let gib = elements as f64 * std::mem::size_of::<f32>() as f64 / (1u64 << 30) as f64;
        return Err(Error::parameter(
            "expression",
            "small enough for a dense working set of 8 GiB",
            format!("{n_cells} cells x {n_genes} genes needs {gib:.1} GiB"),
        ));
    }
    Ok(())
}

/// Gene ranges of roughly [`GENE_BLOCK_ELEMENTS`] each.
fn gene_blocks(n_cells: usize, n_genes: usize) -> Vec<(usize, usize)> {
    let width = (GENE_BLOCK_ELEMENTS / n_cells.max(1)).clamp(1, n_genes.max(1));
    (0..n_genes)
        .step_by(width)
        .map(|start| (start, (start + width).min(n_genes)))
        .collect()
}

fn to_tensor(block: ArrayView2<f32>, device: &Device) -> Result<Tensor> {
    let standard = block.as_standard_layout();
    let values: Vec<f32> = standard.iter().copied().collect();
    Ok(Tensor::from_vec(values, block.dim(), device)?)
}

fn to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
    let (rows, columns) = tensor.dims2()?;
    let values = tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Array2::from_shape_vec((rows, columns), values)
        .map_err(|error| Error::shape(format!("{rows}x{columns}"), error.to_string()))
}

fn to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{gpu_available, DeviceKind};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// A standard normal deviate by Box-Muller, from a seeded generator so that
    /// simulated data is identical on every machine.
    fn standard_normal(rng: &mut StdRng) -> f32 {
        let radius: f32 = rng.gen_range(f32::EPSILON..1.0);
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        (-2.0 * radius.ln()).sqrt() * angle.cos()
    }

    fn normal_matrix(n_rows: usize, n_cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n_rows, n_cols), |_| standard_normal(&mut rng))
    }

    fn gpu() -> Device {
        DeviceKind::Gpu.resolve().unwrap()
    }

    fn max_deviation(left: &Array2<f32>, right: &Array2<f32>) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn a_gene_that_is_a_linear_function_of_the_covariate_regresses_to_zero() {
        let covariates = normal_matrix(40, 2, 1);
        let expression = Array2::from_shape_fn((40, 3), |(cell, gene)| {
            let base = 2.5 + 1.5 * covariates[[cell, 0]] - 0.75 * covariates[[cell, 1]];
            base * (gene as f32 + 1.0)
        });

        let residuals = regress_out(&expression, &covariates, &Device::Cpu).unwrap();

        let largest = residuals.iter().fold(0.0f32, |worst, r| worst.max(r.abs()));
        assert!(largest < 1e-4, "largest residual {largest}");
    }

    #[test]
    fn residuals_are_orthogonal_to_the_design() {
        let covariates = normal_matrix(60, 3, 2);
        let expression = normal_matrix(60, 12, 3);

        let residuals = regress_out(&expression, &covariates, &Device::Cpu).unwrap();

        for gene in 0..expression.ncols() {
            // The intercept is part of the design, so the residuals are centred
            // as well as uncorrelated with every covariate.
            let sum: f32 = residuals.column(gene).sum();
            assert!(sum.abs() < 1e-3, "gene {gene} residuals sum to {sum}");
            for column in 0..covariates.ncols() {
                let product: f32 = residuals
                    .column(gene)
                    .iter()
                    .zip(covariates.column(column))
                    .map(|(r, c)| r * c)
                    .sum();
                assert!(
                    product.abs() < 1e-2,
                    "gene {gene} column {column}: {product}"
                );
            }
        }
    }

    #[test]
    fn regress_out_rejects_a_rank_deficient_design() {
        let mut covariates = normal_matrix(30, 3, 4);
        let duplicate = covariates.column(0).to_owned();
        covariates.column_mut(2).assign(&duplicate);
        let expression = normal_matrix(30, 4, 5);

        let error = regress_out(&expression, &covariates, &Device::Cpu).unwrap_err();

        assert!(
            matches!(error, Error::InvalidParameter { .. }),
            "expected a parameter error, got {error}"
        );
        assert!(error.to_string().contains("full column rank"));
    }

    #[test]
    fn regress_out_rejects_mismatched_lengths() {
        let expression = normal_matrix(20, 4, 6);
        let covariates = normal_matrix(19, 1, 7);

        assert!(regress_out(&expression, &covariates, &Device::Cpu).is_err());
    }

    #[test]
    fn a_constant_covariate_is_rank_deficient_against_the_intercept() {
        let expression = normal_matrix(20, 4, 8);
        let covariates = Array2::from_elem((20, 1), 3.0);

        assert!(regress_out(&expression, &covariates, &Device::Cpu).is_err());
    }

    #[test]
    fn gene_blocks_cover_every_gene_exactly_once() {
        let blocks = gene_blocks(1_000_000, 25);
        assert_eq!(blocks.first(), Some(&(0, 8)));
        assert_eq!(blocks.last(), Some(&(24, 25)));
        assert_eq!(blocks.iter().map(|(s, e)| e - s).sum::<usize>(), 25);
        assert!(blocks.windows(2).all(|pair| pair[0].1 == pair[1].0));
        assert_eq!(gene_blocks(10, 0), Vec::new());
    }

    #[test]
    fn blocking_does_not_change_the_residuals() {
        // Enough cells that the block width is a single gene, so the blocked
        // path is exercised against a design solved exactly once.
        let n_cells = GENE_BLOCK_ELEMENTS;
        assert_eq!(gene_blocks(n_cells, 5).len(), 5);
        let covariates = normal_matrix(50, 2, 9);
        let expression = normal_matrix(50, 7, 10);

        let residuals = regress_out(&expression, &covariates, &Device::Cpu).unwrap();
        let design = Design::new(&with_intercept(&covariates), &Device::Cpu).unwrap();
        let whole = to_array2(
            &design
                .residuals(&to_tensor(expression.view(), &Device::Cpu).unwrap())
                .unwrap(),
        )
        .unwrap();

        assert!(max_deviation(&residuals, &whole) < 1e-6);
    }

    #[test]
    fn refuses_a_dense_working_set_beyond_the_budget() {
        assert!(check_dense_budget(50_000, 20_000, 2).is_ok());
        assert!(check_dense_budget(50_000, 20_000, 5).is_err());
        assert!(check_dense_budget(usize::MAX, 2, 1).is_err());
    }

    /// Two batches, the second shifted up and stretched on every gene.
    fn planted_batch_effect(n_per_batch: usize, n_genes: usize) -> (Array2<f32>, Vec<u32>) {
        let mut rng = StdRng::seed_from_u64(11);
        let n_cells = 2 * n_per_batch;
        let batch: Vec<u32> = (0..n_cells)
            .map(|cell| u32::from(cell >= n_per_batch))
            .collect();
        let expression = Array2::from_shape_fn((n_cells, n_genes), |(cell, gene)| {
            let signal = 5.0 + gene as f32 * 0.1 + standard_normal(&mut rng);
            if batch[cell] == 1 {
                2.0 + 1.5 * signal
            } else {
                signal
            }
        });
        (expression, batch)
    }

    fn batch_mean_gap(matrix: &Array2<f32>, batch: &[u32]) -> f32 {
        (0..matrix.ncols())
            .map(|gene| {
                let mean = |label: u32| {
                    let cells: Vec<f32> = batch
                        .iter()
                        .enumerate()
                        .filter(|(_, &b)| b == label)
                        .map(|(cell, _)| matrix[[cell, gene]])
                        .collect();
                    cells.iter().sum::<f32>() / cells.len() as f32
                };
                (mean(0) - mean(1)).abs()
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn combat_shrinks_the_gap_between_batches() {
        let (expression, batch) = planted_batch_effect(60, 20);

        let corrected = combat(&expression, &batch, 2, None, &Device::Cpu).unwrap();

        let before = batch_mean_gap(&expression, &batch);
        let after = batch_mean_gap(&corrected, &batch);
        assert!(after < before / 20.0, "gap {before} -> {after}");
        assert!(corrected.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn combat_leaves_a_single_batch_almost_untouched() {
        // With one batch there is nothing to correct: the location estimate is
        // zero and the scale one, so the standardisation round trips. Not
        // exactly, and scanpy does not either — the pooled variance divides by
        // n and the batch variance by n - 1, so everything is rescaled by
        // sqrt(n / (n - 1)), 0.6% at 80 cells.
        let n_cells = 80;
        let expression = normal_matrix(n_cells, 15, 12);
        let batch = vec![0u32; n_cells];

        let corrected = combat(&expression, &batch, 1, None, &Device::Cpu).unwrap();

        let bessel = (n_cells as f32 / (n_cells as f32 - 1.0)).sqrt();
        let expected = expression.map(|value| value / bessel);
        assert!(max_deviation(&expected, &corrected) < 1e-2);
    }

    #[test]
    fn combat_rejects_impossible_batch_labels() {
        let expression = normal_matrix(10, 3, 13);
        let labels = |values: Vec<u32>| combat(&expression, &values, 2, None, &Device::Cpu);

        // Out of range, one cell in a batch, and the wrong number of labels.
        assert!(labels(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 2]).is_err());
        assert!(labels(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).is_err());
        assert!(labels(vec![0; 9]).is_err());
    }

    #[test]
    fn combat_keeps_a_covariate_it_was_told_to_preserve() {
        // The covariate's effect must survive the correction; only the batch
        // difference is removed.
        let (mut expression, batch) = planted_batch_effect(50, 10);
        let condition = Array2::from_shape_fn((expression.nrows(), 1), |(cell, _)| {
            f32::from(cell % 2 == 0)
        });
        for cell in 0..expression.nrows() {
            for gene in 0..expression.ncols() {
                expression[[cell, gene]] += 3.0 * condition[[cell, 0]];
            }
        }

        let corrected = combat(&expression, &batch, 2, Some(&condition), &Device::Cpu).unwrap();

        let gap = |matrix: &Array2<f32>, gene: usize| {
            let mean = |wanted: f32| {
                let values: Vec<f32> = (0..matrix.nrows())
                    .filter(|cell| condition[[*cell, 0]] == wanted)
                    .map(|cell| matrix[[cell, gene]])
                    .collect();
                values.iter().sum::<f32>() / values.len() as f32
            };
            mean(1.0) - mean(0.0)
        };
        for gene in 0..expression.ncols() {
            let kept = gap(&corrected, gene);
            assert!(kept > 2.0, "gene {gene} kept only {kept} of the condition");
        }
        assert!(batch_mean_gap(&corrected, &batch) < 0.5);
    }

    #[test]
    fn cpu_and_gpu_agree() {
        if !gpu_available() {
            return;
        }
        let covariates = normal_matrix(200, 2, 14);
        let expression = normal_matrix(200, 50, 15);
        let on_cpu = regress_out(&expression, &covariates, &Device::Cpu).unwrap();
        let on_gpu = regress_out(&expression, &covariates, &gpu()).unwrap();
        assert!(max_deviation(&on_cpu, &on_gpu) < 1e-4);

        let (expression, batch) = planted_batch_effect(100, 40);
        let on_cpu = combat(&expression, &batch, 2, None, &Device::Cpu).unwrap();
        let on_gpu = combat(&expression, &batch, 2, None, &gpu()).unwrap();
        assert!(max_deviation(&on_cpu, &on_gpu) < 1e-3);
    }
}
