//! Removing unwanted variation. Owned by feat/regress-combat.
//!
//! Both algorithms here are least squares against **one design shared by every
//! gene**, which is why they belong in the same module. The normal equations are
//! formed and inverted once — the design is a few columns wide, so that is a
//! host-side f64 factorisation of a tiny matrix — and every gene then costs two
//! matrix products against the result. Tens of thousands of right-hand sides
//! against one small operator is the shape the GPU exists for.
//!
//! Both produce a dense result: the residual of a sparse matrix on a dense design
//! is dense. Neither needs a dense *input*. The matrix is read one block of genes
//! at a time, from a column-major copy of the sparse input when it arrives as
//! CSR, so the only cell-by-gene array ever held is the result. `combat` needs
//! per-batch sufficient statistics of every gene before it can shrink anything,
//! so it makes two passes over the blocks: one to gather `sum x` and `sum x^2`
//! per batch and gene, one to write the corrected values. The empirical Bayes
//! iteration between the passes works on those sums alone.
//!
//! The result must fit in memory. Its size is checked against a budget derived
//! from the machine's physical memory before anything is allocated.
//!
//! Matrices are cells by genes throughout, as everywhere else in the workspace.

use candle_core::{Device, Tensor};
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Fraction of physical memory the dense result may take.
const RESULT_BUDGET_FRACTION: f64 = 0.6;
/// Budget when the physical memory cannot be read: 8 GiB of f32.
const FALLBACK_BUDGET_ELEMENTS: usize = 8 * (1 << 30) / std::mem::size_of::<f32>();

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
const COMBAT_TOLERANCE: f64 = 1e-4;

/// scanpy's empirical Bayes loop has no iteration cap. Ours does, so a pathology
/// is reported rather than hung on.
const COMBAT_MAX_ITERATIONS: usize = 1000;

/// A cell-by-gene matrix read one block of genes at a time.
pub trait GeneBlocks: Sync {
    fn n_cells(&self) -> usize;
    fn n_genes(&self) -> usize;
    /// Genes `start..end` as a dense `(n_cells, end - start)` array.
    fn block(&self, start: usize, end: usize) -> Array2<f32>;
}

impl GeneBlocks for Array2<f32> {
    fn n_cells(&self) -> usize {
        self.nrows()
    }
    fn n_genes(&self) -> usize {
        self.ncols()
    }
    fn block(&self, start: usize, end: usize) -> Array2<f32> {
        self.slice(s![.., start..end]).to_owned()
    }
}

/// A column-major copy of a sparse matrix, so a block of genes densifies from one
/// contiguous range of entries.
pub struct ColumnBlocks {
    n_cells: usize,
    n_genes: usize,
    offsets: Vec<usize>,
    rows: Vec<u32>,
    values: Vec<f32>,
}

impl ColumnBlocks {
    /// Transpose the CSR structure by counting sort. Costs the matrix's own
    /// non-zeros once, nothing proportional to cells times genes.
    pub fn from_csr(matrix: &CsrMatrix) -> Self {
        let (n_cells, n_genes) = (matrix.n_rows(), matrix.n_cols());
        let mut offsets = vec![0usize; n_genes + 1];
        for &gene in matrix.indices() {
            offsets[gene as usize + 1] += 1;
        }
        for gene in 0..n_genes {
            offsets[gene + 1] += offsets[gene];
        }
        let mut cursor = offsets.clone();
        let mut rows = vec![0u32; matrix.nnz()];
        let mut values = vec![0f32; matrix.nnz()];
        let indptr = matrix.indptr();
        for cell in 0..n_cells {
            for entry in indptr[cell] as usize..indptr[cell + 1] as usize {
                let gene = matrix.indices()[entry] as usize;
                rows[cursor[gene]] = cell as u32;
                values[cursor[gene]] = matrix.values()[entry];
                cursor[gene] += 1;
            }
        }
        Self {
            n_cells,
            n_genes,
            offsets,
            rows,
            values,
        }
    }
}

impl GeneBlocks for ColumnBlocks {
    fn n_cells(&self) -> usize {
        self.n_cells
    }
    fn n_genes(&self) -> usize {
        self.n_genes
    }
    fn block(&self, start: usize, end: usize) -> Array2<f32> {
        let width = end - start;
        let mut dense = vec![0f32; self.n_cells * width];
        for gene in start..end {
            for entry in self.offsets[gene]..self.offsets[gene + 1] {
                dense[self.rows[entry] as usize * width + gene - start] = self.values[entry];
            }
        }
        Array2::from_shape_vec((self.n_cells, width), dense).expect("sized above")
    }
}

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
    regress_out_blocks(expression, covariates, device)
}

/// [`regress_out`] on a sparse input, which is never densified as a whole.
pub fn regress_out_csr(
    expression: &CsrMatrix,
    covariates: &Array2<f32>,
    device: &Device,
) -> Result<Array2<f32>> {
    regress_out_blocks(&ColumnBlocks::from_csr(expression), covariates, device)
}

fn regress_out_blocks<B: GeneBlocks>(
    expression: &B,
    covariates: &Array2<f32>,
    device: &Device,
) -> Result<Array2<f32>> {
    let (n_cells, n_genes) = (expression.n_cells(), expression.n_genes());
    if covariates.nrows() != n_cells {
        return Err(Error::shape(
            format!("{n_cells} covariate rows, one per cell"),
            format!("{} rows", covariates.nrows()),
        ));
    }
    if n_cells < 2 {
        return Err(Error::shape("at least 2 cells", format!("{n_cells} cells")));
    }
    check_result_budget(n_cells, n_genes, result_budget_elements())?;

    let design = Design::new(&with_intercept(covariates), device)?;
    let mut residuals = Array2::zeros((n_cells, n_genes));
    for (start, end) in gene_blocks(n_cells, n_genes) {
        let block = to_tensor(expression.block(start, end).view(), device)?;
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
/// The shared least-squares fit and the standardisation run block by block as
/// tensor algebra on `device`. The empirical Bayes step needs, per batch and
/// gene, only the count, the sum and the sum of squares of the standardised
/// values, so it runs on the host in f64 between the two passes over the data.
pub fn combat(
    expression: &Array2<f32>,
    batch: &[u32],
    n_batches: usize,
    covariates: Option<&Array2<f32>>,
    device: &Device,
) -> Result<Array2<f32>> {
    combat_blocks(expression, batch, n_batches, covariates, device)
}

/// [`combat`] on a sparse input, which is never densified as a whole.
pub fn combat_csr(
    expression: &CsrMatrix,
    batch: &[u32],
    n_batches: usize,
    covariates: Option<&Array2<f32>>,
    device: &Device,
) -> Result<Array2<f32>> {
    combat_blocks(
        &ColumnBlocks::from_csr(expression),
        batch,
        n_batches,
        covariates,
        device,
    )
}

fn combat_blocks<B: GeneBlocks>(
    expression: &B,
    batch: &[u32],
    n_batches: usize,
    covariates: Option<&Array2<f32>>,
    device: &Device,
) -> Result<Array2<f32>> {
    let (n_cells, n_genes) = (expression.n_cells(), expression.n_genes());
    let members = batch_members(batch, n_batches, n_cells)?;
    if let Some(covariates) = covariates {
        if covariates.nrows() != n_cells {
            return Err(Error::shape(
                format!("{n_cells} covariate rows, one per cell"),
                format!("{} rows", covariates.nrows()),
            ));
        }
    }
    check_result_budget(n_cells, n_genes, result_budget_elements())?;
    if n_genes == 0 {
        return Ok(Array2::zeros((n_cells, n_genes)));
    }

    let design = Design::new(&combat_design(batch, n_batches, covariates), device)?;
    let weights: Vec<f32> = members
        .iter()
        .map(|cells| cells.len() as f32 / n_cells as f32)
        .collect();
    let batch_weights = Tensor::from_vec(weights, (1, n_batches), device)?;
    let blocks = gene_blocks(n_cells, n_genes);

    // Pass one: the fit, the pooled variance, and the per-batch sums of the
    // standardised values, one block at a time.
    let mut pooled_variance = vec![0f32; n_genes];
    let mut sums = vec![vec![0f64; n_genes]; n_batches];
    let mut squares = vec![vec![0f64; n_genes]; n_batches];
    for &(start, end) in &blocks {
        let block = to_tensor(expression.block(start, end).view(), device)?;
        let coefficients = design.coefficients(&block)?;
        let variance = design
            .residuals_from(&block, &coefficients)?
            .sqr()?
            .mean_keepdim(0)?;
        pooled_variance[start..end].copy_from_slice(&to_vec(&variance)?);
        let mean = standardisation_mean(&design, &batch_weights, &coefficients, n_batches)?;
        let standardised = to_array2(&standardise(&block, &mean, &variance)?)?;
        for (cells, (sum, square)) in members.iter().zip(sums.iter_mut().zip(squares.iter_mut())) {
            let (s, q) = batch_sums(&standardised, cells);
            sum[start..end].copy_from_slice(&s);
            square[start..end].copy_from_slice(&q);
        }
    }

    // The empirical Bayes step, per batch, on the sufficient statistics.
    let effects: Vec<ShrunkEffect> = members
        .iter()
        .zip(sums.iter().zip(squares.iter()))
        .map(|(cells, (sum, square))| shrink_towards_prior(cells.len(), sum, square))
        .collect::<Result<_>>()?;
    let batch_of = batch;

    // Pass two: standardise again and write the corrected block.
    let mut corrected = Array2::<f32>::zeros((n_cells, n_genes));
    for &(start, end) in &blocks {
        let block = to_tensor(expression.block(start, end).view(), device)?;
        let coefficients = design.coefficients(&block)?;
        let variance = Tensor::from_slice(&pooled_variance[start..end], (1, end - start), device)?;
        let mean = standardisation_mean(&design, &batch_weights, &coefficients, n_batches)?;
        let standardised = to_array2(&standardise(&block, &mean, &variance)?)?;
        let mean = to_array2(&mean.broadcast_as((n_cells, end - start))?.contiguous()?)?;
        let deviation: Vec<f32> = pooled_variance[start..end]
            .iter()
            .map(|v| v.sqrt())
            .collect();
        let width = end - start;
        let mut out = vec![0f32; n_cells * width];
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(cell, row)| {
                let effect = &effects[batch_of[cell] as usize];
                for (g, value) in row.iter_mut().enumerate() {
                    let gene = start + g;
                    let scale = (effect.scale[gene] as f32).sqrt().max(f32::MIN_POSITIVE);
                    *value = (standardised[[cell, g]] - effect.location[gene] as f32) / scale
                        * deviation[g]
                        + mean[[cell, g]];
                }
            });
        corrected
            .slice_mut(s![.., start..end])
            .assign(&Array2::from_shape_vec((n_cells, width), out).expect("sized above"));
    }
    Ok(corrected)
}

/// Per-gene sum and sum of squares over one batch's cells.
fn batch_sums(standardised: &Array2<f32>, cells: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let width = standardised.ncols();
    cells
        .par_iter()
        .fold(
            || (vec![0f64; width], vec![0f64; width]),
            |(mut sum, mut square), &cell| {
                for (g, &value) in standardised.row(cell as usize).iter().enumerate() {
                    let v = f64::from(value);
                    sum[g] += v;
                    square[g] += v * v;
                }
                (sum, square)
            },
        )
        .reduce(
            || (vec![0f64; width], vec![0f64; width]),
            |(mut a, mut b), (c, d)| {
                a.iter_mut().zip(&c).for_each(|(x, y)| *x += y);
                b.iter_mut().zip(&d).for_each(|(x, y)| *x += y);
                (a, b)
            },
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

/// The gene-wise mean the correction restores, `(n_cells, width)` when covariates are
/// present and `(1, width)` otherwise: the batch-size-weighted mean over batches, plus
/// the part of the fit the covariates explain.
fn standardisation_mean(
    design: &Design,
    batch_weights: &Tensor,
    coefficients: &Tensor,
    n_batches: usize,
) -> Result<Tensor> {
    let grand_mean = batch_weights.matmul(&coefficients.narrow(0, 0, n_batches)?)?;
    let n_covariates = coefficients.dim(0)? - n_batches;
    if n_covariates == 0 {
        return Ok(grand_mean);
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
    let standardised = expression
        .broadcast_sub(standardisation_mean)?
        .broadcast_div(&safe)?;
    let keep = usable.broadcast_as(standardised.shape())?.contiguous()?;
    Ok(keep.where_cond(&standardised, &standardised.zeros_like()?)?)
}

/// One batch's shrunk location and scale per gene: `gamma*` and `delta*` in
/// Johnson and Li.
struct ShrunkEffect {
    location: Vec<f64>,
    scale: Vec<f64>,
}

/// The empirical Bayes step: shrink one batch's per-gene location and scale
/// towards the prior its genes share, from the batch's count, sum and sum of
/// squares per gene.
///
/// The two conditional posterior means depend on each other, so they are
/// iterated to a fixed point. `sum (x - g)^2` over the batch's cells expands to
/// `sum x^2 - 2 g sum x + n g^2`, which is why the data are not needed here.
fn shrink_towards_prior(n_cells: usize, sum: &[f64], square: &[f64]) -> Result<ShrunkEffect> {
    let n = n_cells as f64;
    let location_hat: Vec<f64> = sum.iter().map(|s| s / n).collect();
    // Bessel's correction: scanpy takes the within-batch variance with pandas'
    // default ddof of 1.
    let scale_hat: Vec<f64> = square
        .iter()
        .zip(&location_hat)
        .map(|(q, g)| (q - n * g * g) / (n - 1.0))
        .collect();
    let (location_mean, location_variance) = mean_and_variance(&location_hat, 0);
    let (mean, variance) = mean_and_variance(&scale_hat, 1);
    // The inverse-gamma hyperparameters that match the observed mean and variance
    // of the per-gene scales.
    let shape = (2.0 * variance + mean * mean) / variance;
    let rate = (mean * variance + mean.powi(3)) / variance;

    let mut location = location_hat.clone();
    let mut scale = scale_hat.clone();
    for _ in 0..COMBAT_MAX_ITERATIONS {
        let mut change = 0.0f64;
        let mut next_location = Vec::with_capacity(sum.len());
        let mut next_scale = Vec::with_capacity(sum.len());
        for g in 0..sum.len() {
            let l = (location_variance * n * location_hat[g] + scale[g] * location_mean)
                / (scale[g] + location_variance * n);
            let deviation = square[g] - 2.0 * l * sum[g] + n * l * l;
            let s = (0.5 * deviation + rate) / (n / 2.0 + shape - 1.0);
            change = change
                .max((l - location[g]).abs() / location[g].abs().max(f64::from(f32::MIN_POSITIVE)))
                .max((s - scale[g]).abs() / scale[g].abs().max(f64::from(f32::MIN_POSITIVE)));
            next_location.push(l);
            next_scale.push(s);
        }
        location = next_location;
        scale = next_scale;
        if change <= COMBAT_TOLERANCE {
            return Ok(ShrunkEffect { location, scale });
        }
    }
    Err(Error::NotConverged {
        operation: "combat empirical Bayes",
        iterations: COMBAT_MAX_ITERATIONS,
    })
}

/// Mean and variance over genes, in f64: a handful of scalars per batch that the
/// whole shrinkage hangs on.
fn mean_and_variance(values: &[f64], correction: usize) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - correction as f64);
    (mean, variance)
}

/// Elements of f32 the dense result may take: a fraction of physical memory.
fn result_budget_elements() -> usize {
    physical_memory_bytes()
        .map(|bytes| {
            (bytes as f64 * RESULT_BUDGET_FRACTION / std::mem::size_of::<f32>() as f64) as usize
        })
        .unwrap_or(FALLBACK_BUDGET_ELEMENTS)
}

/// The machine's physical memory, from `hw.memsize`.
#[cfg(target_os = "macos")]
fn physical_memory_bytes() -> Option<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // Safety: the out-pointer and its length describe one u64 we own.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && size > 0).then_some(size)
}

#[cfg(not(target_os = "macos"))]
fn physical_memory_bytes() -> Option<u64> {
    None
}

/// Refuse an input whose dense result would not fit the budget.
fn check_result_budget(n_cells: usize, n_genes: usize, budget: usize) -> Result<()> {
    let elements = n_cells.saturating_mul(n_genes);
    if elements > budget {
        let gib = |e: usize| e as f64 * std::mem::size_of::<f32>() as f64 / (1u64 << 30) as f64;
        return Err(Error::shape(
            format!(
                "a dense result within {:.1} GiB, {:.0}% of this machine's memory",
                gib(budget),
                RESULT_BUDGET_FRACTION * 100.0
            ),
            format!(
                "{n_cells} cells x {n_genes} genes needs {:.1} GiB",
                gib(elements)
            ),
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

    fn gpu() -> Option<Device> {
        DeviceKind::Gpu.resolve().ok()
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
    fn refuses_a_dense_result_beyond_the_budget() {
        let eight_gib = 8 * (1usize << 30) / 4;
        assert!(check_result_budget(50_000, 20_000, eight_gib).is_ok());
        assert!(check_result_budget(50_000, 50_000, eight_gib).is_err());
        assert!(check_result_budget(usize::MAX, 2, eight_gib).is_err());
        assert!(result_budget_elements() >= eight_gib / 4);
    }

    #[test]
    fn column_blocks_densify_to_the_same_matrix() {
        let dense = Array2::from_shape_fn((7, 5), |(i, j)| {
            if (i * j) % 3 == 0 {
                0.0
            } else {
                (i + j) as f32
            }
        });
        let csr = CsrMatrix::from_dense(dense.as_slice().unwrap(), 7, 5).unwrap();
        let columns = ColumnBlocks::from_csr(&csr);
        assert_eq!(columns.block(0, 5), dense);
        assert_eq!(columns.block(2, 4), dense.slice(s![.., 2..4]).to_owned());
    }

    #[test]
    fn the_sparse_entry_points_match_the_dense_ones() {
        let (expression, batch) = planted_batch_effect(40, 9);
        let csr = CsrMatrix::from_dense(expression.as_slice().unwrap(), 80, 9).unwrap();
        let covariates = normal_matrix(80, 2, 21);
        let dense = regress_out(&expression, &covariates, &Device::Cpu).unwrap();
        let sparse = regress_out_csr(&csr, &covariates, &Device::Cpu).unwrap();
        assert!(max_deviation(&dense, &sparse) < 1e-5);
        let dense = combat(&expression, &batch, 2, None, &Device::Cpu).unwrap();
        let sparse = combat_csr(&csr, &batch, 2, None, &Device::Cpu).unwrap();
        assert!(max_deviation(&dense, &sparse) < 1e-4);
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
        let Some(gpu) = gpu() else {
            return; // no GPU on this machine
        };
        let covariates = normal_matrix(200, 2, 14);
        let expression = normal_matrix(200, 50, 15);
        let on_cpu = regress_out(&expression, &covariates, &Device::Cpu).unwrap();
        let on_gpu = regress_out(&expression, &covariates, &gpu).unwrap();
        assert!(max_deviation(&on_cpu, &on_gpu) < 1e-4);

        let (expression, batch) = planted_batch_effect(100, 40);
        let on_cpu = combat(&expression, &batch, 2, None, &Device::Cpu).unwrap();
        let on_gpu = combat(&expression, &batch, 2, None, &gpu).unwrap();
        assert!(max_deviation(&on_cpu, &on_gpu) < 1e-3);
    }
}
