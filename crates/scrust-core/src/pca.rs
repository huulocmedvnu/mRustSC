use candle_core::{Device, Tensor};
use ndarray::Array2;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Extra sketch columns beyond `n_components`.
///
/// scikit-learn's `randomized_svd` defaults to `n_oversamples=10`; **we
/// deliberately use more**, and the reason is measured rather than assumed.
///
/// Two separate questions were asked of this constant.
///
/// * On a *decaying* spectrum it makes no difference: against an exact SVD on
///   synthetic matrices with `sigma_1/sigma_k` from 1e2 to 1e5, 24 was never
///   better than 10 by more than a factor of two, and was sometimes worse. In
///   particular it does **not** compensate for the range finder losing rank —
///   that failure (see `RANK_TOLERANCE`) leaves the same numerical rank at 10
///   and at 24.
/// * On a *near-degenerate* spectrum it matters a great deal, and the effect is
///   a property of randomised SVD rather than of this implementation. On 300
///   cells x 200 genes of uniform counts, where adjacent singular values differ
///   by well under a percent, correlation with `arpack` on the tenth component
///   is 0.37 for scikit-learn at its own default of 10, 0.99 at 24, and 1.00 at
///   40. Single-cell noise components look exactly like this.
///
/// So the extra columns are bought for the degenerate case, not to paper over a
/// weaker kernel. `tests/test_pca_audit.py::test_oversampling_is_earned_on_a_degenerate_spectrum`
/// holds that justification in place: it fails if 24 stops beating 10.
const OVERSAMPLING: usize = 24;

/// Power iterations applied to the sketch.
///
/// Single-cell spectra decay slowly, so the plain sketch mixes neighbouring
/// components. Each power iteration raises the spectral gap to the power three;
/// the count follows scikit-learn's rule, chosen at the call site.
const POWER_ITERATIONS_WELL_SEPARATED: usize = 7;
const POWER_ITERATIONS_DENSE_SPECTRUM: usize = 4;

/// Rows densified at a time. The dense block is the only large temporary, and at
/// this size it stays a few tens of megabytes for realistic gene counts.
const ROW_BLOCK: usize = 2048;

/// Floor on the Gram eigenvalues used to whiten a sketch, as a fraction of the
/// largest. The Gram matrix squares the condition number, so an f32 Gram carries
/// no information below roughly this ratio and `1/sqrt(lambda)` below it would
/// amplify rounding noise rather than a direction.
///
/// The floor *clamps*; it must not zero. Zeroing deletes the direction from the
/// basis permanently — the column stays zero through every later power iteration
/// — so the range finder silently loses rank and the trailing components come
/// back as exact zeros. Clamping keeps the direction, merely under-normalised,
/// and the next power iteration re-amplifies whatever data it carries.
const RANK_TOLERANCE: f64 = 1e-7;

const JACOBI_SWEEPS: usize = 60;

/// Principal components of a cells-by-genes matrix.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Cell coordinates, `(n_cells, n_components)` — AnnData's `obsm["X_pca"]`.
    pub embedding: Array2<f32>,
    /// Gene loadings, `(n_components, n_genes)` — AnnData's `varm["PCs"]` transposed.
    pub components: Array2<f32>,
    pub explained_variance: Vec<f32>,
    pub explained_variance_ratio: Vec<f32>,
}

/// Truncated PCA by randomised SVD, as `scanpy.pp.pca`.
///
/// Randomised range finding turns the decomposition into a few large matmuls,
/// which is what makes it worth moving to the GPU.
pub fn pca(
    matrix: &CsrMatrix,
    n_components: usize,
    zero_center: bool,
    seed: u64,
    device: &Device,
) -> Result<PcaResult> {
    let n_cells = matrix.n_rows();
    let n_genes = matrix.n_cols();
    let max_components = n_cells.min(n_genes);
    if n_components == 0 {
        return Err(Error::parameter("n_components", "at least 1", n_components));
    }
    if n_components > max_components {
        return Err(Error::parameter(
            "n_components",
            "at most min(n_cells, n_genes)",
            n_components,
        ));
    }
    if n_cells < 2 {
        return Err(Error::parameter("n_cells", "at least 2", n_cells));
    }

    let stats = ColumnStats::of(matrix, zero_center);
    let centred = CentredMatrix::new(matrix, &stats, zero_center, device)?;

    let sketch_width = (n_components + OVERSAMPLING).min(max_components);
    let omega = gaussian_matrix(n_genes, sketch_width, seed, device)?;
    let mut range = orthonormalize(&centred.times(&omega)?)?;
    // scikit-learn's rule: asking for a small share of the spectrum leaves a
    // wide gap at the truncation point and converges quickly, while asking for
    // a large share lands among near-degenerate singular values where extra
    // iterations buy little. Two iterations, the textbook figure, leaves the
    // trailing components visibly noisier than the reference.
    let power_iterations = if (n_components as f64) < 0.1 * max_components as f64 {
        POWER_ITERATIONS_WELL_SEPARATED
    } else {
        POWER_ITERATIONS_DENSE_SPECTRUM
    };
    for _ in 0..power_iterations {
        let genewise = orthonormalize(&centred.transpose_times(&range)?)?;
        range = orthonormalize(&centred.times(&genewise)?)?;
    }

    // `B = Q^T X_centred` is the only decomposition left, and `B B^T` is
    // `(sketch_width, sketch_width)` — small enough to solve exactly on the CPU.
    // We keep `B^T` because that is what the gene loadings are built from.
    //
    // KNOWN DIVERGENCE. scikit-learn takes an exact SVD of `B`
    // (`extmath.py::_randomized_svd`, `linalg.svd(B, full_matrices=False)`);
    // eigendecomposing `B B^T` instead squares the condition number, so the
    // relative error in `sigma_i` grows like `eps * (sigma_1/sigma_i)^2` rather
    // than `eps * (sigma_1/sigma_i)`. Measured on a synthetic spectrum with
    // `sigma_1/sigma_20 = 1e4`, components 9-16 come out 20x-500x further from
    // an exact SVD than scikit-learn's, both in f32. Everything above
    // `sigma_1/sigma_i ~ 30` is indistinguishable, which is why the leading
    // components and the variance ratios agree.
    //
    // Fixing it means a one-sided Jacobi SVD of `B^T` directly — O(n_genes *
    // sketch_width^2) per sweep in f64 on the CPU, giving up the GPU for the
    // final step — or an f64 Gram, which candle cannot do on Metal. Held by a
    // deliberately failing test:
    // `tests/test_pca_audit.py::test_trailing_singular_values_survive_an_ill_conditioned_spectrum`.
    let b_transpose = centred.transpose_times(&range)?;
    let gram = b_transpose.t()?.contiguous()?.matmul(&b_transpose)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen(to_f64_rows(&gram)?, sketch_width)?;
    let leading = descending_order(&eigenvalues);

    let singular: Vec<f64> = leading
        .iter()
        .take(n_components)
        .map(|&index| eigenvalues[index].max(0.0).sqrt())
        .collect();

    let mut scaled = vec![0.0f32; sketch_width * n_components];
    let mut basis = vec![0.0f32; sketch_width * n_components];
    for (column, (&index, &value)) in leading.iter().zip(singular.iter()).enumerate() {
        let inverse = if value > 0.0 { 1.0 / value } else { 0.0 };
        for row in 0..sketch_width {
            let entry = eigenvectors[row * sketch_width + index];
            basis[row * n_components + column] = entry as f32;
            scaled[row * n_components + column] = (entry * inverse) as f32;
        }
    }
    let basis = Tensor::from_vec(basis, (sketch_width, n_components), device)?;
    let scaled = Tensor::from_vec(scaled, (sketch_width, n_components), device)?;

    let singular_row = Tensor::from_vec(
        singular
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>(),
        (1, n_components),
        device,
    )?;
    let embedding = range.matmul(&basis)?.broadcast_mul(&singular_row)?;
    let components = b_transpose.matmul(&scaled)?.t()?.contiguous()?;

    let mut embedding = to_array2(&embedding)?;
    let mut components = to_array2(&components)?;
    fix_component_signs(&mut embedding, &mut components);

    // scanpy dispatches on `zero_center`, and the two branches do not report the
    // same statistic:
    //
    //   zero_center=True  -> sklearn PCA:          S^2 / (n - 1), over
    //                        `total_var = sum_g var(x_g, ddof=1)`
    //   zero_center=False -> sklearn TruncatedSVD: `np.var(X @ V^T, axis=0)`,
    //                        i.e. ddof=0 and with the *mean of each score column
    //                        removed*, over `sum_g var(x_g, ddof=0)`
    //
    // Reporting S^2/(n-1) in the uncentred case makes the first component — the
    // one that carries the grand mean — read as the largest, when scanpy reports
    // it as one of the smallest. See `sklearn/decomposition/_truncated_svd.py`
    // lines 262-273 and `_pca.py` lines 765-779.
    let explained_variance: Vec<f32> = if zero_center {
        let denominator = (n_cells - 1) as f64;
        singular
            .iter()
            .map(|&value| (value * value / denominator) as f32)
            .collect()
    } else {
        column_variances(&embedding)
    };
    let explained_variance_ratio = explained_variance
        .iter()
        .map(|&value| {
            if stats.total_variance > 0.0 {
                (value as f64 / stats.total_variance) as f32
            } else {
                0.0
            }
        })
        .collect();

    Ok(PcaResult {
        embedding,
        components,
        explained_variance,
        explained_variance_ratio,
    })
}

/// Per-gene means and the total variance the ratios are reported against.
struct ColumnStats {
    mean: Vec<f32>,
    /// Sum over **all** genes, not just the retained components: that is the
    /// denominator scanpy reports `variance_ratio` against. `ddof = 1` when
    /// centring (sklearn `PCA`), `ddof = 0` when not (sklearn `TruncatedSVD`).
    total_variance: f64,
}

impl ColumnStats {
    fn of(matrix: &CsrMatrix, zero_center: bool) -> Self {
        let n_genes = matrix.n_cols();
        let n_cells = matrix.n_rows() as f64;
        let mut sums = vec![0.0f64; n_genes];
        let mut squares = vec![0.0f64; n_genes];
        for (&column, &value) in matrix.indices().iter().zip(matrix.values()) {
            let value = value as f64;
            sums[column as usize] += value;
            squares[column as usize] += value * value;
        }
        let mean = sums
            .iter()
            .map(|&sum| (sum / n_cells) as f32)
            .collect::<Vec<_>>();
        // scanpy's uncentred path is sklearn's `TruncatedSVD`, whose `full_var`
        // is `mean_variance_axis(X, axis=0)[1].sum()` — still the *variance* of
        // each gene, just with `ddof = 0`. It is not the second moment.
        let ddof = if zero_center { 1.0 } else { 0.0 };
        let total_variance = sums
            .iter()
            .zip(&squares)
            .map(|(&sum, &square)| (square - sum * sum / n_cells).max(0.0) / (n_cells - ddof))
            .sum();
        Self {
            mean,
            total_variance,
        }
    }
}

/// The sparse matrix with its per-gene mean subtracted, exposed only through the
/// two products the range finder needs.
struct CentredMatrix<'a> {
    matrix: &'a CsrMatrix,
    /// Per-gene mean as `(1, n_genes)`, or `None` when centring is off.
    mean: Option<Tensor>,
    device: &'a Device,
}

impl<'a> CentredMatrix<'a> {
    fn new(
        matrix: &'a CsrMatrix,
        stats: &ColumnStats,
        zero_center: bool,
        device: &'a Device,
    ) -> Result<Self> {
        let mean = if zero_center {
            Some(Tensor::from_vec(
                stats.mean.clone(),
                (1, matrix.n_cols()),
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            matrix,
            mean,
            device,
        })
    }

    /// `X_centred @ rhs`, with `rhs` of shape `(n_genes, width)`.
    fn times(&self, rhs: &Tensor) -> Result<Tensor> {
        let mut blocks = Vec::new();
        for start in (0..self.matrix.n_rows()).step_by(ROW_BLOCK) {
            let end = (start + ROW_BLOCK).min(self.matrix.n_rows());
            let dense = self.matrix.to_tensor_rows(start, end, self.device)?;
            blocks.push(dense.matmul(rhs)?);
        }
        let product = Tensor::cat(&blocks, 0)?;
        match &self.mean {
            // X_centred @ rhs == X @ rhs - ones @ (mean^T @ rhs): centring is a
            // rank-one correction on the sketch, so the dense centred matrix —
            // which would have no zeros left to exploit — is never formed.
            Some(mean) => Ok(product.broadcast_sub(&mean.matmul(rhs)?)?),
            None => Ok(product),
        }
    }

    /// `X_centred^T @ lhs`, with `lhs` of shape `(n_cells, width)`.
    fn transpose_times(&self, lhs: &Tensor) -> Result<Tensor> {
        let mut accumulated: Option<Tensor> = None;
        for start in (0..self.matrix.n_rows()).step_by(ROW_BLOCK) {
            let end = (start + ROW_BLOCK).min(self.matrix.n_rows());
            let dense = self.matrix.to_tensor_rows(start, end, self.device)?;
            let rows = lhs.narrow(0, start, end - start)?.contiguous()?;
            let part = dense.t()?.contiguous()?.matmul(&rows)?;
            accumulated = Some(match accumulated {
                Some(total) => total.add(&part)?,
                None => part,
            });
        }
        let product =
            accumulated.ok_or_else(|| Error::shape("at least one cell", "an empty matrix"))?;
        match &self.mean {
            // The mirror of the correction above: mean (n_genes, 1) times the
            // column sums of `lhs` (1, width).
            Some(mean) => {
                let correction = mean.t()?.contiguous()?.matmul(&lhs.sum_keepdim(0)?)?;
                Ok(product.sub(&correction)?)
            }
            None => Ok(product),
        }
    }
}

/// Gaussian test matrix drawn from `seed`, reproducible run to run.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let count = rows * cols;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(count);
    // Box-Muller: `rand` alone offers no normal distribution, and a new
    // dependency is not worth two transcendental calls per pair of samples.
    while values.len() < count {
        let uniform: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let angle = std::f64::consts::TAU * rng.gen::<f64>();
        let radius = (-2.0 * uniform.ln()).sqrt();
        values.push((radius * angle.cos()) as f32);
        if values.len() < count {
            values.push((radius * angle.sin()) as f32);
        }
    }
    Ok(Tensor::from_vec(values, (rows, cols), device)?)
}

/// Orthonormal basis for the columns of `y`, computed on `y`'s device.
///
/// Whitening by the inverse square root of the Gram matrix is a single matmul,
/// but the Gram matrix squares the condition number and one pass leaves f32
/// visibly non-orthogonal. Repeating it (the CholeskyQR2 trick) restores
/// orthogonality to near machine precision at the cost of one small matmul.
fn orthonormalize(y: &Tensor) -> Result<Tensor> {
    whiten(&whiten(y)?)
}

fn whiten(y: &Tensor) -> Result<Tensor> {
    let width = y.dim(1)?;
    let gram = y.t()?.contiguous()?.matmul(y)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen(to_f64_rows(&gram)?, width)?;
    let largest = eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b));

    let floor = largest * RANK_TOLERANCE;
    let mut transform = vec![0.0f32; width * width];
    for column in 0..width {
        // Clamp, never zero: see `RANK_TOLERANCE`. A direction below the floor is
        // left short of unit norm instead of being deleted from the basis.
        let scale = if floor > 0.0 {
            1.0 / eigenvalues[column].max(floor).sqrt()
        } else {
            0.0
        };
        for row in 0..width {
            transform[row * width + column] = (eigenvectors[row * width + column] * scale) as f32;
        }
    }
    let transform = Tensor::from_vec(transform, (width, width), y.device())?;
    Ok(y.matmul(&transform)?)
}

/// Eigenvalues and eigenvectors of a symmetric row-major matrix by cyclic Jacobi
/// rotations. Column `j` of the returned vectors belongs to eigenvalue `j`.
fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut vectors = vec![0.0f64; n * n];
    for index in 0..n {
        vectors[index * n + index] = 1.0;
    }
    let scale: f64 = a.iter().map(|value| value * value).sum();
    let tolerance = f64::EPSILON * f64::EPSILON * scale.max(f64::MIN_POSITIVE);

    for _ in 0..JACOBI_SWEEPS {
        let mut off_diagonal = 0.0;
        for row in 0..n {
            for column in 0..n {
                if row != column {
                    off_diagonal += a[row * n + column] * a[row * n + column];
                }
            }
        }
        if off_diagonal <= tolerance {
            let eigenvalues = (0..n).map(|index| a[index * n + index]).collect();
            return Ok((eigenvalues, vectors));
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let pivot = a[p * n + q];
                if pivot == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * pivot);
                let tangent =
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()).max(f64::EPSILON);
                let cosine = 1.0 / (tangent * tangent + 1.0).sqrt();
                let sine = tangent * cosine;
                for k in 0..n {
                    let left = a[k * n + p];
                    let right = a[k * n + q];
                    a[k * n + p] = cosine * left - sine * right;
                    a[k * n + q] = sine * left + cosine * right;
                }
                for k in 0..n {
                    let left = a[p * n + k];
                    let right = a[q * n + k];
                    a[p * n + k] = cosine * left - sine * right;
                    a[q * n + k] = sine * left + cosine * right;
                }
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    let left = vectors[k * n + p];
                    let right = vectors[k * n + q];
                    vectors[k * n + p] = cosine * left - sine * right;
                    vectors[k * n + q] = sine * left + cosine * right;
                }
            }
        }
    }
    Err(Error::NotConverged {
        operation: "Jacobi eigenvalue iteration",
        iterations: JACOBI_SWEEPS,
    })
}

/// Population variance (`ddof = 0`) of every column, accumulated in f64.
///
/// This is `np.var(X_transformed, axis=0)`, which is what `TruncatedSVD` reports
/// as `explained_variance_` — the mean of each score column is removed first,
/// which matters precisely because the uncentred first component carries it.
fn column_variances(embedding: &Array2<f32>) -> Vec<f32> {
    let n_rows = embedding.nrows() as f64;
    (0..embedding.ncols())
        .map(|column| {
            let values = embedding.column(column);
            let mean = values.iter().map(|&value| value as f64).sum::<f64>() / n_rows;
            let variance = values
                .iter()
                .map(|&value| {
                    let centred = value as f64 - mean;
                    centred * centred
                })
                .sum::<f64>()
                / n_rows;
            variance as f32
        })
        .collect()
}

/// Indices of `values` sorted from largest to smallest.
fn descending_order(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        values[right]
            .partial_cmp(&values[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// The sign of a singular vector pair is arbitrary. Making the largest-magnitude
/// loading of every component positive fixes it, so repeated runs, different
/// seeds and the two devices agree on orientation.
fn fix_component_signs(embedding: &mut Array2<f32>, components: &mut Array2<f32>) {
    for index in 0..components.nrows() {
        let mut extreme = 0.0f32;
        for &loading in components.row(index) {
            if loading.abs() > extreme.abs() {
                extreme = loading;
            }
        }
        if extreme < 0.0 {
            components
                .row_mut(index)
                .map_inplace(|value| *value = -*value);
            embedding
                .column_mut(index)
                .map_inplace(|value| *value = -*value);
        }
    }
}

fn to_f64_rows(tensor: &Tensor) -> Result<Vec<f64>> {
    Ok(tensor
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .map(f64::from)
        .collect())
}

fn to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
    let (rows, columns) = tensor.dims2()?;
    let flat: Vec<f32> = tensor.to_vec2::<f32>()?.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, columns), flat)
        .map_err(|_| Error::shape(format!("{rows}x{columns}"), "a mismatched buffer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear congruential generator, mirrored exactly in the Python script that
    /// produced the scanpy constants below so both sides see the same matrix.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f64 / (1u64 << 24) as f64
        }
    }

    const N_CELLS: usize = 300;
    const N_GENES: usize = 200;

    /// 300 x 200 sparse counts with four cell groups, each raised in its own
    /// block of 50 genes: three well-separated leading components over noise.
    fn scanpy_fixture() -> CsrMatrix {
        let mut rng = Lcg(42);
        let mut dense = vec![0.0f32; N_CELLS * N_GENES];
        for cell in 0..N_CELLS {
            let group = cell % 4;
            for gene in 0..N_GENES {
                let uniform = rng.next();
                let mut value = if uniform < 0.6 {
                    0.0
                } else {
                    (uniform * 10.0).floor() as f32
                };
                if gene / 50 == group {
                    value += 5.0;
                }
                dense[cell * N_GENES + gene] = value;
            }
        }
        CsrMatrix::from_dense(&dense, N_CELLS, N_GENES).unwrap()
    }

    /// scanpy 1.12.2: `sc.pp.pca(adata, n_comps=10, svd_solver="arpack")` on the
    /// fixture above. Only the three planted components are separated from the
    /// noise floor; the rest are degenerate and not comparable per component.
    const SCANPY_VARIANCE_RATIO: [f32; 3] = [0.09244546, 0.09100571, 0.08773189];
    const SCANPY_VARIANCE: [f32; 3] = [345.06, 339.686, 327.46625];
    const SCANPY_EMBEDDING: [[f32; N_CELLS]; 3] = [
        [
            -22.2898, -7.2824, -1.4166, 22.8133, -16.3447, -1.4756, 10.0057, 27.4164, -20.3634,
            -3.8464, 9.0384, 23.8869, -21.8694, -6.8921, -2.5794, 29.0790, -25.2479, -7.7782,
            -3.3499, 27.6164, -23.9846, -3.5791, 4.8239, 17.2979, -27.1007, -2.3978, -0.4851,
            30.0246, -20.2037, -11.3435, -1.2949, 27.0443, -26.2836, -8.4330, 2.2122, 30.1867,
            -19.0501, -7.7383, 3.0001, 25.9445, -22.6462, -5.5856, 12.1054, 19.0393, -22.0927,
            -9.6480, 10.0131, 25.1649, -30.0345, -3.8086, 3.4683, 24.5739, -22.1518, -11.6148,
            6.6051, 16.3775, -21.6199, -5.9939, -3.1861, 13.5142, -21.7551, -3.2935, -0.0545,
            23.5023, -21.8371, -8.1039, 2.4894, 29.2680, -25.6052, -8.3354, 2.3833, 27.5157,
            -23.0981, -5.1948, 3.4289, 24.4646, -27.1978, -7.3399, 4.1895, 26.2054, -21.9432,
            -7.6868, 3.7615, 25.5468, -21.9680, -6.8676, 0.9118, 30.6317, -25.3952, -5.6741,
            4.8555, 30.3165, -26.5535, -9.0416, 0.6976, 35.3590, -22.8465, -8.9633, 4.0141,
            28.3715, -27.5620, 0.9073, 3.2470, 25.3843, -22.2152, -1.6182, 1.8944, 26.0564,
            -27.6224, -7.0948, 2.0953, 22.4012, -30.3431, -3.4680, 11.5525, 25.3788, -17.6248,
            -6.6098, 4.2827, 27.0276, -26.2991, -7.4631, 4.8601, 28.0892, -28.7417, -1.7773,
            4.0841, 25.7096, -27.7271, -11.9170, 1.0912, 30.8588, -23.5297, -2.4671, -1.5247,
            27.9517, -27.9265, -6.0484, -2.0249, 25.4005, -23.2857, -4.9382, 4.8727, 21.2477,
            -24.8337, -5.4203, 5.5770, 17.4328, -27.0669, -7.7996, 1.6072, 27.2627, -21.6536,
            -3.3223, 7.2043, 31.8081, -25.7813, -6.5785, 7.2987, 22.4870, -17.3467, -6.1056,
            -1.6631, 30.9639, -31.8440, -6.5502, 0.1475, 18.5138, -16.2263, -5.6819, 6.6200,
            20.4548, -22.8969, -1.6210, 8.8413, 28.5865, -30.1353, -2.8206, 10.7276, 29.6840,
            -27.2370, -5.6337, 3.7298, 21.4189, -27.4690, -8.9418, 7.7671, 29.3963, -28.1743,
            -9.7980, 3.4338, 34.0713, -27.2278, -9.4505, 1.2236, 24.9941, -23.2666, 4.1708, 6.1583,
            33.4823, -23.6978, -11.4265, 3.9669, 26.9208, -16.7468, -9.6461, 2.2173, 35.1842,
            -31.0956, -8.7726, 12.5133, 21.4431, -31.0160, -2.9239, 7.5441, 25.3391, -23.7308,
            -1.9051, 5.4466, 27.7033, -23.2582, -8.6981, 3.2056, 30.1285, -15.1363, -5.4608,
            3.9041, 30.6545, -27.7409, -10.0918, 0.9318, 25.7056, -29.4971, -8.5206, -2.7591,
            31.6384, -23.9350, -4.0826, 11.1707, 26.4057, -24.6455, -3.6708, 6.5888, 28.2553,
            -24.0335, -6.1872, 4.3844, 27.9737, -22.1512, -7.2147, -0.9236, 22.7603, -27.2923,
            -0.1272, 4.2452, 24.4048, -20.1311, -5.6131, 3.0017, 23.9862, -19.9291, -0.2724,
            3.2353, 25.7908, -17.7159, -6.1706, 5.7695, 28.4750, -26.5612, -9.5263, 4.0618,
            36.4828, -27.0050, 0.2243, 5.0647, 23.6538, -23.5879, -11.8252, 8.7533, 22.2837,
            -27.6919, -0.9341, 2.3789, 22.8367, -21.9604, -5.8961, 6.6702, 27.2298, -25.5075,
            -6.6425, -1.1961, 25.6523, -19.9050, -3.9428, 6.8797, 23.9327, -22.5511, -7.5507,
            5.0461, 19.9789,
        ],
        [
            -12.3519, 25.2370, -18.2245, -5.2704, -13.2169, 32.4385, -11.0151, 5.5999, -11.6191,
            27.5346, -14.0038, -5.2708, -13.0010, 27.4722, -8.7218, -2.6972, -8.0569, 27.7040,
            -12.2371, -8.6670, -12.2780, 29.5489, -14.1377, -5.3057, -10.4356, 23.5655, -15.2634,
            -4.7675, -9.3765, 32.3479, -20.7575, -3.1190, -12.8929, 28.9843, -12.7539, 0.9969,
            -8.8796, 27.6970, -19.6432, -2.1296, -6.7329, 30.3195, -12.1106, 2.5520, -9.7133,
            27.4059, -11.5222, -5.5085, -13.2548, 28.2862, -23.4979, -4.8595, -10.9710, 27.6442,
            -16.9500, -0.1157, -15.2105, 29.4684, -14.6279, -4.9368, -7.5386, 32.7811, -14.6542,
            2.3092, -12.4903, 30.4946, -14.6692, -0.1110, -8.2628, 35.9245, -20.2686, -0.1241,
            -11.4551, 27.9624, -16.2519, -1.0410, -16.1471, 32.7611, -17.1598, -1.4438, -13.7513,
            21.0848, -11.6735, -0.0876, -12.6193, 27.0514, -24.8344, -0.8697, -5.6209, 27.5382,
            -6.0079, -5.0564, -13.0356, 28.1652, -20.5076, 0.3412, -9.7060, 22.4767, -16.1320,
            -6.9691, -9.9118, 33.9927, -10.5801, -4.2261, -17.3862, 33.3801, -16.8383, -0.4991,
            -10.3978, 30.4010, -20.1127, 2.5957, -13.9038, 25.6152, -26.2621, -1.3333, -6.6952,
            35.5788, -23.8563, -1.9066, -6.9124, 24.7482, -20.5740, 4.0881, -13.4486, 29.0785,
            -20.2519, 6.5299, -12.1972, 30.4998, -12.7887, 5.2176, -7.8175, 29.2727, -14.3234,
            6.2464, -6.0108, 32.7559, -21.0512, -1.8368, -15.2338, 36.5179, -24.0131, -7.1736,
            -15.4552, 28.0717, -17.4020, -5.7455, -2.5964, 31.9406, -15.9130, -1.9470, -2.8045,
            20.6172, -16.3668, 3.4261, -12.3330, 37.0258, -23.6696, -8.0208, -8.5963, 23.0858,
            -26.9109, -3.7885, -18.7471, 23.8774, -11.2632, -5.4150, -3.7108, 30.8628, -21.0552,
            4.6829, -8.0826, 31.2482, -14.2258, 1.2423, -10.3926, 33.8846, -16.7857, -0.8616,
            -14.7610, 33.2703, -14.8934, -1.0274, -10.4043, 26.1538, -11.5031, 4.3199, -16.6664,
            31.1391, -15.2947, 0.3412, -15.2039, 32.1981, -15.3295, 2.1928, -6.4976, 32.2663,
            -22.1268, -0.9824, -8.8469, 28.9696, -25.2830, -4.8551, -11.8138, 30.8292, -16.2347,
            -5.8670, -12.5983, 24.1900, -20.2899, 0.2378, -12.1465, 35.4932, -16.0905, -3.8123,
            -13.6108, 30.6208, -20.2735, 0.7397, -13.1325, 29.7412, -24.5232, -1.2474, -14.5010,
            25.7305, -17.3797, 0.6090, -12.4599, 32.6176, -17.3744, -0.8814, -6.0895, 31.9768,
            -21.9605, -2.4470, -11.0849, 19.5589, -14.5862, 2.3422, -9.2997, 29.0206, -16.5261,
            3.1859, -8.7766, 38.6931, -9.8744, 2.1171, -8.2042, 36.7891, -19.9721, -7.0671,
            -17.5996, 27.2056, -19.8230, 2.0613, -15.0687, 31.6538, -22.1868, 1.2464, -4.1350,
            25.3032, -19.9618, 5.2738, -7.0437, 31.0478, -17.2829, 0.8683, -14.2649, 26.2740,
            -18.1109, -2.5638, -12.8723, 32.9055, -14.1871, -6.0338, -11.4514, 30.1691, -18.0670,
            6.6721, -16.9948, 23.8393, -18.0257, -7.0825, -9.3874, 26.4146, -13.3553, -10.7939,
            -6.3059, 28.8120, -22.0771, -0.3247, -11.2571, 29.0735, -7.5681, -2.4579, -13.3856,
            36.1657, -17.6413, -5.1722,
        ],
        [
            -12.9469, 4.5623, 20.6920, -17.2713, -18.9813, 5.5698, 32.4756, -13.5039, -17.8385,
            7.3763, 26.2966, -16.3504, -21.5301, 9.1113, 28.1981, -12.6257, -15.9931, 3.3986,
            25.5822, -16.7847, -13.6532, -1.7433, 28.8050, -17.2351, -7.4868, 13.3772, 28.0715,
            -16.2639, -18.2614, 9.0637, 24.4223, -16.3206, -14.2556, 16.1050, 27.3520, -18.1983,
            -15.8946, 3.1123, 24.2757, -16.7530, -14.7987, 8.2044, 26.2345, -12.5662, -21.2803,
            6.1962, 29.2229, -13.8887, -13.5653, 7.9842, 29.5403, -16.5975, -12.4310, -4.8675,
            19.0647, -14.5178, -10.9046, 12.5871, 24.3623, -17.6782, -15.0987, 9.0295, 28.2055,
            -14.3982, -16.8909, 6.2018, 32.0042, -14.5238, -20.0025, 11.7787, 21.6910, -19.5339,
            -17.2128, 15.2659, 26.1666, -17.4760, -14.2517, 2.5400, 31.9915, -23.6407, -12.4395,
            4.6375, 23.5546, -16.1180, -15.2597, 10.2228, 30.8864, -17.9372, -15.6110, 5.0483,
            24.5968, -24.8228, -16.0186, 8.8995, 22.9541, -17.5299, -13.3853, 5.3487, 21.4197,
            -13.3152, -22.0492, 11.3919, 25.4921, -16.7326, -11.7687, 7.1357, 27.4494, -14.1088,
            -17.6517, 13.5083, 27.3735, -10.8755, -18.5084, 5.9174, 30.7584, -17.2924, -10.8793,
            12.1128, 22.0839, -21.7669, -19.1249, 7.3767, 26.3268, -24.7430, -20.2357, 8.9913,
            21.6611, -16.2718, -13.9991, 8.2450, 25.4778, -12.7041, -21.0571, 13.8188, 25.0081,
            -18.5289, -22.5533, 13.2965, 27.3106, -18.0605, -18.9490, 2.6412, 25.0225, -14.4453,
            -23.2932, 11.7629, 19.3199, -25.8262, -12.4264, 6.5754, 22.7355, -15.2734, -25.0752,
            5.6607, 21.1598, -14.8483, -13.2579, 8.3047, 28.5181, -10.2255, -9.7718, 13.7630,
            28.2246, -15.8287, -12.0434, 17.2500, 31.3997, -20.4812, -18.1792, 11.8354, 19.5716,
            -16.8357, -13.4597, -2.8537, 25.1720, -16.7541, -19.3962, 6.1866, 24.8264, -20.6220,
            -18.8372, 4.9589, 25.2429, -19.3207, -9.0938, 12.5930, 16.5754, -11.0308, -20.5982,
            8.9485, 29.5289, -11.1441, -13.1831, 7.5738, 30.9057, -19.3157, -11.0214, 12.8279,
            29.2532, -20.9121, -20.2970, 12.0113, 18.1593, -13.3901, -11.8820, 9.3036, 16.4452,
            -18.8830, -20.8512, 5.8295, 28.4477, -20.6755, -13.7319, 6.4258, 26.8095, -12.0215,
            -16.0955, 6.6215, 20.6236, -16.8398, -19.0456, 12.6421, 24.5441, -12.7339, -11.8219,
            3.6710, 18.5289, -17.2645, -16.7255, 8.5243, 22.9562, -18.7371, -16.3591, 8.7163,
            25.8742, -20.5950, -17.0270, 9.7877, 19.8564, -7.4240, -15.7481, 9.1526, 22.6385,
            -20.1021, -16.0608, 6.4156, 24.7352, -13.5117, -18.9007, 7.6520, 24.9074, -17.9935,
            -19.8068, 3.9547, 23.2097, -21.6310, -15.4884, 6.0019, 27.4277, -23.2044, -17.9173,
            15.7274, 25.9275, -17.0500, -10.8616, 9.3415, 27.2511, -19.4069, -18.0943, 5.9777,
            19.4271, -19.1717, -18.4947, -1.1674, 25.8648, -14.0517, -16.7526, 5.3692, 27.9725,
            -10.0403, -15.0345, 6.4832, 21.2247, -15.1753, -12.3809, -3.0850, 18.6810, -18.2740,
            -12.2275, 6.8885, 30.2138, -26.2866, -18.7904, 13.7583, 25.4471, -14.5295, -19.2742,
            7.2130, 26.9781, -19.7974,
        ],
    ];

    fn correlation(left: &[f32], right: &[f32]) -> f32 {
        let n = left.len() as f32;
        let mean_left = left.iter().sum::<f32>() / n;
        let mean_right = right.iter().sum::<f32>() / n;
        let mut covariance = 0.0;
        let mut variance_left = 0.0;
        let mut variance_right = 0.0;
        for (&a, &b) in left.iter().zip(right) {
            covariance += (a - mean_left) * (b - mean_right);
            variance_left += (a - mean_left) * (a - mean_left);
            variance_right += (b - mean_right) * (b - mean_right);
        }
        covariance / (variance_left.sqrt() * variance_right.sqrt())
    }

    fn column(embedding: &Array2<f32>, index: usize) -> Vec<f32> {
        embedding.column(index).to_vec()
    }

    /// Three planted factors on disjoint gene blocks with orthogonal cell
    /// scores, so the exact spectrum is known, plus uniform noise everywhere.
    fn planted() -> CsrMatrix {
        const AMPLITUDES: [f32; 3] = [6.0, 4.0, 2.0];
        const BLOCK: usize = 20;
        let mut rng = Lcg(7);
        let scores = [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ];
        let mut dense = vec![0.0f32; N_CELLS * N_GENES];
        for cell in 0..N_CELLS {
            for gene in 0..N_GENES {
                let mut value = (rng.next() * 2.0 - 1.0) as f32;
                let factor = gene / BLOCK;
                if factor < 3 {
                    value += AMPLITUDES[factor] * scores[factor][cell % 4];
                }
                dense[cell * N_GENES + gene] = value;
            }
        }
        CsrMatrix::from_dense(&dense, N_CELLS, N_GENES).unwrap()
    }

    /// Relative Frobenius error of the rank-`k` reconstruction of the centred
    /// matrix.
    fn reconstruction_error(matrix: &CsrMatrix, result: &PcaResult) -> f32 {
        let dense = matrix.densify_rows(0, matrix.n_rows());
        let n_genes = matrix.n_cols();
        let mut means = vec![0.0f32; n_genes];
        for row in dense.chunks_exact(n_genes) {
            for (mean, &value) in means.iter_mut().zip(row) {
                *mean += value / matrix.n_rows() as f32;
            }
        }
        let approximation = result.embedding.dot(&result.components);
        let mut error = 0.0f64;
        let mut total = 0.0f64;
        for (cell, row) in dense.chunks_exact(n_genes).enumerate() {
            for (gene, &value) in row.iter().enumerate() {
                let centred = (value - means[gene]) as f64;
                let residual = centred - approximation[[cell, gene]] as f64;
                error += residual * residual;
                total += centred * centred;
            }
        }
        (error / total).sqrt() as f32
    }

    #[test]
    fn recovers_the_planted_spectrum() {
        let matrix = planted();
        let result = pca(&matrix, 6, true, 0, &Device::Cpu).unwrap();

        // Planted variances are block_size * amplitude^2, i.e. 720, 320, 80.
        let variance = &result.explained_variance;
        assert!((variance[0] / variance[1] - 720.0 / 320.0).abs() < 0.1);
        assert!((variance[1] / variance[2] - 320.0 / 80.0).abs() < 0.2);
        for pair in variance.windows(2) {
            assert!(pair[0] >= pair[1], "variances must be descending");
        }
        let planted_ratio: f32 = result.explained_variance_ratio[..3].iter().sum();
        assert!(
            planted_ratio > 0.9,
            "planted signal dominates: {planted_ratio}"
        );
        assert!(result.explained_variance_ratio[3] < 0.01);
    }

    #[test]
    fn reconstruction_error_falls_with_more_components() {
        let matrix = planted();
        let errors: Vec<f32> = [2, 3, 8]
            .iter()
            .map(|&k| {
                reconstruction_error(&matrix, &pca(&matrix, k, true, 0, &Device::Cpu).unwrap())
            })
            .collect();
        assert!(errors[0] > errors[1], "{errors:?}");
        assert!(errors[1] > errors[2], "{errors:?}");
        assert!(errors[2] < 0.3, "{errors:?}");
    }

    #[test]
    fn matches_scanpy_on_a_random_matrix() {
        let matrix = scanpy_fixture();
        let result = pca(&matrix, 10, true, 0, &Device::Cpu).unwrap();
        assert_eq!(result.embedding.dim(), (N_CELLS, 10));
        assert_eq!(result.components.dim(), (10, N_GENES));

        for index in 0..3 {
            let ours = column(&result.embedding, index);
            let correlation = correlation(&ours, &SCANPY_EMBEDDING[index]).abs();
            assert!(correlation >= 0.99, "component {index}: corr {correlation}");

            let ratio = result.explained_variance_ratio[index];
            let expected = SCANPY_VARIANCE_RATIO[index];
            assert!(
                (ratio - expected).abs() <= 1e-3 * expected,
                "component {index}: ratio {ratio} vs {expected}"
            );
            let variance = result.explained_variance[index];
            assert!((variance - SCANPY_VARIANCE[index]).abs() <= 1e-3 * SCANPY_VARIANCE[index]);
        }
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let matrix = scanpy_fixture();
        let first = pca(&matrix, 5, true, 3, &Device::Cpu).unwrap();
        let second = pca(&matrix, 5, true, 3, &Device::Cpu).unwrap();
        assert_eq!(first.embedding, second.embedding);
        assert_eq!(first.components, second.components);
        assert_eq!(first.explained_variance, second.explained_variance);
        assert_eq!(
            first.explained_variance_ratio,
            second.explained_variance_ratio
        );
    }

    #[test]
    fn a_different_seed_finds_the_same_subspace() {
        let matrix = scanpy_fixture();
        let first = pca(&matrix, 5, true, 0, &Device::Cpu).unwrap();
        let second = pca(&matrix, 5, true, 999, &Device::Cpu).unwrap();
        for index in 0..3 {
            let correlation = correlation(
                &column(&first.embedding, index),
                &column(&second.embedding, index),
            )
            .abs();
            assert!(correlation >= 0.99, "component {index}: corr {correlation}");
        }
    }

    #[test]
    fn cpu_and_gpu_agree() {
        if !crate::gpu_available() {
            return;
        }
        let device = crate::DeviceKind::Gpu.resolve().unwrap();
        let matrix = scanpy_fixture();
        let cpu = pca(&matrix, 5, true, 0, &Device::Cpu).unwrap();
        let gpu = pca(&matrix, 5, true, 0, &device).unwrap();
        for (left, right) in cpu.embedding.iter().zip(gpu.embedding.iter()) {
            assert!((left - right).abs() <= 1e-2 * left.abs().max(1.0));
        }
        for (left, right) in cpu
            .explained_variance_ratio
            .iter()
            .zip(&gpu.explained_variance_ratio)
        {
            assert!((left - right).abs() <= 1e-4 * left);
        }
    }

    #[test]
    fn uncentred_pca_reports_truncated_svd_statistics() {
        let matrix = scanpy_fixture();
        let result = pca(&matrix, 4, false, 0, &Device::Cpu).unwrap();
        assert!(result.explained_variance.iter().all(|&value| value > 0.0));

        // Without centring the first component carries the grand mean, so its
        // *scores* barely vary: sklearn's `TruncatedSVD` reports
        // `np.var(X @ V^T, axis=0)`, under which component 1 is one of the
        // smallest, not the largest. Reporting `S^2/(n-1)` instead would put it
        // an order of magnitude above the rest.
        assert!(
            result.explained_variance[0] < result.explained_variance[1],
            "uncentred component 1 must be the mean-carrying, low-variance one: {:?}",
            result.explained_variance
        );

        // Each entry must be the population variance of its own score column.
        let n_cells = matrix.n_rows() as f64;
        for index in 0..4 {
            let scores = column(&result.embedding, index);
            let mean = scores.iter().map(|&v| v as f64).sum::<f64>() / n_cells;
            let expected = scores
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / n_cells;
            let reported = result.explained_variance[index] as f64;
            assert!(
                (reported - expected).abs() <= 1e-4 * expected,
                "component {index}: {reported} vs {expected}"
            );
        }
    }

    #[test]
    fn rejects_invalid_component_counts() {
        let matrix = CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        assert!(pca(&matrix, 0, true, 0, &Device::Cpu).is_err());
        assert!(pca(&matrix, 3, true, 0, &Device::Cpu).is_err());
        assert!(pca(&matrix, 2, true, 0, &Device::Cpu).is_ok());
    }
}
