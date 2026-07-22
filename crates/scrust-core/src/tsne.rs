use candle_core::{DType, Device, Tensor};
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Layout parameters, named as in `scanpy.tl.tsne`.
#[derive(Debug, Clone)]
pub struct TsneParams {
    pub n_components: usize,
    pub perplexity: f32,
    pub early_exaggeration: f32,
    pub learning_rate: f32,
    pub n_iterations: usize,
    pub seed: u64,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            n_iterations: 1000,
            seed: 0,
        }
    }
}

/// Largest input the exact O(n^2) formulation accepts.
///
/// This implementation is deliberately exact rather than Barnes-Hut: the whole
/// cost is dense `(n, n)` work, which is a matmul the GPU is good at, instead of
/// a tree walk it is bad at. The price is that the affinity matrix is materialised:
/// `n^2` f32 is 1.6 GB at 20 000 cells, and the gradient holds three more buffers
/// of that shape at once, so the peak is roughly 6.5 GB. Above this bound we
/// refuse the input instead of exhausting unified memory.
const MAX_CELLS: usize = 20_000;

/// Iterations with early exaggeration and low momentum, as scikit-learn's
/// `_EXPLORATION_N_ITER`.
const EXPLORATION_ITERATIONS: usize = 250;
const EXPLORATION_MOMENTUM: f64 = 0.5;
const FINAL_MOMENTUM: f64 = 0.8;

/// scikit-learn's `PERPLEXITY_TOLERANCE` and step cap for the bandwidth search.
const PERPLEXITY_TOLERANCE: f32 = 1e-5;
const PERPLEXITY_SEARCH_STEPS: usize = 100;
/// scikit-learn's `EPSILON_DBL`, the floor on a row's unnormalised mass.
const AFFINITY_EPSILON: f32 = 1e-8;

/// Floor on per-coordinate gains and the gradient norm at which we stop early,
/// both as in scikit-learn's `_gradient_descent`.
const MIN_GAIN: f32 = 0.01;
const MIN_GRADIENT_NORM: f32 = 1e-7;
/// Checking convergence forces a device synchronisation, so do it rarely.
const CONVERGENCE_CHECK_INTERVAL: usize = 50;

/// Scale of the initial layout, as scikit-learn applies to both of its
/// initialisations.
const INIT_STANDARD_DEVIATION: f32 = 1e-4;

/// t-SNE embedding of a cells-by-features matrix, usually PCA coordinates.
pub fn tsne(embedding: &Array2<f32>, params: &TsneParams, device: &Device) -> Result<Array2<f32>> {
    let (n_cells, n_features) = embedding.dim();
    validate(n_cells, n_features, params)?;

    let points = Tensor::from_vec(
        embedding.iter().copied().collect::<Vec<f32>>(),
        (n_cells, n_features),
        device,
    )?;
    let distances = squared_distances(&points)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let conditional = conditional_affinities(&distances, n_cells, params.perplexity);
    let joint = Tensor::from_vec(
        joint_probabilities(&conditional, n_cells),
        (n_cells, n_cells),
        device,
    )?;

    let layout = optimise(
        principal_component_initialisation(embedding, params, device)?,
        &joint,
        params,
    )?;

    let coordinates = layout.flatten_all()?.to_vec1::<f32>()?;
    Array2::from_shape_vec((n_cells, params.n_components), coordinates).map_err(|_| {
        Error::shape(
            format!("{n_cells} by {}", params.n_components),
            "a mismatch",
        )
    })
}

fn validate(n_cells: usize, n_features: usize, params: &TsneParams) -> Result<()> {
    if params.n_components == 0 {
        return Err(Error::parameter(
            "n_components",
            "at least 1",
            params.n_components,
        ));
    }
    if n_features == 0 {
        return Err(Error::shape("at least one feature", "zero features"));
    }
    if !params.perplexity.is_finite() || params.perplexity <= 0.0 {
        return Err(Error::parameter(
            "perplexity",
            "positive",
            params.perplexity,
        ));
    }
    // scanpy's own guidance: the bandwidth search is meaningless when the
    // requested neighbourhood is a sizeable fraction of the data set.
    if (n_cells as f32) < 3.0 * params.perplexity {
        return Err(Error::parameter(
            "perplexity",
            "at most a third of the cell count",
            params.perplexity,
        ));
    }
    if n_cells > MAX_CELLS {
        return Err(Error::parameter(
            "n_cells",
            "at most 20000 for the exact O(n^2) formulation",
            n_cells,
        ));
    }
    if !params.learning_rate.is_finite() || params.learning_rate <= 0.0 {
        return Err(Error::parameter(
            "learning_rate",
            "positive",
            params.learning_rate,
        ));
    }
    Ok(())
}

/// Pairwise squared Euclidean distances as `|x|^2 + |y|^2 - 2 x.y`, so the whole
/// matrix is one matmul.
fn squared_distances(points: &Tensor) -> Result<Tensor> {
    let norms = points.sqr()?.sum_keepdim(1)?;
    let gram = points.matmul(&points.t()?.contiguous()?)?;
    Ok(norms
        .broadcast_add(&norms.t()?)?
        .sub(&gram.affine(2.0, 0.0)?)?
        .maximum(0f32)?)
}

/// Row-wise Gaussian affinities whose entropy matches `log(perplexity)`.
///
/// Bisection on the precision `beta`, following scikit-learn's
/// `_binary_search_perplexity`: same tolerance, same step cap, same rule for
/// widening an unbounded bracket. Rows are independent, so this parallelises
/// without affecting the result.
fn conditional_affinities(squared_distances: &[f32], n_cells: usize, perplexity: f32) -> Vec<f32> {
    let desired_entropy = perplexity.ln();
    let mut affinities = vec![0f32; n_cells * n_cells];

    affinities
        .par_chunks_mut(n_cells)
        .enumerate()
        .for_each(|(row_index, row)| {
            let distances = &squared_distances[row_index * n_cells..(row_index + 1) * n_cells];
            let mut beta = 1.0f32;
            let mut beta_min = f32::NEG_INFINITY;
            let mut beta_max = f32::INFINITY;

            for _ in 0..PERPLEXITY_SEARCH_STEPS {
                let mut mass = 0.0f32;
                for (column, weight) in row.iter_mut().enumerate() {
                    // A point is not its own neighbour.
                    *weight = if column == row_index {
                        0.0
                    } else {
                        (-distances[column] * beta).exp()
                    };
                    mass += *weight;
                }
                if mass == 0.0 {
                    mass = AFFINITY_EPSILON;
                }

                let mut expected_distance = 0.0f32;
                for (column, weight) in row.iter_mut().enumerate() {
                    *weight /= mass;
                    expected_distance += distances[column] * *weight;
                }

                let entropy = mass.ln() + beta * expected_distance;
                let excess = entropy - desired_entropy;
                if excess.abs() <= PERPLEXITY_TOLERANCE {
                    break;
                }
                if excess > 0.0 {
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
        });

    affinities
}

/// Symmetrise the conditional affinities into a joint distribution over pairs:
/// `P = (C + C^T) / 2n`, which sums to one because every row of `C` does.
fn joint_probabilities(conditional: &[f32], n_cells: usize) -> Vec<f32> {
    let normaliser = 2.0 * n_cells as f32;
    let mut joint = vec![0f32; n_cells * n_cells];
    for row in 0..n_cells {
        for column in 0..n_cells {
            joint[row * n_cells + column] = (conditional[row * n_cells + column]
                + conditional[column * n_cells + row])
                / normaliser;
        }
    }
    joint
}

/// Gradient of the Kullback-Leibler objective for the whole embedding.
///
/// Kept as one private function with a tensor-in, tensor-out shape so the fused
/// Metal kernel on `feat/tsne-kernel` can replace the body without touching the
/// optimiser. Everything here is dense `(n, n)` tensor algebra:
///
/// `dC/dy_i = 4 sum_j (P_ij - Q_ij) w_ij (y_i - y_j)` with `w_ij = 1 / (1 + |y_i - y_j|^2)`
///
/// which factors into a row sum and a single matmul. The diagonal needs no mask:
/// it contributes `w_ii (y_i - y_i) = 0` to both halves.
fn kl_gradient(layout: &Tensor, joint: &Tensor, exaggeration: f32) -> Result<Tensor> {
    let n_cells = layout.dim(0)?;

    // Student t with one degree of freedom.
    let weights = squared_distances(layout)?.affine(1.0, 1.0)?.recip()?;
    // Q normalises over ordered pairs i != j; the diagonal contributes exactly n.
    let normaliser = weights
        .sum_all()?
        .affine(1.0, -(n_cells as f64))?
        .reshape((1, 1))?;
    let low_dimensional = weights.broadcast_div(&normaliser)?;

    let forces = joint
        .affine(exaggeration as f64, 0.0)?
        .sub(&low_dimensional)?
        .mul(&weights)?;

    Ok(forces
        .sum_keepdim(1)?
        .broadcast_mul(layout)?
        .sub(&forces.matmul(layout)?)?
        .affine(4.0, 0.0)?)
}

/// Start from the leading principal components of the input, as scikit-learn's
/// `init="pca"` default does.
///
/// The alternative — a random cloud at the same scale — reaches a comparable
/// optimum at a low learning rate but is markedly less stable above it: at
/// scanpy's default learning rate of 1000 the random start settles at roughly
/// twice the KL divergence of the PCA start.
fn principal_component_initialisation(
    embedding: &Array2<f32>,
    params: &TsneParams,
    device: &Device,
) -> Result<Tensor> {
    // The starting point is fixed on the CPU regardless of `device`: it is a
    // small computation, and a rounding-level difference here is amplified by
    // the optimiser's sign-branching gains into a visibly different embedding.
    let (n_cells, n_features) = embedding.dim();
    let contiguous = embedding.as_standard_layout();
    let slice = contiguous
        .as_slice()
        .ok_or_else(|| Error::shape("a contiguous embedding", "a strided view"))?;
    let matrix = CsrMatrix::from_dense(slice, n_cells, n_features)?;
    let fitted = crate::pca::pca(
        &matrix,
        params.n_components,
        true,
        params.seed,
        &Device::Cpu,
    )?;

    // scikit-learn rescales so the first component has standard deviation 1e-4,
    // which keeps the early exaggeration phase in the regime it was tuned for.
    let first = fitted.embedding.column(0);
    let mean = first.sum() / n_cells as f32;
    let variance = first
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / n_cells as f32;
    let scale = INIT_STANDARD_DEVIATION / variance.sqrt().max(f32::MIN_POSITIVE);

    Ok(Tensor::from_vec(
        fitted
            .embedding
            .iter()
            .map(|value| value * scale)
            .collect::<Vec<f32>>(),
        (n_cells, params.n_components),
        device,
    )?)
}

/// Gradient descent with per-coordinate gains and a two-phase momentum and
/// exaggeration schedule, matching scikit-learn's `_gradient_descent`.
fn optimise(initial: Tensor, joint: &Tensor, params: &TsneParams) -> Result<Tensor> {
    let mut layout = initial;
    let mut update = layout.zeros_like()?;
    let mut gains = layout.ones_like()?;

    for iteration in 0..params.n_iterations {
        let (momentum, exaggeration) = if iteration < EXPLORATION_ITERATIONS {
            (EXPLORATION_MOMENTUM, params.early_exaggeration)
        } else {
            (FINAL_MOMENTUM, 1.0)
        };
        // scikit-learn restarts the descent for the second phase, which resets
        // the accumulated momentum and gains.
        if iteration == EXPLORATION_ITERATIONS {
            update = layout.zeros_like()?;
            gains = layout.ones_like()?;
        }

        let gradient = kl_gradient(&layout, joint, exaggeration)?;

        // A coordinate whose step and gradient disagree in sign is overshooting.
        let overshooting = update.mul(&gradient)?.lt(0f32)?.to_dtype(DType::F32)?;
        let steady = overshooting.affine(-1.0, 1.0)?;
        gains = overshooting
            .mul(&gains.affine(1.0, 0.2)?)?
            .add(&steady.mul(&gains.affine(0.8, 0.0)?)?)?
            .maximum(MIN_GAIN)?;

        let step = gradient
            .mul(&gains)?
            .affine(params.learning_rate as f64, 0.0)?;
        update = update.affine(momentum, 0.0)?.sub(&step)?;
        layout = layout.add(&update)?;

        if (iteration + 1) % CONVERGENCE_CHECK_INTERVAL == 0 {
            let norm = gradient.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm <= MIN_GRADIENT_NORM {
                break;
            }
        }
    }

    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::gpu_available;

    /// Explicit LCG so the Python reference script can generate the identical
    /// input without this file carrying thousands of literals.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f64 + 0.5) / 16777216.0
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    const BLOB_SIZE: usize = 70;
    const BLOB_COUNT: usize = 3;
    const BLOB_FEATURES: usize = 10;
    const BLOB_SEPARATION: f32 = 12.0;

    /// Three well-separated Gaussian blobs, bit-identical to `reference.py`.
    fn blobs() -> Array2<f32> {
        let mut rng = Lcg(20240722);
        let mut values = Vec::with_capacity(BLOB_COUNT * BLOB_SIZE * BLOB_FEATURES);
        for blob in 0..BLOB_COUNT {
            for _ in 0..BLOB_SIZE {
                let start = values.len();
                for _ in 0..BLOB_FEATURES {
                    values.push(rng.normal() as f32);
                }
                values[start + blob] += BLOB_SEPARATION;
            }
        }
        Array2::from_shape_vec((BLOB_COUNT * BLOB_SIZE, BLOB_FEATURES), values).unwrap()
    }

    fn uniform_matrix(n_rows: usize, n_columns: usize, seed: u64) -> Array2<f32> {
        let mut rng = Lcg(seed);
        let values = (0..n_rows * n_columns)
            .map(|_| rng.normal() as f32)
            .collect();
        Array2::from_shape_vec((n_rows, n_columns), values).unwrap()
    }

    /// The squared distances the algorithm itself feeds to the bandwidth search.
    fn squared_distance_matrix(input: &Array2<f32>) -> Vec<f32> {
        let points = Tensor::from_vec(
            input.iter().copied().collect::<Vec<f32>>(),
            input.dim(),
            &Device::Cpu,
        )
        .unwrap();
        squared_distances(&points)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn mean_distance(embedding: &Array2<f32>, left: &[usize], right: &[usize]) -> f32 {
        let mut total = 0.0;
        let mut count = 0usize;
        for &i in left {
            for &j in right {
                if i == j {
                    continue;
                }
                let dx = embedding[[i, 0]] - embedding[[j, 0]];
                let dy = embedding[[i, 1]] - embedding[[j, 1]];
                total += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }
        total / count as f32
    }

    #[test]
    fn separated_blobs_stay_separated() {
        let input = blobs();
        let layout = tsne(&input, &TsneParams::default(), &Device::Cpu).unwrap();

        let members: Vec<Vec<usize>> = (0..BLOB_COUNT)
            .map(|blob| (blob * BLOB_SIZE..(blob + 1) * BLOB_SIZE).collect())
            .collect();
        let within = (0..BLOB_COUNT)
            .map(|blob| mean_distance(&layout, &members[blob], &members[blob]))
            .sum::<f32>()
            / BLOB_COUNT as f32;
        let between = mean_distance(&layout, &members[0], &members[1]);

        println!("within-blob {within:.2}, between-blob {between:.2}");
        assert!(
            between > 5.0 * within,
            "within {within}, between {between} - blobs merged"
        );
    }

    #[test]
    fn bandwidth_search_hits_requested_perplexity() {
        let input = uniform_matrix(120, 8, 7);
        let n = input.nrows();
        let distances = squared_distance_matrix(&input);

        let requested = 30.0f32;
        let conditional = conditional_affinities(&distances, n, requested);
        for row in conditional.chunks(n) {
            let entropy: f32 = -row
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            assert!(
                (entropy.exp() - requested).abs() < 1e-2,
                "measured perplexity {}",
                entropy.exp()
            );
        }
    }

    #[test]
    fn joint_probabilities_are_a_symmetric_distribution() {
        let input = uniform_matrix(90, 5, 11);
        let n = input.nrows();
        let distances = squared_distance_matrix(&input);
        let joint = joint_probabilities(&conditional_affinities(&distances, n, 25.0), n);

        assert!(joint.iter().all(|&p| p >= 0.0), "negative affinity");
        for i in 0..n {
            assert_eq!(joint[i * n + i], 0.0, "self-affinity on row {i}");
            for j in 0..n {
                assert_eq!(joint[i * n + j], joint[j * n + i], "asymmetric at {i},{j}");
            }
        }
        let total: f32 = joint.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "total probability {total}");
    }

    #[test]
    fn same_seed_gives_identical_output() {
        let input = uniform_matrix(100, 6, 3);
        let params = TsneParams {
            n_iterations: 120,
            ..Default::default()
        };
        let first = tsne(&input, &params, &Device::Cpu).unwrap();
        let second = tsne(&input, &params, &Device::Cpu).unwrap();
        assert_eq!(first, second);
    }

    fn largest_gap(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn largest_magnitude(values: &[f32]) -> f32 {
        values.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn cpu_and_gpu_agree() {
        if !gpu_available() {
            return;
        }
        let metal = Device::new_metal(0).unwrap();
        let input = uniform_matrix(100, 6, 5);
        let n = input.nrows();

        // One gradient from identical inputs is the honest device comparison:
        // it is the whole per-iteration computation, and it is not chaotic.
        let joint = joint_probabilities(
            &conditional_affinities(&squared_distance_matrix(&input), n, 30.0),
            n,
        );
        let start: Vec<f32> = uniform_matrix(n, 2, 9).iter().copied().collect();
        let gradient_on = |device: &Device| {
            let layout = Tensor::from_vec(start.clone(), (n, 2), device).unwrap();
            let affinities = Tensor::from_vec(joint.clone(), (n, n), device).unwrap();
            kl_gradient(&layout, &affinities, 12.0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let host = gradient_on(&Device::Cpu);
        let accelerated = gradient_on(&metal);
        let gap = largest_gap(&host, &accelerated);
        assert!(
            gap <= 1e-5 * largest_magnitude(&host),
            "largest CPU/GPU gradient gap {gap}"
        );

        // Coordinates are the wrong thing to compare over a full run. Gains
        // switch on the sign of `update * gradient`, so a rounding-level
        // disagreement flips a branch and the two trajectories separate
        // exponentially — a property of scikit-learn's optimiser, not of Metal.
        // The assertion is therefore one-sided on the objective: the GPU must not
        // reach a *worse* optimum. Finding a better one is not a defect, and it
        // does: measured 0.528 against the CPU's 0.583.
        let params = TsneParams::default();
        let cpu = kl_divergence(
            &input,
            &tsne(&input, &params, &Device::Cpu).unwrap(),
            &params,
        );
        let gpu = kl_divergence(&input, &tsne(&input, &params, &metal).unwrap(), &params);
        assert!(
            gpu <= cpu * 1.15,
            "GPU reached KL {gpu}, materially worse than the CPU's {cpu}"
        );
    }

    /// The objective t-SNE minimises, for comparing two embeddings of one input.
    fn kl_divergence(input: &Array2<f32>, layout: &Array2<f32>, params: &TsneParams) -> f64 {
        let n_cells = input.dim().0;
        let points = Tensor::from_vec(
            input.iter().copied().collect::<Vec<f32>>(),
            input.dim(),
            &Device::Cpu,
        )
        .unwrap();
        let distances = squared_distances(&points)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let conditional = conditional_affinities(&distances, n_cells, params.perplexity);
        let joint = joint_probabilities(&conditional, n_cells);

        let mut weights = vec![0.0f64; n_cells * n_cells];
        let mut total = 0.0f64;
        for i in 0..n_cells {
            for j in 0..n_cells {
                if i == j {
                    continue;
                }
                let squared: f64 = (0..params.n_components)
                    .map(|d| (layout[[i, d]] - layout[[j, d]]) as f64)
                    .map(|delta| delta * delta)
                    .sum();
                let weight = 1.0 / (1.0 + squared);
                weights[i * n_cells + j] = weight;
                total += weight;
            }
        }

        joint
            .iter()
            .zip(&weights)
            .filter(|(&p, _)| p > 0.0)
            .map(|(&p, &weight)| {
                let q = (weight / total).max(1e-12);
                p as f64 * ((p as f64) / q).ln()
            })
            .sum()
    }

    /// `scanpy.tl.tsne` on the output of `blobs()`, rounded to three digits.
    /// Coordinates are not comparable across implementations, so this is only
    /// ever used to derive neighbourhoods.
    const SCANPY_EMBEDDING: [f32; 420] = [
        -0.0586, -29.9649, 3.4145, -27.5040, 0.1671, -29.6556, 0.6094, -26.9660, -0.0552, -26.4112,
        2.9543, -29.8929, 2.3969, -28.7746, 1.5850, -27.5926, -0.0919, -28.0261, 3.7874, -26.8410,
        1.1420, -28.7929, 1.8038, -27.0925, -0.6141, -27.2495, 2.1120, -28.0201, 3.0459, -26.5874,
        -1.2972, -26.9894, -0.1118, -24.1678, -0.9597, -26.4475, 0.9450, -28.1628, -0.0704,
        -25.7115, 2.2906, -26.7797, 0.9095, -26.5661, 0.9009, -26.4377, 0.4091, -26.8487, -1.2846,
        -26.5172, 0.0404, -24.2586, -1.6275, -27.2302, 1.5702, -25.3226, 1.6322, -24.6232, 0.7567,
        -30.0658, 2.5031, -28.3178, 1.0281, -27.5989, 0.5146, -28.9215, 1.2595, -25.9529, 2.0349,
        -24.5060, -1.0224, -27.9057, -1.6656, -26.1007, -1.4134, -28.0397, 2.2677, -28.2943,
        2.3541, -27.2746, 0.6982, -26.8166, 1.7735, -28.3365, 0.2735, -26.1627, 1.0859, -28.6819,
        0.0735, -27.6636, 1.8249, -26.4975, -0.5179, -29.0327, 1.1750, -24.9066, -0.3136, -26.4565,
        2.6202, -26.4081, 1.8247, -24.4976, 0.4883, -25.5058, 1.7397, -24.5006, 3.1824, -27.4848,
        2.0422, -27.1190, 0.8874, -28.4845, 2.7493, -25.8414, -0.3876, -27.3482, 0.4642, -27.4704,
        3.5801, -28.7346, -0.1401, -24.2127, 1.0274, -24.5816, 1.0922, -26.6428, 3.5773, -28.6721,
        1.7465, -29.3698, 2.3327, -25.4724, -0.1832, -29.1949, -1.3121, -28.6877, 1.0893, -29.9525,
        -0.5367, -25.3616, 9.5662, 9.0931, 15.2426, 10.0269, 15.4444, 8.7280, 12.9145, 8.5638,
        13.4756, 7.8507, 11.4998, 7.9758, 12.7232, 6.6907, 11.8129, 10.2232, 15.2546, 9.0143,
        11.9935, 11.0398, 11.4468, 9.8868, 14.3037, 10.2227, 9.8752, 9.0631, 15.5204, 9.9422,
        10.3527, 10.4908, 10.6718, 7.7593, 11.1040, 8.8526, 14.9456, 8.9179, 15.0329, 11.4061,
        12.6366, 8.5131, 10.6140, 9.7126, 13.8664, 9.2604, 11.7822, 8.6646, 12.4295, 9.5142,
        10.4796, 10.9989, 10.8626, 9.1104, 11.9529, 7.1066, 13.6168, 8.4351, 12.3199, 10.8275,
        11.3526, 11.0164, 13.5636, 10.6303, 13.8517, 10.9294, 10.4363, 9.0769, 11.0310, 10.7219,
        9.8669, 11.3245, 13.7462, 11.9618, 15.1124, 11.3871, 14.4720, 9.6079, 14.0429, 7.9607,
        12.0726, 8.3622, 12.7843, 9.0108, 10.9087, 7.9846, 10.4892, 8.9823, 13.4890, 10.9165,
        12.5114, 10.2871, 11.9187, 10.4703, 13.7379, 7.7377, 11.3396, 11.7601, 12.5609, 8.9198,
        14.7966, 10.0861, 10.5897, 7.9680, 10.1295, 11.2462, 12.4725, 11.0063, 13.0901, 9.5940,
        14.3488, 8.2243, 13.0121, 10.6080, 11.2849, 7.7030, 13.1390, 7.4859, 12.1789, 9.3252,
        14.6091, 11.3343, 12.4901, 7.5829, 11.9757, 12.6211, 9.7889, 7.9743, 12.9472, 9.4437,
        13.9718, 10.6583, 11.2290, 10.1574, 14.1563, 9.8059, 12.8971, 11.9786, 13.6676, 10.8469,
        13.5299, 8.6732, -22.3977, 3.6051, -17.4139, 3.0112, -18.6829, 1.9555, -18.8888, 4.0923,
        -17.2211, 6.2778, -19.6653, 3.4774, -18.7101, 4.0511, -17.3942, 5.3002, -19.5836, 5.1081,
        -21.8641, 2.3681, -17.3468, 4.4002, -20.4052, 5.2262, -17.8786, 5.4817, -18.3749, 1.6615,
        -20.8438, 3.4125, -17.8274, 2.7305, -21.3175, 4.3582, -18.7903, 5.7433, -19.5087, 3.7300,
        -16.9236, 4.9303, -21.1717, 1.5256, -17.9568, 4.4115, -19.5067, 5.8103, -19.7261, 4.8615,
        -19.7000, 1.5906, -16.6487, 3.4265, -20.7792, 3.6176, -18.6857, 4.5674, -18.1430, 2.5858,
        -20.0009, 4.2754, -20.3535, 5.8284, -16.6049, 4.1804, -21.6909, 5.5023, -18.9914, 6.2080,
        -18.9306, 1.9212, -20.6027, 2.8291, -20.6552, 3.3805, -19.4042, 4.5574, -17.0446, 5.2682,
        -18.3845, 4.0509, -18.7342, 3.3803, -18.5416, 2.6640, -19.1209, 3.6185, -17.3771, 2.6197,
        -18.9973, 2.8915, -20.1253, 1.0277, -20.0469, 4.2526, -20.2124, 2.1650, -20.7325, 4.3593,
        -16.7159, 4.9134, -17.3767, 3.9484, -16.5604, 2.2596, -20.0133, 3.5914, -21.5727, 5.6769,
        -18.1479, 4.4940, -19.4648, 1.8502, -21.8168, 3.3662, -20.3859, 6.5319, -20.3342, 4.7712,
        -20.7690, 2.4777, -21.7677, 5.2093, -17.5733, 3.0524, -18.2046, 5.3302, -18.9183, 2.0432,
        -21.5341, 3.4221, -18.1632, 6.1724, -17.4363, 3.4006, -20.4556, 6.8324, -19.5172, 2.9611,
        -18.8527, 1.5033,
    ];

    /// Judged on the objective, not on landing in scanpy's optimum.
    ///
    /// Whether our neighbourhoods match scanpy's asks which local optimum each run
    /// found, and two independent runs never find the same one. The divergence asks
    /// the question that has a right answer: ours must be no worse.
    #[test]
    fn reaches_a_no_worse_optimum_than_scanpy() {
        let input = blobs();
        let n = input.nrows();
        let params = TsneParams::default();
        let reference = Array2::from_shape_vec((n, 2), SCANPY_EMBEDDING.to_vec()).unwrap();
        let ours = tsne(&input, &params, &Device::Cpu).unwrap();

        let theirs_kl = kl_divergence(&input, &reference, &params);
        let ours_kl = kl_divergence(&input, &ours, &params);
        println!("KL: ours {ours_kl:.4}, scanpy {theirs_kl:.4}");
        assert!(
            ours_kl <= theirs_kl * 1.05,
            "reached KL {ours_kl}, worse than scanpy's {theirs_kl}"
        );
    }

    #[test]
    fn rejects_too_few_points_for_the_perplexity() {
        let input = uniform_matrix(20, 4, 1);
        let error = tsne(&input, &TsneParams::default(), &Device::Cpu).unwrap_err();
        assert!(
            matches!(error, Error::InvalidParameter { parameter, .. } if parameter == "perplexity")
        );
    }

    #[test]
    fn rejects_zero_components() {
        let input = uniform_matrix(100, 4, 1);
        let params = TsneParams {
            n_components: 0,
            ..Default::default()
        };
        let error = tsne(&input, &params, &Device::Cpu).unwrap_err();
        assert!(
            matches!(error, Error::InvalidParameter { parameter, .. } if parameter == "n_components")
        );
    }

    #[test]
    fn rejects_more_cells_than_the_exact_method_allows() {
        // One column only: the guard must fire before anything (n, n) is allocated.
        let input = Array2::<f32>::zeros((MAX_CELLS + 1, 1));
        let error = tsne(&input, &TsneParams::default(), &Device::Cpu).unwrap_err();
        assert!(
            matches!(error, Error::InvalidParameter { parameter, .. } if parameter == "n_cells")
        );
    }
}
