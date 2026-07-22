//! Diffusion maps and pseudotime. Owned by feat/diffusion.

use candle_core::{Device, Tensor};
use ndarray::Array2;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Extra sketch columns beyond `n_comps`, as in `pca.rs`.
///
/// Subspace iteration separates the retained components from the rest at a rate
/// set by `|lambda_width| / |lambda_n_comps|`, so the oversampling is what buys
/// the gap. A diffusion spectrum is flatter than a PCA spectrum, so the same 24
/// columns are worth more here and are still negligible next to `n_cells`.
const OVERSAMPLING: usize = 24;

/// The subspace has converged when no retained Ritz value moves by more than
/// this between iterations.
///
/// The products are `f32`, so the Ritz values stop improving around `1e-7` and a
/// tighter bound only spins until `MAX_ITERATIONS`. Measured against
/// `scipy.sparse.linalg.eigsh` on a 1500-cell neighbour graph, this stops after
/// 67 iterations with the eigenvalues correct to 3e-6 relative — two orders
/// inside the contract's 1e-3.
const RITZ_TOLERANCE: f64 = 1e-6;
const MAX_ITERATIONS: usize = 1000;

/// The `diffmap` signature carries no seed, so the starting block is drawn from
/// a fixed one: same graph, same map, on every machine and every run.
const START_SEED: u64 = 0;

/// Eigenvalues at or above this count as stationary and are weighted 1 in the
/// pseudotime instead of `lambda / (1 - lambda)`, which would divide by nearly
/// zero. The constant is scanpy's, chosen there for `f32` precision.
const STATIONARY_EIGENVALUE: f32 = 0.9994;

/// Upper bound on the padded operator, the one allocation that grows with the
/// graph. See `SparseOperator` for the arithmetic behind the limit.
const MAX_OPERATOR_BYTES: usize = 2 << 30;

/// Sketch directions whose Gram eigenvalue is below this fraction of the largest
/// carry no information in `f32` and are dropped rather than amplified.
const RANK_TOLERANCE: f64 = 1e-7;

const JACOBI_SWEEPS: usize = 60;

/// Eigenvectors and eigenvalues of the diffusion operator.
#[derive(Debug, Clone)]
pub struct DiffusionMap {
    /// `(n_cells, n_comps)`, **including** the trivial first component.
    ///
    /// scanpy's `tl.diffmap` writes `dpt.eigen_basis` unchanged into
    /// `obsm["X_diffmap"]` and the full spectrum into `uns["diffmap_evals"]`;
    /// component 0, the stationary vector with eigenvalue 1, is dropped only by
    /// `pl.diffmap` when it plots. `tl.dpt` reads the stored basis back and
    /// needs that first column, so dropping it here would change the result.
    pub embedding: Array2<f32>,
    /// Descending, starting at 1 for a connected graph.
    pub eigenvalues: Vec<f32>,
}

/// Diffusion map of a connectivity graph, as `scanpy.tl.diffmap`.
///
/// Peak memory is `n_cells * max_row_nnz * 8` bytes for the padded operator plus
/// about `6 * n_cells * (n_comps + 24) * 4` bytes of dense blocks; the
/// `(n_cells, n_cells)` transition matrix is never formed. For PBMC 3k with
/// `n_comps = 15` that is 0.7 MB and 4 MB.
pub fn diffmap(graph: &CsrMatrix, n_comps: usize, device: &Device) -> Result<DiffusionMap> {
    let n_cells = graph.n_rows();
    if graph.n_cols() != n_cells {
        return Err(Error::shape(
            format!("a square {n_cells}x{n_cells} graph"),
            format!("{n_cells}x{}", graph.n_cols()),
        ));
    }
    if n_cells < 2 {
        return Err(Error::parameter("n_cells", "at least 2", n_cells));
    }
    if n_comps == 0 {
        return Err(Error::parameter("n_comps", "at least 1", n_comps));
    }
    // The operator has only `n_cells` eigenvectors, and the last one is never
    // determined by a subspace of the same size, so scipy's `k < n` applies.
    if n_comps >= n_cells {
        return Err(Error::parameter(
            "n_comps",
            "below the number of cells",
            n_comps,
        ));
    }
    let components = component_count(graph);
    if components != 1 {
        // Every component contributes its own eigenvalue 1, so the leading
        // eigenspace is degenerate and its basis is arbitrary; pseudotime
        // between components is infinite. Both are silent wrong answers.
        return Err(Error::parameter(
            "graph",
            "connected: a diffusion map of disconnected components is not defined",
            format!("{components} connected components"),
        ));
    }

    let transitions = symmetric_transitions(graph)?;
    let operator = SparseOperator::new(&transitions, device)?;
    let (eigenvalues, embedding) = leading_eigen(&operator, n_comps, device)?;
    Ok(DiffusionMap {
        embedding,
        eigenvalues,
    })
}

/// Diffusion pseudotime from a root cell, as `scanpy.tl.dpt`.
///
/// The distance is Haghverdi et al.'s Eq. 15: the Euclidean distance from the
/// root in the diffusion components, each scaled by `lambda / (1 - lambda)`.
/// Peak memory is `n_cells` floats; there is no tensor work worth a device here,
/// and the signature carries none.
pub fn dpt(map: &DiffusionMap, root: usize, n_dcs: usize) -> Result<Vec<f32>> {
    let n_cells = map.embedding.nrows();
    if map.eigenvalues.len() != map.embedding.ncols() {
        return Err(Error::shape(
            format!("{} eigenvalues", map.embedding.ncols()),
            format!("{}", map.eigenvalues.len()),
        ));
    }
    if root >= n_cells {
        return Err(Error::parameter(
            "root",
            "below the number of cells",
            format!("{root} of {n_cells}"),
        ));
    }
    if n_dcs == 0 {
        return Err(Error::parameter("n_dcs", "at least 1", n_dcs));
    }
    if n_dcs > map.eigenvalues.len() {
        return Err(Error::parameter(
            "n_dcs",
            "at most the number of diffusion components computed",
            format!("{n_dcs} of {}", map.eigenvalues.len()),
        ));
    }

    let weights: Vec<f32> = map.eigenvalues[..n_dcs]
        .iter()
        .map(|&value| {
            if value < STATIONARY_EIGENVALUE {
                value / (1.0 - value)
            } else {
                1.0
            }
        })
        .collect();

    let mut distances = vec![0.0f32; n_cells];
    for (cell, distance) in distances.iter_mut().enumerate() {
        let squared: f64 = (0..n_dcs)
            .map(|component| {
                let gap = weights[component]
                    * (map.embedding[[root, component]] - map.embedding[[cell, component]]);
                (gap as f64) * (gap as f64)
            })
            .sum();
        *distance = squared.sqrt() as f32;
    }

    // scanpy reports pseudotime on `[0, 1]`. A graph where every cell coincides
    // with the root leaves nothing to scale by, and zero is the honest answer.
    let farthest = distances.iter().fold(0.0f32, |a, &b| a.max(b));
    if farthest > 0.0 {
        for distance in &mut distances {
            *distance /= farthest;
        }
    }
    Ok(distances)
}

/// The symmetrised transition operator scanpy's `Neighbors.compute_transitions`
/// builds, with the density normalisation of Coifman and Lafon on.
///
/// `conn_norm = D^-1 W D^-1` with `D` the degrees, then
/// `T = Z^-1 conn_norm Z^-1` with `Z` the square roots of `conn_norm`'s row
/// sums. `T` is similar to the random walk `Z P Z^-1`, so its spectrum is the
/// walk's while its eigenvectors stay orthogonal. Sparsity is untouched, so the
/// result is the input pattern with rescaled values.
fn symmetric_transitions(graph: &CsrMatrix) -> Result<CsrMatrix> {
    let n_cells = graph.n_rows();
    let degree = column_sums(graph.indices(), graph.values().iter().map(|&v| v as f64), n_cells);
    if let Some(cell) = degree.iter().position(|&value| value <= 0.0) {
        return Err(Error::parameter(
            "graph",
            "a connectivity graph in which every cell has a positive degree",
            format!("cell {cell} has degree {}", degree[cell]),
        ));
    }

    let scaled = |values: &[f64], left: &[f64], right: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(graph.nnz());
        for row in 0..n_cells {
            for entry in graph.indptr()[row] as usize..graph.indptr()[row + 1] as usize {
                let column = graph.indices()[entry] as usize;
                out.push(values[entry] / (left[row] * right[column]));
            }
        }
        out
    };

    let raw: Vec<f64> = graph.values().iter().map(|&value| value as f64).collect();
    let normalised = scaled(&raw, &degree, &degree);
    let z: Vec<f64> = column_sums(graph.indices(), normalised.iter().copied(), n_cells)
        .into_iter()
        .map(|sum| sum.max(0.0).sqrt())
        .collect();
    if let Some(cell) = z.iter().position(|&value| value <= 0.0) {
        return Err(Error::parameter(
            "graph",
            "a connectivity graph with positive weights",
            format!("cell {cell} has no positive transition mass"),
        ));
    }

    let values = scaled(&normalised, &z, &z)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    CsrMatrix::new(
        graph.indptr().to_vec(),
        graph.indices().to_vec(),
        values,
        n_cells,
    )
}

/// Column sums of a CSR matrix, which is what scanpy's `sum(axis=0)` takes.
fn column_sums(indices: &[u32], values: impl Iterator<Item = f64>, n_cols: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; n_cols];
    for (&column, value) in indices.iter().zip(values) {
        sums[column as usize] += value;
    }
    sums
}

/// Weakly connected components of the sparsity pattern, matching what
/// `scipy.sparse.csgraph.connected_components` reports for scanpy.
fn component_count(graph: &CsrMatrix) -> usize {
    let n_cells = graph.n_rows();
    // Reverse edges by counting sort, so a graph that is only nearly symmetric
    // is still judged as the undirected graph scipy sees.
    let mut starts = vec![0u32; n_cells + 1];
    for &column in graph.indices() {
        starts[column as usize + 1] += 1;
    }
    for cell in 0..n_cells {
        starts[cell + 1] += starts[cell];
    }
    let mut cursor = starts.clone();
    let mut sources = vec![0u32; graph.nnz()];
    for row in 0..n_cells {
        for entry in graph.indptr()[row] as usize..graph.indptr()[row + 1] as usize {
            let column = graph.indices()[entry] as usize;
            sources[cursor[column] as usize] = row as u32;
            cursor[column] += 1;
        }
    }

    let mut seen = vec![false; n_cells];
    let mut stack = Vec::new();
    let mut components = 0;
    for start in 0..n_cells {
        if seen[start] {
            continue;
        }
        components += 1;
        seen[start] = true;
        stack.push(start);
        while let Some(cell) = stack.pop() {
            let outgoing =
                &graph.indices()[graph.indptr()[cell] as usize..graph.indptr()[cell + 1] as usize];
            let incoming = &sources[starts[cell] as usize..starts[cell + 1] as usize];
            for &next in outgoing.iter().chain(incoming) {
                let next = next as usize;
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
    }
    components
}

/// The transition operator as the only thing the eigensolver needs of it: a
/// product with a dense `(n_cells, width)` block, on `device`.
///
/// candle has no sparse type, and `pca.rs`'s range finder densifies row blocks
/// of `(rows, n_cols)` — here `n_cols` is `n_cells`, so that would cost
/// `n_cells^2` work and memory for a graph with about 30 entries per row. The
/// rows are instead padded to a common length (the ELLPACK layout) and the
/// product becomes `max_row_nnz` gather-and-scale steps over `(n_cells, width)`
/// tensors: the sparse flop count, expressed as tensor ops candle runs on the
/// GPU. Memory is `n_cells * max_row_nnz * 8` bytes.
struct SparseOperator {
    /// One `(gather indices, weights)` pair per padded slot. Short rows are
    /// padded with weight zero, so they contribute nothing.
    slots: Vec<(Tensor, Tensor)>,
}

impl SparseOperator {
    fn new(matrix: &CsrMatrix, device: &Device) -> Result<Self> {
        let n_rows = matrix.n_rows();
        let widest = (0..n_rows)
            .map(|row| (matrix.indptr()[row + 1] - matrix.indptr()[row]) as usize)
            .max()
            .unwrap_or(0);
        // Padding is what makes the layout tensor-shaped, and a graph with one
        // very dense row would pay for that row on every other row too.
        let bytes = n_rows.saturating_mul(widest).saturating_mul(8);
        if bytes > MAX_OPERATOR_BYTES {
            return Err(Error::parameter(
                "graph",
                "sparse enough that n_cells * max_neighbours * 8 stays under 2 GiB",
                format!("{bytes} bytes"),
            ));
        }

        let mut slots = Vec::with_capacity(widest);
        for slot in 0..widest {
            let mut indices = Vec::with_capacity(n_rows);
            let mut weights = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let from = matrix.indptr()[row] as usize + slot;
                let to = matrix.indptr()[row + 1] as usize;
                if from < to {
                    indices.push(matrix.indices()[from]);
                    weights.push(matrix.values()[from]);
                } else {
                    indices.push(0);
                    weights.push(0.0);
                }
            }
            slots.push((
                Tensor::from_vec(indices, n_rows, device)?,
                Tensor::from_vec(weights, (n_rows, 1), device)?,
            ));
        }
        Ok(Self { slots })
    }

    /// `T @ block`, with `block` of shape `(n_cells, width)`.
    fn times(&self, block: &Tensor) -> Result<Tensor> {
        let mut total: Option<Tensor> = None;
        for (indices, weights) in &self.slots {
            let contribution = block.index_select(indices, 0)?.broadcast_mul(weights)?;
            total = Some(match total {
                Some(sum) => sum.add(&contribution)?,
                None => contribution,
            });
        }
        total.ok_or_else(|| Error::shape("a graph with at least one edge", "an empty graph"))
    }
}

/// The `n_comps` eigenpairs of largest absolute eigenvalue, largest eigenvalue
/// first — the selection `scipy.sparse.linalg.eigsh(which="LM")` makes and the
/// order scanpy stores after reversing it.
///
/// Orthogonal subspace iteration with a Rayleigh-Ritz projection: each step is
/// one sparse-times-dense product, which is the whole reason this runs on the
/// device. The projected matrix `Q^T T Q` falls out of the same product, so the
/// convergence test costs nothing beyond one small matmul.
fn leading_eigen(
    operator: &SparseOperator,
    n_comps: usize,
    device: &Device,
) -> Result<(Vec<f32>, Array2<f32>)> {
    let n_cells = operator.slots[0].1.dim(0)?;
    let width = (n_comps + OVERSAMPLING).min(n_cells);
    let mut block = orthonormalize(&start_block(n_cells, width, device)?)?;
    let mut previous: Option<Vec<f64>> = None;

    for _ in 0..MAX_ITERATIONS {
        let image = operator.times(&block)?;
        let projected = block.t()?.contiguous()?.matmul(&image)?;
        let (values, vectors) = jacobi_eigen(symmetrised(&projected, width)?, width)?;
        let selected = leading_by_magnitude(&values, n_comps);
        let ritz: Vec<f64> = selected.iter().map(|&index| values[index]).collect();

        let converged = previous.as_ref().is_some_and(|last| {
            last.iter()
                .zip(&ritz)
                .all(|(old, new)| (old - new).abs() < RITZ_TOLERANCE)
        });
        if converged {
            let mut embedding = to_array2(&block.matmul(&ritz_basis(
                &vectors, width, &selected, device,
            )?)?)?;
            fix_component_signs(&mut embedding);
            return Ok((ritz.into_iter().map(|value| value as f32).collect(), embedding));
        }
        previous = Some(ritz);
        block = orthonormalize(&image)?;
    }
    Err(Error::NotConverged {
        operation: "diffusion map subspace iteration",
        iterations: MAX_ITERATIONS,
    })
}

/// Starting block for the iteration.
///
/// Uniform entries, not Gaussian: the only property the start needs is a
/// non-zero component along every wanted eigenvector, and unlike a one-shot
/// randomised range finder this block is iterated to convergence, so the
/// rotational invariance of a Gaussian buys nothing here.
fn start_block(n_rows: usize, width: usize, device: &Device) -> Result<Tensor> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(START_SEED);
    let values: Vec<f32> = (0..n_rows * width)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();
    Ok(Tensor::from_vec(values, (n_rows, width), device)?)
}

/// The selected Ritz vectors as a `(width, n_comps)` transform.
fn ritz_basis(
    vectors: &[f64],
    width: usize,
    selected: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let mut basis = vec![0.0f32; width * selected.len()];
    for (column, &index) in selected.iter().enumerate() {
        for row in 0..width {
            basis[row * selected.len() + column] = vectors[row * width + index] as f32;
        }
    }
    Ok(Tensor::from_vec(basis, (width, selected.len()), device)?)
}

/// Indices of the `count` largest eigenvalues by magnitude, then ordered by
/// decreasing eigenvalue.
fn leading_by_magnitude(values: &[f64], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        values[right]
            .abs()
            .partial_cmp(&values[left].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(count);
    order.sort_by(|&left, &right| {
        values[right]
            .partial_cmp(&values[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// `(A + A^T) / 2` as a row-major `f64` buffer.
///
/// `Q^T T Q` is symmetric in exact arithmetic; averaging removes the `f32`
/// asymmetry that would otherwise leave the Jacobi rotations with a residual
/// they cannot drive to zero.
fn symmetrised(tensor: &Tensor, n: usize) -> Result<Vec<f64>> {
    let flat = to_f64_rows(tensor)?;
    let mut symmetric = vec![0.0f64; n * n];
    for row in 0..n {
        for column in 0..n {
            symmetric[row * n + column] = 0.5 * (flat[row * n + column] + flat[column * n + row]);
        }
    }
    Ok(symmetric)
}

/// Orthonormal basis for the columns of `y`, on `y`'s device.
///
/// CholeskyQR2, as in `pca.rs`: whitening by the inverse square root of the Gram
/// matrix is one matmul, but the Gram squares the condition number, so one pass
/// leaves `f32` visibly non-orthogonal and the second restores it.
fn orthonormalize(y: &Tensor) -> Result<Tensor> {
    whiten(&whiten(y)?)
}

fn whiten(y: &Tensor) -> Result<Tensor> {
    let width = y.dim(1)?;
    let gram = y.t()?.contiguous()?.matmul(y)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen(symmetrised(&gram, width)?, width)?;
    let largest = eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b));

    let mut transform = vec![0.0f32; width * width];
    for column in 0..width {
        let scale = if eigenvalues[column] > largest * RANK_TOLERANCE {
            1.0 / eigenvalues[column].sqrt()
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
///
/// `pca.rs` has the same routine, but private to its module, and this branch
/// does not own that file; see the branch report.
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

/// The sign of an eigenvector is arbitrary. As in `pca.rs`, the largest
/// magnitude entry of every component is made positive, so repeated runs and the
/// two devices agree on orientation. Pseudotime is a sign-invariant function of
/// the components, so this only fixes what a caller plots.
fn fix_component_signs(embedding: &mut Array2<f32>) {
    for mut component in embedding.columns_mut() {
        let mut extreme = 0.0f32;
        for &value in component.iter() {
            if value.abs() > extreme.abs() {
                extreme = value;
            }
        }
        if extreme < 0.0 {
            component.map_inplace(|value| *value = -*value);
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

    /// A path of `n` cells with unit weights: the analytic case. The diffusion
    /// components of a path are the discrete cosines, so the first non-trivial
    /// one is monotone along it and pseudotime from an end increases along it.
    ///
    /// The two ends carry a self loop. The density normalisation divides by the
    /// degree, so without it the ends — the only cells with one neighbour — sit
    /// off the cosine and the leading component is monotone everywhere except
    /// there. The self loop is the reflecting boundary that makes every degree
    /// equal and leaves the exact cosine.
    fn path_graph(n: usize) -> CsrMatrix {
        let mut dense = vec![0.0f32; n * n];
        for cell in 0..n - 1 {
            dense[cell * n + cell + 1] = 1.0;
            dense[(cell + 1) * n + cell] = 1.0;
        }
        dense[0] = 1.0;
        dense[n * n - 1] = 1.0;
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    /// Two paths that never meet.
    fn split_graph(n: usize) -> CsrMatrix {
        let mut dense = vec![0.0f32; n * n];
        for cell in 0..n - 1 {
            if cell + 1 == n / 2 {
                continue;
            }
            dense[cell * n + cell + 1] = 1.0;
            dense[(cell + 1) * n + cell] = 1.0;
        }
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    #[test]
    fn the_leading_component_is_stationary() {
        let map = diffmap(&path_graph(60), 5, &Device::Cpu).unwrap();
        assert_eq!(map.embedding.dim(), (60, 5));
        assert!((map.eigenvalues[0] - 1.0).abs() < 1e-4, "{:?}", map.eigenvalues);
        for pair in map.eigenvalues.windows(2) {
            assert!(pair[0] >= pair[1], "{:?}", map.eigenvalues);
        }
        // The stationary vector is positive everywhere, up to the sign
        // convention, and is the only component that does not change sign.
        assert!(map.embedding.column(0).iter().all(|&value| value > 0.0));
    }

    #[test]
    fn the_first_diffusion_component_is_monotone_along_a_path() {
        let map = diffmap(&path_graph(80), 4, &Device::Cpu).unwrap();
        let component = map.embedding.column(1).to_vec();
        let ascending = component[0] < component[79];
        for pair in component.windows(2) {
            assert_eq!(pair[0] < pair[1], ascending, "{component:?}");
        }
    }

    #[test]
    fn pseudotime_increases_away_from_an_endpoint() {
        let map = diffmap(&path_graph(80), 6, &Device::Cpu).unwrap();
        let pseudotime = dpt(&map, 0, 6).unwrap();
        assert_eq!(pseudotime[0], 0.0);
        assert!((pseudotime[79] - 1.0).abs() < 1e-5);
        for pair in pseudotime.windows(2) {
            assert!(pair[0] < pair[1], "{pseudotime:?}");
        }
    }

    #[test]
    fn a_fixed_graph_gives_a_fixed_map() {
        let graph = path_graph(50);
        let first = diffmap(&graph, 5, &Device::Cpu).unwrap();
        let second = diffmap(&graph, 5, &Device::Cpu).unwrap();
        assert_eq!(first.embedding, second.embedding);
        assert_eq!(first.eigenvalues, second.eigenvalues);
    }

    #[test]
    fn cpu_and_gpu_agree() {
        if !crate::gpu_available() {
            return;
        }
        let device = crate::DeviceKind::Gpu.resolve().unwrap();
        let graph = path_graph(64);
        let cpu = diffmap(&graph, 6, &Device::Cpu).unwrap();
        let gpu = diffmap(&graph, 6, &device).unwrap();
        for (left, right) in cpu.eigenvalues.iter().zip(&gpu.eigenvalues) {
            assert!((left - right).abs() <= 1e-4, "{left} vs {right}");
        }
        let cpu_time = dpt(&cpu, 0, 6).unwrap();
        let gpu_time = dpt(&gpu, 0, 6).unwrap();
        for (left, right) in cpu_time.iter().zip(&gpu_time) {
            assert!((left - right).abs() <= 1e-3, "{left} vs {right}");
        }
    }

    #[test]
    fn the_transition_matrix_keeps_the_graph_sparsity() {
        let graph = path_graph(20);
        let transitions = symmetric_transitions(&graph).unwrap();
        assert_eq!(transitions.indptr(), graph.indptr());
        assert_eq!(transitions.indices(), graph.indices());
        // A symmetric operator similar to a stochastic matrix has spectral
        // radius 1, so no entry can exceed it.
        assert!(transitions.values().iter().all(|&value| value.abs() <= 1.0));
    }

    #[test]
    fn rejects_a_disconnected_graph() {
        assert!(diffmap(&split_graph(40), 4, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_impossible_component_counts() {
        let graph = path_graph(10);
        assert!(diffmap(&graph, 0, &Device::Cpu).is_err());
        assert!(diffmap(&graph, 10, &Device::Cpu).is_err());
        assert!(diffmap(&graph, 11, &Device::Cpu).is_err());
        assert!(diffmap(&graph, 9, &Device::Cpu).is_ok());
    }

    #[test]
    fn rejects_a_non_square_graph() {
        let graph = CsrMatrix::from_dense(&[0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 2, 3).unwrap();
        assert!(diffmap(&graph, 1, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_a_root_outside_the_map() {
        let map = diffmap(&path_graph(20), 4, &Device::Cpu).unwrap();
        assert!(dpt(&map, 20, 4).is_err());
        assert!(dpt(&map, 0, 0).is_err());
        assert!(dpt(&map, 0, 5).is_err());
        assert!(dpt(&map, 19, 4).is_ok());
    }
}
