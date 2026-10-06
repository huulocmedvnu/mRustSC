//! Dendrograms, force-directed layouts and densities. Owned by feat/layout.

use std::collections::HashSet;

use candle_core::{DType, Device, Tensor};
use ndarray::Array2;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Upper bound on the elements of one pairwise tile.
///
/// Both quadratic passes below walk a `tile_rows x n_cells` tile rather than the
/// `(n_cells, n_cells)` matrix they conceptually sum over. 4M f32 elements is
/// 16 MB per tile, and a tile step keeps about half a dozen of them alive, so
/// peak device memory is ~100 MB whatever the input size.
const MAX_TILE_ELEMENTS: usize = 4 * 1024 * 1024;

/// Rows of a pairwise tile, given how many points each row spans.
fn tile_rows(n_points: usize) -> usize {
    (MAX_TILE_ELEMENTS / n_points.max(1)).clamp(1, n_points.max(1))
}

/// `(height, n_points)` tile of `block[i][column] - points[j][column]`.
///
/// The signed difference, not the distance: a force needs a direction.
fn difference_tile(block: &Tensor, points: &Tensor, column: usize) -> Result<Tensor> {
    let left = block.narrow(1, column, 1)?;
    let right = points.narrow(1, column, 1)?.t()?.contiguous()?;
    Ok(left.broadcast_sub(&right)?)
}

// ---------------------------------------------------------------------------
// Dendrogram
// ---------------------------------------------------------------------------

/// A merge tree in the form `scipy.cluster.hierarchy` produces.
#[derive(Debug, Clone)]
pub struct Dendrogram {
    /// `(n_groups - 1, 4)` linkage rows: left, right, distance, size.
    pub linkage: Vec<[f64; 4]>,
    pub leaf_order: Vec<u32>,
}

/// Largest number of groups the agglomeration accepts.
///
/// Its working set is one `(n_groups, n_groups)` f64 distance matrix, 8 MB at
/// the limit, and the loop is O(n_groups^3) — about a second there. A real
/// `groupby` has tens of categories; the bound is what turns a per-cell matrix
/// passed here by mistake into an error instead of a cubic-time hang.
const MAX_GROUPS: usize = 1024;

/// Complete-linkage clustering of group centroids, as `scanpy.tl.dendrogram`.
///
/// The distance is `1 - pearson correlation` between two centroids, which is
/// what scanpy feeds `scipy.cluster.hierarchy.linkage`, and the linkage is
/// `complete`, which is what `sc.tl.dendrogram` passes by default.
///
/// This used to be `average`, justified by the two agreeing on the leaf order of
/// PBMC 3k. They do agree there, and that is exactly how far the justification
/// went: on centroids where they part company the leaf order differs outright
/// (`[4, 1, 3, 2, 0, 5]` against `[4, 0, 5, 2, 1, 3]`) and merge heights differ
/// by up to 0.52. Agreement on one dataset is not agreement.
///
/// No `device`: the input is a handful of group centroids, so every tensor
/// involved is smaller than the dispatch that would launch it.
pub fn dendrogram(centroids: &Array2<f32>) -> Result<Dendrogram> {
    let (n_groups, n_dims) = centroids.dim();
    if n_groups < 2 {
        return Err(Error::shape(
            "at least 2 groups to cluster",
            format!("{n_groups} group(s)"),
        ));
    }
    if n_groups > MAX_GROUPS {
        return Err(Error::parameter("n_groups", "at most 1024", n_groups));
    }
    if n_dims < 2 {
        return Err(Error::shape(
            "centroids of at least 2 dimensions, since the correlation of two scalars is undefined",
            format!("{n_dims} dimension(s)"),
        ));
    }

    let distances = correlation_distances(centroids)?;
    let linkage = agglomerate(distances, n_groups);
    let leaf_order = leaf_order(&linkage, n_groups);
    Ok(Dendrogram {
        linkage,
        leaf_order,
    })
}

/// Row-major `(n_groups, n_groups)` matrix of `1 - correlation`.
fn correlation_distances(centroids: &Array2<f32>) -> Result<Vec<f64>> {
    let (n_groups, n_dims) = centroids.dim();
    // Centre and normalise each centroid once; the correlation of two of them is
    // then a plain dot product.
    let mut standardised = vec![0.0f64; n_groups * n_dims];
    for group in 0..n_groups {
        let values: Vec<f64> = centroids.row(group).iter().map(|&v| f64::from(v)).collect();
        let mean = values.iter().sum::<f64>() / n_dims as f64;
        let norm = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            .sqrt();
        if norm.is_nan() || norm <= 0.0 {
            return Err(Error::shape(
                "centroids that vary across dimensions",
                format!("group {group} is constant, so its correlation is undefined"),
            ));
        }
        for (dimension, value) in values.iter().enumerate() {
            standardised[group * n_dims + dimension] = (value - mean) / norm;
        }
    }

    let mut distances = vec![0.0f64; n_groups * n_groups];
    for left in 0..n_groups {
        for right in left + 1..n_groups {
            let correlation: f64 = (0..n_dims)
                .map(|d| standardised[left * n_dims + d] * standardised[right * n_dims + d])
                .sum();
            // pandas clips too: round-off can push a perfect correlation past 1,
            // which would make the distance negative.
            let distance = 1.0 - correlation.clamp(-1.0, 1.0);
            distances[left * n_groups + right] = distance;
            distances[right * n_groups + left] = distance;
        }
    }
    Ok(distances)
}

/// Repeatedly merge the closest pair, in scipy's linkage encoding.
///
/// Deliberately the textbook O(n^3) form. `n_groups` is in the tens, so the
/// nearest-neighbour-chain algorithm scipy uses would buy nothing and cost the
/// reader a page of state.
fn agglomerate(mut distances: Vec<f64>, n_groups: usize) -> Vec<[f64; 4]> {
    // Slot `i` holds a live cluster; `label` is the id scipy would give it —
    // `0..n_groups` for the leaves, then `n_groups + step` per merge.
    let mut label: Vec<usize> = (0..n_groups).collect();
    let mut size = vec![1usize; n_groups];
    let mut live = vec![true; n_groups];

    let mut linkage = Vec::with_capacity(n_groups - 1);
    for step in 0..n_groups - 1 {
        let mut closest = (f64::INFINITY, 0usize, 0usize);
        for left in 0..n_groups {
            for right in left + 1..n_groups {
                if live[left] && live[right] && distances[left * n_groups + right] < closest.0 {
                    closest = (distances[left * n_groups + right], left, right);
                }
            }
        }
        let (distance, left, right) = closest;
        let merged = size[left] + size[right];
        linkage.push([
            label[left].min(label[right]) as f64,
            label[left].max(label[right]) as f64,
            distance,
            merged as f64,
        ]);

        // Complete linkage takes the *furthest* cross pair, so the merged distance is
        // the larger of the two old ones — the Lance-Williams update scipy applies for
        // `method="complete"`, which is what `sc.tl.dendrogram` passes by default.
        for other in 0..n_groups {
            if !live[other] || other == left || other == right {
                continue;
            }
            let updated =
                distances[other * n_groups + left].max(distances[other * n_groups + right]);
            distances[other * n_groups + left] = updated;
            distances[left * n_groups + other] = updated;
        }

        size[left] = merged;
        label[left] = n_groups + step;
        live[right] = false;
    }
    linkage
}

/// Leaves left to right, as `scipy.cluster.hierarchy.dendrogram` orders them.
///
/// With scipy's default `count_sort=False, distance_sort=False` that is a plain
/// depth-first walk from the root, taking each merge's first child before its
/// second. Iterative rather than recursive so the depth of a degenerate tree
/// cannot reach the stack.
fn leaf_order(linkage: &[[f64; 4]], n_leaves: usize) -> Vec<u32> {
    let mut order = Vec::with_capacity(n_leaves);
    let mut stack = vec![2 * n_leaves - 2]; // the root, the last cluster formed
    while let Some(node) = stack.pop() {
        if node < n_leaves {
            order.push(node as u32);
            continue;
        }
        let merge = &linkage[node - n_leaves];
        stack.push(merge[1] as usize); // pushed first, so the left child pops first
        stack.push(merge[0] as usize);
    }
    order
}

// ---------------------------------------------------------------------------
// ForceAtlas2
// ---------------------------------------------------------------------------

/// ForceAtlas2's tuning, at the values `scanpy.tl.draw_graph` passes: repulsion
/// weight, pull towards the origin, and how much oscillation the adaptive step
/// tolerates before it slows down.
const SCALING_RATIO: f32 = 2.0;
const GRAVITY: f32 = 1.0;
const JITTER_TOLERANCE: f32 = 1.0;
/// The speed controller stops punishing itself below this efficiency.
const MIN_SPEED_EFFICIENCY: f32 = 0.05;
/// One iteration may not raise the global speed by more than half.
const MAX_SPEED_RISE: f32 = 0.5;
const LAYOUT_DIMS: usize = 2;

/// Largest graph the exact all-pairs repulsion accepts.
///
/// Memory is not the binding constraint — the tiling above holds it at ~100 MB —
/// but time is: repulsion costs `n_cells^2` per iteration, so 50k cells over
/// scanpy's 500 iterations is already 1.2e12 pair evaluations. Past that the
/// answer is a Barnes-Hut approximation, which this does not implement, so the
/// limit is an error rather than a silent hour.
const MAX_LAYOUT_CELLS: usize = 50_000;

/// ForceAtlas2 layout of the neighbour graph, as `scanpy.tl.draw_graph`.
///
/// Each iteration is three forces and one adaptive step:
///
/// - repulsion between *every* pair, scaled by the product of their masses,
/// - attraction along each edge, linear in the separation and the edge weight,
/// - a constant pull towards the origin, so components cannot drift away.
///
/// `device` carries the repulsion, which is the only quadratic term and the only
/// one that is tensor algebra; see [`repel`]. The other two are sparse scatter
/// updates over the edge list, the same shape of work as UMAP's SGD, and stay on
/// the CPU for the reason `umap` documents.
pub fn force_directed_layout(
    graph: &CsrMatrix,
    n_iterations: usize,
    seed: u64,
    device: &Device,
) -> Result<Array2<f32>> {
    let n_cells = graph.n_rows();
    validate_graph(graph, n_iterations)?;

    let edges = undirected_edges(graph);
    let masses = masses(&edges, n_cells);
    let mut positions = random_positions(n_cells, seed);
    let mut forces = vec![0.0f32; n_cells * LAYOUT_DIMS];
    let mut previous_forces = vec![0.0f32; n_cells * LAYOUT_DIMS];
    let (mut speed, mut efficiency) = (1.0f32, 1.0f32);

    for _ in 0..n_iterations {
        std::mem::swap(&mut forces, &mut previous_forces);
        forces.fill(0.0);
        repel(&positions, &masses, &mut forces, device)?;
        attract(&edges, &positions, &mut forces);
        gravitate(&positions, &masses, &mut forces);
        (speed, efficiency) = advance(
            &mut positions,
            &forces,
            &previous_forces,
            &masses,
            speed,
            efficiency,
        );
    }

    Array2::from_shape_vec((n_cells, LAYOUT_DIMS), positions)
        .map_err(|error| Error::shape("an (n_cells, 2) layout", error.to_string()))
}

fn validate_graph(graph: &CsrMatrix, n_iterations: usize) -> Result<()> {
    if n_iterations == 0 {
        return Err(Error::parameter("n_iterations", "at least 1", 0));
    }
    let n_cells = graph.n_rows();
    if n_cells == 0 || graph.nnz() == 0 {
        return Err(Error::shape(
            "a non-empty connectivity graph",
            "empty graph",
        ));
    }
    if graph.n_cols() != n_cells {
        return Err(Error::shape(
            format!("a square {n_cells}x{n_cells} graph"),
            format!("{n_cells}x{}", graph.n_cols()),
        ));
    }
    if n_cells > MAX_LAYOUT_CELLS {
        return Err(Error::parameter("n_cells", "at most 50000", n_cells));
    }
    Ok(())
}

/// ForceAtlas2's mass: one plus the degree, so hubs repel harder than leaves.
///
/// Counted from the deduplicated edge list rather than from each row's stored entries,
/// so a node's degree is a property of the graph and not of which triangle the caller
/// stored it in. Reading row counts gave a lower-triangular graph node 0 a degree of
/// zero and the last node a degree of n-1, on a graph where every node is equivalent.
fn masses(edges: &[(u32, u32, f32)], n_cells: usize) -> Vec<f32> {
    let mut masses = vec![1.0f32; n_cells];
    for &(left, right, _) in edges {
        masses[left as usize] += 1.0;
        masses[right as usize] += 1.0;
    }
    masses
}

/// Each undirected edge once, as the upper triangle of a symmetric graph.
///
/// Taking both stored directions would double every attraction, which is not the
/// same layout at a different scale: the repulsion it balances is unchanged.
fn undirected_edges(graph: &CsrMatrix) -> Vec<(u32, u32, f32)> {
    // Deduplicate by unordered pair rather than by keeping the upper triangle. Both
    // reject the second copy of a symmetric edge, which is the point -- taking both
    // stored directions would double every attraction -- but only this one survives a
    // graph that is stored the other way up.
    //
    // Keeping `column > row` silently produced a layout with *no attraction at all* from
    // a lower-triangular graph: the call succeeded, because degrees come from row counts
    // and the only structural check is `nnz != 0`, and returned a plausible-looking pure
    // repulsion cloud. Measured on two 8-node cliques, within/between separation went
    // from 10.7/118.7 when stored symmetrically to 82.9/95.6 stored as `np.tril` -- the
    // cliques simply do not form. No scanpy-shaped caller hits it, since `connectivities`
    // is always symmetric, but nothing in the signature says so.
    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    for row in 0..graph.n_rows() {
        for entry in graph.indptr()[row] as usize..graph.indptr()[row + 1] as usize {
            let column = graph.indices()[entry];
            let weight = graph.values()[entry];
            if weight == 0.0 || column as usize == row {
                continue;
            }
            let pair = (
                row.min(column as usize) as u32,
                row.max(column as usize) as u32,
            );
            if seen.insert(pair) {
                edges.push((pair.0, pair.1, weight));
            }
        }
    }
    // Sort so the layout is a function of the graph and not of how it was stored. The
    // forces are applied in this order and accumulated in f32, and the simulation
    // amplifies the difference: the same edges read from lower-triangular storage
    // instead of symmetric storage put nodes tens of units apart.
    edges.sort_unstable_by_key(|&(left, right, _)| (left, right));
    edges
}

/// Uniform draws in `[0, 1)`, the range scanpy initialises `draw_graph` in.
///
/// splitmix64, four lines of state: an initial scatter needs no more, and the
/// layout stays reproducible from `seed` without a dependency.
fn random_positions(n_cells: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n_cells * LAYOUT_DIMS)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            // 24 bits is every f32 in [0, 1) that is exactly representable.
            ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

/// All-pairs repulsion: `SCALING_RATIO * m_i * m_j / d^2` along the separation.
///
/// This is the quadratic half of ForceAtlas2 and the one part of it that is
/// tensor algebra, so it is the part the device earns its keep on. A tile is
/// `tile_rows x n_cells`; the full `(n_cells, n_cells)` matrix is never formed.
fn repel(positions: &[f32], masses: &[f32], forces: &mut [f32], device: &Device) -> Result<()> {
    let n_cells = masses.len();
    let points = Tensor::from_slice(positions, (n_cells, LAYOUT_DIMS), device)?;
    let mass_row = Tensor::from_slice(masses, (1, n_cells), device)?;
    let mass_column = mass_row.reshape((n_cells, 1))?;

    let height = tile_rows(n_cells);
    let mut start = 0;
    while start < n_cells {
        let rows = height.min(n_cells - start);
        let block = points.narrow(0, start, rows)?;
        let dx = difference_tile(&block, &points, 0)?;
        let dy = difference_tile(&block, &points, 1)?;
        let square_distance = (dx.sqr()? + dy.sqr()?)?;

        // A point does not repel itself, and two coincident points have no
        // direction to be pushed apart along. Masking beats nudging the
        // denominator: the answer then owes nothing to the size of an epsilon.
        let separated = square_distance.gt(0f32)?.to_dtype(DType::F32)?;
        let factor = mass_column
            .narrow(0, start, rows)?
            .broadcast_mul(&mass_row)?
            .affine(f64::from(SCALING_RATIO), 0.0)?
            .mul(&separated)?
            .div(&square_distance.maximum(f32::MIN_POSITIVE)?)?;

        let x = dx.mul(&factor)?.sum(1)?.to_vec1::<f32>()?;
        let y = dy.mul(&factor)?.sum(1)?.to_vec1::<f32>()?;
        for row in 0..rows {
            forces[(start + row) * LAYOUT_DIMS] += x[row];
            forces[(start + row) * LAYOUT_DIMS + 1] += y[row];
        }
        start += rows;
    }
    Ok(())
}

/// Hooke's law along every edge, weighted by the edge: ForceAtlas2's
/// `linAttraction` with `outboundAttractionDistribution` off, as scanpy runs it.
fn attract(edges: &[(u32, u32, f32)], positions: &[f32], forces: &mut [f32]) {
    for &(head, tail, weight) in edges {
        let (head, tail) = (head as usize * LAYOUT_DIMS, tail as usize * LAYOUT_DIMS);
        for d in 0..LAYOUT_DIMS {
            let pull = weight * (positions[head + d] - positions[tail + d]);
            forces[head + d] -= pull;
            forces[tail + d] += pull;
        }
    }
}

/// A pull of `mass * GRAVITY` towards the origin, constant in the distance.
///
/// Without it a component with no edge to the rest is pushed away for ever.
fn gravitate(positions: &[f32], masses: &[f32], forces: &mut [f32]) {
    for (cell, &mass) in masses.iter().enumerate() {
        let base = cell * LAYOUT_DIMS;
        let distance = (0..LAYOUT_DIMS)
            .map(|d| positions[base + d] * positions[base + d])
            .sum::<f32>()
            .sqrt();
        if distance <= 0.0 {
            continue; // already at the origin
        }
        let factor = mass * GRAVITY / distance;
        for d in 0..LAYOUT_DIMS {
            forces[base + d] -= positions[base + d] * factor;
        }
    }
}

/// ForceAtlas2's adaptive step: take the accumulated forces with one global
/// speed, tuned from how much they oscillate against how much of them pulls in a
/// consistent direction.
///
/// "Swinging" is how far this iteration's force turned away from the last one's,
/// and "traction" how far the two agree. A layout that is still ordering itself
/// has high traction and speeds up; one that is vibrating around its optimum has
/// high swinging and is slowed down, per node as well as globally.
fn advance(
    positions: &mut [f32],
    forces: &[f32],
    previous_forces: &[f32],
    masses: &[f32],
    speed: f32,
    efficiency: f32,
) -> (f32, f32) {
    let n_cells = masses.len();
    // `sign = -1` measures how far the force turned (swinging), `sign = 1` how
    // much of it repeated (traction).
    let combined = |cell: usize, sign: f32| {
        let base = cell * LAYOUT_DIMS;
        (0..LAYOUT_DIMS)
            .map(|d| {
                let value = previous_forces[base + d] + sign * forces[base + d];
                value * value
            })
            .sum::<f32>()
            .sqrt()
    };

    let mut total_swinging = 0.0f32;
    let mut total_traction = 0.0f32;
    for (cell, &mass) in masses.iter().enumerate() {
        total_swinging += mass * combined(cell, -1.0);
        total_traction += 0.5 * mass * combined(cell, 1.0);
    }
    if total_traction.is_nan() || total_traction <= 0.0 {
        return (speed, efficiency); // nothing is pulling anywhere; leave the layout alone
    }

    let optimal_jitter = 0.05 * (n_cells as f32).sqrt();
    let mut jitter = JITTER_TOLERANCE
        * (optimal_jitter * total_traction / (n_cells * n_cells) as f32)
            .clamp(optimal_jitter.sqrt(), 10.0);
    let mut efficiency = efficiency;
    // Swinging at twice the traction means the layout is fighting itself.
    if total_swinging / total_traction > 2.0 {
        if efficiency > MIN_SPEED_EFFICIENCY {
            efficiency *= 0.5;
        }
        jitter = jitter.max(JITTER_TOLERANCE);
    }
    let target = if total_swinging > 0.0 {
        jitter * efficiency * total_traction / total_swinging
    } else {
        f32::INFINITY
    };
    if total_swinging > jitter * total_traction {
        if efficiency > MIN_SPEED_EFFICIENCY {
            efficiency *= 0.7;
        }
    } else if speed < 1000.0 {
        efficiency *= 1.3;
    }
    // Rising too fast would overshoot everything the last iterations gained.
    let speed = speed + (target - speed).min(MAX_SPEED_RISE * speed);

    for (cell, &mass) in masses.iter().enumerate() {
        let base = cell * LAYOUT_DIMS;
        // The per-node brake: a node whose force keeps reversing moves least.
        let factor = speed / (1.0 + (speed * mass * combined(cell, -1.0)).sqrt());
        for d in 0..LAYOUT_DIMS {
            positions[base + d] += forces[base + d] * factor;
        }
    }
    (speed, efficiency)
}

// ---------------------------------------------------------------------------
// Embedding density
// ---------------------------------------------------------------------------

/// Largest embedding the all-pairs kernel density accepts.
///
/// Peak memory is the tiling's ~100 MB plus the embedding itself; the bound is
/// on the `n_cells^2` kernel evaluations, 1e10 at the limit, which is a few
/// seconds on the GPU and the point past which a grid-based estimator, not this
/// one, is the right tool.
const MAX_DENSITY_CELLS: usize = 100_000;

/// Gaussian kernel density of cells in an embedding, scaled to [0, 1].
///
/// The estimator `scipy.stats.gaussian_kde` implements, which is what
/// `scanpy.tl.embedding_density` calls: a full-covariance Gaussian kernel whose
/// covariance is the *data* covariance scaled by Scott's factor
/// `n^(-2/(d+4))`, evaluated at the data points themselves.
///
/// Scott's rule is `gaussian_kde`'s default and therefore scanpy's; the choice
/// is not cosmetic, since a bandwidth is a smoothing length and Silverman's rule
/// would return a different, equally normalised, curve.
///
/// The kernel's normalising constant is left out: it is one positive factor
/// common to every cell, and the final rescaling to `[0, 1]` divides it away.
pub fn embedding_density(embedding: &Array2<f32>, device: &Device) -> Result<Vec<f32>> {
    let (n_cells, n_dims) = embedding.dim();
    if n_dims == 0 {
        return Err(Error::shape("an embedding with at least 1 column", "0"));
    }
    if n_cells <= n_dims {
        return Err(Error::shape(
            format!("more than {n_dims} cells, so the bandwidth is determined"),
            format!("{n_cells} cells"),
        ));
    }
    if n_cells > MAX_DENSITY_CELLS {
        return Err(Error::parameter("n_cells", "at most 100000", n_cells));
    }

    let whitened = whiten(embedding)?;
    let sums = kernel_sums(&whitened, device)?;
    Ok(scale_to_unit(sums))
}

/// Transform the embedding so the kernel becomes the unit isotropic Gaussian.
///
/// `gaussian_kde` uses a full covariance `C = cov(x) * factor^2`, so its
/// exponent is the Mahalanobis distance. Factoring `C = L L^T` once and mapping
/// every point through `L^-1` turns that into a plain squared distance, which is
/// the form the tiled matmul below can evaluate.
fn whiten(embedding: &Array2<f32>) -> Result<Array2<f32>> {
    let (n_cells, n_dims) = embedding.dim();
    let mut means = vec![0.0f64; n_dims];
    for row in embedding.rows() {
        for (dimension, &value) in row.iter().enumerate() {
            means[dimension] += f64::from(value) / n_cells as f64;
        }
    }

    // Scott's rule, exactly as `gaussian_kde` applies it: the data covariance,
    // unbiased, times `n^(-2/(d+4))`.
    let bandwidth = (n_cells as f64).powf(-1.0 / (n_dims as f64 + 4.0));
    let mut covariance = vec![0.0f64; n_dims * n_dims];
    for row in embedding.rows() {
        for left in 0..n_dims {
            let centred_left = f64::from(row[left]) - means[left];
            for right in 0..=left {
                let centred_right = f64::from(row[right]) - means[right];
                covariance[left * n_dims + right] += centred_left * centred_right;
            }
        }
    }
    for left in 0..n_dims {
        for right in 0..=left {
            let scaled =
                covariance[left * n_dims + right] / (n_cells as f64 - 1.0) * bandwidth * bandwidth;
            covariance[left * n_dims + right] = scaled;
            covariance[right * n_dims + left] = scaled;
        }
    }

    let factor = cholesky(&covariance, n_dims)?;
    let mut whitened = Array2::<f32>::zeros((n_cells, n_dims));
    for (cell, row) in embedding.rows().into_iter().enumerate() {
        // Forward substitution: solve `L z = x - mean` for this point.
        let mut z = vec![0.0f64; n_dims];
        for dimension in 0..n_dims {
            let known: f64 = (0..dimension)
                .map(|before| factor[dimension * n_dims + before] * z[before])
                .sum();
            z[dimension] = (f64::from(row[dimension]) - means[dimension] - known)
                / factor[dimension * n_dims + dimension];
            whitened[[cell, dimension]] = z[dimension] as f32;
        }
    }
    Ok(whitened)
}

/// Lower-triangular `L` with `L L^T = matrix`, row-major.
fn cholesky(matrix: &[f64], n_dims: usize) -> Result<Vec<f64>> {
    let mut factor = vec![0.0f64; n_dims * n_dims];
    for row in 0..n_dims {
        for column in 0..=row {
            let known: f64 = (0..column)
                .map(|k| factor[row * n_dims + k] * factor[column * n_dims + k])
                .sum();
            let value = matrix[row * n_dims + column] - known;
            if row == column {
                if value.is_nan() || value <= 0.0 {
                    return Err(Error::shape(
                        "an embedding whose columns are not collinear, so its covariance is invertible",
                        "a degenerate embedding",
                    ));
                }
                factor[row * n_dims + column] = value.sqrt();
            } else {
                factor[row * n_dims + column] = value / factor[column * n_dims + column];
            }
        }
    }
    Ok(factor)
}

/// `sum_j exp(-|z_i - z_j|^2 / 2)` for every point, tiled over the device.
fn kernel_sums(whitened: &Array2<f32>, device: &Device) -> Result<Vec<f32>> {
    let (n_cells, n_dims) = whitened.dim();
    let flat: Vec<f32> = whitened.iter().copied().collect();
    let points = Tensor::from_vec(flat, (n_cells, n_dims), device)?;
    // |a - b|^2 = |a|^2 + |b|^2 - 2 a.b keeps a whole tile in one matmul.
    let square_norms = points.sqr()?.sum_keepdim(1)?;
    let square_norms_row = square_norms.reshape((1, n_cells))?;
    let transposed = points.t()?.contiguous()?;

    let height = tile_rows(n_cells);
    let mut sums = Vec::with_capacity(n_cells);
    let mut start = 0;
    while start < n_cells {
        let rows = height.min(n_cells - start);
        let square_distance = points
            .narrow(0, start, rows)?
            .matmul(&transposed)?
            .affine(-2.0, 0.0)?
            .broadcast_add(&square_norms.narrow(0, start, rows)?)?
            .broadcast_add(&square_norms_row)?
            // Round-off makes a point's distance to itself slightly negative.
            .clamp(0.0, f32::INFINITY)?;
        sums.extend(
            square_distance
                .affine(-0.5, 0.0)?
                .exp()?
                .sum(1)?
                .to_vec1::<f32>()?,
        );
        start += rows;
    }
    Ok(sums)
}

/// Rescale to `[0, 1]`, as scanpy does, so densities are comparable within a
/// group and not across groups.
fn scale_to_unit(values: Vec<f32>) -> Vec<f32> {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = high - low;
    if range.is_nan() || range <= 0.0 {
        // A perfectly uniform density has no scale to spread over; scanpy's
        // 0/0 gives NaN here, which no plot can use.
        return vec![0.0; values.len()];
    }
    values
        .into_iter()
        .map(|value| (value - low) / range)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Four groups over five dimensions: two nearly parallel, one anti-parallel
    /// to them, one unrelated. Correlation ignores scale, so the pair that must
    /// merge first is the pair with the same *shape*.
    fn centroids() -> Array2<f32> {
        arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.1, 6.0, 8.0, 10.2],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 5.0, 2.0, 8.0, 3.0],
        ])
    }

    #[test]
    fn merges_the_most_correlated_groups_first() {
        let tree = dendrogram(&centroids()).unwrap();
        assert_eq!(tree.linkage.len(), 3);
        assert_eq!(tree.linkage[0][0], 0.0);
        assert_eq!(tree.linkage[0][1], 1.0);
        assert!(tree.linkage[0][2] < 1e-3, "{:?}", tree.linkage[0]);
        assert_eq!(tree.linkage[0][3], 2.0);
        // Distances never decrease: average linkage admits no inversion.
        assert!(tree.linkage.windows(2).all(|pair| pair[0][2] <= pair[1][2]));
        // Every leaf once, and the merged pair adjacent.
        let mut sorted = tree.leaf_order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        let first = tree.leaf_order.iter().position(|&leaf| leaf == 0).unwrap();
        let second = tree.leaf_order.iter().position(|&leaf| leaf == 1).unwrap();
        assert_eq!(first.abs_diff(second), 1);
    }

    #[test]
    fn a_perfectly_correlated_pair_merges_at_zero() {
        // The second row is the first, doubled: correlation 1, distance 0.
        let tree = dendrogram(&arr2(&[
            [1.0f32, 2.0, 4.0],
            [2.0, 4.0, 8.0],
            [4.0, 2.0, 1.0],
        ]))
        .unwrap();
        assert!(tree.linkage[0][2].abs() < 1e-9, "{:?}", tree.linkage[0]);
    }

    #[test]
    fn rejects_input_no_tree_exists_for() {
        assert!(dendrogram(&arr2(&[[1.0f32, 2.0, 3.0]])).is_err()); // one group
        assert!(dendrogram(&arr2(&[[1.0f32], [2.0]])).is_err()); // one dimension
                                                                 // A constant centroid has no correlation with anything.
        assert!(dendrogram(&arr2(&[[1.0f32, 1.0, 1.0], [1.0, 2.0, 3.0]])).is_err());
    }

    /// Two cliques of `per_clique` cells with no edge between them.
    fn two_cliques(per_clique: usize) -> CsrMatrix {
        let n = 2 * per_clique;
        let mut dense = vec![0.0f32; n * n];
        for row in 0..n {
            for column in 0..n {
                if row != column && (row < per_clique) == (column < per_clique) {
                    dense[row * n + column] = 1.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    /// Mean within-clique and between-clique distance.
    fn separation(layout: &Array2<f32>, per_clique: usize) -> (f32, f32) {
        let n = 2 * per_clique;
        let (mut within, mut within_count) = (0.0, 0);
        let (mut between, mut between_count) = (0.0, 0);
        for left in 0..n {
            for right in left + 1..n {
                let distance = ((layout[[left, 0]] - layout[[right, 0]]).powi(2)
                    + (layout[[left, 1]] - layout[[right, 1]]).powi(2))
                .sqrt();
                if (left < per_clique) == (right < per_clique) {
                    within += distance;
                    within_count += 1;
                } else {
                    between += distance;
                    between_count += 1;
                }
            }
        }
        (within / within_count as f32, between / between_count as f32)
    }

    #[test]
    fn separates_two_cliques() {
        let layout = force_directed_layout(&two_cliques(20), 200, 0, &cpu()).unwrap();
        assert_eq!(layout.dim(), (40, 2));
        assert!(layout.iter().all(|value| value.is_finite()));
        let (within, between) = separation(&layout, 20);
        assert!(
            between > 3.0 * within,
            "within={within}, between={between} are not separated"
        );
    }

    #[test]
    fn the_same_seed_reproduces_the_layout_bit_for_bit() {
        let graph = two_cliques(10);
        let first = force_directed_layout(&graph, 50, 7, &cpu()).unwrap();
        let second = force_directed_layout(&graph, 50, 7, &cpu()).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, force_directed_layout(&graph, 50, 8, &cpu()).unwrap());
    }

    #[test]
    fn rejects_a_graph_it_cannot_lay_out() {
        assert!(force_directed_layout(&two_cliques(4), 0, 0, &cpu()).is_err());
        let empty = CsrMatrix::new(vec![0], vec![], vec![], 0).unwrap();
        assert!(force_directed_layout(&empty, 10, 0, &cpu()).is_err());
        let no_edges = CsrMatrix::from_dense(&[0.0; 9], 3, 3).unwrap();
        assert!(force_directed_layout(&no_edges, 10, 0, &cpu()).is_err());
        let rectangular = CsrMatrix::from_dense(&[1.0; 6], 2, 3).unwrap();
        assert!(force_directed_layout(&rectangular, 10, 0, &cpu()).is_err());
    }

    /// Twenty points in a tight scatter, plus one far outlier.
    fn cluster_and_outlier() -> Array2<f32> {
        Array2::from_shape_fn((21, 2), |(row, column)| {
            if row == 20 {
                10.0
            } else {
                // Coprime strides, so no two cluster points coincide or line up.
                ((row * if column == 0 { 7 } else { 11 }) % 13) as f32 * 0.01
            }
        })
    }

    #[test]
    fn a_tight_cluster_is_denser_than_an_isolated_point() {
        let density = embedding_density(&cluster_and_outlier(), &cpu()).unwrap();
        assert_eq!(density.len(), 21);
        assert!(density.iter().all(|value| (0.0..=1.0).contains(value)));
        // The rescaling puts the sparsest point at exactly 0, so this asserts
        // that the isolated point is that one, ahead of all twenty cluster mates.
        assert_eq!(density[20], 0.0, "the outlier must be the sparsest point");
        let cluster_mean = density[..20].iter().sum::<f32>() / 20.0;
        assert!(
            cluster_mean > 0.5,
            "cluster mean {cluster_mean}: {density:?}"
        );
    }

    #[test]
    fn density_is_deterministic() {
        let embedding = cluster_and_outlier();
        assert_eq!(
            embedding_density(&embedding, &cpu()).unwrap(),
            embedding_density(&embedding, &cpu()).unwrap()
        );
    }

    #[test]
    fn rejects_an_embedding_with_no_bandwidth() {
        // Fewer cells than dimensions: the covariance is singular by construction.
        assert!(embedding_density(&arr2(&[[0.0f32, 1.0], [1.0, 0.0]]), &cpu()).is_err());
        // A column that never varies: the covariance is singular by content.
        let flat = Array2::from_shape_fn((10, 2), |(row, column)| (row * (1 - column)) as f32);
        assert!(embedding_density(&flat, &cpu()).is_err());
    }
}
