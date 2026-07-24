use candle_core::Device;
use ndarray::Array2;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Layout parameters, named as in `scanpy.tl.umap`.
#[derive(Debug, Clone)]
pub struct UmapParams {
    pub n_components: usize,
    pub n_epochs: usize,
    pub min_dist: f32,
    pub spread: f32,
    pub learning_rate: f32,
    pub negative_sample_rate: usize,
    pub seed: u64,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_epochs: 200,
            min_dist: 0.5,
            spread: 1.0,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            seed: 0,
        }
    }
}

/// Weight of the repulsive term, umap-learn's `gamma`. Not exposed by
/// `scanpy.tl.umap`, so it is fixed at umap-learn's default.
const REPULSION_STRENGTH: f32 = 1.0;

/// umap-learn clamps every gradient component to this range before applying it,
/// which is what keeps the layout stable at a learning rate of 1.
const GRADIENT_CLIP: f32 = 4.0;

/// Guards the repulsive denominator when two points nearly coincide.
const REPULSION_EPSILON: f32 = 0.001;

/// Extent of the initial layout, umap-learn's `init="random"` range.
const INIT_RANGE: f32 = 10.0;

/// Optimise the UMAP layout of a connectivity graph.
///
/// `device` is unused, and this runs on the CPU whatever the caller asks for. The
/// layout is an irregular scatter-update loop that candle cannot express, so the
/// portable path is scalar CPU code.
///
/// `scrust-gpu` holds a fused Metal kernel written to replace [`optimize_layout`],
/// but nothing calls it: it is not wired into this function or into the bindings, so
/// a caller on Metal still gets the loop below.
///
/// That is a standing decision, not a gap to close on sight. The kernel is Hogwild --
/// it accepts racing writes and so does not reproduce bit for bit even against itself
/// -- and because this function ignores `device` today, wiring it in would make UMAP
/// results depend on whether the caller's machine has a GPU, which nobody asks for by
/// passing `device="auto"`. It would also break the cross-checks in
/// `tests/test_umap_audit.py`, which hold this loop to a transcription of umap-learn's
/// sequential sweep within `2e-3`. See `scrust_gpu`'s crate docs for the full argument.
pub fn umap(
    connectivities: &CsrMatrix,
    params: &UmapParams,
    _device: &Device,
) -> Result<Array2<f32>> {
    validate(connectivities, params)?;
    let n_cells = connectivities.n_rows();

    let (a, b) = fit_ab_params(params.min_dist, params.spread)?;
    let graph = EdgeList::from_graph(connectivities, params.n_epochs);
    if graph.head.is_empty() {
        return Err(Error::shape("a graph with at least one edge", "no edges"));
    }

    let mut embedding = random_layout(n_cells, params.n_components, params.seed);
    rescale_to_init_range(&mut embedding, params.n_components);
    optimize_layout(&mut embedding, &graph, n_cells, params, a, b);

    Array2::from_shape_vec((n_cells, params.n_components), embedding)
        .map_err(|error| Error::shape("an (n_cells, n_components) layout", error.to_string()))
}

fn validate(connectivities: &CsrMatrix, params: &UmapParams) -> Result<()> {
    if params.n_components == 0 {
        return Err(Error::parameter("n_components", "at least 1", 0));
    }
    if params.n_epochs == 0 {
        return Err(Error::parameter("n_epochs", "at least 1", 0));
    }
    let n_cells = connectivities.n_rows();
    if n_cells == 0 || connectivities.nnz() == 0 {
        return Err(Error::shape(
            "a non-empty connectivity graph",
            "empty graph",
        ));
    }
    if connectivities.n_cols() != n_cells {
        return Err(Error::shape(
            format!("a square {n_cells}x{n_cells} graph"),
            format!("{n_cells}x{}", connectivities.n_cols()),
        ));
    }
    Ok(())
}

/// The graph as the flat arrays the SGD consumes, one entry per directed edge.
///
/// Owned `Vec`s of primitives so that the Metal kernel can take the same buffers
/// without a conversion step.
struct EdgeList {
    head: Vec<u32>,
    tail: Vec<u32>,
    /// How many epochs pass between two firings of each edge.
    epochs_per_sample: Vec<f64>,
}

impl EdgeList {
    /// Drop the edges too weak to fire even once in `n_epochs`, then turn the
    /// surviving weights into firing intervals exactly as umap-learn's
    /// `make_epochs_per_sample` (`umap/umap_.py:906`) does: `max / weight`, so
    /// the interval is always at least 1 and the strongest edge fires every
    /// epoch.
    ///
    /// The cutoff is `umap/umap_.py:1089`. One difference: below eleven epochs
    /// umap-learn keeps thresholding at `max / default_epochs` (500 or 200)
    /// rather than at `max / n_epochs`, so a very short run there drops far more
    /// edges than this does. Short runs are not a supported configuration
    /// either side.
    fn from_graph(connectivities: &CsrMatrix, n_epochs: usize) -> Self {
        let max_weight = connectivities
            .values()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_weight / n_epochs as f32;

        let indptr = connectivities.indptr();
        let mut head = Vec::new();
        let mut tail = Vec::new();
        let mut epochs_per_sample = Vec::new();
        for row in 0..connectivities.n_rows() {
            for entry in indptr[row] as usize..indptr[row + 1] as usize {
                let weight = connectivities.values()[entry];
                if weight <= 0.0 || weight < threshold {
                    continue;
                }
                head.push(row as u32);
                tail.push(connectivities.indices()[entry]);
                epochs_per_sample.push(f64::from(max_weight) / f64::from(weight));
            }
        }
        Self {
            head,
            tail,
            epochs_per_sample,
        }
    }
}

/// The three-state Tausworthe generator umap-learn uses (`tau_rand_int`,
/// `umap/utils.py:41`).
///
/// Reimplemented rather than taken from `rand` so that, from the same state, it
/// emits the same words. That equivalence holds for states in `[0, 2^32)`, which
/// is the range this seeding produces; umap-learn declares its own state `i8[:]`
/// and seeds it from `rng_state + head_embedding[:, 0].view(np.int64)`
/// (`umap/layouts.py:367`), so in practice it runs the same recurrence over
/// 64-bit *signed* words and returns an `i4`. The streams therefore agree in
/// distribution, not element by element, and no attempt is made to share a state
/// with umap-learn.
struct TauRng {
    state: [u32; 3],
}

impl TauRng {
    /// Seed from a 64-bit value, respecting tau88's requirement that the three
    /// words exceed 2, 8 and 16 respectively.
    fn new(seed: u64) -> Self {
        let mut splitmix = seed;
        let mut word = || {
            splitmix = splitmix.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = splitmix;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as u32
        };
        Self {
            state: [word() | 0x2, word() | 0x8, word() | 0x10],
        }
    }

    fn next_u32(&mut self) -> u32 {
        let s = &mut self.state;
        s[0] = ((s[0] & 4_294_967_294) << 12) ^ (((s[0] << 13) ^ s[0]) >> 19);
        s[1] = ((s[1] & 4_294_967_288) << 4) ^ (((s[1] << 2) ^ s[1]) >> 25);
        s[2] = ((s[2] & 4_294_967_280) << 17) ^ (((s[2] << 3) ^ s[2]) >> 11);
        s[0] ^ s[1] ^ s[2]
    }

    fn next_unit_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

/// Row-major uniform draws in `[-INIT_RANGE, INIT_RANGE]`.
///
/// umap-learn defaults to a spectral initialisation, which needs an eigensolver
/// this crate does not have. Random initialisation is its documented fallback,
/// `init="random"` (`umap/umap_.py:1095`), and this reproduces it including the
/// rescale that follows at `umap/umap_.py:1188`.
///
/// What that costs, measured against umap-learn driven both ways on the same
/// graph (see `tests/test_umap_audit.py`):
///
/// - The objective is unaffected. On PBMC 3k the fuzzy-set cross entropy is
///   1.5477e6 +- 0.05e6 from spectral and 1.5484e6 +- 0.10e6 from random, and
///   this implementation lands inside that band.
/// - Local structure is unaffected: on labelled blobs every neighbourhood keeps
///   its own label under both.
/// - **Global arrangement is not.** On clusters strung along a trajectory, the
///   fraction of consecutive clusters that stay adjacent in the layout is 0.52
///   from spectral and 0.15 from random; this implementation scores 0.13, i.e.
///   it behaves like umap-learn with `init="random"`, which is what it is.
///
/// So the arrangement of clusters relative to each other must not be read off a
/// layout from this function. Closing that gap means a spectral initialiser: the
/// normalised-Laplacian eigenvectors 1..n_components of the connectivity graph,
/// which is a Lanczos/LOBPCG solve this crate would have to acquire.
fn random_layout(n_cells: usize, n_components: usize, seed: u64) -> Vec<f32> {
    let mut rng = TauRng::new(seed);
    (0..n_cells * n_components)
        .map(|_| (rng.next_unit_f32() * 2.0 - 1.0) * INIT_RANGE)
        .collect()
}

/// Squash each coordinate into `[0, INIT_RANGE]`, as umap-learn does before
/// optimising, so the gradient scale does not depend on how the layout started.
fn rescale_to_init_range(embedding: &mut [f32], n_components: usize) {
    for component in 0..n_components {
        let column = || embedding.iter().skip(component).step_by(n_components);
        let low = column().copied().fold(f32::INFINITY, f32::min);
        let high = column().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = high - low;
        for value in embedding.iter_mut().skip(component).step_by(n_components) {
            *value = if range > 0.0 {
                INIT_RANGE * (*value - low) / range
            } else {
                0.0
            };
        }
    }
}

/// Fit `a` and `b` in `1 / (1 + a * x^(2b))` to the offset exponential decay
/// that `min_dist` and `spread` describe.
///
/// umap-learn calls `scipy.optimize.curve_fit` (`find_ab_params`,
/// `umap/umap_.py:1393`) on `np.linspace(0, 3 * spread, 300)`; this is the same
/// least squares problem on the same 300 points, solved by damped Gauss-Newton
/// from the same starting point of `(1, 1)`. Over `min_dist` in
/// `{0, .001, .05, .1, .25, .5, .8, 1, 1.5, 2, 3}` crossed with `spread` in
/// `{0.3, 0.5, 1, 1.5, 2, 3, 5}` the two agree to 3.7e-5 relative in `a`,
/// 4.2e-6 relative in `b`, and 3.2e-6 sup-norm on the fitted curve itself.
///
/// The exception is `min_dist == 3 * spread`, where the target is identically 1
/// and the fit is degenerate: `curve_fit` returns `a` of order 1e-10 with either
/// sign, this returns exactly 0. Above `3 * spread` this errors where umap-learn
/// would return a meaningless — sometimes negative — `a`.
pub fn fit_ab_params(min_dist: f32, spread: f32) -> Result<(f32, f32)> {
    if spread <= 0.0 {
        return Err(Error::parameter("spread", "greater than 0", spread));
    }
    if !(0.0..=spread * 3.0).contains(&min_dist) {
        return Err(Error::parameter("min_dist", "in [0, 3 * spread]", min_dist));
    }

    const SAMPLES: usize = 300;
    let (min_dist, spread) = (f64::from(min_dist), f64::from(spread));
    let targets: Vec<(f64, f64)> = (0..SAMPLES)
        .map(|i| {
            let x = 3.0 * spread * i as f64 / (SAMPLES - 1) as f64;
            let y = if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            };
            (x, y)
        })
        .collect();

    let (mut a, mut b) = (1.0_f64, 1.0_f64);
    let mut damping = 1e-3_f64;
    let mut cost = sum_squares(&targets, a, b);
    for _ in 0..200 {
        // Normal equations of the Gauss-Newton step, (J^T J + damping) d = -J^T r.
        let (mut jaa, mut jab, mut jbb, mut ga, mut gb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x, y) in &targets {
            let (residual, da, db) = residual_and_jacobian(x, y, a, b);
            jaa += da * da;
            jab += da * db;
            jbb += db * db;
            ga += da * residual;
            gb += db * residual;
        }
        let (damped_aa, damped_bb) = (jaa * (1.0 + damping), jbb * (1.0 + damping));
        let determinant = damped_aa * damped_bb - jab * jab;
        if determinant.abs() < f64::EPSILON {
            break;
        }
        let step_a = -(damped_bb * ga - jab * gb) / determinant;
        let step_b = -(damped_aa * gb - jab * ga) / determinant;

        let (trial_a, trial_b) = (a + step_a, b + step_b);
        // The model is only defined for positive parameters; shrink the step.
        if trial_a <= 0.0 || trial_b <= 0.0 {
            damping *= 10.0;
            continue;
        }
        let trial_cost = sum_squares(&targets, trial_a, trial_b);
        if trial_cost < cost {
            let improvement = cost - trial_cost;
            (a, b, cost) = (trial_a, trial_b, trial_cost);
            damping = (damping * 0.1).max(1e-12);
            if improvement < 1e-14 {
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1e12 {
                break;
            }
        }
    }
    Ok((a as f32, b as f32))
}

/// `x^(2b)` and the `d/db` log factor, with the removable singularity at
/// `x == 0` resolved to its limit: the curve is exactly 1 there for every `b`.
fn power_term(x: f64, b: f64) -> (f64, f64) {
    if x <= 0.0 {
        (0.0, 0.0)
    } else {
        (x.powf(2.0 * b), 2.0 * x.ln())
    }
}

fn model(x: f64, a: f64, b: f64) -> f64 {
    1.0 / (1.0 + a * power_term(x, b).0)
}

fn sum_squares(targets: &[(f64, f64)], a: f64, b: f64) -> f64 {
    targets
        .iter()
        .map(|&(x, y)| {
            let residual = model(x, a, b) - y;
            residual * residual
        })
        .sum()
}

/// Residual and its partial derivatives with respect to `a` and `b`.
fn residual_and_jacobian(x: f64, y: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let (term, log_factor) = power_term(x, b);
    let denominator = 1.0 + a * term;
    let squared = denominator * denominator;
    (
        1.0 / denominator - y,
        -term / squared,
        -a * term * log_factor / squared,
    )
}

/// The SGD: one pass over the edges due to fire per epoch, with the learning
/// rate decaying linearly to zero.
///
/// Deliberately plain scalar Rust. `feat/umap-kernel` swaps this body for a
/// fused Metal kernel with the same effect, so the boundary is the flat
/// `embedding` buffer plus the [`EdgeList`] arrays and nothing else.
///
/// Every random draw comes from a generator seeded from `params.seed` and the
/// head vertex id, as umap-learn does (`rng_state_per_sample[j]`,
/// `umap/layouts.py:161`). Making the negative samples a function of the head
/// vertex alone keeps the result identical whether the edges are visited in
/// order or in parallel.
///
/// The decay is applied *after* an epoch, not before it, matching
/// `umap/layouts.py:431`: epochs 0 and 1 both run at `learning_rate`. And the
/// `+-4` clip lands on the per-dimension gradient before `alpha` scales it
/// (`umap/layouts.py:143,150`), so `alpha` cannot be clipped away.
fn optimize_layout(
    embedding: &mut [f32],
    graph: &EdgeList,
    n_cells: usize,
    params: &UmapParams,
    a: f32,
    b: f32,
) {
    let dim = params.n_components;
    let negative_rate = params.negative_sample_rate;

    let mut rngs: Vec<TauRng> = (0..n_cells)
        .map(|vertex| {
            TauRng::new(params.seed ^ (vertex as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        })
        .collect();

    // umap-learn divides by `negative_sample_rate` unguarded, so a rate of zero
    // is not a configuration it has; here it means what it says, no repulsion.
    // Clamping the divisor to 1 instead would leave the counter drifting behind
    // the epoch and fire a stray repulsion every `epochs_per_sample` epochs.
    let epochs_per_negative_sample: Vec<f64> = graph
        .epochs_per_sample
        .iter()
        .map(|&interval| interval / negative_rate as f64)
        .collect();
    let mut next_negative_sample = epochs_per_negative_sample.clone();
    let mut next_sample = graph.epochs_per_sample.clone();

    let mut alpha = params.learning_rate;
    for epoch in 0..params.n_epochs {
        let now = epoch as f64;
        for edge in 0..graph.head.len() {
            if next_sample[edge] > now {
                continue;
            }
            let current = graph.head[edge] as usize * dim;
            let other = graph.tail[edge] as usize * dim;

            apply_attraction(embedding, current, other, dim, a, b, alpha);
            next_sample[edge] += graph.epochs_per_sample[edge];

            if negative_rate == 0 {
                continue;
            }
            let n_negative =
                ((now - next_negative_sample[edge]) / epochs_per_negative_sample[edge]) as usize;
            let rng = &mut rngs[graph.head[edge] as usize];
            for _ in 0..n_negative {
                let sampled = (rng.next_u32() as usize % n_cells) * dim;
                if sampled != current {
                    apply_repulsion(embedding, current, sampled, dim, a, b, alpha);
                }
            }
            next_negative_sample[edge] += n_negative as f64 * epochs_per_negative_sample[edge];
        }
        alpha = params.learning_rate * (1.0 - epoch as f32 / params.n_epochs as f32);
    }
}

/// Pull the two endpoints of a firing edge together, moving both.
fn apply_attraction(
    embedding: &mut [f32],
    current: usize,
    other: usize,
    dim: usize,
    a: f32,
    b: f32,
    alpha: f32,
) {
    let distance = squared_distance(embedding, current, other, dim);
    let coefficient = if distance > 0.0 {
        -2.0 * a * b * distance.powf(b - 1.0) / (a * distance.powf(b) + 1.0)
    } else {
        0.0
    };
    for d in 0..dim {
        let gradient = clip(coefficient * (embedding[current + d] - embedding[other + d]));
        embedding[current + d] += gradient * alpha;
        embedding[other + d] -= gradient * alpha;
    }
}

/// Push a negative sample away, moving only the head vertex.
fn apply_repulsion(
    embedding: &mut [f32],
    current: usize,
    sampled: usize,
    dim: usize,
    a: f32,
    b: f32,
    alpha: f32,
) {
    let distance = squared_distance(embedding, current, sampled, dim);
    if distance <= 0.0 {
        return;
    }
    let coefficient = 2.0 * REPULSION_STRENGTH * b
        / ((REPULSION_EPSILON + distance) * (a * distance.powf(b) + 1.0));
    for d in 0..dim {
        let gradient = clip(coefficient * (embedding[current + d] - embedding[sampled + d]));
        embedding[current + d] += gradient * alpha;
    }
}

fn squared_distance(embedding: &[f32], left: usize, right: usize, dim: usize) -> f32 {
    (0..dim)
        .map(|d| {
            let difference = embedding[left + d] - embedding[right + d];
            difference * difference
        })
        .sum()
}

fn clip(value: f32) -> f32 {
    value.clamp(-GRADIENT_CLIP, GRADIENT_CLIP)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The connectivity graph and embedding `scanpy` produced for 80 cells drawn
    /// from four Gaussian blobs in 10-D, via `pp.neighbors(n_neighbors=10)` and
    /// `tl.umap(random_state=0)`. Only the upper triangle of the symmetric graph is
    /// stored; the test mirrors it.
    mod reference {
        pub const N_CELLS: usize = 80;

        pub const EDGES: &[(u32, u32, f32)] = &[
            (0, 1, 0.45072),
            (0, 2, 0.2003),
            (0, 3, 1.0),
            (0, 4, 0.14481),
            (0, 5, 0.17857),
            (0, 6, 1.0),
            (0, 8, 0.29773),
            (0, 9, 0.38459),
            (0, 10, 0.15111),
            (0, 11, 0.32115),
            (0, 12, 0.45168),
            (0, 13, 0.22822),
            (0, 14, 0.3878),
            (0, 15, 0.12129),
            (0, 16, 1.0),
            (0, 17, 0.44746),
            (0, 18, 0.8065),
            (0, 19, 0.26929),
            (0, 25, 0.23758),
            (1, 2, 0.34203),
            (1, 3, 0.156),
            (1, 6, 0.57169),
            (1, 9, 1.0),
            (1, 12, 1.0),
            (1, 13, 1.0),
            (1, 16, 0.37878),
            (1, 18, 0.15407),
            (2, 3, 0.62198),
            (2, 8, 0.33999),
            (2, 9, 0.46856),
            (2, 11, 0.22908),
            (2, 12, 0.31439),
            (2, 15, 1.0),
            (2, 16, 0.44769),
            (3, 4, 0.85736),
            (3, 6, 0.57568),
            (3, 7, 0.24954),
            (3, 8, 0.28114),
            (3, 9, 0.24135),
            (3, 11, 0.051863),
            (3, 12, 0.14082),
            (3, 13, 0.60511),
            (3, 16, 0.4151),
            (3, 17, 0.23697),
            (3, 18, 0.49531),
            (3, 19, 0.39603),
            (4, 7, 1.0),
            (4, 8, 0.43523),
            (4, 11, 1.0),
            (4, 14, 0.16009),
            (4, 17, 0.23625),
            (4, 18, 0.059752),
            (4, 19, 0.31047),
            (5, 6, 0.27799),
            (5, 7, 0.93204),
            (5, 8, 0.45493),
            (5, 9, 0.23095),
            (5, 10, 1.0),
            (5, 11, 0.30611),
            (5, 14, 0.40469),
            (5, 15, 0.6839),
            (5, 18, 1.0),
            (5, 25, 0.29366),
            (6, 9, 0.31737),
            (6, 12, 0.51958),
            (6, 13, 0.37899),
            (6, 14, 0.34288),
            (6, 17, 0.28227),
            (6, 18, 1.0),
            (6, 19, 0.22085),
            (6, 25, 0.29572),
            (7, 8, 0.63696),
            (7, 10, 0.037528),
            (7, 11, 0.18912),
            (7, 12, 0.091284),
            (7, 14, 0.07701),
            (7, 15, 0.039662),
            (7, 18, 0.17427),
            (8, 10, 0.60029),
            (8, 11, 1.0),
            (8, 14, 0.87725),
            (8, 15, 1.0),
            (8, 16, 0.59831),
            (8, 17, 1.0),
            (8, 18, 0.61794),
            (8, 19, 0.34941),
            (8, 25, 0.23258),
            (9, 10, 0.38353),
            (9, 12, 0.063665),
            (9, 13, 0.37659),
            (9, 18, 0.24125),
            (10, 11, 0.085345),
            (10, 14, 0.065389),
            (10, 15, 0.086123),
            (10, 17, 0.35201),
            (10, 18, 0.94944),
            (11, 14, 1.0),
            (11, 15, 0.67783),
            (11, 17, 0.96702),
            (11, 18, 0.11226),
            (11, 25, 0.24731),
            (12, 13, 0.065518),
            (12, 16, 0.98398),
            (13, 15, 0.11781),
            (13, 16, 0.35127),
            (13, 18, 0.62403),
            (13, 19, 0.58393),
            (14, 15, 0.13933),
            (14, 17, 0.84502),
            (14, 18, 0.62047),
            (14, 25, 1.0),
            (15, 16, 0.33339),
            (15, 17, 0.20004),
            (15, 18, 0.16632),
            (16, 18, 0.21265),
            (16, 19, 0.26358),
            (17, 18, 0.60076),
            (17, 19, 0.20918),
            (18, 19, 1.0),
            (18, 25, 0.31445),
            (20, 22, 0.10333),
            (20, 24, 0.94914),
            (20, 27, 0.2502),
            (20, 29, 1.0),
            (20, 31, 0.10737),
            (20, 35, 0.60709),
            (20, 38, 0.13391),
            (20, 65, 0.11259),
            (20, 66, 1.0),
            (21, 22, 0.46616),
            (21, 23, 0.10801),
            (21, 24, 0.28098),
            (21, 26, 0.23015),
            (21, 28, 0.27568),
            (21, 30, 0.60042),
            (21, 31, 0.40987),
            (21, 35, 0.30011),
            (21, 38, 0.31662),
            (21, 39, 1.0),
            (22, 23, 0.18622),
            (22, 24, 1.0),
            (22, 26, 0.28955),
            (22, 27, 0.078829),
            (22, 28, 0.47895),
            (22, 30, 0.23157),
            (22, 31, 0.69377),
            (22, 33, 0.19197),
            (22, 35, 1.0),
            (22, 37, 0.70599),
            (22, 38, 0.13002),
            (22, 39, 0.55467),
            (23, 24, 0.090733),
            (23, 28, 1.0),
            (23, 29, 0.17785),
            (23, 34, 0.46243),
            (23, 35, 0.66208),
            (23, 38, 0.66748),
            (23, 39, 0.41069),
            (24, 27, 0.75208),
            (24, 29, 0.018681),
            (24, 30, 0.92515),
            (24, 35, 0.25761),
            (24, 36, 0.17036),
            (24, 37, 0.20303),
            (24, 65, 0.53541),
            (25, 29, 0.45355),
            (25, 32, 0.094419),
            (25, 33, 0.21102),
            (25, 34, 0.12076),
            (25, 35, 0.36307),
            (26, 28, 0.42473),
            (26, 29, 0.45032),
            (26, 30, 1.0),
            (26, 31, 1.0),
            (26, 32, 0.26621),
            (26, 34, 1.0),
            (26, 35, 0.65846),
            (26, 36, 0.32991),
            (26, 39, 0.34853),
            (27, 29, 0.09596),
            (27, 32, 0.73344),
            (27, 33, 0.2022),
            (27, 35, 1.0),
            (27, 37, 0.3018),
            (27, 38, 0.27708),
            (28, 30, 0.29923),
            (28, 31, 0.20838),
            (28, 34, 0.26944),
            (28, 35, 1.0),
            (28, 37, 0.49474),
            (28, 38, 0.69187),
            (28, 39, 0.99698),
            (29, 31, 0.38679),
            (29, 32, 0.73325),
            (29, 33, 0.17648),
            (29, 34, 0.23776),
            (29, 35, 1.0),
            (29, 36, 0.15901),
            (29, 62, 0.1637),
            (29, 65, 0.75912),
            (29, 66, 0.48442),
            (29, 68, 0.33724),
            (29, 70, 0.50193),
            (29, 74, 0.14692),
            (29, 78, 0.0663),
            (30, 31, 0.15834),
            (30, 34, 0.14155),
            (30, 35, 0.37529),
            (30, 36, 0.16989),
            (30, 37, 0.45202),
            (30, 38, 0.4834),
            (31, 32, 0.6654),
            (31, 33, 0.32753),
            (31, 35, 0.80687),
            (31, 36, 1.0),
            (31, 37, 0.14236),
            (31, 39, 0.1827),
            (31, 65, 0.25308),
            (31, 78, 0.042597),
            (32, 33, 1.0),
            (32, 34, 0.12935),
            (32, 35, 1.0),
            (32, 36, 0.781),
            (32, 38, 0.20602),
            (33, 35, 0.79712),
            (33, 38, 0.28634),
            (33, 65, 0.18118),
            (34, 35, 0.99981),
            (34, 77, 0.12014),
            (35, 36, 0.23782),
            (35, 37, 0.55712),
            (35, 38, 1.0),
            (35, 39, 0.34824),
            (36, 65, 0.42285),
            (36, 75, 0.20611),
            (37, 38, 1.0),
            (37, 39, 0.37718),
            (38, 39, 0.34665),
            (40, 43, 0.26979),
            (40, 45, 0.14822),
            (40, 49, 0.22272),
            (40, 52, 0.22538),
            (40, 53, 0.63381),
            (40, 54, 1.0),
            (40, 55, 0.43958),
            (40, 56, 0.58147),
            (40, 58, 0.50099),
            (40, 59, 0.3732),
            (40, 71, 0.39635),
            (41, 45, 0.73077),
            (41, 52, 0.057946),
            (41, 54, 0.37427),
            (41, 55, 1.0),
            (41, 56, 0.69207),
            (41, 58, 0.36061),
            (41, 59, 0.16909),
            (41, 67, 0.080284),
            (41, 76, 0.059834),
            (42, 43, 0.14243),
            (42, 45, 0.13333),
            (42, 50, 0.70511),
            (42, 51, 0.78513),
            (42, 52, 0.25083),
            (42, 53, 0.64983),
            (42, 54, 0.23309),
            (42, 59, 1.0),
            (42, 60, 0.62636),
            (42, 72, 0.20388),
            (43, 44, 0.23635),
            (43, 46, 0.4199),
            (43, 48, 0.086689),
            (43, 51, 0.073991),
            (43, 52, 0.72183),
            (43, 53, 0.26644),
            (43, 54, 1.0),
            (43, 55, 0.96348),
            (43, 56, 0.34856),
            (43, 60, 0.40147),
            (43, 71, 0.16187),
            (44, 45, 0.13636),
            (44, 46, 0.27704),
            (44, 47, 0.19192),
            (44, 48, 0.86235),
            (44, 50, 0.32571),
            (44, 51, 0.043064),
            (44, 52, 1.0),
            (44, 54, 0.21852),
            (44, 55, 0.82805),
            (44, 56, 0.58728),
            (44, 57, 0.15823),
            (44, 58, 0.10797),
            (44, 59, 0.13784),
            (45, 47, 0.14516),
            (45, 51, 0.76267),
            (45, 52, 0.9859),
            (45, 53, 0.53338),
            (45, 54, 1.0),
            (45, 55, 0.34214),
            (45, 57, 0.29286),
            (45, 58, 0.12137),
            (45, 59, 0.5006),
            (46, 48, 1.0),
            (46, 49, 0.6611),
            (46, 50, 1.0),
            (46, 52, 0.14966),
            (46, 54, 0.15618),
            (46, 55, 0.2565),
            (46, 71, 1.0),
            (47, 49, 0.21743),
            (47, 50, 0.35807),
            (47, 52, 0.25176),
            (47, 53, 0.30766),
            (47, 54, 0.92022),
            (47, 55, 0.26928),
            (47, 56, 0.14743),
            (47, 57, 1.0),
            (48, 50, 0.66728),
            (48, 52, 0.21763),
            (48, 54, 0.17134),
            (48, 55, 0.29232),
            (48, 58, 0.17604),
            (48, 71, 0.63184),
            (49, 50, 0.1867),
            (49, 52, 0.19989),
            (49, 54, 0.5899),
            (49, 55, 1.0),
            (49, 56, 0.16491),
            (49, 73, 0.14827),
            (50, 52, 0.21842),
            (50, 54, 0.23734),
            (50, 59, 0.4106),
            (50, 71, 0.37477),
            (51, 52, 0.82904),
            (51, 54, 0.077351),
            (51, 55, 0.059085),
            (51, 56, 0.0088291),
            (51, 59, 1.0),
            (51, 72, 0.21948),
            (52, 53, 0.42305),
            (52, 54, 1.0),
            (52, 55, 0.42283),
            (52, 56, 0.27594),
            (52, 57, 0.3261),
            (52, 59, 0.52136),
            (52, 71, 0.13534),
            (53, 54, 1.0),
            (53, 57, 0.11831),
            (53, 59, 0.35293),
            (53, 60, 0.16224),
            (53, 71, 0.1628),
            (54, 55, 0.80343),
            (54, 56, 0.40129),
            (54, 57, 0.57526),
            (54, 58, 0.76255),
            (54, 59, 0.445),
            (54, 60, 0.11347),
            (54, 71, 0.21636),
            (55, 56, 1.0),
            (55, 57, 0.40103),
            (55, 58, 1.0),
            (56, 57, 0.17378),
            (56, 58, 0.83886),
            (56, 59, 0.16918),
            (57, 58, 0.346),
            (59, 63, 0.13973),
            (59, 64, 0.27078),
            (59, 69, 0.12673),
            (59, 71, 0.66695),
            (59, 72, 0.26083),
            (59, 74, 0.087399),
            (60, 67, 1.0),
            (60, 68, 0.14389),
            (60, 71, 0.80154),
            (60, 72, 0.2313),
            (60, 78, 0.38423),
            (60, 79, 0.22736),
            (61, 62, 0.384),
            (61, 64, 0.13811),
            (61, 65, 0.11393),
            (61, 67, 0.26174),
            (61, 68, 0.49289),
            (61, 70, 0.47141),
            (61, 73, 0.43611),
            (61, 75, 0.30818),
            (61, 78, 1.0),
            (62, 63, 1.0),
            (62, 65, 0.38796),
            (62, 66, 0.021697),
            (62, 68, 0.74521),
            (62, 69, 0.14827),
            (62, 70, 1.0),
            (62, 72, 0.27036),
            (62, 74, 0.11683),
            (62, 75, 0.1162),
            (62, 76, 0.32334),
            (62, 77, 0.26755),
            (63, 65, 0.24938),
            (63, 67, 0.11757),
            (63, 68, 0.14636),
            (63, 71, 0.16544),
            (63, 75, 0.85924),
            (63, 76, 0.59851),
            (63, 79, 0.75059),
            (64, 65, 0.3321),
            (64, 66, 0.20661),
            (64, 67, 0.29831),
            (64, 68, 0.25672),
            (64, 69, 0.7458),
            (64, 70, 0.20984),
            (64, 72, 0.20261),
            (64, 73, 0.8146),
            (64, 74, 0.43897),
            (64, 75, 0.66407),
            (64, 76, 1.0),
            (64, 77, 0.44052),
            (64, 79, 0.065002),
            (65, 66, 0.462),
            (65, 67, 0.13554),
            (65, 68, 1.0),
            (65, 69, 0.3905),
            (65, 72, 0.67046),
            (65, 74, 0.19581),
            (65, 75, 0.73524),
            (65, 76, 0.34995),
            (65, 78, 0.84706),
            (66, 69, 0.1899),
            (66, 70, 0.016256),
            (66, 74, 0.86723),
            (66, 75, 0.35429),
            (66, 76, 0.15342),
            (67, 68, 1.0),
            (67, 70, 0.20501),
            (67, 73, 0.99405),
            (67, 75, 0.29521),
            (67, 76, 0.28291),
            (67, 77, 0.42121),
            (67, 78, 1.0),
            (67, 79, 0.22047),
            (68, 70, 0.59403),
            (68, 72, 0.263),
            (68, 73, 1.0),
            (68, 76, 0.29683),
            (68, 77, 0.65239),
            (68, 78, 0.81956),
            (68, 79, 1.0),
            (69, 72, 0.14324),
            (69, 74, 1.0),
            (69, 75, 0.22192),
            (69, 76, 0.42721),
            (70, 73, 0.22762),
            (70, 74, 0.17589),
            (70, 76, 0.30837),
            (70, 77, 0.38256),
            (71, 73, 0.12588),
            (71, 75, 0.16421),
            (71, 79, 0.2502),
            (72, 74, 1.0),
            (73, 76, 0.2599),
            (73, 77, 0.5268),
            (73, 78, 0.18785),
            (73, 79, 0.44502),
            (75, 76, 1.0),
            (75, 77, 0.38705),
            (75, 78, 0.25148),
            (76, 77, 1.0),
            (76, 79, 0.36601),
            (77, 79, 0.4152),
        ];

        /// Row-major `(N_CELLS, 2)`.
        pub const EMBEDDING: &[f32] = &[
            5.6895, -0.4933, 5.5979, -1.6203, 5.5275, -0.035601, 6.0625, -0.56295, 6.6219, 0.55763,
            6.8029, -0.057722, 5.7192, -1.0147, 6.6383, 0.69891, 7.2903, 0.22201, 6.614, -1.2315,
            7.3725, -0.5585, 7.1605, 0.92711, 5.0997, -1.4851, 6.1877, -1.422, 7.7416, 0.41128,
            6.1484, 0.3835, 5.2013, -0.81974, 7.0861, 0.05919, 6.7211, -0.48191, 6.3969, -0.92047,
            10.42, 3.623, 10.667, 2.1862, 10.883, 2.4903, 10.366, 1.275, 11.098, 3.2233, 8.0916,
            0.84903, 10.164, 2.2284, 10.405, 2.945, 10.569, 1.4148, 9.6656, 3.5837, 11.3, 2.5553,
            9.5749, 2.592, 9.8001, 2.8405, 9.3185, 2.4296, 9.8265, 1.8036, 10.389, 2.0433, 9.2642,
            3.1575, 11.456, 2.1449, 11.155, 1.5013, 10.884, 1.6712, 12.356, 10.386, 11.766, 11.156,
            11.355, 8.642, 11.082, 10.497, 10.444, 10.717, 11.992, 9.6247, 10.291, 9.8252, 11.401,
            10.285, 10.092, 10.079, 11.448, 10.694, 10.617, 9.5943, 11.694, 8.9145, 11.423, 9.7672,
            12.095, 9.6926, 11.96, 10.263, 11.133, 11.018, 11.467, 11.535, 12.034, 10.834, 12.096,
            11.17, 11.139, 9.084, 10.381, 7.8227, 9.6952, 6.4307, 8.7027, 6.2236, 8.4976, 6.2296,
            9.3326, 5.4061, 9.1962, 4.6545, 9.9515, 4.442, 9.8892, 6.6719, 9.8906, 6.1095, 9.8137,
            5.0406, 8.9011, 5.7113, 10.185, 8.8709, 10.596, 5.5416, 9.5537, 6.023, 10.266, 4.9359,
            8.6551, 5.3148, 8.7785, 5.5004, 9.2685, 5.997, 10.384, 6.3641, 9.305, 6.7164,
        ];
    }

    fn cpu() -> Device {
        Device::Cpu
    }

    /// scanpy's own defaults for a small dataset: 500 epochs, min_dist 0.5.
    fn params(seed: u64) -> UmapParams {
        UmapParams {
            n_epochs: 500,
            seed,
            ..UmapParams::default()
        }
    }

    /// Two blobs of `per_cluster` cells, fully connected inside and joined by a
    /// single weak bridge so the graph stays connected.
    fn two_clusters(per_cluster: usize) -> CsrMatrix {
        let n = 2 * per_cluster;
        let mut dense = vec![0.0_f32; n * n];
        for row in 0..n {
            for column in 0..n {
                if row != column && (row < per_cluster) == (column < per_cluster) {
                    dense[row * n + column] = 1.0;
                }
            }
        }
        dense[(per_cluster - 1) * n + per_cluster] = 0.01;
        dense[per_cluster * n + per_cluster - 1] = 0.01;
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    fn symmetric_from_edges(n: usize, edges: &[(u32, u32, f32)]) -> CsrMatrix {
        let mut dense = vec![0.0_f32; n * n];
        for &(head, tail, weight) in edges {
            dense[head as usize * n + tail as usize] = weight;
            dense[tail as usize * n + head as usize] = weight;
        }
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    fn distance(embedding: &Array2<f32>, left: usize, right: usize) -> f32 {
        embedding
            .row(left)
            .iter()
            .zip(embedding.row(right).iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Mean within-cluster and between-cluster distance for two equal clusters.
    fn separation(embedding: &Array2<f32>, per_cluster: usize) -> (f32, f32) {
        let n = 2 * per_cluster;
        let (mut within, mut within_count) = (0.0, 0);
        let (mut between, mut between_count) = (0.0, 0);
        for left in 0..n {
            for right in left + 1..n {
                let d = distance(embedding, left, right);
                if (left < per_cluster) == (right < per_cluster) {
                    within += d;
                    within_count += 1;
                } else {
                    between += d;
                    between_count += 1;
                }
            }
        }
        (within / within_count as f32, between / between_count as f32)
    }

    fn nearest(embedding: &Array2<f32>, cell: usize, k: usize) -> Vec<usize> {
        let mut others: Vec<usize> = (0..embedding.nrows()).filter(|&i| i != cell).collect();
        others.sort_by(|&left, &right| {
            distance(embedding, cell, left)
                .total_cmp(&distance(embedding, cell, right))
                .then(left.cmp(&right))
        });
        others.truncate(k);
        others
    }

    #[test]
    fn fitted_curve_matches_umap_learn() {
        // Printed by umap.umap_.find_ab_params, which uses scipy curve_fit. The
        // grid is deliberately wider than scanpy's defaults, and includes the
        // corners where the fit is stiff: min_dist near 3 * spread drives b past
        // 10, and small spread drives a past 13.
        #[rustfmt::skip]
        const EXPECTED: &[(f32, f32, f64, f64)] = &[
            (0.0,   0.3, 12.967_457_651_095_764, 0.790_494_977_503_261_7),
            (0.05,  0.3, 13.972_623_215_241_446, 0.966_823_547_967_981_1),
            (0.25,  0.3, 13.098_268_687_207_58,  1.721_437_138_872_51),
            (0.5,   0.3,  7.014_077_264_183_128, 2.997_854_849_523_842),
            (0.8,   0.3,  4.209_116_100_989_761, 10.346_527_894_157_395),
            (0.05,  0.5,  5.453_765_163_911_532, 0.895_060_799_997_703_4),
            (0.5,   0.5,  1.667_716_866_662_288_6, 1.929_240_435_347_153),
            (1.0,   0.5,  0.095_683_520_251_482_1, 3.891_226_528_641_106),
            (0.0,   1.0,  1.932_808_397_545_408, 0.790_494_973_590_513_9),
            (0.001, 1.0,  1.929_073_395_323_564_8, 0.791_504_533_140_477_3),
            (0.1,   1.0,  1.576_943_460_575_499_3, 0.895_060_878_168_034_7),
            (0.25,  1.0,  1.121_436_342_630_349, 1.057_499_876_751_258_7),
            (0.5,   1.0,  0.583_030_020_757_172_3, 1.334_166_992_130_317_2),
            (1.0,   1.0,  0.114_975_682_735_773_67, 1.929_237_147_503_818_6),
            (2.0,   1.0,  0.000_434_603_788_570_385_84, 3.891_216_958_023_864),
            (0.25,  1.5,  0.621_894_460_430_009_6, 0.966_823_341_554_800_9),
            (1.5,   1.5,  0.024_052_795_629_159_844, 1.929_233_955_740_086_4),
            (3.0,   1.5,  0.000_018_521_313_761_547_716, 3.891_219_549_999_578),
            (0.5,   2.0,  0.258_878_741_072_124_9, 1.057_499_699_759_442),
            (2.0,   2.0,  0.007_926_773_267_380_772, 1.929_231_522_961_066_8),
            (1.0,   3.0,  0.073_078_624_402_803_34, 1.148_889_435_769_294_3),
            (3.0,   3.0,  0.001_658_266_410_031_984_3, 1.929_231_849_591_078_3),
            (0.0,   5.0,  0.151_748_584_544_517_8, 0.790_494_774_442_064_2),
            (3.0,   5.0,  0.004_131_336_598_340_305, 1.447_462_930_438_019_8),
        ];
        // Relative, because `a` ranges over five decades across the grid; the
        // measured worst case is 3.7e-5 in `a` and 4.2e-6 in `b`.
        const TOLERANCE: f64 = 1e-4;
        for &(min_dist, spread, expected_a, expected_b) in EXPECTED {
            let (a, b) = fit_ab_params(min_dist, spread).unwrap();
            let (a, b) = (f64::from(a), f64::from(b));
            assert!(
                (a - expected_a).abs() <= TOLERANCE * expected_a.abs()
                    && (b - expected_b).abs() <= TOLERANCE * expected_b.abs(),
                "min_dist={min_dist} spread={spread}: got ({a}, {b}), want ({expected_a}, {expected_b})"
            );
        }
    }

    /// umap-learn divides by `negative_sample_rate` unguarded, so zero is not a
    /// setting it has; here it must mean no repulsion at all, which is what
    /// makes a run comparable term by term with a transcribed reference.
    #[test]
    fn a_zero_negative_sample_rate_switches_repulsion_off() {
        let graph = two_clusters(6);
        let params = UmapParams {
            n_epochs: 30,
            negative_sample_rate: 0,
            ..params(0)
        };
        let embedding = umap(&graph, &params, &cpu()).unwrap();
        // With attraction only and no repulsion every connected vertex collapses
        // onto its cluster; a stray repulsion would leave them spread out.
        let (within, _) = separation(&embedding, 6);
        assert!(
            within < 1e-2,
            "within-cluster spread {within} is not a collapse"
        );
    }

    #[test]
    fn output_has_the_requested_shape_and_is_finite() {
        let graph = two_clusters(20);
        for n_components in [1, 2, 3] {
            let embedding = umap(
                &graph,
                &UmapParams {
                    n_components,
                    ..params(0)
                },
                &cpu(),
            )
            .unwrap();
            assert_eq!(embedding.dim(), (40, n_components));
            assert!(embedding.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn clusters_stay_separated() {
        let embedding = umap(&two_clusters(25), &params(0), &cpu()).unwrap();
        let (within, between) = separation(&embedding, 25);
        assert!(
            between > 5.0 * within,
            "within={within}, between={between} are not well separated"
        );
    }

    #[test]
    fn the_same_seed_reproduces_the_layout_bit_for_bit() {
        let graph = two_clusters(20);
        let first = umap(&graph, &params(42), &cpu()).unwrap();
        let second = umap(&graph, &params(42), &cpu()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn a_different_seed_gives_a_different_but_equally_separated_layout() {
        let graph = two_clusters(25);
        let first = umap(&graph, &params(1), &cpu()).unwrap();
        let second = umap(&graph, &params(2), &cpu()).unwrap();
        assert_ne!(first, second);

        let (within, between) = separation(&second, 25);
        assert!(
            between > 5.0 * within,
            "within={within}, between={between} are not well separated"
        );
    }

    #[test]
    fn neighbourhoods_agree_with_scanpy() {
        const NEAR: usize = 15;
        const WIDE: usize = 30;
        let graph = symmetric_from_edges(reference::N_CELLS, reference::EDGES);
        let scanpy =
            Array2::from_shape_vec((reference::N_CELLS, 2), reference::EMBEDDING.to_vec()).unwrap();
        let ours = umap(&graph, &params(0), &cpu()).unwrap();

        let mut preserved = 0;
        for cell in 0..reference::N_CELLS {
            let wide = nearest(&ours, cell, WIDE);
            preserved += nearest(&scanpy, cell, NEAR)
                .iter()
                .filter(|neighbour| wide.contains(neighbour))
                .count();
        }
        let fraction = preserved as f32 / (reference::N_CELLS * NEAR) as f32;
        assert!(
            fraction >= 0.80,
            "neighbourhood preservation {:.1}% is below 80%",
            fraction * 100.0
        );
    }

    #[test]
    fn rejects_degenerate_input() {
        let graph = two_clusters(4);
        let zero_epochs = UmapParams {
            n_epochs: 0,
            ..params(0)
        };
        let zero_components = UmapParams {
            n_components: 0,
            ..params(0)
        };
        assert!(umap(&graph, &zero_epochs, &cpu()).is_err());
        assert!(umap(&graph, &zero_components, &cpu()).is_err());

        let empty = CsrMatrix::new(vec![0], vec![], vec![], 0).unwrap();
        assert!(umap(&empty, &params(0), &cpu()).is_err());

        let no_edges = CsrMatrix::from_dense(&[0.0; 9], 3, 3).unwrap();
        assert!(umap(&no_edges, &params(0), &cpu()).is_err());
    }
}
