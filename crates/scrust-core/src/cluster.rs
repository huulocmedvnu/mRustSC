//! Community detection on the neighbour graph. Owned by feat/leiden.
//!
//! # The objective
//!
//! Both algorithms maximise the same quantity, RBConfiguration — Newman
//! modularity with a resolution parameter scaling the configuration null model:
//!
//! ```text
//! Q = (1 / 2m) * sum_ij (A_ij - resolution * k_i k_j / 2m) * delta(c_i, c_j)
//! ```
//!
//! with `k_i` the strength of node `i` and `2m` the sum of every entry of `A`.
//! That is exactly what `scanpy.tl.leiden` optimises, through
//! `leidenalg.RBConfigurationVertexPartition` or, with `flavor="igraph"`,
//! through `community_leiden(objective_function="modularity")`.
//!
//! # Leiden versus Louvain
//!
//! The two share everything here except one pass. Louvain is local moving plus
//! aggregation; Leiden inserts a *refinement* between them, which re-splits each
//! community into well-connected subcommunities and aggregates on those. That is
//! what stops Leiden from returning internally disconnected communities, which
//! Louvain demonstrably does, so it is not an optional speed-up.
//!
//! # Where this runs, and why
//!
//! On the CPU, all of it. Local moving and refinement are pointer-chasing loops
//! whose next memory access depends on the result of the last one: a node moves,
//! which changes the community strengths its neighbours will read, which changes
//! which of them are re-queued. There is no tensor to form and no independent
//! work to fan out over; a GPU port would serialise on the same dependency chain
//! while paying kernel-launch latency per node. Aggregation and the modularity
//! evaluation *are* sparse reductions, but they are single linear passes over
//! the stored entries and cost a few milliseconds at single-cell graph sizes —
//! measured against a candle formulation of the modularity reduction, the tensor
//! version was slower on both backends at every size the neighbour graph
//! reaches. `device` is therefore accepted and not used; see the report.
//!
//! # Memory
//!
//! Peak memory is `O(n + nnz)`: the symmetrised adjacency is stored once as flat
//! arrays and every aggregate level is strictly smaller. Nothing here is dense
//! in the cell count.

use std::collections::VecDeque;

use candle_core::Device;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Community labels plus the modularity the partition achieves.
#[derive(Debug, Clone)]
pub struct Partition {
    pub labels: Vec<u32>,
    pub modularity: f64,
    pub n_communities: usize,
}

/// Leiden's `theta`: how much randomness the refinement merge allows.
///
/// leidenalg's default. Zero would make refinement greedy and reproduce
/// Louvain's tendency to lock in a bad early split.
const REFINEMENT_RANDOMNESS: f64 = 0.01;

/// Leiden community detection, as `scanpy.tl.leiden`.
///
/// `n_iterations` restarts the whole algorithm from the partition the previous
/// pass found, as `leidenalg` does; it stops early once a pass changes nothing.
pub fn leiden(
    graph: &CsrMatrix,
    resolution: f64,
    n_iterations: usize,
    seed: u64,
    _device: &Device,
) -> Result<Partition> {
    check_resolution(resolution)?;
    if n_iterations == 0 {
        return Err(Error::parameter("n_iterations", "at least 1", n_iterations));
    }
    let graph = Graph::from_csr(graph)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut labels: Vec<u32> = (0..graph.n_nodes() as u32).collect();
    let mut n_communities = graph.n_nodes();
    for _ in 0..n_iterations {
        let (found, found_communities) = optimise(
            &graph,
            &labels,
            n_communities,
            resolution,
            Refinement::Enabled,
            &mut rng,
        );
        // A pass that returns its own input has converged; the labels are
        // compacted the same way both times, so equality is exact.
        if found == labels {
            break;
        }
        labels = found;
        n_communities = found_communities;
    }
    Ok(describe(&graph, labels, n_communities, resolution))
}

/// Louvain community detection, as `scanpy.tl.louvain`.
pub fn louvain(
    graph: &CsrMatrix,
    resolution: f64,
    seed: u64,
    _device: &Device,
) -> Result<Partition> {
    check_resolution(resolution)?;
    let graph = Graph::from_csr(graph)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let singletons: Vec<u32> = (0..graph.n_nodes() as u32).collect();
    let (labels, n_communities) = optimise(
        &graph,
        &singletons,
        graph.n_nodes(),
        resolution,
        Refinement::Disabled,
        &mut rng,
    );
    Ok(describe(&graph, labels, n_communities, resolution))
}

/// Newman modularity of an existing labelling.
pub fn modularity(graph: &CsrMatrix, labels: &[u32], resolution: f64) -> Result<f64> {
    check_resolution(resolution)?;
    let graph = Graph::from_csr(graph)?;
    if labels.len() != graph.n_nodes() {
        return Err(Error::shape(
            format!("{} labels, one per node", graph.n_nodes()),
            format!("{} labels", labels.len()),
        ));
    }
    let n_communities = labels.iter().max().map_or(0, |&id| id as usize + 1);
    Ok(score(&graph, labels, n_communities, resolution))
}

fn check_resolution(resolution: f64) -> Result<()> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(Error::parameter(
            "resolution",
            "finite and greater than zero",
            resolution,
        ));
    }
    Ok(())
}

/// Relabel by descending community size and attach the objective reached.
fn describe(graph: &Graph, labels: Vec<u32>, n_communities: usize, resolution: f64) -> Partition {
    let labels = by_descending_size(&labels, n_communities);
    let modularity = score(graph, &labels, n_communities, resolution);
    Partition {
        labels,
        modularity,
        n_communities,
    }
}

// ---------------------------------------------------------------------------
// The graph the optimisation works on
// ---------------------------------------------------------------------------

/// An undirected weighted graph as the optimisation needs it: flat neighbour
/// lists, node strengths, and the total edge weight, all in `f64`.
///
/// Aggregation returns the same type, so local moving and refinement are written
/// once and run unchanged on the original graph and on every aggregate of it.
/// Weights are `f64` because modularity subtracts two nearly equal sums over all
/// `nnz` entries, which `f32` cannot do without losing the difference.
#[derive(Debug, Clone)]
struct Graph {
    indptr: Vec<usize>,
    neighbors: Vec<u32>,
    weights: Vec<f64>,
    /// `k_i`, the sum of row `i`.
    strength: Vec<f64>,
    /// `2m`, the sum of every stored entry.
    total: f64,
}

impl Graph {
    /// Build from the square, symmetric connectivities `pp.neighbors` writes.
    ///
    /// The input is symmetrised as `(A + A^T) / 2`, which leaves a symmetric
    /// matrix untouched and gives a well-defined undirected graph for anything
    /// else, rather than silently optimising a quantity that is not modularity.
    fn from_csr(matrix: &CsrMatrix) -> Result<Self> {
        let n = matrix.n_rows();
        if n == 0 || matrix.n_cols() != n {
            return Err(Error::shape(
                "a square graph with at least one node",
                format!("{} x {}", n, matrix.n_cols()),
            ));
        }
        if let Some(&bad) = matrix
            .values()
            .iter()
            .find(|value| !value.is_finite() || **value < 0.0)
        {
            return Err(Error::parameter(
                "graph weights",
                "finite and non-negative",
                bad,
            ));
        }

        let mut indptr = vec![0usize; n + 1];
        for row in 0..n {
            for &column in row_indices(matrix, row) {
                indptr[row + 1] += 1;
                indptr[column as usize + 1] += 1;
            }
        }
        for node in 0..n {
            indptr[node + 1] += indptr[node];
        }

        let mut neighbors = vec![0u32; indptr[n]];
        let mut weights = vec![0.0f64; indptr[n]];
        let mut cursor = indptr[..n].to_vec();
        for row in 0..n {
            let from = matrix.indptr()[row] as usize;
            for offset in 0..row_indices(matrix, row).len() {
                let column = matrix.indices()[from + offset] as usize;
                // Half to each direction, so a symmetric input round-trips
                // exactly and a self loop keeps its full weight.
                let half = f64::from(matrix.values()[from + offset]) / 2.0;
                neighbors[cursor[row]] = column as u32;
                weights[cursor[row]] = half;
                cursor[row] += 1;
                neighbors[cursor[column]] = row as u32;
                weights[cursor[column]] = half;
                cursor[column] += 1;
            }
        }

        let strength: Vec<f64> = (0..n)
            .map(|node| weights[indptr[node]..indptr[node + 1]].iter().sum())
            .collect();
        let total: f64 = strength.iter().sum();
        if total <= 0.0 {
            return Err(Error::parameter(
                "graph",
                "at least one edge of positive weight",
                total,
            ));
        }

        Ok(Self {
            indptr,
            neighbors,
            weights,
            strength,
            total,
        })
    }

    fn n_nodes(&self) -> usize {
        self.indptr.len() - 1
    }

    fn row(&self, node: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let span = self.indptr[node]..self.indptr[node + 1];
        self.neighbors[span.clone()]
            .iter()
            .copied()
            .zip(self.weights[span].iter().copied())
    }
}

fn row_indices(matrix: &CsrMatrix, row: usize) -> &[u32] {
    let from = matrix.indptr()[row] as usize;
    let to = matrix.indptr()[row + 1] as usize;
    &matrix.indices()[from..to]
}

/// `Q` for a labelling, community by community.
fn score(graph: &Graph, labels: &[u32], n_communities: usize, resolution: f64) -> f64 {
    let mut internal = vec![0.0f64; n_communities];
    let mut strength = vec![0.0f64; n_communities];
    for node in 0..graph.n_nodes() {
        let community = labels[node] as usize;
        strength[community] += graph.strength[node];
        for (neighbour, weight) in graph.row(node) {
            if labels[neighbour as usize] as usize == community {
                internal[community] += weight;
            }
        }
    }
    internal
        .iter()
        .zip(&strength)
        .map(|(&inside, &total)| {
            inside / graph.total - resolution * (total / graph.total) * (total / graph.total)
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Local moving, shared by both algorithms
// ---------------------------------------------------------------------------

/// A community assignment plus the bookkeeping local moving reads on every node.
///
/// Community ids live in `0..n_nodes` so that a community can always split back
/// into singletons without renumbering.
struct Communities {
    of_node: Vec<u32>,
    strength: Vec<f64>,
    size: Vec<usize>,
    /// Ids currently unused, so a node can leave for an empty community.
    free: Vec<u32>,
}

impl Communities {
    fn from_labels(labels: &[u32], n_communities: usize, strength: &[f64]) -> Self {
        let n = labels.len();
        let mut communities = Self {
            of_node: labels.to_vec(),
            strength: vec![0.0; n],
            size: vec![0; n],
            free: (n_communities..n).map(|id| id as u32).rev().collect(),
        };
        for (node, &community) in labels.iter().enumerate() {
            communities.strength[community as usize] += strength[node];
            communities.size[community as usize] += 1;
        }
        communities
    }

    fn leave(&mut self, node: usize, strength: f64) {
        let community = self.of_node[node] as usize;
        self.strength[community] -= strength;
        self.size[community] -= 1;
        if self.size[community] == 0 {
            self.free.push(community as u32);
        }
    }

    fn join(&mut self, node: usize, community: u32, strength: f64) {
        self.of_node[node] = community;
        self.strength[community as usize] += strength;
        self.size[community as usize] += 1;
    }
}

/// Move each node to the neighbouring community that most improves `Q`,
/// re-queueing a node's neighbours whenever it moves. Returns whether anything
/// moved at all.
///
/// Visiting order is a single seeded shuffle and every tie is broken towards the
/// smaller community id, so the result depends on `seed` and nothing else.
fn move_nodes(
    graph: &Graph,
    communities: &mut Communities,
    resolution: f64,
    rng: &mut StdRng,
) -> bool {
    let n = graph.n_nodes();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<u32> = order.into_iter().collect();
    let mut queued = vec![true; n];

    let mut edge_to = vec![0.0f64; n];
    let mut seen = vec![false; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut moved_any = false;

    while let Some(node) = queue.pop_front() {
        let node = node as usize;
        queued[node] = false;
        let node_strength = graph.strength[node];
        let previous = communities.of_node[node];
        communities.leave(node, node_strength);

        for (neighbour, weight) in graph.row(node) {
            if neighbour as usize == node {
                continue;
            }
            let community = communities.of_node[neighbour as usize] as usize;
            if !seen[community] {
                seen[community] = true;
                touched.push(community as u32);
            }
            edge_to[community] += weight;
        }
        touched.sort_unstable();

        // Only the terms that vary with the target matter, so the gain of moving
        // node i into community C is e(i, C) - resolution * K_C * k_i / 2m, with
        // K_C already excluding i.
        let gain = |community: u32, communities: &Communities| {
            edge_to[community as usize]
                - resolution * communities.strength[community as usize] * node_strength
                    / graph.total
        };
        let mut best = previous;
        let mut best_gain = gain(previous, communities);
        for &community in &touched {
            let candidate = gain(community, communities);
            if candidate > best_gain {
                best_gain = candidate;
                best = community;
            }
        }
        // Leaving for a community of its own is what lets a node escape a
        // community it was merged into at a coarser level.
        let empty = communities.free.last().copied();
        if let Some(id) = empty.filter(|_| best_gain < 0.0) {
            best = id;
        }

        if best != previous {
            if Some(best) == empty {
                communities.free.pop();
            }
            communities.join(node, best, node_strength);
            moved_any = true;
            for (neighbour, _) in graph.row(node) {
                let neighbour = neighbour as usize;
                if neighbour != node && !queued[neighbour] && communities.of_node[neighbour] != best
                {
                    queued[neighbour] = true;
                    queue.push_back(neighbour as u32);
                }
            }
        } else {
            communities.join(node, previous, node_strength);
        }

        for &community in &touched {
            edge_to[community as usize] = 0.0;
            seen[community as usize] = false;
        }
        touched.clear();
    }
    moved_any
}

// ---------------------------------------------------------------------------
// Refinement: the whole of the difference between Leiden and Louvain
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refinement {
    Enabled,
    Disabled,
}

/// A node or subcommunity counts as well connected to the rest of its community
/// when its cut into that remainder is at least what the null model expects.
fn well_connected(cut: f64, part: f64, subset: f64, resolution: f64, total: f64) -> bool {
    cut >= resolution * part * (subset - part) / total
}

/// Split every community of `partition` into well-connected subcommunities.
///
/// Aggregating on the result rather than on `partition` is what guarantees the
/// communities Leiden returns are internally connected: a community that local
/// moving glued together out of two barely touching halves is torn apart here,
/// and the halves are then free to leave separately at the next level.
fn refine(
    graph: &Graph,
    partition: &[u32],
    n_communities: usize,
    resolution: f64,
    rng: &mut StdRng,
) -> Vec<u32> {
    let n = graph.n_nodes();
    let (offsets, members) = group_by(partition, n_communities);

    let mut of_node: Vec<u32> = (0..n as u32).collect();
    let mut sub_strength = graph.strength.clone();
    let mut sub_size = vec![1usize; n];
    // Per subcommunity: the summed node-to-subset weight of its members, and the
    // weight of the edges inside it counted in both directions. Their difference
    // is the subcommunity's cut into the rest of the community.
    let mut sub_boundary = vec![0.0f64; n];
    let mut sub_internal = vec![0.0f64; n];

    let mut in_subset = vec![false; n];
    let mut to_subset = vec![0.0f64; n];
    let mut edge_to = vec![0.0f64; n];
    let mut seen = vec![false; n];

    for community in 0..n_communities {
        let subset = &members[offsets[community]..offsets[community + 1]];
        if subset.len() < 2 {
            continue;
        }
        for &node in subset {
            in_subset[node as usize] = true;
        }

        let mut subset_strength = 0.0;
        for &node in subset {
            let node = node as usize;
            let mut inside = 0.0;
            for (neighbour, weight) in graph.row(node) {
                if neighbour as usize != node && in_subset[neighbour as usize] {
                    inside += weight;
                }
            }
            to_subset[node] = inside;
            sub_boundary[node] = inside;
            sub_internal[node] = 0.0;
            subset_strength += graph.strength[node];
        }

        let mut visit: Vec<u32> = subset
            .iter()
            .copied()
            .filter(|&node| {
                let node = node as usize;
                well_connected(
                    to_subset[node],
                    graph.strength[node],
                    subset_strength,
                    resolution,
                    graph.total,
                )
            })
            .collect();
        visit.shuffle(rng);

        for node in visit {
            let node = node as usize;
            // Only nodes still on their own may merge; that keeps every
            // subcommunity a union of merges into one seed and the bookkeeping
            // above exact.
            if sub_size[node] != 1 || of_node[node] != node as u32 {
                continue;
            }

            let mut targets: Vec<u32> = Vec::new();
            for (neighbour, weight) in graph.row(node) {
                let neighbour = neighbour as usize;
                if neighbour == node || !in_subset[neighbour] {
                    continue;
                }
                let target = of_node[neighbour];
                if !seen[target as usize] {
                    seen[target as usize] = true;
                    targets.push(target);
                }
                edge_to[target as usize] += weight;
            }
            targets.sort_unstable();

            let mut choices: Vec<(u32, f64)> = Vec::new();
            for &target in &targets {
                let target_index = target as usize;
                let gain = edge_to[target_index]
                    - resolution * sub_strength[target_index] * graph.strength[node] / graph.total;
                let cut = sub_boundary[target_index] - sub_internal[target_index];
                if gain >= 0.0
                    && well_connected(
                        cut,
                        sub_strength[target_index],
                        subset_strength,
                        resolution,
                        graph.total,
                    )
                {
                    choices.push((target, gain));
                }
            }
            if let Some(target) = sample_merge(&choices, rng) {
                let target_index = target as usize;
                sub_internal[target_index] += 2.0 * edge_to[target_index];
                sub_boundary[target_index] += to_subset[node];
                sub_strength[target_index] += graph.strength[node];
                sub_size[target_index] += 1;
                sub_size[node] = 0;
                of_node[node] = target;
            }

            for &target in &targets {
                edge_to[target as usize] = 0.0;
                seen[target as usize] = false;
            }
        }

        for &node in subset {
            in_subset[node as usize] = false;
        }
    }
    of_node
}

/// Pick a merge target with probability proportional to `exp(gain / theta)`.
///
/// The maximum is subtracted before exponentiating: `theta` is 0.01, so any gain
/// of order one overflows `exp`, and only the ratios between the choices matter.
fn sample_merge(choices: &[(u32, f64)], rng: &mut StdRng) -> Option<u32> {
    let &(best, largest) = choices
        .iter()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let weights: Vec<f64> = choices
        .iter()
        .map(|&(_, gain)| ((gain - largest) / REFINEMENT_RANDOMNESS).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Some(best);
    }

    let mut draw = rng.gen::<f64>() * total;
    for (&(community, _), &weight) in choices.iter().zip(&weights) {
        draw -= weight;
        if draw <= 0.0 {
            return Some(community);
        }
    }
    Some(best)
}

// ---------------------------------------------------------------------------
// Aggregation and the level loop, shared by both algorithms
// ---------------------------------------------------------------------------

/// Collapse each community to one node, summing the weights between them.
///
/// The result has one entry per distinct community pair rather than per original
/// edge, so every level is smaller than the last and the loop terminates.
fn aggregate(graph: &Graph, labels: &[u32], n_communities: usize) -> Graph {
    let (offsets, members) = group_by(labels, n_communities);
    let mut indptr = Vec::with_capacity(n_communities + 1);
    let mut neighbors: Vec<u32> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    indptr.push(0usize);

    let mut edge_to = vec![0.0f64; n_communities];
    let mut seen = vec![false; n_communities];
    let mut touched: Vec<u32> = Vec::new();
    for community in 0..n_communities {
        for &node in &members[offsets[community]..offsets[community + 1]] {
            for (neighbour, weight) in graph.row(node as usize) {
                let target = labels[neighbour as usize];
                if !seen[target as usize] {
                    seen[target as usize] = true;
                    touched.push(target);
                }
                edge_to[target as usize] += weight;
            }
        }
        touched.sort_unstable();
        for &target in &touched {
            neighbors.push(target);
            weights.push(edge_to[target as usize]);
            edge_to[target as usize] = 0.0;
            seen[target as usize] = false;
        }
        touched.clear();
        indptr.push(neighbors.len());
    }

    let strength: Vec<f64> = (0..n_communities)
        .map(|community| {
            weights[indptr[community]..indptr[community + 1]]
                .iter()
                .sum()
        })
        .collect();
    let total = graph.total;
    Graph {
        indptr,
        neighbors,
        weights,
        strength,
        total,
    }
}

/// One run of the level loop: local moving, optional refinement, aggregation,
/// repeated until a level cannot merge anything.
fn optimise(
    graph: &Graph,
    initial: &[u32],
    n_initial: usize,
    resolution: f64,
    refinement: Refinement,
    rng: &mut StdRng,
) -> (Vec<u32>, usize) {
    let mut level = graph.clone();
    let mut level_labels = initial.to_vec();
    let mut level_communities = n_initial;
    // Which node of the current level each original node has been folded into.
    let mut fold: Vec<u32> = (0..graph.n_nodes() as u32).collect();

    loop {
        let mut communities =
            Communities::from_labels(&level_labels, level_communities, &level.strength);
        let moved = move_nodes(&level, &mut communities, resolution, rng);
        let (partition, n_communities) = compact(&communities.of_node);

        if !moved || n_communities == level.n_nodes() {
            let labels = fold.iter().map(|&node| partition[node as usize]).collect();
            return (labels, n_communities);
        }

        let (collapse, n_collapsed) = match refinement {
            Refinement::Enabled => {
                compact(&refine(&level, &partition, n_communities, resolution, rng))
            }
            Refinement::Disabled => (partition.clone(), n_communities),
        };
        let next = aggregate(&level, &collapse, n_collapsed);
        // The next level starts from the partition local moving found, so
        // refinement can only split communities, never undo a profitable merge.
        // Without refinement this is the singleton partition, which is Louvain.
        let mut next_labels = vec![0u32; n_collapsed];
        for node in 0..level.n_nodes() {
            next_labels[collapse[node] as usize] = partition[node];
        }
        for node in fold.iter_mut() {
            *node = collapse[*node as usize];
        }

        level = next;
        level_labels = next_labels;
        level_communities = n_communities;
    }
}

// ---------------------------------------------------------------------------
// Label bookkeeping
// ---------------------------------------------------------------------------

/// Members of each label, as a flat array with per-label offsets.
fn group_by(labels: &[u32], n_labels: usize) -> (Vec<usize>, Vec<u32>) {
    let mut offsets = vec![0usize; n_labels + 1];
    for &label in labels {
        offsets[label as usize + 1] += 1;
    }
    for label in 0..n_labels {
        offsets[label + 1] += offsets[label];
    }
    let mut members = vec![0u32; labels.len()];
    let mut cursor = offsets[..n_labels].to_vec();
    for (node, &label) in labels.iter().enumerate() {
        members[cursor[label as usize]] = node as u32;
        cursor[label as usize] += 1;
    }
    (offsets, members)
}

/// Renumber sparse labels to `0..n`, in ascending order of the ids they replace.
///
/// Ascending rather than first-seen, so the numbering does not depend on the
/// seeded visit order and a converged run reproduces its own input exactly.
fn compact(labels: &[u32]) -> (Vec<u32>, usize) {
    let width = labels.iter().max().map_or(0, |&id| id as usize + 1);
    let mut used = vec![false; width];
    for &label in labels {
        used[label as usize] = true;
    }
    let mut renumbered = vec![0u32; width];
    let mut assigned = 0u32;
    for (id, &present) in used.iter().enumerate() {
        if present {
            renumbered[id] = assigned;
            assigned += 1;
        }
    }
    (
        labels.iter().map(|&id| renumbered[id as usize]).collect(),
        assigned as usize,
    )
}

/// Relabel so that community 0 is the largest, as scanpy's users expect.
fn by_descending_size(labels: &[u32], n_communities: usize) -> Vec<u32> {
    let mut sizes = vec![0usize; n_communities];
    for &label in labels {
        sizes[label as usize] += 1;
    }
    let mut order: Vec<u32> = (0..n_communities as u32).collect();
    order.sort_by(|&a, &b| {
        sizes[b as usize]
            .cmp(&sizes[a as usize])
            .then_with(|| a.cmp(&b))
    });
    let mut rank = vec![0u32; n_communities];
    for (new, &old) in order.iter().enumerate() {
        rank[old as usize] = new as u32;
    }
    labels.iter().map(|&id| rank[id as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n_cliques` cliques of `size` nodes each, with no edge between them.
    fn disconnected_cliques(n_cliques: usize, size: usize) -> CsrMatrix {
        let n = n_cliques * size;
        let mut dense = vec![0.0f32; n * n];
        for clique in 0..n_cliques {
            for a in 0..size {
                for b in 0..size {
                    if a != b {
                        dense[(clique * size + a) * n + clique * size + b] = 1.0;
                    }
                }
            }
        }
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    /// Two triangles joined by a single edge: small enough to reason about by
    /// hand, connected enough that a bad algorithm can merge them.
    fn barbell() -> CsrMatrix {
        let mut dense = vec![0.0f32; 36];
        let edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)];
        for (a, b) in edges {
            dense[a * 6 + b] = 1.0;
            dense[b * 6 + a] = 1.0;
        }
        CsrMatrix::from_dense(&dense, 6, 6).unwrap()
    }

    fn n_labels(labels: &[u32]) -> usize {
        let mut ids = labels.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    #[test]
    fn modularity_matches_a_hand_computed_value() {
        // The barbell has 7 edges, so 2m = 14. Split into the two triangles:
        // each side holds 3 internal edges, one of which (2-3) is cut, so the
        // internal weight is 2*3 = 6 per side. Strengths are 2,2,3,3,2,2, so
        // each side sums to 7. Q = 2 * (6/14 - (7/14)^2) = 6/7 - 1/2.
        let expected = 6.0 / 7.0 - 0.5;
        let labels = [0, 0, 0, 1, 1, 1];
        let got = modularity(&barbell(), &labels, 1.0).unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "expected {expected}, got {got}"
        );

        // Everything in one community: internal weight is all of 2m.
        let one = modularity(&barbell(), &[0; 6], 1.0).unwrap();
        assert!((one - 0.0).abs() < 1e-12, "one community should score 0");

        // Every node alone: no internal weight, only the null model term.
        let alone = modularity(&barbell(), &[0, 1, 2, 3, 4, 5], 1.0).unwrap();
        let expected_alone: f64 = -[2.0, 2.0, 3.0, 3.0, 2.0, 2.0]
            .iter()
            .map(|k: &f64| (k / 14.0) * (k / 14.0))
            .sum::<f64>();
        assert!((alone - expected_alone).abs() < 1e-12);
    }

    #[test]
    fn resolution_scales_only_the_null_model_term() {
        let labels = [0, 0, 0, 1, 1, 1];
        let at_one = modularity(&barbell(), &labels, 1.0).unwrap();
        let at_two = modularity(&barbell(), &labels, 2.0).unwrap();
        assert!((at_two - (6.0 / 7.0 - 2.0 * 0.5)).abs() < 1e-12);
        assert!(at_two < at_one);
    }

    #[test]
    fn leiden_recovers_disconnected_cliques() {
        let graph = disconnected_cliques(4, 8);
        let partition = leiden(&graph, 1.0, 2, 0, &Device::Cpu).unwrap();
        assert_eq!(partition.n_communities, 4);
        for clique in 0..4 {
            let first = partition.labels[clique * 8];
            assert!(partition.labels[clique * 8..(clique + 1) * 8]
                .iter()
                .all(|&label| label == first));
        }
        assert_eq!(n_labels(&partition.labels), 4);
    }

    #[test]
    fn louvain_recovers_disconnected_cliques() {
        let graph = disconnected_cliques(4, 8);
        let partition = louvain(&graph, 1.0, 0, &Device::Cpu).unwrap();
        assert_eq!(partition.n_communities, 4);
        for clique in 0..4 {
            let first = partition.labels[clique * 8];
            assert!(partition.labels[clique * 8..(clique + 1) * 8]
                .iter()
                .all(|&label| label == first));
        }
    }

    #[test]
    fn leiden_scores_no_worse_than_louvain() {
        let graph = disconnected_cliques(6, 5);
        for seed in 0..8 {
            let leiden = leiden(&graph, 1.0, 2, seed, &Device::Cpu).unwrap();
            let louvain = louvain(&graph, 1.0, seed, &Device::Cpu).unwrap();
            assert!(
                leiden.modularity >= louvain.modularity - 1e-9,
                "seed {seed}: leiden {} < louvain {}",
                leiden.modularity,
                louvain.modularity
            );
        }
    }

    #[test]
    fn reported_modularity_is_the_partitions_modularity() {
        let graph = disconnected_cliques(5, 6);
        let partition = leiden(&graph, 1.0, 2, 3, &Device::Cpu).unwrap();
        let recomputed = modularity(&graph, &partition.labels, 1.0).unwrap();
        assert!((partition.modularity - recomputed).abs() < 1e-12);
    }

    #[test]
    fn the_same_seed_gives_the_same_labels() {
        let graph = disconnected_cliques(4, 7);
        let first = leiden(&graph, 1.0, 2, 11, &Device::Cpu).unwrap();
        let second = leiden(&graph, 1.0, 2, 11, &Device::Cpu).unwrap();
        assert_eq!(first.labels, second.labels);
        assert_eq!(first.modularity.to_bits(), second.modularity.to_bits());
    }

    #[test]
    fn higher_resolution_never_finds_fewer_communities() {
        let graph = disconnected_cliques(4, 10);
        let mut previous = 0;
        for resolution in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let partition = leiden(&graph, resolution, 2, 0, &Device::Cpu).unwrap();
            assert!(
                partition.n_communities >= previous,
                "resolution {resolution} found {} communities after {previous}",
                partition.n_communities
            );
            previous = partition.n_communities;
        }
    }

    #[test]
    fn communities_are_labelled_by_descending_size() {
        // One clique of 12 and one of 3: the larger must become community 0.
        let mut dense = vec![0.0f32; 15 * 15];
        for (start, size) in [(0usize, 3usize), (3, 12)] {
            for a in start..start + size {
                for b in start..start + size {
                    if a != b {
                        dense[a * 15 + b] = 1.0;
                    }
                }
            }
        }
        let graph = CsrMatrix::from_dense(&dense, 15, 15).unwrap();
        let partition = leiden(&graph, 1.0, 2, 0, &Device::Cpu).unwrap();
        assert_eq!(partition.n_communities, 2);
        assert_eq!(partition.labels[3], 0);
        assert_eq!(partition.labels[0], 1);
    }

    #[test]
    fn an_asymmetric_graph_is_symmetrised() {
        // Only the upper triangle stored: the undirected graph is the barbell.
        let mut dense = vec![0.0f32; 36];
        for (a, b) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)] {
            dense[a * 6 + b] = 2.0;
        }
        let half = CsrMatrix::from_dense(&dense, 6, 6).unwrap();
        let labels = [0, 0, 0, 1, 1, 1];
        let got = modularity(&half, &labels, 1.0).unwrap();
        assert!((got - (6.0 / 7.0 - 0.5)).abs() < 1e-12);
    }

    #[test]
    fn rejects_degenerate_input() {
        let empty = CsrMatrix::new(vec![0], vec![], vec![], 0).unwrap();
        assert!(leiden(&empty, 1.0, 2, 0, &Device::Cpu).is_err());
        assert!(louvain(&empty, 1.0, 0, &Device::Cpu).is_err());
        assert!(modularity(&empty, &[], 1.0).is_err());

        let negative = CsrMatrix::new(vec![0, 1, 2], vec![1, 0], vec![-1.0, -1.0], 2).unwrap();
        assert!(leiden(&negative, 1.0, 2, 0, &Device::Cpu).is_err());
        assert!(modularity(&negative, &[0, 0], 1.0).is_err());

        let graph = disconnected_cliques(2, 3);
        assert!(leiden(&graph, 0.0, 2, 0, &Device::Cpu).is_err());
        assert!(louvain(&graph, 0.0, 0, &Device::Cpu).is_err());
        assert!(modularity(&graph, &[0; 6], 0.0).is_err());
        assert!(leiden(&graph, f64::NAN, 2, 0, &Device::Cpu).is_err());
        assert!(leiden(&graph, 1.0, 0, 0, &Device::Cpu).is_err());

        // A graph with no edges at all has no modularity to speak of.
        let edgeless = CsrMatrix::new(vec![0, 0, 0], vec![], vec![], 2).unwrap();
        assert!(leiden(&edgeless, 1.0, 2, 0, &Device::Cpu).is_err());

        // Rectangular input is not a graph.
        let rectangular = CsrMatrix::new(vec![0, 1], vec![2], vec![1.0], 4).unwrap();
        assert!(leiden(&rectangular, 1.0, 2, 0, &Device::Cpu).is_err());
        assert!(modularity(&disconnected_cliques(2, 3), &[0, 0], 1.0).is_err());
    }

    #[test]
    fn communities_are_internally_connected() {
        // Two cliques joined at a single node: Leiden's refinement must not
        // return a community that falls apart when that node is removed.
        let graph = disconnected_cliques(3, 9);
        let partition = leiden(&graph, 1.0, 2, 5, &Device::Cpu).unwrap();
        let built = Graph::from_csr(&graph).unwrap();
        let (offsets, members) = group_by(&partition.labels, partition.n_communities);
        for community in 0..partition.n_communities {
            let subset = &members[offsets[community]..offsets[community + 1]];
            assert!(is_connected(&built, subset));
        }
    }

    /// Breadth-first search restricted to `subset`.
    fn is_connected(graph: &Graph, subset: &[u32]) -> bool {
        let mut inside = vec![false; graph.n_nodes()];
        for &node in subset {
            inside[node as usize] = true;
        }
        let mut visited = vec![false; graph.n_nodes()];
        let mut stack = vec![subset[0]];
        visited[subset[0] as usize] = true;
        let mut reached = 1;
        while let Some(node) = stack.pop() {
            for (neighbour, weight) in graph.row(node as usize) {
                let neighbour = neighbour as usize;
                if weight > 0.0 && inside[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    reached += 1;
                    stack.push(neighbour as u32);
                }
            }
        }
        reached == subset.len()
    }

    #[test]
    fn aggregation_preserves_the_objective() {
        let graph = Graph::from_csr(&disconnected_cliques(4, 6)).unwrap();
        let labels: Vec<u32> = (0..24).map(|node| node / 6).collect();
        let before = score(&graph, &labels, 4, 1.0);
        let collapsed = aggregate(&graph, &labels, 4);
        let after = score(&collapsed, &[0, 1, 2, 3], 4, 1.0);
        assert!((before - after).abs() < 1e-12);
        assert!((collapsed.total - graph.total).abs() < 1e-12);
    }
}
