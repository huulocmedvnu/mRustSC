//! Approximate k nearest neighbours by NN-descent (Dong et al., 2011) seeded with a
//! random-projection forest, the construction pynndescent uses behind scanpy.
//!
//! The exact search in [`crate::neighbors`] compares every pair of cells, so its cost
//! is quadratic and, on the GPU of a laptop, overtakes an approximate index near
//! 400 000 cells. This module trades a few per cent of recall for a cost that is
//! close to linear in the number of cells: at a million cells and 50 dimensions it
//! runs in well under a minute on eleven cores.
//!
//! Every random choice is drawn from a counter-based generator keyed by the seed, the
//! cell and the iteration, so the result does not depend on how rayon schedules the
//! work: the same seed gives the same graph on the same machine.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::neighbors::KnnGraph;

/// Tuning parameters of the approximate search.
#[derive(Debug, Clone, Copy)]
pub struct NnDescentParams {
    /// Random-projection trees used to seed the graph. `None` picks
    /// `5 + round(n^0.25)`, capped at 32, as pynndescent does.
    pub n_trees: Option<usize>,
    /// Largest leaf of a tree. `None` picks `max(k + 1, 30)`.
    pub leaf_size: Option<usize>,
    /// Neighbours sampled per cell and per direction in the local join.
    pub max_candidates: usize,
    /// Upper bound on NN-descent iterations.
    pub n_iters: usize,
    /// Stop when one iteration changes fewer than `delta * n * k` entries.
    pub delta: f32,
    /// Seed of the counter-based generator.
    pub seed: u64,
}

impl Default for NnDescentParams {
    fn default() -> Self {
        Self {
            n_trees: None,
            leaf_size: None,
            max_candidates: 50,
            n_iters: 10,
            delta: 0.001,
            seed: 0,
        }
    }
}

/// One entry of a cell's neighbour list: squared distance, neighbour id, and whether
/// the entry has not yet been used as a source of candidates.
#[derive(Debug, Clone, Copy)]
struct Entry {
    square: f32,
    id: u32,
    fresh: bool,
}

/// A cell's neighbour list, sorted nearest first, never longer than `k`.
struct Heap {
    entries: Vec<Entry>,
}

impl Heap {
    fn with_capacity(k: usize) -> Self {
        Self {
            entries: Vec::with_capacity(k + 1),
        }
    }

    /// Worst retained squared distance, or infinity while the list is short.
    fn threshold(&self, k: usize) -> f32 {
        if self.entries.len() < k {
            f32::INFINITY
        } else {
            self.entries[k - 1].square
        }
    }

    /// Insert `id` at `square` if it beats the worst entry and is not present already.
    /// Returns whether the list changed.
    fn push(&mut self, square: f32, id: u32, k: usize, fresh: bool) -> bool {
        if square >= self.threshold(k) {
            return false;
        }
        if self.entries.iter().any(|e| e.id == id) {
            return false;
        }
        let at = self
            .entries
            .partition_point(|e| (e.square, e.id) < (square, id));
        self.entries.insert(at, Entry { square, id, fresh });
        if self.entries.len() > k {
            self.entries.pop();
        }
        true
    }
}

/// splitmix64 keyed on (seed, stream, counter): a stateless generator whose output
/// depends only on its key, so parallel callers never share state.
#[inline]
fn mix(seed: u64, stream: u64, counter: u64) -> u64 {
    let mut z = seed
        .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(counter.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A small sequential generator for one cell or one tree, keyed as above.
struct Rng {
    seed: u64,
    stream: u64,
    counter: u64,
}

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
        Self {
            seed,
            stream,
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.counter += 1;
        mix(self.seed, self.stream, self.counter)
    }

    /// Uniform in `0..bound`.
    fn below(&mut self, bound: usize) -> usize {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * bound as f64) as usize
            % bound.max(1)
    }
}

#[inline]
fn square_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Leaves of one random-projection tree: `members` is a permutation of the cells and
/// `leaves` holds `(start, end)` ranges into it.
struct Tree {
    members: Vec<u32>,
    leaves: Vec<(u32, u32)>,
    /// Leaf index of every cell.
    leaf_of: Vec<u32>,
}

/// Build one tree by recursive random hyperplane splits until every leaf is at most
/// `leaf_size` cells.
fn build_tree(points: &Array2<f32>, leaf_size: usize, seed: u64, tree: u64) -> Tree {
    let n = points.nrows();
    let dims = points.ncols();
    let data = points.as_slice().expect("centred points are contiguous");
    let mut members: Vec<u32> = (0..n as u32).collect();
    let mut leaves = Vec::new();
    let mut rng = Rng::new(seed, 0x7A3E_0000 + tree);
    let mut stack = vec![(0usize, n)];
    let mut normal = vec![0.0f32; dims];
    while let Some((start, end)) = stack.pop() {
        let len = end - start;
        if len <= leaf_size {
            leaves.push((start as u32, end as u32));
            continue;
        }
        let a = members[start + rng.below(len)] as usize;
        let mut b = members[start + rng.below(len)] as usize;
        let mut tries = 0;
        while b == a && tries < 8 {
            b = members[start + rng.below(len)] as usize;
            tries += 1;
        }
        let pa = &data[a * dims..(a + 1) * dims];
        let pb = &data[b * dims..(b + 1) * dims];
        let mut split = 0.0f32;
        for d in 0..dims {
            normal[d] = pa[d] - pb[d];
            split += normal[d] * (pa[d] + pb[d]) * 0.5;
        }
        // Partition in place: cells on the positive side first.
        let slice = &mut members[start..end];
        let mut left = 0usize;
        for i in 0..len {
            let id = slice[i] as usize;
            let p = &data[id * dims..(id + 1) * dims];
            let side: f32 = p.iter().zip(&normal).map(|(x, w)| x * w).sum::<f32>() - split;
            let positive = if side == 0.0 {
                rng.next_u64() & 1 == 0
            } else {
                side > 0.0
            };
            if positive {
                slice.swap(i, left);
                left += 1;
            }
        }
        // A degenerate hyperplane leaves everything on one side; halve at random.
        if left == 0 || left == len {
            left = len / 2;
        }
        stack.push((start, start + left));
        stack.push((start + left, end));
    }
    let mut leaf_of = vec![0u32; n];
    for (leaf, &(s, e)) in leaves.iter().enumerate() {
        for &id in &members[s as usize..e as usize] {
            leaf_of[id as usize] = leaf as u32;
        }
    }
    Tree {
        members,
        leaves,
        leaf_of,
    }
}

/// Approximate k nearest neighbours by Euclidean distance.
///
/// The embedding is centred as in the exact search. The output has the shape of
/// [`crate::neighbors::knn`]: `k` neighbours per cell, nearest first, the cell itself
/// excluded.
pub fn knn_approximate(
    embedding: &Array2<f32>,
    k: usize,
    params: &NnDescentParams,
) -> Result<KnnGraph> {
    let (n_cells, n_dims) = embedding.dim();
    if n_cells == 0 || n_dims == 0 {
        return Err(Error::shape(
            "a non-empty (cells, dimensions) embedding",
            format!("{n_cells} x {n_dims}"),
        ));
    }
    if k == 0 {
        return Err(Error::parameter("k", "at least 1", k));
    }
    if k >= n_cells {
        return Err(Error::parameter(
            "k",
            "smaller than the number of cells, since a cell is not its own neighbour",
            k,
        ));
    }
    let points = Array2::from_shape_vec((n_cells, n_dims), crate::neighbors::centred(embedding))
        .map_err(|_| Error::shape(format!("{n_cells} x {n_dims}"), "a mismatched buffer"))?;
    let data = points.as_slice().expect("contiguous");
    let row = |id: usize| &data[id * n_dims..(id + 1) * n_dims];

    let n_trees = params
        .n_trees
        .unwrap_or_else(|| (5 + (n_cells as f64).powf(0.25).round() as usize).min(32));
    let leaf_size = params.leaf_size.unwrap_or((k + 1).max(30));
    let seed = params.seed;

    // ---- forest initialisation: every cell's candidates are its leaf mates in each tree
    let trees: Vec<Tree> = (0..n_trees as u64)
        .into_par_iter()
        .map(|t| build_tree(&points, leaf_size, seed, t))
        .collect();
    let heaps: Vec<Mutex<Heap>> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let mut heap = Heap::with_capacity(k);
            let pi = row(i);
            for tree in &trees {
                let (s, e) = tree.leaves[tree.leaf_of[i] as usize];
                for &j in &tree.members[s as usize..e as usize] {
                    if j as usize != i {
                        heap.push(square_distance(pi, row(j as usize)), j, k, true);
                    }
                }
            }
            // A cell whose leaves held fewer than k distinct mates is topped up at random.
            let mut rng = Rng::new(seed, 0x1A17_0000 + i as u64);
            let mut tries = 0;
            while heap.entries.len() < k && tries < 16 * k {
                let j = rng.below(n_cells);
                if j != i {
                    heap.push(square_distance(pi, row(j)), j as u32, k, true);
                }
                tries += 1;
            }
            Mutex::new(heap)
        })
        .collect();
    drop(trees);

    // ---- NN-descent iterations
    let thresholds: Vec<AtomicU32> = (0..n_cells)
        .map(|i| AtomicU32::new(heaps[i].lock().unwrap().threshold(k).to_bits()))
        .collect();
    let max_candidates = params.max_candidates.max(1);
    let stop_at = (params.delta * n_cells as f32 * k as f32) as usize;
    for iteration in 0..params.n_iters {
        // Forward candidates: a sample of each cell's fresh and stale neighbours. The
        // sampled fresh ones become stale, as in pynndescent.
        let forward: Vec<(Vec<u32>, Vec<u32>)> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let mut heap = heaps[i].lock().unwrap();
                let mut rng = Rng::new(seed, 0x2B00_0000 + i as u64 + ((iteration as u64) << 40));
                let mut fresh = Vec::new();
                let mut stale = Vec::new();
                let mut seen_fresh = 0usize;
                let mut seen_stale = 0usize;
                for entry in heap.entries.iter_mut() {
                    if entry.fresh {
                        seen_fresh += 1;
                        if fresh.len() < max_candidates {
                            fresh.push(entry.id);
                            entry.fresh = false;
                        } else {
                            let slot = rng.below(seen_fresh);
                            if slot < max_candidates {
                                fresh[slot] = entry.id;
                                entry.fresh = false;
                            }
                        }
                    } else {
                        seen_stale += 1;
                        if stale.len() < max_candidates {
                            stale.push(entry.id);
                        } else {
                            let slot = rng.below(seen_stale);
                            if slot < max_candidates {
                                stale[slot] = entry.id;
                            }
                        }
                    }
                }
                (fresh, stale)
            })
            .collect();
        // Reverse candidates by a counting sort over the forward lists.
        let mut reverse_fresh: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
        let mut reverse_stale: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
        for (i, (fresh, stale)) in forward.iter().enumerate() {
            for &j in fresh {
                reverse_fresh[j as usize].push(i as u32);
            }
            for &j in stale {
                reverse_stale[j as usize].push(i as u32);
            }
        }
        let candidates: Vec<(Vec<u32>, Vec<u32>)> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let mut rng = Rng::new(seed, 0x3C00_0000 + i as u64 + ((iteration as u64) << 40));
                let merge = |own: &Vec<u32>, rev: &Vec<u32>, rng: &mut Rng| -> Vec<u32> {
                    let mut all: Vec<u32> = own.iter().chain(rev.iter()).copied().collect();
                    all.sort_unstable();
                    all.dedup();
                    // Fisher-Yates down to the cap, so the sample is uniform.
                    if all.len() > max_candidates {
                        for p in 0..max_candidates {
                            let q = p + rng.below(all.len() - p);
                            all.swap(p, q);
                        }
                        all.truncate(max_candidates);
                    }
                    all
                };
                let fresh = merge(&forward[i].0, &reverse_fresh[i], &mut rng);
                let stale = merge(&forward[i].1, &reverse_stale[i], &mut rng);
                (fresh, stale)
            })
            .collect();
        drop(forward);
        drop(reverse_fresh);
        drop(reverse_stale);

        // Local join: every pair that shares a cell is a candidate pair.
        let updates = AtomicUsize::new(0);
        let try_insert = |p: u32, q: u32, square: f32| {
            let threshold = f32::from_bits(thresholds[p as usize].load(Ordering::Relaxed));
            if square >= threshold {
                return;
            }
            let mut heap = heaps[p as usize].lock().unwrap();
            if heap.push(square, q, k, true) {
                thresholds[p as usize].store(heap.threshold(k).to_bits(), Ordering::Relaxed);
                updates.fetch_add(1, Ordering::Relaxed);
            }
        };
        candidates.par_iter().for_each(|(fresh, stale)| {
            for (a, &p) in fresh.iter().enumerate() {
                let pp = row(p as usize);
                for &q in &fresh[a + 1..] {
                    let square = square_distance(pp, row(q as usize));
                    try_insert(p, q, square);
                    try_insert(q, p, square);
                }
                for &q in stale {
                    if q == p {
                        continue;
                    }
                    let square = square_distance(pp, row(q as usize));
                    try_insert(p, q, square);
                    try_insert(q, p, square);
                }
            }
        });
        if updates.load(Ordering::Relaxed) <= stop_at {
            break;
        }
    }

    // ---- output, nearest first
    let mut indices = Array2::<u32>::zeros((n_cells, k));
    let mut distances = Array2::<f32>::zeros((n_cells, k));
    let rows: Vec<Vec<Entry>> = heaps
        .into_iter()
        .map(|m| m.into_inner().unwrap().entries)
        .collect();
    for (i, entries) in rows.iter().enumerate() {
        if entries.len() < k {
            return Err(Error::parameter(
                "k",
                "fewer neighbours than requested could be found; the input has too few distinct cells",
                k,
            ));
        }
        for (j, entry) in entries.iter().enumerate() {
            indices[(i, j)] = entry.id;
            distances[(i, j)] = entry.square.max(0.0).sqrt();
        }
    }
    Ok(KnnGraph { indices, distances })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::knn;
    use candle_core::Device;

    fn clustered(n: usize, dims: usize, seed: u64) -> Array2<f32> {
        // Ten Gaussian blobs, the shape a PCA embedding of cells has.
        let mut rng = Rng::new(seed, 1);
        let centres: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dims).map(|_| rng.below(1000) as f32 / 50.0).collect())
            .collect();
        Array2::from_shape_fn((n, dims), |(i, d)| {
            let c = &centres[i % 10];
            let u = (mix(seed, i as u64, d as u64) >> 11) as f32 / (1u64 << 53) as f32;
            c[d] + (u - 0.5) * 2.0
        })
    }

    fn recall(approx: &KnnGraph, exact: &KnnGraph) -> f64 {
        let (n, k) = exact.indices.dim();
        let mut hits = 0usize;
        for i in 0..n {
            let truth: std::collections::HashSet<u32> =
                exact.indices.row(i).iter().copied().collect();
            hits += approx
                .indices
                .row(i)
                .iter()
                .filter(|j| truth.contains(j))
                .count();
        }
        hits as f64 / (n * k) as f64
    }

    #[test]
    fn recall_against_the_exact_search_is_high() {
        let points = clustered(4000, 20, 7);
        let exact = knn(&points, 15, &Device::Cpu).unwrap();
        let approx = knn_approximate(&points, 15, &NnDescentParams::default()).unwrap();
        assert_eq!(approx.indices.dim(), (4000, 15));
        let r = recall(&approx, &exact);
        assert!(r >= 0.95, "recall {r:.3}");
        // Rows are sorted nearest first, exclude the cell, and carry no duplicate.
        for i in 0..4000 {
            let row = approx.indices.row(i);
            let d = approx.distances.row(i);
            assert!(row.iter().all(|&j| j as usize != i));
            assert!(d.windows(2).into_iter().all(|w| w[0] <= w[1]));
            let mut ids: Vec<u32> = row.to_vec();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), 15);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_graph() {
        let points = clustered(2000, 10, 3);
        let a = knn_approximate(&points, 10, &NnDescentParams::default()).unwrap();
        let b = knn_approximate(&points, 10, &NnDescentParams::default()).unwrap();
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn rejects_impossible_k() {
        let points = clustered(20, 3, 1);
        assert!(knn_approximate(&points, 20, &NnDescentParams::default()).is_err());
        assert!(knn_approximate(&points, 0, &NnDescentParams::default()).is_err());
    }
}
