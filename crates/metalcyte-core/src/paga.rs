//! Partition-based graph abstraction. Owned by feat/paga.
//!
//! PAGA coarse-grains a single-cell neighbour graph onto the groups of a
//! partition. The whole computation is a scatter-add of the graph's stored
//! entries into a `(n_groups, n_groups)` accumulator, so peak memory is
//! quadratic in the number of *groups* — a few dozen — and never in the number
//! of cells. There is no `Device` argument for that reason: one streaming pass
//! over `nnz` edges, memory bound, writing into a matrix small enough to sit in
//! cache. Handing it to the GPU would cost a buffer round trip and an atomic
//! scatter to accelerate arithmetic that is already free.

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// The abstracted graph over groups.
#[derive(Debug, Clone)]
pub struct AbstractedGraph {
    /// `(n_groups, n_groups)` connectivity, symmetric with a zero diagonal.
    pub connectivities: Vec<f32>,
    /// `(n_groups, n_groups)` maximum spanning tree, each edge stored once in
    /// the upper triangle — the layout `scipy.sparse.csgraph` returns and the
    /// one scanpy stores as `connectivities_tree`.
    pub connectivities_tree: Vec<f32>,
    pub n_groups: usize,
}

/// Both returned matrices are dense `n_groups` by `n_groups`; refuse to allocate
/// past a gigabyte rather than fail on an accidental `n_groups`.
const MAX_GROUPS: usize = 10_000;

/// PAGA connectivities, as `scanpy.tl.paga` with `model="v1.2"`.
///
/// `graph` is the square, binarised neighbour graph (scanpy binarises
/// `obsp["distances"]`, which is directed: an entry `(cell, neighbour)` is one
/// edge). `groups[cell]` is a group id below `n_groups`.
pub fn paga(graph: &CsrMatrix, groups: &[u32], n_groups: usize) -> Result<AbstractedGraph> {
    let sizes = validate(graph, groups, n_groups)?;
    let (edges_per_group, between) = count_edges(graph, groups, n_groups);
    let connectivities = confidence(&sizes, &edges_per_group, &between, n_groups);
    let connectivities_tree = maximum_spanning_tree(&connectivities, n_groups);
    Ok(AbstractedGraph {
        connectivities,
        connectivities_tree,
        n_groups,
    })
}

/// Check the inputs and return the cell count of each group.
fn validate(graph: &CsrMatrix, groups: &[u32], n_groups: usize) -> Result<Vec<u64>> {
    // Emptiness first: a graph with no cells also has no groups, and "an empty
    // graph" is the useful half of that to report.
    if graph.n_rows() == 0 {
        return Err(Error::shape(
            "a graph with at least one cell",
            "an empty graph",
        ));
    }
    if n_groups == 0 || n_groups > MAX_GROUPS {
        return Err(Error::parameter(
            "n_groups",
            "between 1 and 10000, the abstracted graph being dense",
            n_groups,
        ));
    }
    if graph.n_cols() != graph.n_rows() {
        return Err(Error::shape(
            format!("a square {0} by {0} neighbour graph", graph.n_rows()),
            format!("{} by {}", graph.n_rows(), graph.n_cols()),
        ));
    }
    if groups.len() != graph.n_rows() {
        return Err(Error::shape(
            format!("{} group labels", graph.n_rows()),
            format!("{}", groups.len()),
        ));
    }

    let mut sizes = vec![0u64; n_groups];
    for &group in groups {
        let group = group as usize;
        if group >= n_groups {
            return Err(Error::parameter("groups", "a label below n_groups", group));
        }
        sizes[group] += 1;
    }
    // An empty group has no expected edge count, so its row of the connectivity
    // matrix would be meaningless rather than merely zero.
    if let Some(empty) = sizes.iter().position(|&size| size == 0) {
        return Err(Error::parameter(
            "groups",
            "non-empty for every group",
            format!("group {empty} has no cells"),
        ));
    }
    Ok(sizes)
}

/// One pass over the stored entries, grouped by label.
///
/// Returns the number of edges leaving each group — counting the ones that stay
/// inside it — and the symmetric count of edges between each pair of groups.
///
/// scanpy adds the within-group edge count to the group's outgoing count, which
/// together is just every stored entry in a row belonging to that group.
fn count_edges(graph: &CsrMatrix, groups: &[u32], n_groups: usize) -> (Vec<u64>, Vec<u64>) {
    let mut edges_per_group = vec![0u64; n_groups];
    let mut between = vec![0u64; n_groups * n_groups];
    let indptr = graph.indptr();
    let indices = graph.indices();
    // No `values`: an edge is a stored entry, and its weight never enters the count.
    for cell in 0..graph.n_rows() {
        let group = groups[cell] as usize;
        for entry in indptr[cell] as usize..indptr[cell + 1] as usize {
            // Every *stored* entry is an edge, whatever its value. scanpy binarises
            // before it builds the graph -- `ones.data = np.ones(len(ones.data))`,
            // `_paga.py:182-183` -- so the `nonzero()` inside
            // `get_igraph_from_adjacency` then sees nothing but ones and drops none
            // of them. Skipping zero-valued entries here (which this used to do,
            // citing that `nonzero()`) loses real edges.
            //
            // It is not a rare shape. This runs on `obsp["distances"]`, where two
            // identical cells are a stored zero: on 120 cells of which 60 are exact
            // duplicates, `sc.pp.neighbors(n_neighbors=10)` stores 540 zeros out of
            // 1080 entries -- half the graph -- and dropping them moved a real
            // connectivity by 0.096.
            edges_per_group[group] += 1;
            let neighbor = groups[indices[entry] as usize] as usize;
            if neighbor != group {
                // Both directions of a pair land on the same unordered slot,
                // which is scanpy's `inter_es + inter_es.T`.
                between[group.min(neighbor) * n_groups + group.max(neighbor)] += 1;
            }
        }
    }
    (edges_per_group, between)
}

/// Observed edges between two groups over the count expected under a null model
/// that rewires the graph at random, capped at 1.
fn confidence(
    sizes: &[u64],
    edges_per_group: &[u64],
    between: &[u64],
    n_groups: usize,
) -> Vec<f32> {
    let n_cells: u64 = sizes.iter().sum();
    let mut connectivities = vec![0.0f32; n_groups * n_groups];
    for i in 0..n_groups {
        for j in i + 1..n_groups {
            let observed = between[i * n_groups + j];
            if observed == 0 {
                continue;
            }
            // A group-`i` edge lands on a given group-`j` cell with probability
            // `n_j / (n - 1)`, so the two groups expect
            // `(e_i * n_j + e_j * n_i) / (n - 1)` edges between them.
            let expected = (edges_per_group[i] as f64 * sizes[j] as f64
                + edges_per_group[j] as f64 * sizes[i] as f64)
                / (n_cells - 1) as f64;
            // `expected == 0` needs both groups to be isolated, and then
            // `observed` is zero too; the guard above has already skipped it.
            let value = (observed as f64 / expected).min(1.0) as f32;
            connectivities[i * n_groups + j] = value;
            connectivities[j * n_groups + i] = value;
        }
    }
    connectivities
}

/// Maximum spanning forest of the connectivity matrix, by Prim over the dense
/// matrix.
///
/// scanpy takes the *minimum* spanning tree of the reciprocal connectivities,
/// which is the same edge set: `1/x` is decreasing on the positive weights, and
/// a missing connection stays missing under it. `O(n_groups^2)` is the right
/// shape of algorithm for a graph of a few dozen dense nodes.
fn maximum_spanning_tree(connectivities: &[f32], n_groups: usize) -> Vec<f32> {
    let mut tree = vec![0.0f32; n_groups * n_groups];
    let mut in_tree = vec![false; n_groups];
    let mut best = vec![0.0f32; n_groups];
    let mut parent = vec![usize::MAX; n_groups];

    for _ in 0..n_groups {
        // The strongest edge from the tree to a node outside it; when nothing
        // reaches out, the graph is disconnected and a new component starts.
        let strongest = (0..n_groups)
            .filter(|&node| !in_tree[node] && best[node] > 0.0)
            .max_by(|&a, &b| best[a].total_cmp(&best[b]));
        let node = match strongest.or_else(|| (0..n_groups).find(|&node| !in_tree[node])) {
            Some(node) => node,
            None => break,
        };

        in_tree[node] = true;
        if parent[node] != usize::MAX {
            let (row, column) = (parent[node].min(node), parent[node].max(node));
            tree[row * n_groups + column] = connectivities[row * n_groups + column];
        }
        for other in 0..n_groups {
            let weight = connectivities[node * n_groups + other];
            if !in_tree[other] && weight > best[other] {
                best[other] = weight;
                parent[other] = node;
            }
        }
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three groups in a chain: 0-1-2, with no edge between the ends.
    fn chain() -> (CsrMatrix, Vec<u32>) {
        let groups = vec![0u32, 0, 1, 1, 2, 2];
        // Every cell is linked to its group mate; cell 1-2 and 3-4 bridge the
        // neighbouring groups. Undirected, so both directions are stored.
        let pairs = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)];
        let mut dense = vec![0.0f32; 36];
        for (a, b) in pairs {
            dense[a * 6 + b] = 1.0;
            dense[b * 6 + a] = 1.0;
        }
        (CsrMatrix::from_dense(&dense, 6, 6).unwrap(), groups)
    }

    #[test]
    fn chain_connects_only_neighbouring_groups() {
        let (graph, groups) = chain();
        let abstracted = paga(&graph, &groups, 3).unwrap();
        let c = &abstracted.connectivities;
        assert!(c[1] > 0.0 && c[3] > 0.0, "groups 0 and 1 are connected");
        assert!(c[5] > 0.0 && c[7] > 0.0, "groups 1 and 2 are connected");
        assert_eq!(c[2], 0.0, "the ends of the chain are not connected");
        assert_eq!(c[6], 0.0);
        assert_eq!(c[0] + c[4] + c[8], 0.0, "the diagonal stays empty");
    }

    #[test]
    fn chain_tree_is_the_chain() {
        let (graph, groups) = chain();
        let tree = paga(&graph, &groups, 3).unwrap().connectivities_tree;
        assert!(tree[1] > 0.0 && tree[5] > 0.0, "0-1 and 1-2 are kept");
        assert_eq!(tree[2], 0.0);
        // Each edge is stored once, in the upper triangle.
        assert_eq!(tree[3], 0.0);
        assert_eq!(tree[7], 0.0);
    }

    #[test]
    fn expected_edges_use_the_group_sizes() {
        // Four cells in group 0, two in group 1, joined by a single pair. The
        // groups then hold 7 and 3 directed edges, so the null model expects
        // (7*2 + 3*4)/5 = 5.2 edges between them against the 2 observed.
        let groups = vec![0u32, 0, 0, 0, 1, 1];
        let pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (3, 4)];
        let mut dense = vec![0.0f32; 36];
        for (a, b) in pairs {
            dense[a * 6 + b] = 1.0;
            dense[b * 6 + a] = 1.0;
        }
        let graph = CsrMatrix::from_dense(&dense, 6, 6).unwrap();
        let abstracted = paga(&graph, &groups, 2).unwrap();
        assert!((abstracted.connectivities[1] - 2.0 / 5.2).abs() < 1e-6);
    }

    #[test]
    fn a_disconnected_group_gets_its_own_component() {
        // The chain plus a seventh cell in a fourth group, with no edges at all:
        // the spanning forest still terminates, and keeps only the chain's edges.
        let groups = vec![0u32, 0, 1, 1, 2, 2, 3];
        let pairs = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)];
        let mut dense = vec![0.0f32; 49];
        for (a, b) in pairs {
            dense[a * 7 + b] = 1.0;
            dense[b * 7 + a] = 1.0;
        }
        let graph = CsrMatrix::from_dense(&dense, 7, 7).unwrap();
        let tree = paga(&graph, &groups, 4).unwrap().connectivities_tree;
        assert_eq!(tree.iter().filter(|&&value| value > 0.0).count(), 2);
    }

    #[test]
    fn rejects_bad_input() {
        let (graph, groups) = chain();
        assert!(paga(&graph, &groups, 2).is_err(), "label out of range");
        assert!(paga(&graph, &groups, 4).is_err(), "group with no cells");
        assert!(paga(&graph, &groups, 0).is_err(), "no groups");
        assert!(
            paga(&graph, &groups[..3], 3).is_err(),
            "labels do not match"
        );
        let empty = CsrMatrix::new(vec![0], vec![], vec![], 0).unwrap();
        assert!(paga(&empty, &[], 1).is_err(), "empty graph");
    }
}
