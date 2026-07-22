use candle_core::{Device, Tensor};
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Nearest neighbours of every cell, excluding the cell itself.
#[derive(Debug, Clone)]
pub struct KnnGraph {
    /// Neighbour ids, `(n_cells, k)`, nearest first.
    pub indices: Array2<u32>,
    /// Distances to those neighbours, `(n_cells, k)`.
    pub distances: Array2<f32>,
}

/// Upper bound on the elements of one distance tile.
///
/// A tile is `tile_rows x n_cells` f32, so 64M elements is 256 MB: large enough
/// that even a million cells needs only a few thousand tiles, small enough to
/// sit in unified memory beside the embedding without crowding out the matmul's
/// own workspace.
const MAX_TILE_ELEMENTS: usize = 64 * 1024 * 1024;

/// Exact k nearest neighbours by Euclidean distance.
///
/// Exact rather than approximate: on the GPU the distance matrix is one tiled
/// matmul, so the usual reason to approximate does not apply at this scale.
pub fn knn(embedding: &Array2<f32>, k: usize, device: &Device) -> Result<KnnGraph> {
    let tile_rows = (MAX_TILE_ELEMENTS / embedding.nrows().max(1)).max(1);
    knn_tiled(embedding, k, device, tile_rows)
}

/// `knn` with an explicit tile height, so tests can force many small tiles.
fn knn_tiled(
    embedding: &Array2<f32>,
    k: usize,
    device: &Device,
    tile_rows: usize,
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

    let flat: Vec<f32> = embedding.iter().copied().collect();
    let points = Tensor::from_vec(flat, (n_cells, n_dims), device)?;
    // |a - b|^2 = |a|^2 + |b|^2 - 2 a.b keeps the whole thing in one matmul.
    let square_norms = points.sqr()?.sum_keepdim(1)?;
    let square_norms_row = square_norms.reshape((1, n_cells))?;
    let points_t = points.t()?.contiguous()?;

    let mut indices = Array2::<u32>::zeros((n_cells, k));
    let mut distances = Array2::<f32>::zeros((n_cells, k));
    let mut start = 0;
    while start < n_cells {
        let height = tile_rows.min(n_cells - start);
        let tile = points.narrow(0, start, height)?;
        let tile_norms = square_norms.narrow(0, start, height)?;
        let square_distances = tile
            .matmul(&points_t)?
            .affine(-2.0, 0.0)?
            .broadcast_add(&tile_norms)?
            .broadcast_add(&square_norms_row)?
            .to_vec2::<f32>()?;

        let selected: Vec<(Vec<u32>, Vec<f32>)> = square_distances
            .par_iter()
            .enumerate()
            .map(|(offset, row)| select_nearest(row, start + offset, k))
            .collect();
        for (offset, (neighbours, neighbour_distances)) in selected.into_iter().enumerate() {
            indices
                .row_mut(start + offset)
                .assign(&ArrayView1::from(&neighbours));
            distances
                .row_mut(start + offset)
                .assign(&ArrayView1::from(&neighbour_distances));
        }
        start += height;
    }

    Ok(KnnGraph { indices, distances })
}

/// The `k` smallest entries of one row of squared distances, nearest first,
/// excluding the cell itself.
///
/// Isolated from the tiling above so the fused Metal selection on
/// `feat/knn-kernel` can replace exactly this step. Ties break towards the
/// smaller cell id, so the output does not depend on the order a partial sort
/// happens to leave equal keys in.
fn select_nearest(square_distances: &[f32], cell: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut candidates: Vec<(f32, u32)> = square_distances
        .iter()
        .enumerate()
        .filter(|(column, _)| *column != cell)
        // |a|^2 + |b|^2 - 2 a.b cancels to a small negative for near-identical
        // points; clamping keeps the square root real.
        .map(|(column, &square)| (square.max(0.0), column as u32))
        .collect();

    let by_distance_then_id =
        |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    candidates.select_nth_unstable_by(k - 1, by_distance_then_id);
    candidates.truncate(k);
    candidates.sort_unstable_by(by_distance_then_id);

    candidates
        .iter()
        .map(|&(square, column)| (column, square.sqrt()))
        .unzip()
}

/// umap-learn's `SMOOTH_K_TOLERANCE`: the binary search stops once the fuzzy set
/// cardinality is this close to `log2(k)`.
const SMOOTH_K_TOLERANCE: f32 = 1e-5;
/// umap-learn's `MIN_K_DIST_SCALE`: `sigma` is floored at this fraction of a
/// mean distance so that a tight cluster cannot produce a degenerate kernel.
const MIN_K_DIST_SCALE: f32 = 1e-3;
/// umap-learn's `n_iter` for the bandwidth binary search.
const SMOOTH_K_ITERATIONS: usize = 64;

/// UMAP's fuzzy simplicial set, the weighted graph `scanpy.pp.neighbors` stores
/// in `obsp["connectivities"]`.
pub fn connectivities(graph: &KnnGraph) -> Result<CsrMatrix> {
    let (n_cells, k) = graph.distances.dim();
    if graph.indices.dim() != (n_cells, k) {
        return Err(Error::shape(
            format!("indices of shape {n_cells} x {k}"),
            format!("{:?}", graph.indices.dim()),
        ));
    }
    if n_cells == 0 || k == 0 {
        return Err(Error::shape(
            "a neighbour graph with at least one cell and one neighbour",
            format!("{n_cells} x {k}"),
        ));
    }
    if graph.indices.iter().any(|&id| id as usize >= n_cells) {
        return Err(Error::shape(
            format!("neighbour ids below {n_cells}"),
            "an out-of-range neighbour id".to_string(),
        ));
    }

    let (sigmas, rhos) = smooth_knn_distances(&graph.distances);
    symmetrise(directed_weights(graph, &sigmas, &rhos), n_cells)
}

/// Per cell, the bandwidth `sigma` solving `sum_j exp(-(d_ij - rho_i)/sigma_i)
/// == log2(k)`, and the local connectivity offset `rho`.
///
/// umap-learn passes rows that start with the cell itself at distance zero; a
/// `KnnGraph` excludes that entry, so the implied row width is `k + 1` and the
/// self entry contributes only to the means and to the target cardinality.
fn smooth_knn_distances(distances: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let (n_cells, k) = distances.dim();
    let width = k + 1;
    let target = (width as f32).log2();
    let mean_distance = distances.sum() / (n_cells * width) as f32;

    let mut sigmas = Vec::with_capacity(n_cells);
    let mut rhos = Vec::with_capacity(n_cells);
    for row in distances.rows() {
        // local_connectivity = 1: rho is the nearest neighbour that is not a
        // duplicate of the cell, so every cell has at least one weight of 1.
        let rho = row.iter().copied().find(|&d| d > 0.0).unwrap_or(0.0);

        let mut low = 0.0f32;
        let mut high = f32::MAX;
        let mut mid = 1.0f32;
        for _ in 0..SMOOTH_K_ITERATIONS {
            let cardinality: f32 = row
                .iter()
                .map(|&d| {
                    let offset = d - rho;
                    if offset > 0.0 {
                        (-offset / mid).exp()
                    } else {
                        1.0
                    }
                })
                .sum();
            if (cardinality - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if cardinality > target {
                high = mid;
                mid = (low + high) / 2.0;
            } else {
                low = mid;
                mid = if high >= f32::MAX {
                    mid * 2.0
                } else {
                    (low + high) / 2.0
                };
            }
        }

        let reference = if rho > 0.0 {
            row.sum() / width as f32
        } else {
            mean_distance
        };
        sigmas.push(mid.max(MIN_K_DIST_SCALE * reference));
        rhos.push(rho);
    }
    (sigmas, rhos)
}

/// The directed membership strengths, one row per cell, sorted by column.
fn directed_weights(graph: &KnnGraph, sigmas: &[f32], rhos: &[f32]) -> Vec<Vec<(u32, f32)>> {
    let (n_cells, k) = graph.indices.dim();
    (0..n_cells)
        .map(|cell| {
            let mut row = Vec::with_capacity(k);
            for j in 0..k {
                let neighbour = graph.indices[[cell, j]];
                // umap gives a cell zero affinity to itself and then drops
                // zeros, which is what keeps the diagonal empty.
                if neighbour as usize == cell {
                    continue;
                }
                let offset = graph.distances[[cell, j]] - rhos[cell];
                let weight = if offset <= 0.0 || sigmas[cell] == 0.0 {
                    1.0
                } else {
                    (-offset / sigmas[cell]).exp()
                };
                if weight != 0.0 {
                    row.push((neighbour, weight));
                }
            }
            row.sort_unstable_by_key(|&(neighbour, _)| neighbour);
            row
        })
        .collect()
}

/// Fuzzy union of the directed graph with its transpose: `a + a^T - a * a^T`.
fn symmetrise(directed: Vec<Vec<(u32, f32)>>, n_cells: usize) -> Result<CsrMatrix> {
    let mut transposed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];
    for (cell, row) in directed.iter().enumerate() {
        for &(neighbour, weight) in row {
            transposed[neighbour as usize].push((cell as u32, weight));
        }
    }
    for row in transposed.iter_mut() {
        row.sort_unstable_by_key(|&(cell, _)| cell);
    }

    let mut indptr = Vec::with_capacity(n_cells + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0);
    for (forward, backward) in directed.iter().zip(transposed.iter()) {
        let (mut f, mut b) = (0, 0);
        while f < forward.len() || b < backward.len() {
            let forward_column = forward.get(f).map(|&(column, _)| column);
            let backward_column = backward.get(b).map(|&(column, _)| column);
            let column = match (forward_column, backward_column) {
                (Some(x), Some(y)) => x.min(y),
                (Some(x), None) => x,
                (None, Some(y)) => y,
                (None, None) => unreachable!("the loop condition keeps one side live"),
            };
            let mut weight = 0.0;
            let mut reverse = 0.0;
            if forward_column == Some(column) {
                weight = forward[f].1;
                f += 1;
            }
            if backward_column == Some(column) {
                reverse = backward[b].1;
                b += 1;
            }
            indices.push(column);
            values.push(weight + reverse - weight * reverse);
        }
        indptr.push(values.len() as u32);
    }
    CsrMatrix::new(indptr, indices, values, n_cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64, so the same embedding can be regenerated in Python to
    /// recompute the scanpy constants below.
    fn reference_embedding(n_cells: usize, n_dims: usize, seed: u64) -> Array2<f32> {
        let mut state = seed;
        let mut values = Vec::with_capacity(n_cells * n_dims);
        for _ in 0..n_cells * n_dims {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            values.push((z >> 40) as f32 / (1u32 << 24) as f32);
        }
        Array2::from_shape_vec((n_cells, n_dims), values).unwrap()
    }

    fn line(n_cells: usize) -> Array2<f32> {
        Array2::from_shape_fn((n_cells, 1), |(cell, _)| cell as f32)
    }

    fn grid(side: usize) -> Array2<f32> {
        Array2::from_shape_fn((side * side, 2), |(cell, axis)| {
            if axis == 0 {
                (cell / side) as f32
            } else {
                (cell % side) as f32
            }
        })
    }

    #[test]
    fn neighbours_on_a_line_are_the_adjacent_points() {
        let graph = knn(&line(6), 2, &Device::Cpu).unwrap();
        assert_eq!(graph.indices.row(0).to_vec(), vec![1, 2]);
        assert_eq!(graph.distances.row(0).to_vec(), vec![1.0, 2.0]);
        // 1 and 3 are both at distance 1 from 2; the smaller id wins the tie.
        assert_eq!(graph.indices.row(2).to_vec(), vec![1, 3]);
        assert_eq!(graph.indices.row(5).to_vec(), vec![4, 3]);
    }

    #[test]
    fn neighbours_on_a_grid_follow_the_lattice() {
        let graph = knn(&grid(3), 4, &Device::Cpu).unwrap();
        // Centre: the four orthogonal cells, all at distance 1, so id order.
        assert_eq!(graph.indices.row(4).to_vec(), vec![1, 3, 5, 7]);
        // Corner: two at 1, the diagonal at sqrt(2), then 2 beats 6 on id.
        assert_eq!(graph.indices.row(0).to_vec(), vec![1, 3, 4, 2]);
        let corner = graph.distances.row(0).to_vec();
        assert!((corner[2] - 2.0f32.sqrt()).abs() < 1e-6);
        assert!((corner[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tiling_does_not_change_the_result() {
        let embedding = reference_embedding(200, 8, 7);
        let single = knn_tiled(&embedding, 10, &Device::Cpu, 200).unwrap();
        let many = knn_tiled(&embedding, 10, &Device::Cpu, 7).unwrap();
        assert_eq!(single.indices, many.indices);
        assert_eq!(single.distances, many.distances);
    }

    #[test]
    fn rejects_degenerate_arguments() {
        assert!(knn(&line(6), 0, &Device::Cpu).is_err());
        assert!(knn(&line(6), 6, &Device::Cpu).is_err());
        assert!(knn(&line(6), 7, &Device::Cpu).is_err());
        assert!(knn(&Array2::<f32>::zeros((0, 4)), 3, &Device::Cpu).is_err());
        assert!(knn(&Array2::<f32>::zeros((5, 0)), 3, &Device::Cpu).is_err());
    }

    #[test]
    fn connectivities_rejects_a_degenerate_graph() {
        let empty = KnnGraph {
            indices: Array2::zeros((0, 3)),
            distances: Array2::zeros((0, 3)),
        };
        assert!(connectivities(&empty).is_err());

        let out_of_range = KnnGraph {
            indices: Array2::from_shape_vec((2, 1), vec![1, 9]).unwrap(),
            distances: Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
        };
        assert!(connectivities(&out_of_range).is_err());
    }

    fn dense(matrix: &CsrMatrix) -> Array2<f32> {
        let n_rows = matrix.n_rows();
        Array2::from_shape_vec((n_rows, matrix.n_cols()), matrix.densify_rows(0, n_rows)).unwrap()
    }

    #[test]
    fn connectivities_are_symmetric_with_an_empty_diagonal() {
        let graph = knn(&reference_embedding(120, 5, 3), 10, &Device::Cpu).unwrap();
        let matrix = connectivities(&graph).unwrap();
        let weights = dense(&matrix);

        for cell in 0..weights.nrows() {
            assert_eq!(weights[[cell, cell]], 0.0);
            for other in 0..weights.ncols() {
                assert!((weights[[cell, other]] - weights[[other, cell]]).abs() < 1e-6);
            }
        }
        for &weight in matrix.values() {
            assert!(
                weight > 0.0 && weight <= 1.0,
                "weight {weight} outside (0, 1]"
            );
        }
        for cell in 0..matrix.n_rows() {
            let from = matrix.indptr()[cell] as usize;
            let to = matrix.indptr()[cell + 1] as usize;
            assert!(matrix.indices()[from..to].windows(2).all(|w| w[0] < w[1]));
        }
    }

    // scanpy.pp.neighbors(n_neighbors=15) on `reference_embedding(300, 20, 42)`.
    // scanpy stores 14 non-self neighbours per cell, so the comparison is
    // against `knn(.., 14)`. Full rows for ten spread-out cells plus exact
    // aggregates over the whole matrix, which pin all 300 rows without putting
    // 4200 literals in the source.
    const SCANPY_SAMPLE_CELLS: [usize; 10] = [0, 1, 7, 42, 99, 137, 150, 211, 255, 299];
    const SCANPY_NEIGHBORS: [[u32; 14]; 10] = [
        [
            119, 99, 297, 284, 270, 287, 34, 173, 109, 67, 100, 40, 134, 9,
        ],
        [
            198, 226, 173, 42, 110, 241, 94, 250, 98, 58, 274, 284, 156, 68,
        ],
        [
            151, 189, 51, 228, 64, 244, 203, 273, 53, 67, 98, 272, 289, 117,
        ],
        [
            28, 198, 232, 173, 274, 24, 1, 110, 33, 126, 229, 14, 287, 11,
        ],
        [
            284, 0, 266, 130, 249, 220, 181, 109, 34, 58, 118, 208, 169, 98,
        ],
        [
            110, 52, 222, 147, 83, 192, 97, 272, 249, 156, 118, 109, 70, 225,
        ],
        [
            56, 263, 207, 226, 149, 23, 186, 84, 124, 183, 209, 290, 204, 144,
        ],
        [176, 298, 44, 104, 64, 45, 189, 269, 51, 6, 226, 30, 18, 205],
        [
            296, 105, 287, 80, 39, 269, 291, 78, 217, 73, 101, 180, 127, 122,
        ],
        [
            154, 120, 263, 289, 3, 218, 117, 246, 228, 82, 101, 159, 53, 160,
        ],
    ];
    #[rustfmt::skip]
    const SCANPY_CONNECTIVITY_ROWS: [&[(u32, f32)]; 10] = [
        &[(9, 6.0651936e-02), (34, 2.7650505e-01), (35, 1.503_178e-1), (40, 7.689_591e-2), (67, 1.0919636e-01), (99, 9.8126864e-01), (100, 8.302_305e-2), (109, 2.003_29e-1), (119, 1.0), (134, 2.220_671e-1), (173, 1.1610797e-01), (199, 1.699_634e-1), (270, 3.868_359e-1), (284, 5.186_652e-1), (287, 1.4172494e-01), (297, 1.0)],
        &[(42, 4.8017377e-01), (58, 1.514_928e-1), (68, 1.2612756e-01), (94, 2.316_605e-1), (98, 1.6800594e-01), (110, 2.9406378e-01), (156, 2.452_443e-1), (173, 3.3317405e-01), (198, 1.0), (226, 3.483_126e-1), (241, 4.9488765e-01), (250, 3.4408212e-01), (274, 1.4485574e-01), (284, 1.4298384e-01)],
        &[(12, 2.5005543e-01), (41, 2.4051532e-01), (51, 1.0), (53, 2.870_779e-1), (64, 4.3026882e-01), (67, 2.6128945e-01), (82, 9.165_373e-2), (85, 1.689_396e-1), (98, 1.0684434e-01), (117, 1.0203358e-01), (151, 1.0), (153, 9.005_759e-2), (169, 1.4084813e-01), (170, 8.202_439e-2), (189, 8.526_736e-1), (203, 8.029_593e-1), (214, 3.040_719e-1), (228, 5.5598927e-01), (236, 1.2959956e-01), (244, 8.4076804e-01), (257, 1.6502915e-01), (272, 1.0673708e-01), (273, 3.9501154e-01), (289, 1.0214462e-01)],
        &[(1, 4.8017377e-01), (11, 3.620_879e-1), (14, 2.5035965e-01), (24, 2.358_025e-1), (28, 1.0), (33, 3.312_486e-1), (38, 1.0150236e-01), (91, 1.0990662e-01), (110, 2.0037375e-01), (126, 1.6709046e-01), (173, 2.919_618e-1), (198, 8.497_894e-1), (213, 5.0851576e-02), (229, 2.219_679e-1), (232, 6.2977695e-01), (274, 3.6633128e-01), (287, 1.568_282e-1)],
        &[(0, 9.8126864e-01), (34, 1.3121346e-01), (58, 2.9475835e-01), (98, 8.7079905e-02), (104, 8.393_145e-2), (109, 2.2142047e-01), (111, 1.5512483e-01), (118, 1.1709028e-01), (130, 2.7555096e-01), (169, 2.0864783e-01), (181, 2.796_788e-1), (208, 2.8623974e-01), (220, 3.4435192e-01), (244, 3.483_044e-2), (249, 2.086_834e-1), (266, 1.0), (284, 1.0)],
        &[(52, 5.2801704e-01), (70, 1.6394348e-01), (83, 2.1547337e-01), (97, 2.0943844e-01), (109, 1.7134176e-01), (110, 9.9999994e-01), (118, 1.7420667e-01), (147, 3.1099957e-01), (156, 1.7462766e-01), (157, 1.9715303e-01), (192, 2.1144849e-01), (222, 5.327_993e-1), (223, 9.219_883e-2), (225, 1.6369617e-01), (249, 1.898_673e-1), (272, 1.989_204e-1)],
        &[(23, 4.307_694e-1), (26, 6.129_617e-2), (56, 1.0), (84, 4.1045457e-01), (124, 3.7398565e-01), (144, 9.998_135e-2), (149, 7.8936344e-01), (183, 1.5666927e-01), (186, 4.4473392e-01), (204, 2.0071255e-01), (207, 1.0), (209, 3.9630747e-01), (226, 4.2224774e-01), (263, 5.961_615e-1), (290, 5.451_274e-1)],
        &[(6, 1.8733603e-01), (18, 8.831_03e-2), (30, 9.4160534e-02), (44, 5.666_34e-1), (45, 3.1965548e-01), (51, 1.3588169e-01), (64, 2.4524283e-01), (104, 3.3380222e-01), (176, 1.0), (189, 1.5607502e-01), (205, 8.6127155e-02), (226, 1.0655477e-01), (269, 1.3756935e-01), (298, 9.036_035e-1)],
        &[(39, 3.7870023e-01), (71, 1.0324326e-01), (73, 1.3537657e-01), (78, 4.2596117e-01), (80, 7.0739526e-01), (101, 1.3476756e-01), (105, 7.8216493e-01), (122, 1.1666572e-01), (127, 1.5828831e-01), (180, 1.3227737e-01), (199, 1.4529586e-01), (217, 1.3984427e-01), (264, 1.123_739e-1), (269, 2.1950261e-01), (287, 4.2271453e-01), (291, 2.1035857e-01), (296, 9.9999994e-01)],
        &[(3, 4.2273083e-01), (46, 7.192_994e-2), (53, 2.5759345e-01), (82, 3.9567703e-01), (101, 1.7714095e-01), (117, 2.1023947e-01), (120, 1.0), (154, 1.0), (159, 3.389_756e-1), (160, 4.9865252e-01), (176, 1.4622718e-01), (218, 1.0), (228, 3.118_579e-1), (246, 3.3656424e-01), (263, 4.6396673e-01), (289, 4.057_48e-1)],
    ];
    const SCANPY_NNZ: usize = 5798;
    const SCANPY_WEIGHT_SUM: f32 = 1.915_624e3;
    const SCANPY_WEIGHT_SQUARE_SUM: f32 = 1.082_613e3;

    #[test]
    fn matches_scanpy_neighbour_sets() {
        let graph = knn(&reference_embedding(300, 20, 42), 14, &Device::Cpu).unwrap();
        for (sample, &cell) in SCANPY_SAMPLE_CELLS.iter().enumerate() {
            let ours: Vec<u32> = graph.indices.row(cell).to_vec();
            let shared = SCANPY_NEIGHBORS[sample]
                .iter()
                .filter(|id| ours.contains(id))
                .count();
            assert!(
                shared * 10 >= ours.len() * 9,
                "cell {cell}: only {shared} of {} neighbours shared",
                ours.len()
            );
        }
    }

    #[test]
    fn matches_scanpy_connectivities() {
        let graph = knn(&reference_embedding(300, 20, 42), 14, &Device::Cpu).unwrap();
        let matrix = connectivities(&graph).unwrap();

        for (sample, &cell) in SCANPY_SAMPLE_CELLS.iter().enumerate() {
            let from = matrix.indptr()[cell] as usize;
            let to = matrix.indptr()[cell + 1] as usize;
            let ours = &matrix.indices()[from..to];
            for &(column, expected) in SCANPY_CONNECTIVITY_ROWS[sample] {
                // Only edges both graphs have are comparable.
                let Some(offset) = ours.iter().position(|&id| id == column) else {
                    continue;
                };
                let actual = matrix.values()[from + offset];
                assert!(
                    (actual - expected).abs() <= 1e-3 * expected.abs(),
                    "cell {cell} -> {column}: expected {expected}, got {actual}"
                );
            }
        }

        assert_eq!(matrix.nnz(), SCANPY_NNZ);
        let sum: f32 = matrix.values().iter().sum();
        let square_sum: f32 = matrix.values().iter().map(|w| w * w).sum();
        assert!((sum - SCANPY_WEIGHT_SUM).abs() <= 1e-3 * SCANPY_WEIGHT_SUM);
        assert!((square_sum - SCANPY_WEIGHT_SQUARE_SUM).abs() <= 1e-3 * SCANPY_WEIGHT_SQUARE_SUM);
    }
}
