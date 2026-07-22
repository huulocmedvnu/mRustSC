//! Spatial autocorrelation over the neighbour graph. Owned by feat/metrics.
//!
//! Moran's I and Geary's C are the same computation with two different
//! numerators. Both need the graph's total weight, and per gene the mean-centred
//! expression `z`, its sum of squares, and the cross term `z' G z`. Geary's C
//! needs one extra reduction against the weight incident on each cell, because
//!
//! ```text
//! sum_ij w_ij (x_i - x_j)^2 = sum_i (rowsum_i + colsum_i) z_i^2 - 2 z' G z
//! ```
//!
//! so the pairwise difference never has to be formed. Writing them as one
//! routine parameterised by `Statistic` is what keeps the shared reduction from
//! drifting between the two.
//!
//! The work is a sparse-times-dense product — the graph is the sparse operand,
//! a block of genes the dense one — expressed in candle as gather, scale,
//! scatter-add, so the same source runs on the CPU and on the Apple GPU. The
//! gathered operand is `nnz(graph) x block`, which is what bounds the block
//! width; the expression matrix itself is never densified beyond that block.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Moran's I for every column of `values`, as `scanpy.metrics.morans_i`.
pub fn morans_i(graph: &CsrMatrix, values: &CsrMatrix, device: &Device) -> Result<Vec<f32>> {
    autocorrelation(graph, values, device, Statistic::MoransI)
}

/// Geary's C for every column of `values`, as `scanpy.metrics.gearys_c`.
pub fn gearys_c(graph: &CsrMatrix, values: &CsrMatrix, device: &Device) -> Result<Vec<f32>> {
    autocorrelation(graph, values, device, Statistic::GearysC)
}

/// Which of the two numerators to form from the shared reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Statistic {
    MoransI,
    GearysC,
}

/// Elements in the largest dense intermediate a single block may materialise.
///
/// The gathered neighbour values are `nnz(graph) x block`, so the block width is
/// set from the graph's edge count and not from the gene count. 16M f32 is
/// 64 MB, small enough to sit in unified memory beside the graph and large
/// enough that a typical `k = 15` neighbour graph still takes hundreds of genes
/// per pass.
const MAX_BLOCK_ELEMENTS: usize = 16 * 1024 * 1024;

fn autocorrelation(
    graph: &CsrMatrix,
    values: &CsrMatrix,
    device: &Device,
    statistic: Statistic,
) -> Result<Vec<f32>> {
    let n_cells = graph.n_rows();
    if graph.n_cols() != n_cells {
        return Err(Error::shape(
            format!("a square graph, {n_cells} x {n_cells}"),
            format!("{n_cells} x {}", graph.n_cols()),
        ));
    }
    if values.n_rows() != n_cells {
        return Err(Error::shape(
            format!("{n_cells} rows of values, one per cell"),
            format!("{} rows", values.n_rows()),
        ));
    }
    if n_cells == 0 {
        return Err(Error::shape("a graph with at least one cell", "no cells"));
    }
    let n_genes = values.n_cols();
    if n_genes == 0 {
        return Ok(Vec::new());
    }

    let (total_weight, incident_weight) = graph_weights(graph)?;
    let edges = graph.nnz();
    // Accumulating the graph reductions in f64 would need an f64 device path,
    // which Metal does not have; the block reductions stay f32 and only the
    // final per-gene combination is done in f64.
    let edge_rows = Tensor::from_vec(edge_rows(graph.indptr()), edges, device)?;
    let edge_columns = Tensor::from_vec(graph.indices().to_vec(), edges, device)?;
    let edge_weights = Tensor::from_vec(graph.values().to_vec(), (edges, 1), device)?;
    let incident = Tensor::from_vec(incident_weight, (n_cells, 1), device)?;

    let columns = ColumnView::of(values);
    let width = (MAX_BLOCK_ELEMENTS / edges.max(n_cells)).clamp(1, n_genes);
    let mut result = Vec::with_capacity(n_genes);
    for start in (0..n_genes).step_by(width) {
        let end = (start + width).min(n_genes);
        let block = end - start;
        let centred = Tensor::from_vec(
            columns.centred_block(n_cells, start, end),
            (n_cells, block),
            device,
        )?;

        let squares = centred.sqr()?;
        let sum_squares = squares.sum(0)?.to_vec1::<f32>()?;
        // Gather-scale-scatter is `G z` for a whole block of genes at once: one
        // pass over the stored edges, no branching, nothing to densify beyond
        // the block itself.
        let neighbour_values = centred.index_select(&edge_columns, 0)?;
        let weighted = neighbour_values.broadcast_mul(&edge_weights)?;
        let neighbour_sums = Tensor::zeros((n_cells, block), DType::F32, device)?
            .index_add(&edge_rows, &weighted, 0)?;
        let cross = (&centred * &neighbour_sums)?.sum(0)?.to_vec1::<f32>()?;

        let incident_squares = match statistic {
            Statistic::MoransI => vec![0.0; block],
            Statistic::GearysC => squares.broadcast_mul(&incident)?.sum(0)?.to_vec1::<f32>()?,
        };
        for gene in 0..block {
            result.push(combine(
                statistic,
                n_cells,
                total_weight,
                sum_squares[gene] as f64,
                cross[gene] as f64,
                incident_squares[gene] as f64,
            ));
        }
    }
    Ok(result)
}

/// One gene's statistic from the reductions the two share.
///
/// A constant gene has no variance to explain, so both statistics are 0/0;
/// scanpy returns NaN for it and warns, and the value is what callers compare.
fn combine(
    statistic: Statistic,
    n_cells: usize,
    total_weight: f64,
    sum_squares: f64,
    cross: f64,
    incident_squares: f64,
) -> f32 {
    if sum_squares <= 0.0 {
        return f32::NAN;
    }
    let n = n_cells as f64;
    let value = match statistic {
        Statistic::MoransI => n / total_weight * cross / sum_squares,
        // scanpy's normalisation, not the textbook one: (n - 1) on top and 2W
        // below, so a perfectly smooth signal reaches 0 and white noise 1.
        Statistic::GearysC => {
            (n - 1.0) * (incident_squares - 2.0 * cross) / (2.0 * total_weight * sum_squares)
        }
    };
    value as f32
}

/// The graph's total weight and, per cell, the weight incident on it in either
/// direction — `rowsum + colsum`, which is `2 * rowsum` for a symmetric graph.
///
/// Summed in f64: this is one reduction over every stored edge, and it divides
/// every statistic, so it is the one place a large graph's f32 drift would show
/// up in every gene at once.
fn graph_weights(graph: &CsrMatrix) -> Result<(f64, Vec<f32>)> {
    let n_cells = graph.n_rows();
    let mut incident = vec![0.0f64; n_cells];
    let mut total = 0.0f64;
    for row in 0..n_cells {
        let from = graph.indptr()[row] as usize;
        let to = graph.indptr()[row + 1] as usize;
        for entry in from..to {
            let weight = graph.values()[entry] as f64;
            total += weight;
            incident[row] += weight;
            incident[graph.indices()[entry] as usize] += weight;
        }
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(Error::parameter(
            "graph",
            "a graph with positive total edge weight",
            total,
        ));
    }
    Ok((total, incident.into_iter().map(|w| w as f32).collect()))
}

/// The row each stored edge belongs to, one entry per stored edge.
fn edge_rows(indptr: &[u32]) -> Vec<u32> {
    let mut rows = Vec::with_capacity(*indptr.last().unwrap_or(&0) as usize);
    for row in 0..indptr.len() - 1 {
        let span = (indptr[row + 1] - indptr[row]) as usize;
        rows.extend(std::iter::repeat_n(row as u32, span));
    }
    rows
}

/// `values` transposed into columns, so a whole gene can be read at once.
///
/// Both statistics reduce over all cells of one gene, which is a column of a
/// cells-by-genes CSR matrix and therefore strided across every row. Paying
/// O(nnz) once to store the same entries by column keeps the matrix sparse;
/// densifying it to get column access would cost `n_cells * n_genes * 4` bytes,
/// 345 MB for PBMC 3k's full gene set and far more for a real atlas.
struct ColumnView {
    starts: Vec<usize>,
    rows: Vec<u32>,
    data: Vec<f32>,
}

impl ColumnView {
    fn of(values: &CsrMatrix) -> Self {
        let n_genes = values.n_cols();
        let mut starts = vec![0usize; n_genes + 1];
        for &gene in values.indices() {
            starts[gene as usize + 1] += 1;
        }
        for gene in 0..n_genes {
            starts[gene + 1] += starts[gene];
        }

        let mut cursor = starts.clone();
        let mut rows = vec![0u32; values.nnz()];
        let mut data = vec![0.0f32; values.nnz()];
        for cell in 0..values.n_rows() {
            let from = values.indptr()[cell] as usize;
            let to = values.indptr()[cell + 1] as usize;
            for entry in from..to {
                let gene = values.indices()[entry] as usize;
                rows[cursor[gene]] = cell as u32;
                data[cursor[gene]] = values.values()[entry];
                cursor[gene] += 1;
            }
        }
        Self { starts, rows, data }
    }

    /// Genes `start..end` densified and mean-centred, row-major `(n_cells, block)`.
    ///
    /// Centring here rather than on the device means the zeros the sparse form
    /// omits are written exactly once, as `-mean`.
    fn centred_block(&self, n_cells: usize, start: usize, end: usize) -> Vec<f32> {
        let block = end - start;
        let means: Vec<f32> = (start..end)
            .map(|gene| {
                let sum: f64 = self.data[self.starts[gene]..self.starts[gene + 1]]
                    .iter()
                    .map(|&value| value as f64)
                    .sum();
                (sum / n_cells as f64) as f32
            })
            .collect();

        let mut dense = Vec::with_capacity(n_cells * block);
        for _ in 0..n_cells {
            dense.extend(means.iter().map(|mean| -mean));
        }
        for (offset, gene) in (start..end).enumerate() {
            for entry in self.starts[gene]..self.starts[gene + 1] {
                dense[self.rows[entry] as usize * block + offset] += self.data[entry];
            }
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// scanpy's formulas transcribed literally in f64: the oracle for both the
    /// tensor path and the f32 error measurement.
    fn reference(graph: &CsrMatrix, values: &CsrMatrix, statistic: Statistic) -> Vec<f64> {
        let n = graph.n_rows();
        let dense = values.densify_rows(0, n);
        let total: f64 = graph.values().iter().map(|&w| w as f64).sum();
        (0..values.n_cols())
            .map(|gene| {
                let x: Vec<f64> = (0..n)
                    .map(|cell| dense[cell * values.n_cols() + gene] as f64)
                    .collect();
                let mean = x.iter().sum::<f64>() / n as f64;
                let z: Vec<f64> = x.iter().map(|value| value - mean).collect();
                let sum_squares: f64 = z.iter().map(|value| value * value).sum();
                let mut cross = 0.0;
                let mut differences = 0.0;
                for row in 0..n {
                    for entry in graph.indptr()[row] as usize..graph.indptr()[row + 1] as usize {
                        let column = graph.indices()[entry] as usize;
                        let weight = graph.values()[entry] as f64;
                        cross += weight * z[row] * z[column];
                        differences += weight * (z[row] - z[column]).powi(2);
                    }
                }
                match statistic {
                    Statistic::MoransI => n as f64 / total * cross / sum_squares,
                    Statistic::GearysC => {
                        (n as f64 - 1.0) * differences / (2.0 * total * sum_squares)
                    }
                }
            })
            .collect()
    }

    /// A ring of `n` cells, each joined to its two neighbours with weight 1.
    fn ring(n: usize) -> CsrMatrix {
        let mut dense = vec![0.0f32; n * n];
        for cell in 0..n {
            dense[cell * n + (cell + 1) % n] = 1.0;
            dense[cell * n + (cell + n - 1) % n] = 1.0;
        }
        CsrMatrix::from_dense(&dense, n, n).unwrap()
    }

    /// splitmix64 in [0, 1), so a test graph is reproducible without a rng dep.
    fn pseudo_random(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u32 << 24) as f32
            })
            .collect()
    }

    /// Smooth, alternating and constant signals over a ring of `n` cells.
    fn ring_signals(n: usize) -> CsrMatrix {
        let mut dense = vec![0.0f32; n * 3];
        for cell in 0..n {
            let angle = std::f32::consts::TAU * cell as f32 / n as f32;
            dense[cell * 3] = angle.cos();
            dense[cell * 3 + 1] = if cell % 2 == 0 { 1.0 } else { -1.0 };
            dense[cell * 3 + 2] = 2.5;
        }
        CsrMatrix::from_dense(&dense, n, 3).unwrap()
    }

    #[test]
    fn a_smooth_signal_is_correlated_and_an_alternating_one_is_not() {
        let n = 64;
        let (graph, signals) = (ring(n), ring_signals(n));
        let i = morans_i(&graph, &signals, &Device::Cpu).unwrap();
        let c = gearys_c(&graph, &signals, &Device::Cpu).unwrap();

        // cos over a 64-cell ring is nearly constant between neighbours.
        assert!(i[0] > 0.99, "smooth Moran's I {}", i[0]);
        assert!(c[0] < 0.01, "smooth Geary's C {}", c[0]);
        // +-1 alternating is the exact opposite: every edge crosses the sign, so
        // Moran's I is exactly -1 and Geary's C its own maximum, 2(n-1)/n.
        assert!((i[1] + 1.0).abs() < 1e-5, "alternating Moran's I {}", i[1]);
        let maximum = 2.0 * (n as f32 - 1.0) / n as f32;
        assert!(
            (c[1] - maximum).abs() < 1e-5,
            "alternating Geary's C {}",
            c[1]
        );
    }

    #[test]
    fn a_constant_gene_has_no_statistic() {
        let (graph, signals) = (ring(16), ring_signals(16));
        assert!(morans_i(&graph, &signals, &Device::Cpu).unwrap()[2].is_nan());
        assert!(gearys_c(&graph, &signals, &Device::Cpu).unwrap()[2].is_nan());
    }

    /// An asymmetric graph, to pin the `rowsum + colsum` identity Geary's C uses
    /// instead of forming the pairwise differences.
    #[test]
    fn matches_the_reference_on_an_asymmetric_graph() {
        let graph =
            CsrMatrix::from_dense(&[0.0, 2.0, 0.0, 0.5, 0.0, 1.5, 0.0, 0.0, 3.0], 3, 3).unwrap();
        let values = CsrMatrix::from_dense(&[1.0, 4.0, -2.0, 0.0, 5.0, 6.0], 3, 2).unwrap();
        for (statistic, ours) in [
            (
                Statistic::MoransI,
                morans_i(&graph, &values, &Device::Cpu).unwrap(),
            ),
            (
                Statistic::GearysC,
                gearys_c(&graph, &values, &Device::Cpu).unwrap(),
            ),
        ] {
            for (gene, expected) in reference(&graph, &values, statistic).iter().enumerate() {
                assert!(
                    (ours[gene] as f64 - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "{statistic:?} gene {gene}: {} vs {expected}",
                    ours[gene]
                );
            }
        }
    }

    /// A graph big enough that f32 accumulation has somewhere to drift, held
    /// against the f64 oracle. The relative error is printed so a regression
    /// shows up as a number and not just as a pass.
    #[test]
    fn f32_accumulation_stays_within_the_contract_tolerance() {
        let (n_cells, n_genes) = (400, 40);
        let graph = {
            let mut dense = vec![0.0f32; n_cells * n_cells];
            let weights = pseudo_random(n_cells * 20, 11);
            for cell in 0..n_cells {
                for step in 1..=20 {
                    let other = (cell + step * 7) % n_cells;
                    if other != cell {
                        let weight = weights[cell * 20 + step - 1];
                        dense[cell * n_cells + other] = weight;
                        dense[other * n_cells + cell] = weight;
                    }
                }
            }
            CsrMatrix::from_dense(&dense, n_cells, n_cells).unwrap()
        };
        // Sparse and skewed like real expression: most entries zero, a few large.
        let values = {
            let mut dense = pseudo_random(n_cells * n_genes, 5);
            for (index, value) in dense.iter_mut().enumerate() {
                *value = if *value < 0.85 {
                    0.0
                } else {
                    *value * (1 + index % 7) as f32
                };
            }
            CsrMatrix::from_dense(&dense, n_cells, n_genes).unwrap()
        };

        for (name, statistic, ours) in [
            (
                "morans_i",
                Statistic::MoransI,
                morans_i(&graph, &values, &Device::Cpu).unwrap(),
            ),
            (
                "gearys_c",
                Statistic::GearysC,
                gearys_c(&graph, &values, &Device::Cpu).unwrap(),
            ),
        ] {
            let expected = reference(&graph, &values, statistic);
            let worst = expected
                .iter()
                .zip(&ours)
                .filter(|(reference, _)| reference.is_finite())
                .map(|(reference, &ours)| (ours as f64 - reference).abs() / reference.abs())
                .fold(0.0f64, f64::max);
            println!("{name}: worst f32-vs-f64 relative error {worst:.3e}");
            assert!(worst < 1e-4, "{name} drifts by {worst:e}");
        }
    }

    #[test]
    fn the_gpu_agrees_with_the_cpu() {
        let Ok(gpu) = Device::new_metal(0) else {
            return;
        };
        let (graph, signals) = (ring(128), ring_signals(128));
        for (cpu, metal) in [
            (
                morans_i(&graph, &signals, &Device::Cpu).unwrap(),
                morans_i(&graph, &signals, &gpu).unwrap(),
            ),
            (
                gearys_c(&graph, &signals, &Device::Cpu).unwrap(),
                gearys_c(&graph, &signals, &gpu).unwrap(),
            ),
        ] {
            for (gene, (cpu, metal)) in cpu.iter().zip(&metal).enumerate() {
                assert!(
                    (cpu - metal).abs() <= 1e-5 || (cpu.is_nan() && metal.is_nan()),
                    "gene {gene}: cpu {cpu}, metal {metal}"
                );
            }
        }
    }

    #[test]
    fn blocking_does_not_change_the_result() {
        // 4000 genes over a 40-cell ring forces several passes at any sane block
        // width, and the answers must not depend on where the block boundary fell.
        let n_cells = 40;
        let graph = ring(n_cells);
        let wide = CsrMatrix::from_dense(&pseudo_random(n_cells * 4000, 3), n_cells, 4000).unwrap();
        let ours = morans_i(&graph, &wide, &Device::Cpu).unwrap();
        let narrow = CsrMatrix::from_dense(
            &wide.densify_rows(0, n_cells)[..n_cells * 4000],
            n_cells,
            4000,
        )
        .unwrap();
        assert_eq!(ours, morans_i(&graph, &narrow, &Device::Cpu).unwrap());
        assert_eq!(ours.len(), 4000);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let graph = ring(8);
        // Values with the wrong number of cells.
        let wrong = CsrMatrix::from_dense(&pseudo_random(9, 1), 9, 1).unwrap();
        assert!(morans_i(&graph, &wrong, &Device::Cpu).is_err());
        assert!(gearys_c(&graph, &wrong, &Device::Cpu).is_err());
        // A graph that is not square.
        let oblong = CsrMatrix::from_dense(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], 2, 3).unwrap();
        let values = CsrMatrix::from_dense(&[1.0, 2.0], 2, 1).unwrap();
        assert!(morans_i(&oblong, &values, &Device::Cpu).is_err());
        // A graph with no edges at all: every statistic would divide by zero.
        let empty = CsrMatrix::new(vec![0, 0, 0], vec![], vec![], 2).unwrap();
        assert!(morans_i(&empty, &values, &Device::Cpu).is_err());
        assert!(gearys_c(&empty, &values, &Device::Cpu).is_err());
        // No cells at all.
        let nothing = CsrMatrix::new(vec![0], vec![], vec![], 0).unwrap();
        assert!(morans_i(&nothing, &nothing, &Device::Cpu).is_err());
    }
}
