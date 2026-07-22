use candle_core::Device;

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Which dispersion definition to use, matching scanpy's `flavor` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvgFlavor {
    /// Binned normalised dispersion on log data, scanpy's default.
    Seurat,
    /// Loess-free variance-stabilising ranking on raw counts.
    CellRanger,
}

/// Per-gene statistics behind the highly-variable flag.
#[derive(Debug, Clone)]
pub struct HighlyVariableGenes {
    pub means: Vec<f32>,
    pub dispersions: Vec<f32>,
    pub normalised_dispersions: Vec<f32>,
    pub highly_variable: Vec<bool>,
}

/// scanpy's `n_bins` default; both flavours cut the mean into this many bins.
const N_BINS: usize = 20;

/// What scanpy substitutes for a zero mean before dividing, so a gene expressed
/// in no cell yields a finite dispersion instead of a division by zero.
const ZERO_MEAN_SUBSTITUTE: f64 = 1e-12;

/// `scipy.stats.norm.ppf(0.75)`. statsmodels' `mad` divides by it so the median
/// absolute deviation estimates the standard deviation of a normal sample.
const MAD_SCALE: f64 = 0.674_489_750_196_081_7;

/// Flag the most variable genes, as `scanpy.pp.highly_variable_genes`.
///
/// `Seurat` expects log-transformed data and `CellRanger` raw counts, as in
/// scanpy. Both bin the genes by mean expression and rank them by how far their
/// dispersion sits from what their bin considers typical.
pub fn highly_variable_genes(
    matrix: &CsrMatrix,
    n_top_genes: usize,
    flavor: HvgFlavor,
    _device: &Device,
) -> Result<HighlyVariableGenes> {
    if matrix.n_rows() == 0 {
        return Err(Error::shape("at least one cell", "0 cells"));
    }
    if matrix.n_cols() == 0 {
        return Err(Error::shape("at least one gene", "0 genes"));
    }
    if n_top_genes == 0 {
        return Err(Error::parameter("n_top_genes", "at least 1", n_top_genes));
    }

    let (means, dispersions) = gene_statistics(matrix, flavor);
    let edges = match flavor {
        HvgFlavor::Seurat => equal_width_edges(&means, N_BINS),
        HvgFlavor::CellRanger => percentile_edges(&means)?,
    };
    let bins = assign_bins(&means, &edges);
    let normalised = normalise_within_bins(&dispersions, &bins, flavor);
    let highly_variable = select_top(&normalised, n_top_genes);

    Ok(HighlyVariableGenes {
        means: means.iter().map(|&value| value as f32).collect(),
        dispersions: dispersions.iter().map(|&value| value as f32).collect(),
        normalised_dispersions: normalised.iter().map(|&value| value as f32).collect(),
        highly_variable,
    })
}

/// Per-gene mean and dispersion, in the form the flavour's binning expects.
///
/// One pass over the stored values gives the sum and the sum of squares per
/// gene; the implicit zeros enter only through the cell count. Nothing is
/// densified and no tensor is built, so `device` stays unused.
fn gene_statistics(matrix: &CsrMatrix, flavor: HvgFlavor) -> (Vec<f64>, Vec<f64>) {
    let n_genes = matrix.n_cols();
    let mut sums = vec![0.0f64; n_genes];
    let mut squared_sums = vec![0.0f64; n_genes];

    for (&gene, &value) in matrix.indices().iter().zip(matrix.values()) {
        // Seurat dispersions are defined on the counts behind the log data,
        // which scanpy recovers with an f32 `expm1` before accumulating in f64.
        let value = match flavor {
            HvgFlavor::Seurat => value.exp_m1() as f64,
            HvgFlavor::CellRanger => value as f64,
        };
        let gene = gene as usize;
        sums[gene] += value;
        squared_sums[gene] += value * value;
    }

    let n_cells = matrix.n_rows() as f64;
    let mut means = Vec::with_capacity(n_genes);
    let mut dispersions = Vec::with_capacity(n_genes);
    for gene in 0..n_genes {
        let mean = sums[gene] / n_cells;
        let mut variance = squared_sums[gene] / n_cells - mean * mean;
        if matrix.n_rows() > 1 {
            variance *= n_cells / (n_cells - 1.0); // scanpy's correction=1
        }
        let mean = if mean == 0.0 {
            ZERO_MEAN_SUBSTITUTE
        } else {
            mean
        };
        let dispersion = variance / mean;

        match flavor {
            // Seurat ranks on the log of both quantities. A zero dispersion has
            // no log, and scanpy carries it as NaN rather than -inf.
            HvgFlavor::Seurat => {
                means.push(mean.ln_1p());
                dispersions.push(if dispersion == 0.0 {
                    f64::NAN
                } else {
                    dispersion.ln()
                });
            }
            HvgFlavor::CellRanger => {
                means.push(mean);
                dispersions.push(dispersion);
            }
        }
    }
    (means, dispersions)
}

/// `n_bins` equal-width bin edges over the observed range, as `pandas.cut` with
/// an integer `bins`: the lower edge is nudged out by 0.1% of the range so the
/// smallest value falls inside the first bin rather than on its open boundary.
fn equal_width_edges(values: &[f64], n_bins: usize) -> Vec<f64> {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let degenerate = minimum == maximum;

    let (low, high) = if degenerate {
        let widen = |value: f64| {
            if value != 0.0 {
                0.001 * value.abs()
            } else {
                0.001
            }
        };
        (minimum - widen(minimum), maximum + widen(maximum))
    } else {
        (minimum, maximum)
    };

    let step = (high - low) / n_bins as f64;
    let mut edges: Vec<f64> = (0..=n_bins).map(|i| low + step * i as f64).collect();
    edges[n_bins] = high; // np.linspace pins the last point exactly
    if !degenerate {
        edges[0] -= (maximum - minimum) * 0.001;
    }
    edges
}

/// Bin edges at the 10th to 100th percentile in steps of 5, flanked by
/// infinities, as scanpy's `cell_ranger` flavour builds them.
fn percentile_edges(values: &[f64]) -> Result<Vec<f64>> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let mut edges = vec![f64::NEG_INFINITY];
    edges.extend((10..105).step_by(5).map(|q| percentile(&sorted, q as f64)));
    edges.push(f64::INFINITY);

    // `pandas.cut` refuses repeated edges, which happens once more than 5% of
    // the genes share one mean. Report it rather than silently binning
    // differently from scanpy.
    if edges.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::parameter(
            "flavor",
            "cell_ranger with distinct mean percentiles; too many genes share one mean",
            "cell_ranger",
        ));
    }
    Ok(edges)
}

/// `numpy.percentile` with its default linear interpolation, on sorted input.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q / 100.0;
    let below = position.floor();
    let lower = sorted[below as usize];
    lower + (position - below) * (sorted[position.ceil() as usize] - lower)
}

/// Bin index per gene, `None` when the mean falls outside every bin.
///
/// `pandas.cut` builds right-closed intervals, so a value's bin is
/// `searchsorted(edges, value, side="left") - 1`.
fn assign_bins(values: &[f64], edges: &[f64]) -> Vec<Option<usize>> {
    values
        .iter()
        .map(|&value| {
            let upper = edges.partition_point(|&edge| edge < value);
            if upper == 0 || upper == edges.len() {
                None
            } else {
                Some(upper - 1)
            }
        })
        .collect()
}

/// Centre and scale each gene's dispersion by the statistics of its own bin.
///
/// The two flavours share this shape and differ only in the pair of statistics,
/// so the bin bookkeeping lives here once.
fn normalise_within_bins(
    dispersions: &[f64],
    bins: &[Option<usize>],
    flavor: HvgFlavor,
) -> Vec<f64> {
    let n_bins = bins
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |bin| bin + 1);
    let mut grouped = vec![Vec::new(); n_bins];
    for (bin, &dispersion) in bins.iter().zip(dispersions) {
        // pandas' groupby aggregations skip NaN, so a gene without a dispersion
        // neither shifts nor widens the bin it sits in.
        if let (Some(bin), false) = (bin, dispersion.is_nan()) {
            grouped[*bin].push(dispersion);
        }
    }

    let statistics: Vec<(f64, f64)> = grouped
        .iter()
        .map(|bin| match flavor {
            HvgFlavor::Seurat => mean_and_deviation(bin),
            HvgFlavor::CellRanger => median_and_mad(bin),
        })
        .collect();

    bins.iter()
        .zip(dispersions)
        .map(|(bin, &dispersion)| match bin {
            Some(bin) => {
                let (centre, spread) = statistics[*bin];
                (dispersion - centre) / spread
            }
            None => f64::NAN,
        })
        .collect()
}

/// Bin mean and unbiased standard deviation, scanpy's `seurat` statistics.
///
/// A bin holding one gene has no standard deviation; scanpy then centres at zero
/// and scales by the mean, which normalises that lone gene to exactly 1.
fn mean_and_deviation(dispersions: &[f64]) -> (f64, f64) {
    let count = dispersions.len();
    let mean = dispersions.iter().sum::<f64>() / count as f64;
    if count < 2 {
        return (0.0, mean);
    }
    let variance = dispersions
        .iter()
        .map(|&dispersion| (dispersion - mean).powi(2))
        .sum::<f64>()
        / (count - 1) as f64;
    (mean, variance.sqrt())
}

/// Bin median and scaled median absolute deviation, scanpy's `cell_ranger`
/// statistics: robust to the few enormous dispersions raw counts produce.
fn median_and_mad(dispersions: &[f64]) -> (f64, f64) {
    let centre = median(dispersions);
    let deviations: Vec<f64> = dispersions
        .iter()
        .map(|&dispersion| (dispersion - centre).abs())
        .collect();
    (centre, median(&deviations) / MAD_SCALE)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// Flag the `n_top_genes` largest normalised dispersions.
///
/// scanpy takes the n-th largest value as a cut-off and keeps everything at or
/// above it, so genes tied on the boundary are all kept and the result can hold
/// more than `n_top_genes`. A gene without a normalised dispersion ranks last.
fn select_top(normalised: &[f64], n_top_genes: usize) -> Vec<bool> {
    let mut ranked: Vec<f64> = normalised
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    let wanted = n_top_genes.min(ranked.len());
    if wanted == 0 {
        return vec![false; normalised.len()];
    }
    ranked.sort_by(|a, b| b.total_cmp(a));
    let cutoff = ranked[wanted - 1];

    normalised
        .iter()
        .map(|&value| {
            if value.is_nan() {
                f64::NEG_INFINITY
            } else {
                value
            }
        })
        .map(|value| value >= cutoff)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RTOL: f64 = 1e-4;
    const N_CELLS: usize = 200;
    const N_GENES: usize = 300;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// SplitMix64, so the reference matrix here is identical to the one the
    /// scanpy script that produced the constants below generated.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn reference_counts() -> Vec<f32> {
        let mut rng = SplitMix64(20_240_722);
        let mut counts = vec![0.0f32; N_CELLS * N_GENES];
        for cell in 0..N_CELLS {
            for gene in 0..N_GENES {
                let draw = rng.next() % 100;
                let size = rng.next();
                if draw < 5 + (gene as u64 * 7) % 60 {
                    counts[cell * N_GENES + gene] = (1 + size % (2 + gene as u64 % 12)) as f32;
                }
            }
        }
        counts
    }

    fn logged(counts: &[f32]) -> Vec<f32> {
        counts.iter().map(|value| value.ln_1p()).collect()
    }

    /// scanpy reports normalised dispersions in f64; ours are the same numbers
    /// rounded to f32, so the comparison happens in f64.
    fn assert_close(actual: &[f32], expected: &[f64], rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (gene, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            let got = got as f64;
            assert!(
                (got - want).abs() <= rtol * want.abs().max(1.0),
                "gene {gene}: got {got}, want {want}"
            );
        }
    }

    fn overlap(actual: &[bool], expected: &[usize]) -> f32 {
        let shared = expected.iter().filter(|&&gene| actual[gene]).count();
        shared as f32 / expected.len() as f32
    }

    #[test]
    fn means_and_dispersions_match_hand_computation() {
        // 4 cells x 2 genes of raw counts.
        // gene 0: 1, 3, 0, 4 -> mean 2, var (1+1+4+4)/3 = 10/3, dispersion 5/3
        // gene 1: 2, 2, 2, 2 -> mean 2, var 0,                  dispersion 0
        let matrix =
            CsrMatrix::from_dense(&[1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0], 4, 2).unwrap();
        let (means, dispersions) = gene_statistics(&matrix, HvgFlavor::CellRanger);

        assert!((means[0] - 2.0).abs() < 1e-9);
        assert!((means[1] - 2.0).abs() < 1e-9);
        assert!((dispersions[0] - 5.0 / 3.0).abs() < 1e-9);
        assert!(dispersions[1].abs() < 1e-9);
    }

    #[test]
    fn seurat_statistics_undo_the_log_first() {
        // The same counts, handed over log1p-transformed as scanpy expects.
        let matrix =
            CsrMatrix::from_dense(&logged(&[1.0, 2.0, 3.0, 2.0, 0.0, 2.0, 4.0, 2.0]), 4, 2)
                .unwrap();
        let (means, dispersions) = gene_statistics(&matrix, HvgFlavor::Seurat);

        assert!((means[0] - 2.0f64.ln_1p()).abs() < 1e-6);
        assert!((dispersions[0] - (5.0f64 / 3.0).ln()).abs() < 1e-6);
        // A constant gene has zero dispersion, which has no log.
        assert!(dispersions[1].is_nan());
    }

    #[test]
    fn seurat_matches_scanpy() {
        let matrix = CsrMatrix::from_dense(&logged(&reference_counts()), N_CELLS, N_GENES).unwrap();
        let result = highly_variable_genes(&matrix, 100, HvgFlavor::Seurat, &cpu()).unwrap();

        assert_close(&result.normalised_dispersions, &SCANPY_SEURAT_NORM, RTOL);
        assert!(overlap(&result.highly_variable, &SCANPY_SEURAT_SELECTED) >= 0.95);
    }

    #[test]
    fn cell_ranger_matches_scanpy() {
        let matrix = CsrMatrix::from_dense(&reference_counts(), N_CELLS, N_GENES).unwrap();
        let result = highly_variable_genes(&matrix, 100, HvgFlavor::CellRanger, &cpu()).unwrap();

        assert_close(
            &result.normalised_dispersions,
            &SCANPY_CELL_RANGER_NORM,
            RTOL,
        );
        assert!(overlap(&result.highly_variable, &SCANPY_CELL_RANGER_SELECTED) >= 0.95);
    }

    #[test]
    fn a_gene_expressed_nowhere_is_never_highly_variable() {
        let mut counts = reference_counts();
        for cell in 0..N_CELLS {
            counts[cell * N_GENES + 7] = 0.0;
        }
        let matrix = CsrMatrix::from_dense(&logged(&counts), N_CELLS, N_GENES).unwrap();
        let result = highly_variable_genes(&matrix, 100, HvgFlavor::Seurat, &cpu()).unwrap();

        assert!(result.means[7].abs() < 1e-6);
        assert!(result.dispersions[7].is_nan());
        assert!(result.normalised_dispersions[7].is_nan());
        assert!(!result.highly_variable[7]);
    }

    #[test]
    fn more_top_genes_than_genes_selects_everything_rankable() {
        let matrix = CsrMatrix::from_dense(&reference_counts(), N_CELLS, N_GENES).unwrap();
        let result =
            highly_variable_genes(&matrix, N_GENES * 10, HvgFlavor::CellRanger, &cpu()).unwrap();

        let rankable = result
            .normalised_dispersions
            .iter()
            .filter(|value| !value.is_nan())
            .count();
        let selected = result.highly_variable.iter().filter(|flag| **flag).count();
        assert_eq!(selected, rankable);
    }

    #[test]
    fn a_lone_gene_in_its_bin_normalises_to_one() {
        // Gene 2 is expressed far above the others, so it sits alone in the top
        // bin of the mean; scanpy defines its normalised dispersion as 1.
        let counts = [
            1.0f32, 2.0, 500.0, 3.0, 1.0, 900.0, 2.0, 4.0, 100.0, 1.0, 3.0, 700.0,
        ];
        let matrix = CsrMatrix::from_dense(&logged(&counts), 4, 3).unwrap();
        let result = highly_variable_genes(&matrix, 1, HvgFlavor::Seurat, &cpu()).unwrap();

        assert!((result.normalised_dispersions[2] - 1.0).abs() < 1e-5);
        assert!(result.highly_variable[2]);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let matrix = CsrMatrix::from_dense(&[1.0, 2.0], 1, 2).unwrap();
        assert!(highly_variable_genes(&matrix, 0, HvgFlavor::Seurat, &cpu()).is_err());

        let no_cells = CsrMatrix::new(vec![0], vec![], vec![], 2).unwrap();
        assert!(highly_variable_genes(&no_cells, 1, HvgFlavor::Seurat, &cpu()).is_err());
    }

    const SCANPY_SEURAT_NORM: [f64; 300] = [
        -5.88736712e-01,
        -6.22189431e-01,
        -1.86152323e-01,
        -1.45281075e-01,
        3.06923989e-01,
        2.03390607e-01,
        1.18290002e-01,
        -9.00363557e-01,
        -1.14771688e+00,
        1.33973348e+00,
        9.49690341e-01,
        1.32869507e+00,
        -1.22129588e+00,
        -1.03798511e+00,
        -6.91215398e-01,
        -7.14088715e-01,
        -9.70155060e-01,
        -4.89837812e-01,
        8.40967539e-01,
        1.02625880e+00,
        8.59307496e-01,
        7.95402055e-01,
        8.18196674e-01,
        1.41002129e+00,
        -1.80249328e+00,
        -1.85918472e+00,
        -1.87946036e-01,
        5.78636743e-02,
        2.20508297e-01,
        5.71908304e-01,
        2.56931706e-01,
        4.84650917e-02,
        -4.88505645e-01,
        -4.78021081e-01,
        2.46637677e-01,
        1.68580780e+00,
        -9.37965890e-01,
        -9.99301609e-01,
        -4.86971205e-01,
        -4.92446839e-01,
        -8.88760341e-01,
        -6.22047053e-01,
        -1.53009444e+00,
        1.70474720e+00,
        1.19619807e+00,
        1.13537747e+00,
        1.17444074e+00,
        1.90740864e+00,
        -1.33574425e+00,
        -1.44070740e+00,
        -1.31487634e+00,
        -1.45662716e+00,
        6.89090129e-01,
        6.04159185e-01,
        6.67870652e-01,
        1.09546888e+00,
        5.16455373e-01,
        3.17016602e-01,
        3.50367217e-01,
        4.65223745e-01,
        -6.64095683e-01,
        -1.17204159e+00,
        -1.08721054e-01,
        6.56782464e-02,
        -1.41166213e-01,
        -3.33450772e-01,
        1.04899422e-01,
        -7.96094741e-01,
        -8.21430137e-01,
        1.26317409e+00,
        1.33275409e+00,
        1.48963487e+00,
        -1.45232578e+00,
        -9.23132881e-01,
        -7.51477874e-01,
        -5.78279780e-01,
        -8.57484929e-01,
        -1.16851230e+00,
        6.64502178e-01,
        9.71535469e-01,
        9.89895381e-01,
        1.10216595e+00,
        7.68504296e-01,
        1.03550755e+00,
        -1.79048154e+00,
        -1.72862100e+00,
        7.59319721e-01,
        4.03957322e-01,
        4.11427360e-01,
        1.51441173e-01,
        4.22982233e-01,
        4.64551779e-01,
        6.13519127e-02,
        -4.77499914e-01,
        -5.72277901e-01,
        1.01701093e+00,
        -1.00035061e+00,
        -6.49539981e-01,
        -2.79837827e-01,
        -1.85388295e-01,
        -2.35706718e-01,
        -4.17271862e-01,
        -5.32129187e-01,
        1.34204239e+00,
        1.22168917e+00,
        9.11923014e-01,
        1.10717792e+00,
        1.51917999e+00,
        -1.21765343e+00,
        -9.22540655e-01,
        -1.28589379e+00,
        -1.68200243e+00,
        4.41403074e-01,
        5.23659639e-01,
        4.44868637e-01,
        7.94127046e-01,
        9.10428910e-01,
        5.14940703e-01,
        4.94667941e-01,
        6.42557266e-01,
        -6.41912061e-01,
        -1.05359448e+00,
        -3.30463272e-01,
        4.00636673e-02,
        8.79899216e-02,
        1.17587486e-01,
        7.48117608e-02,
        -1.94202203e-01,
        -9.27352096e-02,
        1.36321097e+00,
        9.12473828e-01,
        1.24346890e+00,
        -1.04643289e+00,
        -7.64354763e-01,
        -8.17835153e-01,
        -6.53195215e-01,
        -1.09486454e+00,
        -2.21142091e+00,
        7.86613737e-01,
        5.42166695e-01,
        1.00611214e+00,
        9.38866322e-01,
        9.34224429e-01,
        1.55838884e+00,
        -1.77288589e+00,
        -1.79297414e+00,
        -1.44235020e-01,
        3.05377210e-01,
        4.28450901e-01,
        6.08421905e-01,
        3.63977553e-01,
        -1.15981796e-01,
        8.49355437e-01,
        -7.73903293e-01,
        -7.21523290e-01,
        1.32922290e+00,
        -9.18911199e-01,
        -4.08482945e-01,
        -3.71129973e-01,
        -5.56393091e-01,
        -1.53113829e-01,
        -2.41193211e-01,
        -1.39030399e+00,
        1.96877484e+00,
        1.16227291e+00,
        8.66016895e-01,
        1.11601415e+00,
        1.36502199e+00,
        -1.72927898e+00,
        -1.03341123e+00,
        -1.14207903e+00,
        -2.17497419e+00,
        7.77412549e-01,
        8.03653896e-01,
        1.13138265e+00,
        4.96360924e-01,
        1.04503516e+00,
        4.00175189e-01,
        3.06653084e-01,
        7.48095704e-01,
        -4.01877741e-01,
        -1.02669674e+00,
        -1.03460903e-01,
        -1.62705860e-01,
        -2.61151296e-01,
        9.84401344e-02,
        -1.31094195e-02,
        -4.70224595e-01,
        -2.11127374e+00,
        1.68366320e+00,
        1.05224713e+00,
        1.50453510e+00,
        -1.08331425e+00,
        -9.47528409e-01,
        -6.46263832e-01,
        -1.34881783e+00,
        -9.71024859e-01,
        -1.16352322e+00,
        9.56608063e-01,
        6.56374074e-01,
        9.88188879e-01,
        9.30980900e-01,
        9.74653012e-01,
        1.19810979e+00,
        -1.78203790e+00,
        -1.74434452e+00,
        -5.21173084e-01,
        1.15949394e-01,
        1.35388140e-01,
        2.50797753e-01,
        -3.66271155e-02,
        3.44789529e-01,
        -5.15324672e-01,
        -9.24857136e-01,
        -9.19691720e-01,
        1.15770259e+00,
        -1.10898004e+00,
        -6.26342618e-01,
        -5.33246968e-01,
        -3.35657779e-01,
        -6.79287483e-01,
        -6.72571116e-01,
        -1.44922268e+00,
        1.14245917e+00,
        9.50474684e-01,
        9.13961632e-01,
        8.98056991e-01,
        1.57354840e+00,
        -1.22263327e+00,
        -9.53734280e-01,
        -1.10434739e+00,
        -1.81684927e+00,
        8.99273539e-01,
        3.33409391e-01,
        1.06488521e+00,
        7.45584986e-01,
        6.21625051e-01,
        7.39380162e-01,
        5.08606604e-01,
        1.35620846e+00,
        -4.31472361e-01,
        -7.95488950e-01,
        -6.24305915e-02,
        1.64962987e-01,
        2.69898102e-01,
        -4.07892861e-01,
        1.55509892e-01,
        -7.48523664e-01,
        -9.62509146e-01,
        1.44281318e+00,
        1.81276050e+00,
        1.11893752e+00,
        -1.54318268e+00,
        -1.05149398e+00,
        -6.26500208e-01,
        -1.28522239e+00,
        -1.13646963e+00,
        -1.14467481e+00,
        8.61896814e-01,
        1.10439140e+00,
        9.67306601e-01,
        6.92621035e-01,
        9.00894816e-01,
        1.10836653e+00,
        -1.70371168e+00,
        -1.94545147e+00,
        -1.39226722e-01,
        2.07011647e-01,
        3.09464824e-01,
        5.06803672e-01,
        6.12914634e-02,
        2.31868036e-01,
        -3.02355938e-01,
        4.28202697e-02,
        2.97204605e-01,
        1.60393050e+00,
        -1.09670441e+00,
        -5.67953302e-01,
        -4.01706957e-01,
        -8.40327888e-02,
        -8.94382236e-01,
        -4.16041021e-01,
        -5.98014673e-01,
        1.43128179e+00,
        1.16274783e+00,
        1.19973037e+00,
        1.02652871e+00,
        1.56142838e+00,
        -1.78553001e+00,
        -1.48458656e+00,
        -1.17869212e+00,
        -1.57450815e+00,
        3.05301339e-02,
        5.27839686e-01,
        3.87687148e-01,
        6.17803517e-01,
        7.46733863e-01,
        1.09322949e+00,
        4.38948540e-01,
        1.33398314e+00,
    ];
    const SCANPY_SEURAT_SELECTED: [usize; 100] = [
        9, 10, 11, 18, 19, 20, 21, 22, 23, 29, 35, 43, 44, 45, 46, 47, 52, 53, 54, 55, 69, 70, 71,
        78, 79, 80, 81, 82, 83, 86, 95, 103, 104, 105, 106, 107, 115, 116, 119, 129, 130, 131, 138,
        140, 141, 142, 143, 149, 152, 155, 163, 164, 165, 166, 167, 172, 173, 174, 176, 179, 189,
        190, 191, 198, 199, 200, 201, 202, 203, 215, 223, 224, 225, 226, 227, 232, 234, 235, 236,
        237, 239, 249, 250, 251, 258, 259, 260, 261, 262, 263, 275, 283, 284, 285, 286, 287, 295,
        296, 297, 299,
    ];
    const SCANPY_CELL_RANGER_NORM: [f64; 300] = [
        -6.43530416e-01,
        -4.51001293e-02,
        -5.12749256e-01,
        0.00000000e+00,
        5.21032958e-01,
        0.00000000e+00,
        -4.11388159e-02,
        -4.57124636e-01,
        -1.06008359e+00,
        1.57123232e+00,
        3.68181946e+00,
        9.34273595e-01,
        -7.32578072e-01,
        -9.74198216e-01,
        -6.22898513e-01,
        -7.49585450e-01,
        -1.02167240e+00,
        -6.37844962e-01,
        9.35029069e-01,
        1.11229887e+00,
        3.51542520e-01,
        6.74489750e-01,
        1.62785570e+00,
        1.28544939e+00,
        -1.27772830e+00,
        -8.49353173e-01,
        3.76146541e-01,
        -2.08036494e-01,
        0.00000000e+00,
        9.40729298e-01,
        5.46169036e-02,
        2.78410053e-01,
        -7.46282762e-02,
        0.00000000e+00,
        0.00000000e+00,
        3.35594016e+00,
        -6.69867215e-01,
        -1.27191440e+00,
        -2.19863505e-01,
        -4.40579077e-01,
        -5.04647662e-01,
        -3.68020230e-01,
        -1.04373380e+00,
        3.72032242e+00,
        1.41474424e+00,
        6.74489750e-01,
        1.20376844e+00,
        2.75906819e+00,
        -6.90826964e-01,
        -1.16104239e+00,
        -1.00538637e+00,
        -7.25064591e-01,
        1.26217943e+00,
        6.47621750e-01,
        5.85349982e-01,
        7.06538073e-01,
        8.71221011e-01,
        1.03431133e+00,
        1.43769566e-03,
        3.67100653e-01,
        -6.93640255e-01,
        -4.65358115e-01,
        -4.28015593e-02,
        -9.28317410e-02,
        -7.03265995e-02,
        -2.00359488e-01,
        -5.57991531e-02,
        -3.91302820e-01,
        -8.79062376e-01,
        2.11531023e+00,
        1.68301884e+00,
        1.05446516e+00,
        -1.56582725e+00,
        -7.54410782e-01,
        -6.74489750e-01,
        -6.89720879e-01,
        -4.85304835e-01,
        -6.76934956e-01,
        7.21441764e-01,
        1.02176388e+00,
        6.74489750e-01,
        8.24353891e-01,
        1.74106505e+00,
        4.21211352e-01,
        -1.32149850e+00,
        -8.41568626e-01,
        7.34875618e-01,
        2.17031502e-02,
        2.57899929e-01,
        1.60752644e-01,
        2.35451733e-01,
        1.89722224e-01,
        6.74489750e-01,
        -6.74489750e-01,
        -1.22528327e+00,
        1.55222924e+00,
        -7.12730726e-01,
        -9.87506447e-01,
        -3.73191619e-01,
        6.99154086e-02,
        -5.15436189e-01,
        -5.73594986e-01,
        -6.74489750e-01,
        2.82124772e+00,
        2.01310909e+00,
        1.33477546e+00,
        8.28563108e-01,
        1.53867538e+00,
        -6.57616009e-01,
        -4.36381331e-01,
        -9.93445719e-01,
        -8.10361884e-01,
        3.64533091e-01,
        3.89654688e-01,
        4.36482655e-01,
        7.10110666e-01,
        7.92050236e-01,
        7.00358961e-01,
        4.17538144e-01,
        6.74489750e-01,
        -6.79112286e-01,
        -3.84082765e-01,
        -6.74139045e-01,
        -1.15830896e-01,
        1.07182785e-01,
        -1.89702161e-01,
        -8.84720392e-02,
        -1.05774754e-01,
        3.75879788e-01,
        2.37343279e+00,
        1.33580670e+00,
        1.53425855e+00,
        -6.57935078e-01,
        -6.74489750e-01,
        -5.64300831e-01,
        -6.74489750e-01,
        -7.69549874e-01,
        -8.29310981e-01,
        1.11026544e+00,
        4.03525670e-01,
        5.16681365e-01,
        8.22067501e-01,
        6.60793741e-01,
        8.87370481e-01,
        -6.89095924e-01,
        -8.31732650e-01,
        4.23692621e-01,
        3.97425341e-01,
        2.85274214e-01,
        1.00331515e+00,
        1.69147052e-01,
        -5.06655649e-02,
        8.82287613e-01,
        -1.37610660e+00,
        -1.33716640e+00,
        1.66946502e+00,
        -6.56451435e-01,
        -2.87232836e-01,
        -1.42721171e-01,
        -3.36470485e-01,
        -3.22664164e-01,
        -1.49814103e-01,
        -9.29765980e-01,
        3.28702321e+00,
        1.46175129e+00,
        1.25046198e+00,
        1.01798873e+00,
        1.14402928e+00,
        -6.74489750e-01,
        -4.80171785e-01,
        -9.31065507e-01,
        -1.08424596e+00,
        6.74489750e-01,
        1.03209320e+00,
        2.04606149e+00,
        0.00000000e+00,
        1.17560340e+00,
        5.35050001e-02,
        -1.49607817e-01,
        8.61608247e-01,
        -5.09579294e-01,
        -3.64964156e-01,
        -3.80111711e-02,
        0.00000000e+00,
        -5.29633849e-01,
        -1.02162108e-01,
        0.00000000e+00,
        0.00000000e+00,
        -3.24196954e+00,
        3.35579929e+00,
        1.61317706e+00,
        1.19789434e+00,
        -6.74489750e-01,
        -9.27385155e-01,
        -5.83329110e-01,
        -7.00560807e-01,
        -5.37562145e-01,
        -6.74489750e-01,
        1.42899822e+00,
        9.03461466e-01,
        1.00309424e+00,
        6.35658039e-01,
        1.10495125e+00,
        2.25399147e+00,
        -1.06949687e+00,
        -8.18342808e-01,
        4.51001293e-02,
        1.93131514e-01,
        -9.35350289e-02,
        -1.98355128e-01,
        0.00000000e+00,
        8.95851793e-02,
        -1.03357347e-01,
        -1.76306317e+00,
        -1.48360199e+00,
        1.84462128e+00,
        -7.83638191e-01,
        -4.33853658e-01,
        -6.74489750e-01,
        -2.82601647e-01,
        -3.99558588e-01,
        -5.74215624e-01,
        -1.45986822e+00,
        2.39018689e+00,
        1.05199721e+00,
        8.80856216e-01,
        8.62455011e-01,
        1.72290362e+00,
        -6.59075723e-01,
        -4.24763503e-01,
        -9.21413013e-01,
        -1.03362744e+00,
        1.67913554e+00,
        1.32232620e-01,
        6.74489750e-01,
        6.41940830e-01,
        3.29536806e-01,
        6.64674716e-01,
        0.00000000e+00,
        3.52115061e-01,
        -5.31757173e-01,
        -1.89878218e-01,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        -2.46161343e-01,
        0.00000000e+00,
        -2.99394372e-01,
        -9.58839981e-01,
        3.05533040e+00,
        1.62821154e+00,
        1.13143116e+00,
        -1.61624098e+00,
        -9.80958498e-01,
        -2.97836494e-01,
        -6.76060557e-01,
        -6.09070490e-01,
        -6.65212290e-01,
        1.60108701e+00,
        7.15986073e-01,
        6.74489750e-01,
        5.74486538e-01,
        1.96708274e+00,
        8.79654229e-01,
        -6.74489750e-01,
        -1.13391962e+00,
        4.29205210e-01,
        0.00000000e+00,
        1.03380310e-01,
        8.32094815e-01,
        7.28168789e-02,
        0.00000000e+00,
        1.05222809e-01,
        -3.04195605e-01,
        -5.45212452e-01,
        2.19611459e+00,
        -7.75855351e-01,
        -4.12809096e-01,
        -4.58068888e-01,
        0.00000000e+00,
        -6.82719295e-01,
        -4.95464189e-01,
        -3.66917083e-01,
        3.00293754e+00,
        1.35255880e+00,
        1.36327045e+00,
        1.22231910e+00,
        1.59337708e+00,
        -6.88031018e-01,
        -1.18641474e+00,
        -9.47454258e-01,
        -9.55089113e-01,
        6.24229716e-01,
        3.83720284e-01,
        5.37932679e-01,
        1.09306508e-01,
        7.36869042e-01,
        1.25121548e+00,
        3.22292317e-01,
        3.32462172e-01,
    ];
    const SCANPY_CELL_RANGER_SELECTED: [usize; 100] = [
        9, 10, 11, 18, 19, 21, 22, 23, 29, 35, 43, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 69, 70,
        71, 78, 79, 80, 81, 82, 86, 92, 95, 103, 104, 105, 106, 107, 115, 116, 117, 119, 129, 130,
        131, 138, 141, 142, 143, 149, 152, 155, 163, 164, 165, 166, 167, 172, 173, 174, 176, 179,
        189, 190, 191, 198, 199, 200, 201, 202, 203, 215, 223, 224, 225, 226, 227, 232, 234, 235,
        237, 249, 250, 251, 258, 259, 260, 261, 262, 263, 269, 275, 283, 284, 285, 286, 287, 292,
        294, 296, 297,
    ];
}
