use ndarray::Array1;

use crate::de::glm::GlmFit;
use crate::error::{Error, Result};

/// A test applied to one contrast of a fit.
#[derive(Debug, Clone)]
pub struct TestStatistics {
    pub statistic: Array1<f32>,
    pub p_values: Array1<f32>,
    /// Contrast estimate on the natural-log scale.
    pub effect: Array1<f32>,
    pub effect_standard_error: Array1<f32>,
}

/// A gene whose standard error is zero, negative or non-finite carries no usable
/// evidence: the fit degenerated (all-zero counts, separation, a singular
/// information matrix). Reporting p = 1 keeps it in the table and lets multiple
/// testing treat it as uninformative, whereas NaN would silently propagate into
/// every downstream summary.
const UNINFORMATIVE_P_VALUE: f32 = 1.0;

/// Wald test of `contrast @ beta == 0`, using the normal approximation.
pub fn wald_test(fit: &GlmFit, contrast: &[f32]) -> Result<TestStatistics> {
    let (n_genes, n_coefficients) = fit.coefficients.dim();
    if contrast.len() != n_coefficients {
        return Err(Error::shape(
            format!("a contrast with {n_coefficients} entries"),
            format!("{}", contrast.len()),
        ));
    }
    if fit.covariance.dim() != (n_genes, n_coefficients, n_coefficients) {
        return Err(Error::shape(
            format!("a covariance of ({n_genes}, {n_coefficients}, {n_coefficients})"),
            format!("{:?}", fit.covariance.dim()),
        ));
    }

    let mut effect = Array1::zeros(n_genes);
    let mut standard_error = Array1::zeros(n_genes);
    let mut statistic = Array1::zeros(n_genes);
    let mut p_values = Array1::zeros(n_genes);

    for gene in 0..n_genes {
        let estimate: f32 = (0..n_coefficients)
            .map(|c| fit.coefficients[[gene, c]] * contrast[c])
            .sum();
        let variance: f32 = (0..n_coefficients)
            .map(|c| {
                let row: f32 = (0..n_coefficients)
                    .map(|d| fit.covariance[[gene, c, d]] * contrast[d])
                    .sum();
                contrast[c] * row
            })
            .sum();

        // Rounding can push a variance a hair below zero; the square root of a
        // negative would be indistinguishable from a genuinely failed fit.
        let error = if variance > 0.0 {
            variance.sqrt()
        } else {
            f32::NAN
        };
        let usable = error.is_finite() && error > 0.0;

        effect[gene] = estimate;
        standard_error[gene] = error;
        statistic[gene] = if usable { estimate / error } else { 0.0 };
        p_values[gene] = if usable {
            2.0 * normal_survival(statistic[gene].abs())
        } else {
            UNINFORMATIVE_P_VALUE
        };
    }

    Ok(TestStatistics {
        statistic,
        p_values,
        effect,
        effect_standard_error: standard_error,
    })
}

/// Upper tail of the standard normal, `P(Z > z)`.
///
/// Written through `erfc` rather than `1 - cdf` so the far tail keeps its
/// relative accuracy instead of cancelling against one.
fn normal_survival(z: f32) -> f32 {
    (0.5 * erfc(f64::from(z) / std::f64::consts::SQRT_2)) as f32
}

/// The complementary error function, by Cody's rational approximations.
///
/// Three fits — a polynomial ratio near zero, and two `exp(-x^2)`-scaled ratios
/// beyond it — reproduce `erfc` to near double precision without a series, which
/// is why this is preferred to the shorter textbook Chebyshev form.
pub(crate) fn erfc(x: f64) -> f64 {
    const A: [f64; 5] = [
        3.161_123_743_870_565_6e0,
        1.138_641_541_510_501_6e2,
        3.774_852_376_853_02e2,
        3.209_377_589_138_469_5e3,
        1.857_777_061_846_031_5e-1,
    ];
    const B: [f64; 4] = [
        2.360_129_095_234_412_1e1,
        2.440_246_379_344_441_7e2,
        1.282_616_526_077_372_3e3,
        2.844_236_833_439_171e3,
    ];
    const C: [f64; 9] = [
        5.641_884_969_886_701e-1,
        8.883_149_794_388_376e0,
        6.611_919_063_714_163e1,
        2.986_351_381_974_001e2,
        8.819_522_212_417_69e2,
        1.712_047_612_634_070_6e3,
        2.051_078_377_826_071_5e3,
        1.230_339_354_797_997_2e3,
        2.153_115_354_744_038_5e-8,
    ];
    const D: [f64; 8] = [
        1.574_492_611_070_983_5e1,
        1.176_939_508_913_125e2,
        5.371_811_018_620_099e2,
        1.621_389_574_566_690_2e3,
        3.290_799_235_733_46e3,
        4.362_619_090_143_247e3,
        3.439_367_674_143_722e3,
        1.230_339_354_803_749_4e3,
    ];
    const P: [f64; 6] = [
        3.053_266_349_612_323_4e-1,
        3.603_448_999_498_044_4e-1,
        1.257_817_261_112_292_5e-1,
        1.608_378_514_874_228e-2,
        6.587_491_615_298_378e-4,
        1.631_538_713_730_209_8e-2,
    ];
    const Q: [f64; 5] = [
        2.568_520_192_289_822,
        1.872_952_849_923_460_5e0,
        5.279_051_029_514_284e-1,
        6.051_834_131_244_132e-2,
        2.335_204_976_268_692e-3,
    ];
    /// `1 / sqrt(pi)`, the asymptotic leading coefficient.
    const INVERSE_SQRT_PI: f64 = 5.641_895_835_477_563e-1;

    let magnitude = x.abs();
    if magnitude <= 0.46875 {
        let square = magnitude * magnitude;
        let mut numerator = A[4] * square;
        let mut denominator = square;
        for index in 0..3 {
            numerator = (numerator + A[index]) * square;
            denominator = (denominator + B[index]) * square;
        }
        return 1.0 - x * (numerator + A[3]) / (denominator + B[3]);
    }

    let tail = if magnitude <= 4.0 {
        let mut numerator = C[8] * magnitude;
        let mut denominator = magnitude;
        for index in 0..7 {
            numerator = (numerator + C[index]) * magnitude;
            denominator = (denominator + D[index]) * magnitude;
        }
        (numerator + C[7]) / (denominator + D[7])
    } else {
        let inverse_square = 1.0 / (magnitude * magnitude);
        let mut numerator = P[5] * inverse_square;
        let mut denominator = inverse_square;
        for index in 0..4 {
            numerator = (numerator + P[index]) * inverse_square;
            denominator = (denominator + Q[index]) * inverse_square;
        }
        let series = inverse_square * (numerator + P[4]) / (denominator + Q[4]);
        (INVERSE_SQRT_PI - series) / magnitude
    };

    // Splitting exp(-x^2) at 1/16 of x keeps the exponent exact in the first
    // factor, so the rounding of x^2 costs no significant digits.
    let truncated = (magnitude * 16.0).trunc() / 16.0;
    let remainder = (magnitude - truncated) * (magnitude + truncated);
    let scaled = (-truncated * truncated).exp() * (-remainder).exp() * tail;

    if x < 0.0 {
        2.0 - scaled
    } else {
        scaled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// `x`, `scipy.special.erfc(x)`, `scipy.stats.norm.sf(x)`.
    ///
    /// Every `x` is an exact binary fraction, so the f32 wrapper is measured on
    /// its own error rather than on the rounding of its argument.
    const REFERENCE: [(f64, f64, f64); 16] = [
        (0.0, 1.0, 0.5),
        (0.25, 0.7236736098317631, 0.4012936743170763),
        (0.5, 0.4795001221869535, 0.3085375387259869),
        (1.0, 0.15729920705028516, 0.15865525393145707),
        (1.5, 0.033894853524689274, 0.06680720126885806),
        (2.0, 0.004677734981047266, 0.022750131948179198),
        (2.5, 0.0004069520174449589, 0.006209665325776134),
        (3.0, 2.2090496998585445e-5, 0.001349898031630093),
        (4.0, 1.541725790028002e-8, 3.167124183311986e-5),
        (5.0, 1.5374597944280353e-12, 2.8665157187919344e-7),
        (6.0, 2.151973671249891e-17, 9.865876450376944e-10),
        (8.0, 1.1224297172982928e-29, 6.22096057427174e-16),
        (-0.5, 1.5204998778130465, 0.6914624612740131),
        (-1.0, 1.8427007929497148, 0.8413447460685429),
        (-2.5, 1.999593047982555, 0.9937903346742238),
        (-4.0, 1.999999984582742, 0.9999683287581669),
    ];

    fn fit_from(coefficients: Array2<f32>, covariance: Array3<f32>) -> GlmFit {
        let n_genes = coefficients.dim().0;
        GlmFit {
            coefficients,
            covariance,
            dispersions: vec![0.1; n_genes],
            fitted_means: Array2::zeros((n_genes, 1)),
            converged: vec![true; n_genes],
            n_iterations: 1,
        }
    }

    /// One gene, two coefficients, a diagonal covariance: everything by hand.
    fn single_gene_fit(effect: f32, variance: f32) -> GlmFit {
        let coefficients = Array2::from_shape_vec((1, 2), vec![0.7, effect]).unwrap();
        let mut covariance = Array3::zeros((1, 2, 2));
        covariance[[0, 0, 0]] = 0.25;
        covariance[[0, 1, 1]] = variance;
        fit_from(coefficients, covariance)
    }

    #[test]
    fn erfc_matches_scipy_to_double_precision() {
        for (x, expected, _) in REFERENCE {
            let relative = (erfc(x) - expected).abs() / expected;
            assert!(relative < 1e-14, "erfc({x}): relative error {relative}");
        }
    }

    #[test]
    fn normal_survival_matches_scipy_to_f32_precision() {
        for (x, _, expected) in REFERENCE {
            let actual = f64::from(normal_survival(x as f32));
            let relative = (actual - expected).abs() / expected;
            assert!(relative < 1e-7, "norm.sf({x}): relative error {relative}");
        }
    }

    #[test]
    fn reproduces_a_hand_computed_z_and_p() {
        // effect 1.2, variance 0.36 => se 0.6, z = 2, p = 2 * norm.sf(2).
        let statistics = wald_test(&single_gene_fit(1.2, 0.36), &[0.0, 1.0]).unwrap();

        assert!((statistics.effect[0] - 1.2).abs() < 1e-6);
        assert!((statistics.effect_standard_error[0] - 0.6).abs() < 1e-6);
        assert!((statistics.statistic[0] - 2.0).abs() < 1e-5);
        assert!((statistics.p_values[0] - 0.045_500_264).abs() < 1e-6);
    }

    #[test]
    fn a_contrast_combines_coefficients() {
        // contrast [1, 1] over independent coefficients: se = sqrt(0.25 + 0.36).
        let statistics = wald_test(&single_gene_fit(1.2, 0.36), &[1.0, 1.0]).unwrap();

        assert!((statistics.effect[0] - 1.9).abs() < 1e-6);
        assert!((statistics.effect_standard_error[0] - 0.61_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn a_null_effect_gives_a_p_value_near_one() {
        let statistics = wald_test(&single_gene_fit(0.0, 0.36), &[0.0, 1.0]).unwrap();
        assert!((statistics.p_values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_standard_errors_give_one_not_nan() {
        for variance in [0.0, f32::NAN, -1.0, f32::INFINITY] {
            let statistics = wald_test(&single_gene_fit(1.2, variance), &[0.0, 1.0]).unwrap();
            let p = statistics.p_values[0];
            assert!(!p.is_nan() && (p - 1.0).abs() < 1e-6, "variance {variance}");
            assert_eq!(statistics.statistic[0], 0.0);
        }
    }

    #[test]
    fn rejects_a_contrast_of_the_wrong_length() {
        assert!(wald_test(&single_gene_fit(1.2, 0.36), &[1.0]).is_err());
    }

    #[test]
    fn p_values_are_uniform_under_the_null() {
        let n_genes = 4000;
        let variance = 0.25_f32;
        let mut normals = StandardNormals::new(7);

        let coefficients = Array2::from_shape_fn((n_genes, 1), |_| {
            variance.sqrt() * normals.next_value() as f32
        });
        let covariance = Array3::from_shape_fn((n_genes, 1, 1), |_| variance);
        let statistics = wald_test(&fit_from(coefficients, covariance), &[1.0]).unwrap();

        assert!(statistics.p_values.iter().all(|p| (0.0..=1.0).contains(p)));
        for level in [0.01_f64, 0.05, 0.25, 0.5] {
            let below = statistics
                .p_values
                .iter()
                .filter(|&&p| f64::from(p) <= level)
                .count();
            let observed = below as f64 / n_genes as f64;
            // Three standard errors of the binomial proportion.
            let tolerance = 3.0 * (level * (1.0 - level) / n_genes as f64).sqrt();
            assert!((observed - level).abs() < tolerance, "{level}: {observed}");
        }
    }

    /// Box-Muller over a seeded uniform stream: the tests need a null sample,
    /// not a distribution library.
    struct StandardNormals {
        rng: rand::rngs::StdRng,
        spare: Option<f64>,
    }

    impl StandardNormals {
        fn new(seed: u64) -> Self {
            use rand::SeedableRng;
            Self {
                rng: rand::rngs::StdRng::seed_from_u64(seed),
                spare: None,
            }
        }

        fn next_value(&mut self) -> f64 {
            use rand::Rng;
            if let Some(value) = self.spare.take() {
                return value;
            }
            let radius = (-2.0 * self.rng.gen::<f64>().max(f64::MIN_POSITIVE).ln()).sqrt();
            let angle = std::f64::consts::TAU * self.rng.gen::<f64>();
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }
    }
}
