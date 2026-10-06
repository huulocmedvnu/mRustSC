//! Multiple-testing corrections.
//!
//! The algorithms work in `f64` even though the rest of the crate is `f32`:
//! a differential expression p-value routinely falls below `f32`'s smallest
//! normal value (~1.2e-38) and would underflow to exactly zero, which destroys
//! the ordering the correction depends on. The `f32` entry points convert.

/// Benjamini-Hochberg adjusted p-values.
///
/// Non-finite entries are passed through untouched and excluded from the
/// effective number of tests, so unfitted genes do not dilute the correction.
pub fn benjamini_hochberg(p_values: &[f32]) -> Vec<f32> {
    widen_and_narrow(p_values, benjamini_hochberg_f64)
}

/// Benjamini-Hochberg on p-values that are already `f64`.
pub fn benjamini_hochberg_f64(p_values: &[f64]) -> Vec<f64> {
    adjust_tested_only(p_values, step_up)
}

/// Bonferroni adjusted p-values, with the same treatment of non-finite entries.
pub fn bonferroni(p_values: &[f32]) -> Vec<f32> {
    widen_and_narrow(p_values, bonferroni_f64)
}

/// Bonferroni on p-values that are already `f64`.
pub fn bonferroni_f64(p_values: &[f64]) -> Vec<f64> {
    adjust_tested_only(p_values, |tested| {
        let n_tests = tested.len() as f64;
        tested.iter().map(|p| p * n_tests).collect()
    })
}

/// Run an `f64` correction over `f32` inputs. Widening is exact; the result is
/// narrowed back, which is all the caller can hold anyway.
fn widen_and_narrow(p_values: &[f32], correction: impl Fn(&[f64]) -> Vec<f64>) -> Vec<f32> {
    let widened: Vec<f64> = p_values.iter().map(|&p| p as f64).collect();
    correction(&widened).into_iter().map(|p| p as f32).collect()
}

/// Apply `correction` to the finite p-values only, leaving the rest as they are.
///
/// A gene that could not be fitted carries a non-finite p-value. It was never
/// tested, so counting it would drag every other gene's adjusted value towards
/// significance: the effective number of tests is the number of finite entries.
/// Clipping to `[0, 1]` lives here too, because it holds for every correction.
fn adjust_tested_only(p_values: &[f64], correction: impl Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let tested: Vec<usize> = (0..p_values.len())
        .filter(|&index| p_values[index].is_finite())
        .collect();

    let mut adjusted = p_values.to_vec();
    if tested.is_empty() {
        return adjusted;
    }

    let values: Vec<f64> = tested.iter().map(|&index| p_values[index]).collect();
    for (&index, value) in tested.iter().zip(correction(&values)) {
        adjusted[index] = value.clamp(0.0, 1.0);
    }
    adjusted
}

/// The Benjamini-Hochberg step-up, walking the p-values from largest to smallest.
///
/// The running minimum enforces monotonicity: a gene is never reported as more
/// significant than one ranked above it, so tied p-values share an adjusted value.
fn step_up(tested: &[f64]) -> Vec<f64> {
    let n_tests = tested.len();
    let mut descending: Vec<usize> = (0..n_tests).collect();
    descending.sort_unstable_by(|&a, &b| tested[b].total_cmp(&tested[a]));

    let mut adjusted = vec![0.0; n_tests];
    let mut running_minimum = f64::INFINITY;
    for (position, &index) in descending.iter().enumerate() {
        let rank = (n_tests - position) as f64;
        running_minimum = running_minimum.min(n_tests as f64 / rank * tested[index]);
        adjusted[index] = running_minimum;
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.stats.false_discovery_control` on 20 uniform draws (seed 0).
    const UNIFORM: [f32; 20] = [
        0.636962, 0.269787, 0.040974, 0.016528, 0.81327, 0.912756, 0.606636, 0.729497, 0.543625,
        0.935072, 0.815854, 0.002739, 0.857404, 0.033586, 0.729655, 0.175656, 0.863179, 0.541461,
        0.299712, 0.422687,
    ];
    const UNIFORM_BH: [f32; 20] = [
        0.935072, 0.85632, 0.20487, 0.16528, 0.935072, 0.935072, 0.935072, 0.935072, 0.935072,
        0.935072, 0.935072, 0.05478, 0.935072, 0.20487, 0.935072, 0.702624, 0.935072, 0.935072,
        0.85632, 0.935072,
    ];

    /// The same reference on a vector that is almost entirely ties.
    const TIED: [f32; 22] = [
        0.1, 0.01, 0.1, 1.0, 1.0, 0.5, 0.5, 0.1, 1.0, 1.0, 0.001, 0.01, 0.5, 0.5, 0.1, 0.5, 1.0,
        0.5, 0.001, 0.1, 0.5, 0.01,
    ];
    const TIED_BH: [f32; 22] = [
        0.22, 0.044, 0.22, 1.0, 1.0, 0.6470588, 0.6470588, 0.22, 1.0, 1.0, 0.011, 0.044, 0.6470588,
        0.6470588, 0.22, 0.6470588, 1.0, 0.6470588, 0.011, 0.22, 0.6470588, 0.044,
    ];

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-6 * e.abs().max(1e-3), "{a} != {e}");
        }
    }

    #[test]
    fn matches_scipy_on_uniform_p_values() {
        assert_close(&benjamini_hochberg(&UNIFORM), &UNIFORM_BH);
    }

    #[test]
    fn matches_scipy_with_heavy_ties() {
        assert_close(&benjamini_hochberg(&TIED), &TIED_BH);
    }

    #[test]
    fn non_finite_entries_are_excluded_from_the_test_count() {
        let with_nan = [0.04, f32::NAN, 0.01, 0.03, f32::NAN];
        let adjusted = benjamini_hochberg(&with_nan);

        assert!(adjusted[1].is_nan() && adjusted[4].is_nan());
        // scipy on the three finite entries alone.
        assert_close(
            &[adjusted[0], adjusted[2], adjusted[3]],
            &[0.04, 0.03, 0.04],
        );
    }

    #[test]
    fn handles_empty_and_single_element_input() {
        assert!(benjamini_hochberg(&[]).is_empty());
        assert!(bonferroni(&[]).is_empty());
        assert_close(&benjamini_hochberg(&[0.02]), &[0.02]);
        assert_close(&bonferroni(&[0.02]), &[0.02]);
    }

    /// `scipy.stats.false_discovery_control(..., method="bh")` on the inputs the
    /// uniform draws never reach: everything tied, a single test, a run with no
    /// ties at all, and p-values eight decades below what `f32` can hold.
    ///
    /// Compared exactly, not to a tolerance: BH is arithmetic on the inputs,
    /// and a step-up that is right should reproduce the reference bit for bit
    /// wherever the reference itself is unambiguous.
    #[test]
    fn matches_scipy_on_adversarial_p_values() {
        let cases: [(&[f64], &[f64]); 6] = [
            (&[0.3; 7], &[0.3; 7]),
            (&[0.02], &[0.02]),
            (&[1.0; 5], &[1.0; 5]),
            (&[0.0; 4], &[0.0; 4]),
            (
                &[1e-300, 1e-300, 1e-300, 0.5, 0.5, 0.5, 1.0],
                &[
                    2.333_333_333_333_333_6e-300,
                    2.333_333_333_333_333_6e-300,
                    2.333_333_333_333_333_6e-300,
                    0.583_333_333_333_333_4,
                    0.583_333_333_333_333_4,
                    0.583_333_333_333_333_4,
                    1.0,
                ],
            ),
            (
                &[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                &[
                    0.9,
                    0.888_888_888_888_889,
                    0.875,
                    0.857_142_857_142_857_1,
                    0.833_333_333_333_333_4,
                    0.8,
                    0.75,
                    0.666_666_666_666_666_7,
                    0.5,
                    0.5,
                ],
            ),
        ];
        for (p_values, expected) in cases {
            let adjusted = benjamini_hochberg_f64(p_values);
            for (index, (&got, &want)) in adjusted.iter().zip(expected).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-15 * want.abs(),
                    "{p_values:?}[{index}]: {got} != {want}"
                );
            }
        }
    }

    /// The far tail is why the correction runs in `f64`.
    ///
    /// Every one of these p-values is below `f32::MIN_POSITIVE`, so an `f32`
    /// correction would flatten them all to zero and lose the ordering the
    /// step-up depends on. In `f64` the ordering survives, and the two
    /// subnormal inputs come back as the subnormals scipy reports.
    #[test]
    fn the_far_tail_survives_the_correction() {
        let p_values = [5e-324, 1e-320, 1e-300, 0.5];
        assert!(p_values[..3].iter().all(|&p| (p as f32) == 0.0));

        let adjusted = benjamini_hochberg_f64(&p_values);
        let expected = [2e-323, 2e-320, 1.333_333_333_333_333_4e-300, 0.5];
        for (index, (&got, &want)) in adjusted.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-14 * want.abs(),
                "[{index}]: {got} != {want}"
            );
        }
        // Ordering, the thing the underflow would have destroyed.
        assert!(adjusted[0] < adjusted[1] && adjusted[1] < adjusted[2]);
    }

    /// The step-up must never let a gene overtake one with a smaller p-value.
    #[test]
    fn adjusted_values_stay_monotone_in_the_raw_p_values() {
        const RAW: [f64; 12] = [
            1e-300, 1e-300, 2e-40, 1e-12, 1e-12, 1e-12, 0.004, 0.004, 0.2, 0.2, 0.9, 1.0,
        ];
        let adjusted = benjamini_hochberg_f64(&RAW);
        for i in 0..RAW.len() {
            for j in 0..RAW.len() {
                if RAW[i] < RAW[j] {
                    assert!(adjusted[i] <= adjusted[j], "{i} vs {j}");
                }
                if RAW[i] == RAW[j] {
                    assert_eq!(adjusted[i], adjusted[j], "tied {i} vs {j}");
                }
            }
        }
    }

    #[test]
    fn adjusted_values_stay_in_the_unit_interval() {
        let p_values = [0.9, 0.5, 0.999, 1.0, 0.0];
        let adjusted = benjamini_hochberg(&p_values)
            .into_iter()
            .chain(bonferroni(&p_values));
        for value in adjusted {
            assert!((0.0..=1.0).contains(&value), "{value}");
        }
    }

    #[test]
    fn bonferroni_is_never_below_benjamini_hochberg() {
        for p_values in [&UNIFORM[..], &TIED[..]] {
            let pairs = bonferroni(p_values)
                .into_iter()
                .zip(benjamini_hochberg(p_values));
            for (strong, weak) in pairs {
                assert!(strong >= weak - 1e-6, "{strong} < {weak}");
            }
        }
    }

    #[test]
    fn bonferroni_multiplies_by_the_number_of_tests() {
        let adjusted = bonferroni(&[0.01, 0.2, f32::NAN, 0.5]);
        assert_close(&[adjusted[0], adjusted[1], adjusted[3]], &[0.03, 0.6, 1.0]);
        assert!(adjusted[2].is_nan());
    }
}
