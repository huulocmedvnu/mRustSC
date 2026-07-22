/// Benjamini-Hochberg adjusted p-values.
///
/// Non-finite entries are passed through untouched and excluded from the
/// effective number of tests, so unfitted genes do not dilute the correction.
pub fn benjamini_hochberg(p_values: &[f32]) -> Vec<f32> {
    adjust_tested_only(p_values, step_up)
}

/// Bonferroni adjusted p-values, with the same treatment of non-finite entries.
pub fn bonferroni(p_values: &[f32]) -> Vec<f32> {
    adjust_tested_only(p_values, |tested| {
        let n_tests = tested.len() as f32;
        tested.iter().map(|p| p * n_tests).collect()
    })
}

/// Apply `correction` to the finite p-values only, leaving the rest as they are.
///
/// A gene that could not be fitted carries a non-finite p-value. It was never
/// tested, so counting it would drag every other gene's adjusted value towards
/// significance: the effective number of tests is the number of finite entries.
/// Clipping to `[0, 1]` lives here too, because it holds for every correction.
fn adjust_tested_only(p_values: &[f32], correction: impl Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
    let tested: Vec<usize> = (0..p_values.len())
        .filter(|&index| p_values[index].is_finite())
        .collect();

    let mut adjusted = p_values.to_vec();
    if tested.is_empty() {
        return adjusted;
    }

    let values: Vec<f32> = tested.iter().map(|&index| p_values[index]).collect();
    for (&index, value) in tested.iter().zip(correction(&values)) {
        adjusted[index] = value.clamp(0.0, 1.0);
    }
    adjusted
}

/// The Benjamini-Hochberg step-up, walking the p-values from largest to smallest.
///
/// The running minimum enforces monotonicity: a gene is never reported as more
/// significant than one ranked above it, so tied p-values share an adjusted value.
fn step_up(tested: &[f32]) -> Vec<f32> {
    let n_tests = tested.len();
    let mut descending: Vec<usize> = (0..n_tests).collect();
    descending.sort_unstable_by(|&a, &b| tested[b].total_cmp(&tested[a]));

    let mut adjusted = vec![0.0; n_tests];
    let mut running_minimum = f32::INFINITY;
    for (position, &index) in descending.iter().enumerate() {
        let rank = (n_tests - position) as f32;
        running_minimum = running_minimum.min(n_tests as f32 / rank * tested[index]);
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
