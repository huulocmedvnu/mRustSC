/// Benjamini-Hochberg adjusted p-values.
///
/// Non-finite entries are passed through untouched and excluded from the
/// effective number of tests, so unfitted genes do not dilute the correction.
pub fn benjamini_hochberg(_p_values: &[f32]) -> Vec<f32> {
    todo!("feat/multiple-testing")
}

/// Bonferroni adjusted p-values, with the same treatment of non-finite entries.
pub fn bonferroni(_p_values: &[f32]) -> Vec<f32> {
    todo!("feat/multiple-testing")
}
