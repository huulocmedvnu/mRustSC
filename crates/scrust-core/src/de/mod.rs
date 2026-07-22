//! Differential expression: rank-sum over cells, and a negative binomial GLM
//! over pseudobulk replicates.

pub mod dispersion;
pub mod glm;
pub mod hypothesis;
pub mod multiple_testing;
pub mod wilcoxon;
