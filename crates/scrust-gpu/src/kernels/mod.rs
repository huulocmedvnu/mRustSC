//! One module per hand written kernel. Each owns its Metal source and the Rust
//! function that dispatches it.

pub mod knn;
pub mod tsne_gradient;
pub mod umap_sgd;
