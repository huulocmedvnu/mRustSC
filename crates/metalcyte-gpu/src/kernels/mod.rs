//! One module per hand written kernel. Each owns its Metal source and the Rust
//! function that dispatches it.

pub mod knn;
pub mod raster;
pub mod spmm;
pub mod tsne_fft_gpu;
pub mod tsne_gradient;
pub mod umap_sgd;
