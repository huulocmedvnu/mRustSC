//! Algorithms and data types for single-cell analysis.
//!
//! Every algorithm is written once against candle tensors and takes a
//! `candle_core::Device`, so the CPU and Apple GPU paths are the same code. Only
//! the inner loops that candle cannot express live in `scrust-gpu` as hand
//! written Metal kernels.

pub mod de;
pub mod device;
pub mod error;
pub mod neighbors;
pub mod pca;
pub mod preprocess;
pub mod sparse;
pub mod tsne;
pub mod umap;

pub use device::{gpu_available, DeviceKind};
pub use error::{Error, Result};
pub use sparse::CsrMatrix;
