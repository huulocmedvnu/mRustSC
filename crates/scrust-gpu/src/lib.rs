//! Hand written Metal kernels for the loops candle cannot express.
//!
//! Everything expressible as tensor algebra lives in `scrust-core` and runs on
//! either device. Only the irregular inner loops — nearest-neighbour selection,
//! UMAP's negative sampling, t-SNE's repulsive forces — need their own kernel,
//! and they live here.

pub mod context;
pub mod kernels;

pub use context::MetalContext;
