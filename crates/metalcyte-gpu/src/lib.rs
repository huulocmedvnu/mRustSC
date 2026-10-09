//! Hand written Metal kernels for the loops candle cannot express.
//!
//! Everything expressible as tensor algebra lives in `metalcyte-core` and runs on
//! either device. Only the irregular inner loops — nearest-neighbour selection,
//! UMAP's negative sampling, t-SNE's repulsive forces — need their own kernel,
//! and they live here.
//!
//! # What is wired in, and what is not
//!
//! **`knn` is wired.** `crates/metalcyte-py` depends on this crate and dispatches a Metal
//! caller's k-NN to [`kernels::knn::knn_metal`] (`metalcyte-py/src/embedding.rs`); the CPU
//! path in `core::neighbors` stays the oracle. To match it bit for bit on degenerate
//! input the kernel reproduces the CPU path's two numerical safeguards — f64
//! mean-centering and snapping a squared distance below
//! `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` to zero — so both devices treat a
//! knot tighter than `f32` can resolve as coincident points. `tests/test_device_parity.py`
//! holds the two devices to this: 4 of 4 pass. Measured on an M3 Pro the kernel is
//! ~2-2.5x faster than the candle path it replaces.
//!
//! **`spmm` and `tsne_gradient` are not wired.** They are finished and tested against
//! their `metalcyte-core` counterparts, but no call site reaches them yet: `spmm`'s only
//! natural consumer, `core::pca`, does a *centred* product with a rank-one correction
//! rather than the plain sparse×dense this kernel offers.
//!
//! **`umap_sgd` is wired only behind `tl.umap(..., parallel=True)`.** It is Hogwild: it
//! accepts racing writes between threads, so it does not reproduce bit for bit against
//! itself. The opt-in CPU optimiser behind `parallel=True` has the same property, so the
//! GPU takes that path when the device is Metal and the default sequential layout, which
//! the `umap-learn` cross-checks in `tests/test_umap_audit.py` hold to, never reaches it.
//! The kernel needs Metal Shading Language 3.0 (`atomic_float`) and asks for it
//! explicitly, because the wheel's macOS 11 deployment target defaults to an older one.

// Force-link Accelerate for ndarray's BLAS backend when the feature is on.
#[cfg(feature = "accelerate")]
extern crate blas_src;

pub mod context;
pub mod kernels;

pub use context::MetalContext;
