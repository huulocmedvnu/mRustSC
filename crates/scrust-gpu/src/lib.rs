//! Hand written Metal kernels for the loops candle cannot express.
//!
//! Everything expressible as tensor algebra lives in `scrust-core` and runs on
//! either device. Only the irregular inner loops — nearest-neighbour selection,
//! UMAP's negative sampling, t-SNE's repulsive forces — need their own kernel,
//! and they live here.
//!
//! # What is wired in, and what is not
//!
//! **`knn` is wired.** `crates/scrust-py` depends on this crate and dispatches a Metal
//! caller's k-NN to [`kernels::knn::knn_metal`] (`scrust-py/src/embedding.rs`); the CPU
//! path in `core::neighbors` stays the oracle. To match it bit for bit on degenerate
//! input the kernel reproduces the CPU path's two numerical safeguards — f64
//! mean-centering and snapping a squared distance below
//! `(n_dims + 2) * f32::EPSILON * (|a|^2 + |b|^2)` to zero — so both devices treat a
//! knot tighter than `f32` can resolve as coincident points. `tests/test_device_parity.py`
//! holds the two devices to this: 4 of 4 pass. Measured on an M3 Pro the kernel is
//! ~2-2.5x faster than the candle path it replaces.
//!
//! **`spmm` and `tsne_gradient` are not wired.** They are finished and tested against
//! their `scrust-core` counterparts, but no call site reaches them yet: `spmm`'s only
//! natural consumer, `core::pca`, does a *centred* product with a rank-one correction
//! rather than the plain sparse×dense this kernel offers.
//!
//! **`umap_sgd` is deliberately not wired, and should stay that way for now.** It is
//! Hogwild — it accepts racing writes between threads, so it does not reproduce bit for
//! bit against itself, let alone against the sequential CPU sweep (see its module docs).
//! `core::umap` also ignores `device` today, so wiring the kernel in would make a UMAP
//! layout depend on whether the caller's machine has a GPU — the same failure mode the
//! `knn` fix closes for k-NN, but across a whole stochastic algorithm — and it would
//! break the `umap-learn` cross-checks in `tests/test_umap_audit.py`. Whoever revisits
//! it should start with `docs/API_CONTRACT.md`, where the reproducibility promise lives.

pub mod context;
pub mod kernels;

pub use context::MetalContext;
