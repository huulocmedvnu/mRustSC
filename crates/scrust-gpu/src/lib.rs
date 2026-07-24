//! Hand written Metal kernels for the loops candle cannot express.
//!
//! Everything expressible as tensor algebra lives in `scrust-core` and runs on
//! either device. Only the irregular inner loops — nearest-neighbour selection,
//! UMAP's negative sampling, t-SNE's repulsive forces — need their own kernel,
//! and they live here.
//!
//! # Nothing here is wired in, and that is a decision rather than an oversight
//!
//! No code outside this crate calls any of these kernels. `scrust-py` does not even
//! depend on it. `git log -S"scrust_gpu::" -- crates/scrust-core/src crates/scrust-py/src`
//! is empty across every branch: there has never been a call site. The kernels were
//! each written on their own branch under a file-ownership split that put the call
//! sites — `core::umap`, `core::tsne`, `core::neighbors` — in *other* branches'
//! territory, so no branch owned the join.
//!
//! Connecting them is deliberately deferred. Two reasons, both about correctness
//! rather than effort:
//!
//! 1. **`umap_sgd` is Hogwild.** It accepts racing writes between threads, so it does
//!    not reproduce bit for bit against itself, let alone against the sequential CPU
//!    sweep. See its own module docs for the measurements.
//! 2. **`core::umap` currently ignores `device` entirely.** Wiring the kernel in makes
//!    it start reading `device`, and the default is `"auto"`, which resolves to Metal
//!    wherever one exists. Callers would silently get different layouts on different
//!    machines without asking for anything — the same failure mode as the duplicate
//!    distance bug in `core::neighbors`, but across a whole algorithm.
//!
//! Both would also break the scanpy cross-checks in `tests/test_*_audit.py`, which
//! compare against a transcription of `umap-learn`'s sequential loop within `2e-3`.
//!
//! So this is not dead code to delete on sight. It is finished, tested work waiting on
//! a decision about non-determinism that has not been made. Whoever makes it should
//! start with `docs/API_CONTRACT.md`, which is where the reproducibility promise lives.

pub mod context;
pub mod kernels;

pub use context::MetalContext;
