use ndarray::Array2;
use scrust_core::error::Result;
use scrust_core::umap::UmapParams;

use crate::context::MetalContext;

/// One epoch of UMAP's attractive and repulsive updates.
///
/// `head`/`tail` are the graph edges and `epochs_per_sample` the schedule that
/// decides which edges fire this epoch.
pub fn umap_epoch(
    _context: &MetalContext,
    _embedding: &mut Array2<f32>,
    _head: &[u32],
    _tail: &[u32],
    _epochs_per_sample: &[f32],
    _epoch: usize,
    _params: &UmapParams,
) -> Result<()> {
    todo!("feat/umap-kernel")
}
