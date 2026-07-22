//! Processing matrices larger than memory. Owned by feat/out-of-core.
//!
//! The in-memory `CsrMatrix` is the fast path. Above a configured ceiling the
//! same operations run over row blocks streamed from disk, so a run is bounded
//! by the block size rather than by the dataset.

use crate::error::Result;
use crate::sparse::CsrMatrix;

/// A source of row blocks, whether from memory or from a file.
///
/// Small enough that a caller needing only sequential access does not depend on
/// random access, and vice versa.
pub trait RowBlocks {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    /// The next block of at most `max_rows` rows, or `None` at the end.
    fn next_block(&mut self, max_rows: usize) -> Result<Option<CsrMatrix>>;
    fn restart(&mut self) -> Result<()>;
}

/// How many rows fit in `budget_bytes` given the matrix's density.
pub fn rows_per_block(_n_cols: usize, _density: f64, _budget_bytes: usize) -> usize {
    todo!("feat/out-of-core")
}

/// Per-gene mean and variance accumulated over blocks, in one pass.
pub fn streaming_gene_statistics(_blocks: &mut dyn RowBlocks) -> Result<(Vec<f32>, Vec<f32>)> {
    todo!("feat/out-of-core")
}

/// Per-cell totals accumulated over blocks.
pub fn streaming_cell_totals(_blocks: &mut dyn RowBlocks) -> Result<Vec<f32>> {
    todo!("feat/out-of-core")
}
