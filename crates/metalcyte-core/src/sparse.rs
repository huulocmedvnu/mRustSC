use candle_core::{Device, Tensor};

use crate::error::{Error, Result};

/// Cells-by-genes counts in compressed sparse row form, the layout AnnData uses.
///
/// Single-cell matrices are 90-95% zeros, so the sparse form is what crosses the
/// Python boundary. Algorithms densify per row block when they need a tensor.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f32>,
    n_cols: usize,
}

impl CsrMatrix {
    pub fn new(
        indptr: Vec<u32>,
        indices: Vec<u32>,
        values: Vec<f32>,
        n_cols: usize,
    ) -> Result<Self> {
        if indptr.is_empty() {
            return Err(Error::shape(
                "indptr with at least one entry",
                "empty indptr",
            ));
        }
        if indices.len() != values.len() {
            return Err(Error::shape(
                format!("{} values", indices.len()),
                format!("{} values", values.len()),
            ));
        }
        // `indptr` has to start at zero and never decrease. Only its *last* entry used
        // to be checked, which let three broken shapes through, all reachable from the
        // Python boundary since every binding builds a matrix from caller-supplied
        // arrays:
        //
        //  - `[1, 3]` over three values: entry 0 belongs to no row. No error, and the
        //    damage is silent and inconsistent -- row readers (`row_reductions`, and so
        //    `normalize_total`) never see the orphan, while column readers
        //    (`column_reductions`, `scale`'s gene moments) zip `indices` with `values`
        //    and do. `normalize_total(target_sum=6)` returned `[5, 3, 3]` where the
        //    same matrix written correctly gives `[4.29, 0.857, 0.857]`.
        //  - `[0, 2, 1, 3]`: the second row's range runs backwards, and the first
        //    consumer to slice it panics. A panic crosses into Python as
        //    `PanicException`, which derives from `BaseException`, so a caller's
        //    `except Exception` does not catch it.
        //  - `[0, 5, 3]` over three values: row 0 claims `values[0..5]`, out of bounds.
        //
        // `scipy.sparse.csr_matrix.check_format` rejects all three with a `ValueError`,
        // which is what these now do.
        if indptr[0] != 0 {
            return Err(Error::shape(
                "indptr starting at 0",
                format!("indptr starting at {}", indptr[0]),
            ));
        }
        if let Some(row) = indptr.windows(2).position(|pair| pair[1] < pair[0]) {
            return Err(Error::shape(
                "a non-decreasing indptr",
                format!(
                    "indptr[{row}] = {} followed by {}, so row {row} runs backwards",
                    indptr[row],
                    indptr[row + 1]
                ),
            ));
        }
        let expected_nnz = *indptr.last().expect("checked non-empty") as usize;
        if expected_nnz != values.len() {
            return Err(Error::shape(
                format!("{expected_nnz} stored entries (from indptr)"),
                format!("{} stored entries", values.len()),
            ));
        }
        if indices.iter().any(|&column| column as usize >= n_cols) {
            return Err(Error::shape(
                format!("column indices below {n_cols}"),
                "an out-of-range column index".to_string(),
            ));
        }
        Ok(Self {
            indptr,
            indices,
            values,
            n_cols,
        })
    }

    /// Build from a dense row-major slice; convenient for tests and small inputs.
    pub fn from_dense(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Self> {
        if data.len() != n_rows * n_cols {
            return Err(Error::shape(
                format!("{} values", n_rows * n_cols),
                format!("{} values", data.len()),
            ));
        }
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for row in data.chunks_exact(n_cols) {
            for (column, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    indices.push(column as u32);
                    values.push(value);
                }
            }
            indptr.push(values.len() as u32);
        }
        Self::new(indptr, indices, values, n_cols)
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Mutable access to the stored values, for kernels that scale in place.
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Densify rows `start..end` into a row-major buffer.
    pub fn densify_rows(&self, start: usize, end: usize) -> Vec<f32> {
        let end = end.min(self.n_rows());
        let mut dense = vec![0.0; (end - start) * self.n_cols];
        for row in start..end {
            let from = self.indptr[row] as usize;
            let to = self.indptr[row + 1] as usize;
            let offset = (row - start) * self.n_cols;
            for entry in from..to {
                dense[offset + self.indices[entry] as usize] = self.values[entry];
            }
        }
        dense
    }

    /// Densify rows `start..end` straight onto `device`.
    pub fn to_tensor_rows(&self, start: usize, end: usize, device: &Device) -> Result<Tensor> {
        let end = end.min(self.n_rows());
        let dense = self.densify_rows(start, end);
        Ok(Tensor::from_vec(dense, (end - start, self.n_cols), device)?)
    }

    /// Densify the whole matrix onto `device`. Only for matrices known to fit.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        self.to_tensor_rows(0, self.n_rows(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> CsrMatrix {
        // 3 cells x 4 genes, two zero entries per row.
        CsrMatrix::from_dense(
            &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0],
            3,
            4,
        )
        .unwrap()
    }

    #[test]
    fn from_dense_keeps_only_stored_entries() {
        let matrix = example();
        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(matrix.n_cols(), 4);
        assert_eq!(matrix.nnz(), 5);
        assert_eq!(matrix.indptr(), &[0, 2, 4, 5]);
    }

    #[test]
    fn densify_round_trips() {
        let matrix = example();
        assert_eq!(
            matrix.densify_rows(0, 3),
            vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn densify_handles_a_row_block() {
        let matrix = example();
        assert_eq!(matrix.densify_rows(1, 2), vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        assert!(CsrMatrix::new(vec![0, 1], vec![0], vec![1.0, 2.0], 4).is_err());
        assert!(CsrMatrix::new(vec![0, 1], vec![9], vec![1.0], 4).is_err());
        assert!(CsrMatrix::new(vec![], vec![], vec![], 4).is_err());
    }
}
