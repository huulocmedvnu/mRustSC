//! Zero-copy, multi-core preprocessing kernels.
//!
//! The owned-`CsrMatrix` versions in `normalize` and `scale` copy every array in and
//! out, which costs more than the arithmetic for cheap elementwise steps. These
//! variants work straight on borrowed slices (in practice numpy's own buffers), write
//! in place or into a caller-provided output, and split the work across every core
//! with rayon. On Apple silicon that is the performance and efficiency cores together.
//!
//! Numerics match the owned versions: each row is still summed sequentially in `f32`,
//! so `normalize_total` and `log1p` give bit-identical results; `scale` accumulates
//! gene moments in `f64` exactly as before and differs only in how the per-thread
//! partial sums are combined.

use rayon::prelude::*;

use crate::error::{Error, Result};

/// A CSR offset or index of any width numpy hands over (`int32`, `uint32`, `int64`).
pub trait Offset: Copy + Send + Sync {
    fn to_usize(self) -> usize;
}

macro_rules! offset {
    ($($t:ty),*) => {$(
        impl Offset for $t {
            #[inline(always)]
            fn to_usize(self) -> usize {
                self as usize
            }
        }
    )*};
}
offset!(i32, u32, i64, u64);

/// Rows below this many stored values per chunk are not worth a task of their own.
const MIN_ROWS_PER_TASK: usize = 256;

fn check_indptr<O: Offset>(indptr: &[O], nnz: usize) -> Result<usize> {
    let Some(last) = indptr.last() else {
        return Err(Error::shape(
            "indptr of length n_rows + 1",
            "an empty indptr",
        ));
    };
    if last.to_usize() != nnz {
        return Err(Error::shape(
            format!("indptr ending at {nnz} stored values"),
            format!("indptr ending at {}", last.to_usize()),
        ));
    }
    Ok(indptr.len() - 1)
}

/// Per-row sum of stored values, each row summed sequentially (as the owned version).
pub fn row_totals<O: Offset>(indptr: &[O], values: &[f32]) -> Result<Vec<f32>> {
    let n_rows = check_indptr(indptr, values.len())?;
    Ok((0..n_rows)
        .into_par_iter()
        .with_min_len(MIN_ROWS_PER_TASK)
        .map(|row| {
            let span = indptr[row].to_usize()..indptr[row + 1].to_usize();
            values[span].iter().sum::<f32>()
        })
        .collect())
}

/// Median of the totals over all cells (zero-count cells included), as scanpy's CSR path.
fn median_of(totals: &[f32]) -> Option<f32> {
    if totals.is_empty() {
        return None;
    }
    let mut sorted = totals.to_vec();
    sorted.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    })
}

/// `scanpy.pp.normalize_total` on the stored values, in place. Returns the target used.
///
/// Zero-count cells are left untouched. A non-positive target (e.g. the median of a
/// matrix that is mostly empty cells) is refused rather than silently zeroing the data.
pub fn normalize_total_inplace<O: Offset>(
    indptr: &[O],
    values: &mut [f32],
    target_sum: Option<f32>,
) -> Result<Option<f32>> {
    let totals = row_totals(indptr, values)?;
    let Some(target) = target_sum.or_else(|| median_of(&totals)) else {
        return Ok(None);
    };
    if target <= 0.0 || !target.is_finite() {
        return Err(Error::parameter("target_sum", "a positive count", target));
    }
    for_each_row_mut(indptr, values, |row, slice| {
        let total = totals[row];
        if total > 0.0 {
            let factor = target / total;
            for value in slice {
                *value *= factor;
            }
        }
    });
    Ok(Some(target))
}

/// `ln(1 + x)` on every stored value, in place, across all cores.
pub fn log1p_inplace(values: &mut [f32]) {
    values
        .par_chunks_mut(1 << 16)
        .for_each(|chunk| chunk.iter_mut().for_each(|v| *v = v.ln_1p()));
}

/// Split `values` into its rows and hand each to `f` in parallel.
fn for_each_row_mut<O: Offset, F>(indptr: &[O], values: &mut [f32], f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    // Carve the value buffer into disjoint row slices first, then process them in
    // parallel; no unsafe aliasing is needed.
    let n_rows = indptr.len() - 1;
    let mut rows: Vec<(usize, &mut [f32])> = Vec::with_capacity(n_rows);
    let mut rest = values;
    let mut consumed = 0usize;
    for row in 0..n_rows {
        let end = indptr[row + 1].to_usize();
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(end - consumed);
        rows.push((row, head));
        rest = tail;
        consumed = end;
    }
    rows.into_par_iter()
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(row, slice)| f(row, slice));
}

/// Per-gene mean and corrected standard deviation over all cells (implicit zeros included).
///
/// `f64` accumulation and the two-pass squared-deviation form, exactly as
/// `scale::gene_mean_and_deviation`; the passes are split by row blocks and the
/// per-block partial sums are added together.
pub fn gene_moments<O: Offset, I: Offset>(
    indptr: &[O],
    indices: &[I],
    values: &[f32],
    n_cols: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n_rows = check_indptr(indptr, values.len())?;
    if indices.len() != values.len() {
        return Err(Error::shape(
            format!("{} indices", values.len()),
            format!("{} indices", indices.len()),
        ));
    }
    let block = (n_rows / rayon::current_num_threads().max(1)).max(MIN_ROWS_PER_TASK);
    let blocks: Vec<(usize, usize)> = (0..n_rows)
        .step_by(block)
        .map(|s| (s, (s + block).min(n_rows)))
        .collect();

    let fold = |pass: &(dyn Fn(usize, f32) -> f64 + Sync)| -> (Vec<f64>, Vec<usize>) {
        blocks
            .par_iter()
            .map(|&(start, end)| {
                let mut sums = vec![0f64; n_cols];
                let mut stored = vec![0usize; n_cols];
                for k in indptr[start].to_usize()..indptr[end].to_usize() {
                    let gene = indices[k].to_usize();
                    sums[gene] += pass(gene, values[k]);
                    stored[gene] += 1;
                }
                (sums, stored)
            })
            .reduce(
                || (vec![0f64; n_cols], vec![0usize; n_cols]),
                |(mut a, mut ca), (b, cb)| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    ca.iter_mut().zip(&cb).for_each(|(x, y)| *x += y);
                    (a, ca)
                },
            )
    };

    let (sums, stored) = fold(&|_, v| f64::from(v));
    let means: Vec<f32> = sums.iter().map(|s| (s / n_rows as f64) as f32).collect();
    let (squares, _) = fold(&|gene, v| {
        let d = f64::from(v) - f64::from(means[gene]);
        d * d
    });
    let deviations = (0..n_cols)
        .map(|gene| {
            let mean = f64::from(means[gene]);
            let implicit = (n_rows - stored[gene]) as f64;
            let variance = (squares[gene] + implicit * mean * mean) / (n_rows as f64 - 1.0);
            let deviation = variance.sqrt() as f32;
            if deviation > 0.0 {
                deviation
            } else {
                1.0
            }
        })
        .collect();
    Ok((means, deviations))
}

/// `scanpy.pp.scale` from CSR straight into a row-major dense `out` (n_rows x n_cols).
///
/// One fused pass per row: fill the row with the value an implicit zero scales to, then
/// overwrite the stored entries. No intermediate dense matrix, no device round trip.
pub fn scale_into<O: Offset, I: Offset>(
    indptr: &[O],
    indices: &[I],
    values: &[f32],
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f32>,
    out: &mut [f32],
) -> Result<()> {
    let n_rows = check_indptr(indptr, values.len())?;
    if n_rows < 2 {
        return Err(Error::shape("at least 2 cells", format!("{n_rows} cells")));
    }
    if out.len() != n_rows * n_cols {
        return Err(Error::shape(
            format!("an output of {} values", n_rows * n_cols),
            format!("{} values", out.len()),
        ));
    }
    let (means, deviations) = gene_moments(indptr, indices, values, n_cols)?;
    let clip = |x: f32| match max_value {
        None => x,
        Some(limit) if zero_center => x.clamp(-limit, limit),
        Some(limit) => x.min(limit),
    };
    let zero_row: Vec<f32> = (0..n_cols)
        .map(|g| {
            clip(if zero_center {
                (0.0 - means[g]) / deviations[g]
            } else {
                0.0 / deviations[g]
            })
        })
        .collect();
    out.par_chunks_mut(n_cols)
        .with_min_len(64)
        .enumerate()
        .for_each(|(row, dest)| {
            dest.copy_from_slice(&zero_row);
            for k in indptr[row].to_usize()..indptr[row + 1].to_usize() {
                let g = indices[k].to_usize();
                let v = values[k];
                dest[g] = clip(if zero_center {
                    (v - means[g]) / deviations[g]
                } else {
                    v / deviations[g]
                });
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::{normalize, scale};
    use crate::sparse::CsrMatrix;
    use candle_core::Device;

    fn random_csr(rows: usize, cols: usize, seed: u64) -> CsrMatrix {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dense: Vec<f32> = (0..rows * cols)
            .map(|_| {
                if rng.gen::<f32>() < 0.2 {
                    rng.gen_range(1..9) as f32
                } else {
                    0.0
                }
            })
            .collect();
        CsrMatrix::from_dense(&dense, rows, cols).unwrap()
    }

    #[test]
    fn normalize_and_log1p_are_bit_identical_to_the_owned_versions() {
        let m = random_csr(1500, 80, 3);
        let reference =
            normalize::log1p(&normalize::normalize_total(&m, None, &Device::Cpu).unwrap()).unwrap();
        let mut values = m.values().to_vec();
        normalize_total_inplace(m.indptr(), &mut values, None).unwrap();
        log1p_inplace(&mut values);
        assert_eq!(values, reference.values());
    }

    #[test]
    fn scale_into_matches_the_owned_version() {
        let m = random_csr(1200, 60, 5);
        for (zc, mv) in [(true, Some(10.0)), (true, None), (false, Some(3.0))] {
            let reference = scale::scale(&m, zc, mv, &Device::Cpu).unwrap();
            let reference: Vec<f32> = reference.flatten_all().unwrap().to_vec1().unwrap();
            let mut out = vec![0f32; m.n_rows() * m.n_cols()];
            scale_into(
                m.indptr(),
                m.indices(),
                m.values(),
                m.n_cols(),
                zc,
                mv,
                &mut out,
            )
            .unwrap();
            let worst = out
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-5, "max abs diff {worst}");
        }
    }

    #[test]
    fn zero_median_is_refused() {
        let indptr: Vec<u32> = vec![0, 0, 0, 1];
        let mut values = vec![2.0f32];
        assert!(normalize_total_inplace(&indptr, &mut values, None).is_err());
    }
}
