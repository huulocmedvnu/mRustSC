//! Parallel reads of fixed-width elements from a chunked file layout, such as an
//! uncompressed HDF5 dataset whose chunk byte offsets are known.
//!
//! The out-of-core head reads each row block of a CSR matrix as a set of byte ranges.
//! This module reads those ranges with `pread` on all cores and narrows `int64` gene
//! indices to `int32`, so both on-disk index widths give the same block.

use std::fs::File;
use std::os::unix::fs::FileExt;

use rayon::prelude::*;

/// Fill `out` with elements `[lo, lo + out.len() / itemsize)` of a dataset whose chunk `c`
/// holds elements `[c * chunk_len, (c + 1) * chunk_len)` starting at byte `offsets[c]`.
pub fn read_elements(
    file: &File,
    offsets: &[u64],
    chunk_len: u64,
    itemsize: usize,
    lo: u64,
    out: &mut [u8],
) -> std::io::Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let hi = lo + (out.len() / itemsize) as u64;
    let mut pieces: Vec<(u64, &mut [u8])> = Vec::new();
    let mut rest = out;
    let mut start = lo;
    while start < hi {
        let chunk = start / chunk_len;
        let end = hi.min((chunk + 1) * chunk_len);
        let Some(&base) = offsets.get(chunk as usize) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("element {start} lies past the last chunk"),
            ));
        };
        let (head, tail) = rest.split_at_mut(((end - start) as usize) * itemsize);
        pieces.push((base + (start - chunk * chunk_len) * itemsize as u64, head));
        rest = tail;
        start = end;
    }
    pieces
        .into_par_iter()
        .try_for_each(|(at, buf)| file.read_exact_at(buf, at))
}

pub fn as_bytes_mut<T: Copy>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: `T` is a plain number type (f32, i32, i64); every byte pattern is valid.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

/// Read stored entries `[lo, lo + n)` of a CSR matrix: the `float32` values and the gene
/// indices as `int32`. `index_itemsize` is 4 or 8, the width the file stores them in.
#[allow(clippy::too_many_arguments)]
pub fn read_csr_entries(
    file: &File,
    data_offsets: &[u64],
    data_chunk_len: u64,
    index_offsets: &[u64],
    index_chunk_len: u64,
    index_itemsize: usize,
    lo: u64,
    n: usize,
) -> std::io::Result<(Vec<f32>, Vec<i32>)> {
    let mut values = vec![0f32; n];
    let mut indices = vec![0i32; n];
    let (r_values, r_indices) = rayon::join(
        || {
            read_elements(
                file,
                data_offsets,
                data_chunk_len,
                4,
                lo,
                as_bytes_mut(&mut values),
            )
        },
        || -> std::io::Result<()> {
            if index_itemsize == 4 {
                return read_elements(
                    file,
                    index_offsets,
                    index_chunk_len,
                    4,
                    lo,
                    as_bytes_mut(&mut indices),
                );
            }
            let mut wide = vec![0i64; n];
            read_elements(
                file,
                index_offsets,
                index_chunk_len,
                8,
                lo,
                as_bytes_mut(&mut wide),
            )?;
            indices
                .par_iter_mut()
                .zip(wide.par_iter())
                .try_for_each(|(narrow, &w)| {
                    *narrow = i32::try_from(w).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("gene index {w} does not fit in int32"),
                        )
                    })?;
                    Ok(())
                })
        },
    );
    r_values?;
    r_indices?;
    Ok((values, indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_ranges_across_scattered_chunks() {
        // Three chunks of 4 u32 values, written out of order with a gap, as HDF5 may.
        let dir = std::env::temp_dir().join(format!("mc_io_{}", std::process::id()));
        let mut f = File::create(&dir).unwrap();
        let chunk = |c: u32| {
            (0..4)
                .flat_map(move |i| (c * 4 + i).to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let mut bytes = vec![0xffu8; 8];
        let mut offsets = [0u64; 3];
        for c in [2u32, 0, 1] {
            offsets[c as usize] = bytes.len() as u64;
            bytes.extend(chunk(c));
            bytes.extend([0xee; 5]);
        }
        f.write_all(&bytes).unwrap();
        drop(f);
        let file = File::open(&dir).unwrap();
        for (lo, hi) in [(0u64, 12u64), (3, 9), (5, 6), (4, 8), (7, 7)] {
            let mut out = vec![0u32; (hi - lo) as usize];
            read_elements(&file, &offsets, 4, 4, lo, as_bytes_mut(&mut out)).unwrap();
            assert_eq!(out, (lo as u32..hi as u32).collect::<Vec<_>>());
        }
        let mut past = vec![0u32; 2];
        assert!(read_elements(&file, &offsets, 4, 4, 11, as_bytes_mut(&mut past)).is_err());
        std::fs::remove_file(&dir).unwrap();
    }
}
