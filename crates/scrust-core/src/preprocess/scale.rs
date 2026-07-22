use candle_core::{Device, Tensor};

use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Standardise each gene to zero mean and unit variance, as `scanpy.pp.scale`.
///
/// Returns a dense tensor: scaling destroys sparsity, which is why scanpy also
/// densifies here.
///
/// The standard deviation carries Bessel's correction (ddof = 1), which is what
/// scanpy computes — `mean_var(..., correction=1)`, its "R convention".
pub fn scale(
    matrix: &CsrMatrix,
    zero_center: bool,
    max_value: Option<f32>,
    device: &Device,
) -> Result<Tensor> {
    let n_rows = matrix.n_rows();
    if n_rows < 2 {
        // With one cell the corrected variance is 0/0; refuse rather than hand
        // back a tensor of NaN.
        return Err(Error::shape("at least 2 cells", format!("{n_rows} cells")));
    }

    let data = matrix.to_tensor(device)?;
    let mean = data.mean_keepdim(0)?;
    let centered = data.broadcast_sub(&mean)?;
    let variance = (centered.sqr()?.sum_keepdim(0)? / (n_rows as f64 - 1.0))?;
    let deviation = variance.sqrt()?;

    // A gene with no variance would divide by zero; scanpy substitutes 1, which
    // leaves the centred column exactly zero.
    let ones = deviation.ones_like()?;
    let deviation = deviation.gt(0.0)?.where_cond(&deviation, &ones)?;

    let numerator = if zero_center { centered } else { data };
    let scaled = numerator.broadcast_div(&deviation)?;

    let Some(limit) = max_value else {
        return Ok(scaled);
    };
    // Without zero-centering scanpy clips only from above, since the values are
    // still non-negative.
    Ok(if zero_center {
        scaled.clamp(-limit, limit)?
    } else {
        scaled.minimum(limit)?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sc.pp.normalize_total` then `sc.pp.log1p` on the shared 6 x 5 reference
    /// matrix; the input to the scanpy `scale` calls below.
    const LOGGED: [f32; 30] = [
        1.0296195, 0.0, 0.0, 0.64185387, 1.0296195, //
        0.0, 0.0, 0.0, 0.0, 1.704748, //
        1.3862944, 0.0, 0.5596158, 0.0, 0.5596158, //
        1.178655, 0.7537718, 0.0, 0.0, 0.7537718, //
        0.0, 0.0, 1.178655, 0.0, 1.178655, //
        1.0296195, 0.0, 0.64185387, 0.64185387, 0.64185387,
    ];

    /// `sc.pp.scale(LOGGED)`.
    const SCALED: [f32; 30] = [
        0.42367423,
        -0.40824828,
        -0.8199774,
        1.2909944,
        0.12093413, //
        -1.2610966,
        -0.40824828,
        -0.8199774,
        -0.6454972,
        1.7039754, //
        1.007303,
        -0.40824828,
        0.33678293,
        -0.6454972,
        -0.9811303, //
        0.6675418,
        2.0412414,
        -0.8199774,
        -0.6454972,
        -0.5258734, //
        -1.2610966,
        -0.40824828,
        1.6163752,
        -0.6454972,
        0.47039267, //
        0.42367423,
        -0.40824828,
        0.50677407,
        1.2909944,
        -0.7882985,
    ];

    /// `sc.pp.scale(LOGGED, max_value=1.0)`.
    const SCALED_CLIPPED_1: [f32; 30] = [
        0.42367423,
        -0.40824828,
        -0.8199774,
        1.0,
        0.12093413, //
        -1.0,
        -0.40824828,
        -0.8199774,
        -0.6454972,
        1.0, //
        1.0,
        -0.40824828,
        0.33678293,
        -0.6454972,
        -0.9811303, //
        0.6675418,
        1.0,
        -0.8199774,
        -0.6454972,
        -0.5258734, //
        -1.0,
        -0.40824828,
        1.0,
        -0.6454972,
        0.47039267, //
        0.42367423,
        -0.40824828,
        0.50677407,
        1.0,
        -0.7882985,
    ];

    /// `sc.pp.scale(LOGGED, zero_center=False)`.
    const SCALED_NO_CENTER: [f32; 30] = [
        1.6847708, 0.0, 0.0, 1.9364916, 2.4142513, //
        0.0, 0.0, 0.0, 0.0, 3.9972925, //
        2.2683995, 0.0, 1.1567603, 0.0, 1.312187, //
        1.9286385, 2.4494896, 0.0, 0.0, 1.7674438, //
        0.0, 0.0, 2.4363525, 0.0, 2.76371, //
        1.6847708, 0.0, 1.3267515, 1.9364916, 1.5050187,
    ];

    const RTOL: f32 = 1e-5;

    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= RTOL * e.abs().max(1.0),
                "element {i}: {a} != {e}"
            );
        }
    }

    #[test]
    fn hand_checked_standardisation() {
        // One gene with values 1, 3: mean 2, corrected deviation sqrt(2).
        let matrix = CsrMatrix::from_dense(&[1.0, 3.0], 2, 1).unwrap();
        let out = scale(&matrix, true, None, &Device::Cpu).unwrap();
        let root_two = 2.0_f32.sqrt();
        assert_close(&flat(&out), &[-1.0 / root_two, 1.0 / root_two]);
    }

    #[test]
    fn constant_gene_and_all_zero_row_stay_zero() {
        // Gene 0 is constant, gene 1 is all zero, and cell 2 has no counts.
        let matrix = CsrMatrix::from_dense(&[7.0, 0.0, 7.0, 0.0, 7.0, 0.0], 3, 2).unwrap();
        let out = flat(&scale(&matrix, true, None, &Device::Cpu).unwrap());
        assert_eq!(out, vec![0.0; 6]);
    }

    #[test]
    fn without_centering_a_constant_gene_keeps_its_value() {
        let matrix = CsrMatrix::from_dense(&[7.0, 7.0], 2, 1).unwrap();
        let out = flat(&scale(&matrix, false, None, &Device::Cpu).unwrap());
        assert_eq!(out, vec![7.0, 7.0]);
    }

    #[test]
    fn matches_scanpy_scale() {
        let matrix = CsrMatrix::from_dense(&LOGGED, 6, 5).unwrap();
        assert_close(
            &flat(&scale(&matrix, true, None, &Device::Cpu).unwrap()),
            &SCALED,
        );
    }

    #[test]
    fn matches_scanpy_scale_with_clipping() {
        let matrix = CsrMatrix::from_dense(&LOGGED, 6, 5).unwrap();
        assert_close(
            &flat(&scale(&matrix, true, Some(1.0), &Device::Cpu).unwrap()),
            &SCALED_CLIPPED_1,
        );
    }

    #[test]
    fn matches_scanpy_scale_without_centering() {
        let matrix = CsrMatrix::from_dense(&LOGGED, 6, 5).unwrap();
        assert_close(
            &flat(&scale(&matrix, false, None, &Device::Cpu).unwrap()),
            &SCALED_NO_CENTER,
        );
    }

    #[test]
    fn the_resolved_device_agrees_with_the_cpu() {
        // Falls back to the CPU where there is no Metal device, so it always runs.
        let device = crate::DeviceKind::Auto.resolve().unwrap();
        let matrix = CsrMatrix::from_dense(&LOGGED, 6, 5).unwrap();
        assert_close(
            &flat(&scale(&matrix, true, Some(1.0), &device).unwrap()),
            &SCALED_CLIPPED_1,
        );
    }

    #[test]
    fn clipping_without_centering_only_bounds_from_above() {
        let matrix = CsrMatrix::from_dense(&LOGGED, 6, 5).unwrap();
        let out = flat(&scale(&matrix, false, Some(2.0), &Device::Cpu).unwrap());
        let expected: Vec<f32> = SCALED_NO_CENTER.iter().map(|v| v.min(2.0)).collect();
        assert_close(&out, &expected);
    }

    #[test]
    fn rejects_a_single_cell() {
        let matrix = CsrMatrix::from_dense(&[1.0, 2.0], 1, 2).unwrap();
        assert!(scale(&matrix, true, None, &Device::Cpu).is_err());
    }
}
