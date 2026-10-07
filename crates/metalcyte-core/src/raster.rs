//! Rasterising a point cloud into an image on the cores: the reference for, and the
//! fallback behind, the Metal renderer in `metalcyte-gpu`.
//!
//! A million cells drawn through a general plotting library are a million path objects;
//! here they are a million splats into one RGBA bitmap, which any notebook or figure
//! shows as a single image. Points are drawn in input order with "over" compositing, as
//! a scatter plot draws them, so the two renderers agree to the rounding of the blend.

use crate::error::{Error, Result};

/// The data window mapped onto the image: `x_min..x_max` across, `y_min..y_max` up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
}

impl Viewport {
    /// The window that holds every point, with `margin` of its span added on each side.
    pub fn fitting(xy: &[f32], margin: f32) -> Self {
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        for p in xy.chunks_exact(2) {
            x_min = x_min.min(p[0]);
            x_max = x_max.max(p[0]);
            y_min = y_min.min(p[1]);
            y_max = y_max.max(p[1]);
        }
        if !x_min.is_finite() {
            return Self {
                x_min: 0.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 1.0,
            };
        }
        let (dx, dy) = ((x_max - x_min).max(1e-6), (y_max - y_min).max(1e-6));
        Self {
            x_min: x_min - margin * dx,
            x_max: x_max + margin * dx,
            y_min: y_min - margin * dy,
            y_max: y_max + margin * dy,
        }
    }
}

/// What a render needs besides the points.
#[derive(Debug, Clone, Copy)]
pub struct RenderSpec {
    pub width: usize,
    pub height: usize,
    /// Point diameter in pixels.
    pub point_size: f32,
    pub viewport: Viewport,
    /// Background as `0xRRGGBBAA`.
    pub background: u32,
}

fn unpack(rgba: u32) -> [f32; 4] {
    [
        ((rgba >> 24) & 0xff) as f32 / 255.0,
        ((rgba >> 16) & 0xff) as f32 / 255.0,
        ((rgba >> 8) & 0xff) as f32 / 255.0,
        (rgba & 0xff) as f32 / 255.0,
    ]
}

/// Coverage of a pixel centre at distance `d` from a point's centre of radius `r`: one
/// inside, zero outside, a one-pixel linear edge between. The Metal fragment shader uses
/// the same function.
#[inline]
pub fn coverage(d: f32, r: f32) -> f32 {
    ((r + 0.5 - d).clamp(0.0, 1.0)).min(1.0)
}

/// Draw `xy` (`(n, 2)` flat) with one `0xRRGGBBAA` colour per point into an RGBA8 image,
/// row-major from the top, `width * height * 4` bytes.
pub fn render_points(xy: &[f32], rgba: &[u32], spec: &RenderSpec) -> Result<Vec<u8>> {
    let n = xy.len() / 2;
    if rgba.len() != n {
        return Err(Error::shape(
            format!("one colour per point ({n})"),
            format!("{} colours", rgba.len()),
        ));
    }
    if spec.width == 0 || spec.height == 0 {
        return Err(Error::parameter("image size", "at least 1 x 1", 0));
    }
    let (w, h) = (spec.width, spec.height);
    let v = spec.viewport;
    let sx = w as f32 / (v.x_max - v.x_min).max(1e-12);
    let sy = h as f32 / (v.y_max - v.y_min).max(1e-12);
    let r = (spec.point_size / 2.0).max(0.5);
    let bg = unpack(spec.background);
    let mut image: Vec<[f32; 4]> = vec![bg; w * h];
    let reach = r.ceil() as i64 + 1;
    for (p, &colour) in xy.chunks_exact(2).zip(rgba) {
        let c = unpack(colour);
        if c[3] <= 0.0 {
            continue;
        }
        let px = (p[0] - v.x_min) * sx;
        let py = (v.y_max - p[1]) * sy;
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        let (cx, cy) = (px.floor() as i64, py.floor() as i64);
        for yy in (cy - reach).max(0)..(cy + reach + 1).min(h as i64) {
            for xx in (cx - reach).max(0)..(cx + reach + 1).min(w as i64) {
                let d = ((xx as f32 + 0.5 - px).powi(2) + (yy as f32 + 0.5 - py).powi(2)).sqrt();
                let a = coverage(d, r) * c[3];
                if a <= 0.0 {
                    continue;
                }
                let pixel = &mut image[yy as usize * w + xx as usize];
                for k in 0..3 {
                    pixel[k] = c[k] * a + pixel[k] * (1.0 - a);
                }
                pixel[3] = a + pixel[3] * (1.0 - a);
            }
        }
    }
    Ok(image
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_lands_where_the_viewport_puts_it() {
        let spec = RenderSpec {
            width: 10,
            height: 10,
            point_size: 1.0,
            viewport: Viewport {
                x_min: 0.0,
                x_max: 10.0,
                y_min: 0.0,
                y_max: 10.0,
            },
            background: 0xffffffff,
        };
        // Data (2.5, 7.5) is column 2, row 2 from the top.
        let image = render_points(&[2.5, 7.5], &[0xff0000ff], &spec).unwrap();
        let at = |x: usize, y: usize| &image[(y * 10 + x) * 4..(y * 10 + x) * 4 + 4];
        assert_eq!(at(2, 2), &[255, 0, 0, 255]);
        assert_eq!(at(7, 7), &[255, 255, 255, 255]);
    }

    #[test]
    fn alpha_composites_over_the_background() {
        let spec = RenderSpec {
            width: 1,
            height: 1,
            point_size: 1.0,
            viewport: Viewport {
                x_min: 0.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 1.0,
            },
            background: 0x000000ff,
        };
        let image = render_points(&[0.5, 0.5], &[0xffffff80], &spec).unwrap();
        assert!((image[0] as i32 - 128).abs() <= 1, "{:?}", image);
    }

    #[test]
    fn fitting_viewport_holds_every_point() {
        let v = Viewport::fitting(&[0.0, 0.0, 4.0, 2.0], 0.0);
        assert_eq!((v.x_min, v.x_max, v.y_min, v.y_max), (0.0, 4.0, 0.0, 2.0));
    }
}
