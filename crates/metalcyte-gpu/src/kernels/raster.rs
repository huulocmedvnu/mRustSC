//! A point-cloud renderer on the GPU: one render pass of point primitives into an
//! offscreen RGBA8 texture in unified memory, read back as bytes.
//!
//! The vertex shader maps each point through the viewport and sets its size; the
//! fragment shader draws an anti-aliased disc with the same coverage function as the
//! renderer on the cores and blends it over what is there. A million points take a few
//! milliseconds; the result is a bitmap a notebook shows as one image.

use metal::{
    MTLBlendFactor, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLResourceOptions, MTLStorageMode, MTLStoreAction, MTLTextureUsage, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState, TextureDescriptor,
};
use metalcyte_core::error::{Error, Result};
use metalcyte_core::raster::RenderSpec;

use crate::context::MetalContext;

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    scale: [f32; 2],
    offset: [f32; 2],
    point_size: f32,
    _pad: [f32; 3],
}

/// The render pipeline, built once per context: compiling the shaders costs more than
/// drawing a million points, so a caller plotting repeatedly must not pay it each time.
fn pipeline(context: &MetalContext) -> Result<RenderPipelineState> {
    thread_local! {
        static PIPELINE: std::cell::RefCell<Option<(usize, RenderPipelineState)>> =
            const { std::cell::RefCell::new(None) };
    }
    let key = context as *const MetalContext as usize;
    if let Some((cached_key, state)) = PIPELINE.with(|slot| slot.borrow().clone()) {
        if cached_key == key {
            return Ok(state);
        }
    }
    let state = build_pipeline(context)?;
    PIPELINE.with(|slot| *slot.borrow_mut() = Some((key, state.clone())));
    Ok(state)
}

fn build_pipeline(context: &MetalContext) -> Result<RenderPipelineState> {
    let kernel = |message: String| Error::Kernel {
        name: "point_vertex",
        message,
    };
    let library = context
        .device()
        .new_library_with_source(SOURCE, &metal::CompileOptions::new())
        .map_err(kernel)?;
    let vertex = library.get_function("point_vertex", None).map_err(kernel)?;
    let fragment = library
        .get_function("point_fragment", None)
        .map_err(kernel)?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    let attachment = descriptor
        .color_attachments()
        .object_at(0)
        .expect("attachment 0");
    attachment.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    attachment.set_blending_enabled(true);
    attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
    attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
    attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
    attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
    context
        .device()
        .new_render_pipeline_state(&descriptor)
        .map_err(kernel)
}

/// Draw `xy` (`(n, 2)` flat) with one `0xRRGGBBAA` colour per point into an RGBA8 image,
/// row-major from the top, `width * height * 4` bytes.
pub fn render_points(
    context: &MetalContext,
    xy: &[f32],
    rgba: &[u32],
    spec: &RenderSpec,
) -> Result<Vec<u8>> {
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
    let pipeline = pipeline(context)?;

    let texture_descriptor = TextureDescriptor::new();
    texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_descriptor.set_width(w as u64);
    texture_descriptor.set_height(h as u64);
    texture_descriptor.set_storage_mode(MTLStorageMode::Shared);
    texture_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let texture = context.device().new_texture(&texture_descriptor);

    let bg = spec.background;
    let channel = |shift: u32| f64::from((bg >> shift) & 0xff) / 255.0;
    let pass = RenderPassDescriptor::new();
    let attachment = pass.color_attachments().object_at(0).expect("attachment 0");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(
        channel(24),
        channel(16),
        channel(8),
        channel(0),
    ));

    let v = spec.viewport;
    let uniforms = Uniforms {
        // Data to clip space: x in [x_min, x_max] -> [-1, 1], y likewise (y up).
        scale: [
            2.0 / (v.x_max - v.x_min).max(1e-12),
            2.0 / (v.y_max - v.y_min).max(1e-12),
        ],
        offset: [
            -1.0 - 2.0 * v.x_min / (v.x_max - v.x_min).max(1e-12),
            -1.0 - 2.0 * v.y_min / (v.y_max - v.y_min).max(1e-12),
        ],
        point_size: spec.point_size.max(1.0),
        _pad: [0.0; 3],
    };

    let positions = context.device().new_buffer_with_data(
        xy.as_ptr() as *const std::ffi::c_void,
        (xy.len().max(2) * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let colours = context.device().new_buffer_with_data(
        rgba.as_ptr() as *const std::ffi::c_void,
        (rgba.len().max(1) * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command = context.queue().new_command_buffer();
    let encoder = command.new_render_command_encoder(pass);
    encoder.set_render_pipeline_state(&pipeline);
    encoder.set_vertex_buffer(0, Some(&positions), 0);
    encoder.set_vertex_buffer(1, Some(&colours), 0);
    encoder.set_vertex_bytes(
        2,
        size_of::<Uniforms>() as u64,
        &uniforms as *const Uniforms as *const std::ffi::c_void,
    );
    encoder.set_fragment_bytes(
        2,
        size_of::<Uniforms>() as u64,
        &uniforms as *const Uniforms as *const std::ffi::c_void,
    );
    if n > 0 {
        encoder.draw_primitives(MTLPrimitiveType::Point, 0, n as u64);
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != metal::MTLCommandBufferStatus::Completed {
        return Err(Error::Kernel {
            name: "point_vertex",
            message: format!("render ended in state {:?}", command.status()),
        });
    }
    let mut image = vec![0u8; w * h * 4];
    texture.get_bytes(
        image.as_mut_ptr() as *mut std::ffi::c_void,
        (w * 4) as u64,
        MTLRegion {
            origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal::MTLSize {
                width: w as u64,
                height: h as u64,
                depth: 1,
            },
        },
        0,
    );
    Ok(image)
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float2 scale;
    float2 offset;
    float point_size;
    float3 pad;
};

struct PointOut {
    float4 position [[position]];
    float size [[point_size]];
    float4 colour;
};

vertex PointOut point_vertex(device const float2 *xy [[buffer(0)]],
                             device const uint *rgba [[buffer(1)]],
                             constant Uniforms &u [[buffer(2)]],
                             uint id [[vertex_id]]) {
    PointOut out;
    const float2 p = xy[id];
    out.position = float4(p * u.scale + u.offset, 0.0, 1.0);
    // The disc is drawn into a sprite one pixel wider on each side for the edge.
    out.size = u.point_size + 1.0;
    const uint c = rgba[id];
    out.colour = float4((c >> 24) & 0xff, (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff) / 255.0;
    if (!all(isfinite(p))) {
        out.position = float4(2.0, 2.0, 0.0, 1.0); // off screen
    }
    return out;
}

fragment float4 point_fragment(PointOut in [[stage_in]],
                               float2 coord [[point_coord]],
                               constant Uniforms &u [[buffer(2)]]) {
    // The sprite is point_size + 1 pixels wide (see the vertex stage); the size is
    // taken from the uniforms because a [[point_size]] output is not readable here.
    const float r = max(u.point_size * 0.5, 0.5);
    const float d = length((coord - 0.5) * (u.point_size + 1.0));
    const float cover = clamp(r + 0.5 - d, 0.0, 1.0);
    const float a = cover * in.colour.a;
    if (a <= 0.0) {
        discard_fragment();
    }
    return float4(in.colour.rgb, a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use metalcyte_core::raster::{self, Viewport};

    #[test]
    fn matches_the_renderer_on_the_cores() {
        let Ok(context) = MetalContext::new() else {
            return;
        };
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 5000;
        let xy: Vec<f32> = (0..2 * n).map(|_| (next() * 100.0) as f32).collect();
        let rgba: Vec<u32> = (0..n)
            .map(|i| [0xe4572eff, 0x2f6fd0ff, 0x3cb371c0][i % 3])
            .collect();
        let spec = RenderSpec {
            width: 400,
            height: 300,
            point_size: 4.0,
            viewport: Viewport::fitting(&xy, 0.02),
            background: 0xffffffff,
        };
        let cores = raster::render_points(&xy, &rgba, &spec).unwrap();
        let device = render_points(&context, &xy, &rgba, &spec).unwrap();
        assert_eq!(cores.len(), device.len());
        // The two rasterisers differ by sub-pixel sampling at disc edges, so compare the
        // images as a whole: mean absolute difference per channel under 2 of 255, and no
        // pixel that one leaves at the background and the other paints solid.
        let mean: f64 = cores
            .iter()
            .zip(&device)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .sum::<f64>()
            / cores.len() as f64;
        assert!(mean < 2.0, "mean channel difference {mean}");
        let solid_mismatch = cores
            .chunks(4)
            .zip(device.chunks(4))
            .filter(|(a, b)| {
                let a_bg = a[0] == 255 && a[1] == 255 && a[2] == 255;
                let b_bg = b[0] == 255 && b[1] == 255 && b[2] == 255;
                a_bg != b_bg && (a[0] as i32 - b[0] as i32).abs() > 128
            })
            .count();
        assert!(
            solid_mismatch < n / 100,
            "{solid_mismatch} pixels disagree outright"
        );
    }
}
