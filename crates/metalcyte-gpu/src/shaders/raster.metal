
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
