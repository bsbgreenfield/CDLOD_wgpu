struct Params {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(0.04045));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3(0.0031308));
}

fn tent(i: i32) -> f32 {
    return select(0.25, 0.5, i == 0);
}

// tent filter
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= params.dst_size)) { return; }
    let center = vec2<i32>(gid.xy * 2u);
    let max_coord = vec2<i32>(params.src_size) - 1;
    var sum = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureLoad(src, clamp(center + vec2(x, y), vec2(0), max_coord), 0);
            sum += vec4(to_linear(s.rgb), s.a) * (tent(x) * tent(y));
        }
    }
    textureStore(dst, gid.xy, vec4(to_srgb(sum.rgb), sum.a));
}
