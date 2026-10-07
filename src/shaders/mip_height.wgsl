struct Params {
	src_size: vec2<u32>,
	dst_size: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<u32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32uint, write>;


@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id): gid: vec3<u32>) {
	if (any(gid.xy >= params.dst_size)) { return; }
	let h = textureLoad(src, gid.xy * 2u, 0).r;
	textureStore(dst, gid.xy, vec4<u32>(h, 0u,0u, 0u));
}

