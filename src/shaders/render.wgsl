
struct Camera {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    pos: vec3<f32>,
}

struct RenderParams {
    map_min: vec3<f32>,
    slot_shift: u32, // the power of two factor of the tile per axis count
    scale_factors: vec3<f32>, // sample space -> world space
    nodes_per_tile_axis: u32,
    texel_scale_factor: vec2<f32>, 
    node_size: u32,           // number of samples per axis for every node
    tile_texels: u32,        // tile_size + 1
}

struct SelectedNode {
	xz: u32,
	info: u32,
}



struct SelectedNodes {
 	index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
    nodes:          array<SelectedNode>,	
}

const GUTTER: u32 = 1u;

@group(0) @binding(0) var<uniform>       camera:   Camera;
@group(0) @binding(1) var<uniform>       params:   RenderParams;
@group(0) @binding(2) var<storage, read> selected: SelectedNodes;
@group(0) @binding(3) var                atlas:    texture_2d<u32>; 
@group(0) @binding(4) var                diffuse:  texture_2d<f32>;
@group(0) @binding(5) var                diffuse_sampler: sampler;


struct VsOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
}


@vertex
fn vs_main(@builtin(instance_index) instance_idx: u32, @location(0) v_coords: vec2<u32>) -> VsOut {

	// unpack data from node
	let selected_node = selected.nodes[instance_idx];
	let lod = selected_node.info >> 16u;
	let tile_slot = selected_node.info & 0xFFFFu;
	let node_coords = vec2<u32>(selected_node.xz >> 16u, node.xz & 0xFFFFu);

	// use data to sample texture atlas
	let slot_mask = (1u << params.slot_shift) - 1u;
	let slot_origin = vec2<u32>(tile_slot & slot_mask, tile_slot >> params.slot_shift) * params.tile_texels;
	let tile_off = (node_coords % params.nodes_per_tile_axis) * params.node_size + v_coords;
	let texel = slot_origin + GUTTER + tile_off;
	let height_val = textureLoad(atlas, texel, 0).r;

	// lod space:    one unit = one node
	// mip space:    one unit = one decimated sample
	// sample space: one unit = one source sample
	// world space:  one unit = one world unit
	// lod space -> mip space -> sample space -> world space
	let mip_coords = node_coords * params.node_size + v_coords;
	let sample_coords = mip_coords << vec2<u32>(lod);
	let world_coords = params.map_min + vec3<f32>(sample_coords.x, f32(height_val), sample_coords.y) * params.scale_factors;

	var out: VsOut;
	out.clip = camera.view_proj * vec4<f32>(world_coords, 1.0);
	out.uv = (f32( sample_coords ) + 0.5) * params.texel_scale_factor;

	return out;
}

@fragment
	fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
	return textureSample(diffuse, diffuse_sampler, in.uv);
}
