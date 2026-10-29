enable wgpu_mesh_shader;

const MAX_ROWS: u32 = 4u; 
const BLADE_COUNT: u32 = 32u;
const GRASS_VERT_COUNT: u32 = BLADE_COUNT * MAX_ROWS * 2u;
const GRASS_PRIM_COUNT: u32 = BLADE_COUNT * (MAX_ROWS - 1u) * 2u;
const GUTTER: u32 = 1u;

const SUN: vec3<f32> = vec3<f32>(0.4, 0.8, 0.3);
const ROOT_COLOR: vec3<f32> = vec3<f32>(0.05, 0.12, 0.02);
const TIP_A: vec3<f32> = vec3<f32>(0.25, 0.45, 0.08);
const TIP_B: vec3<f32> = vec3<f32>(0.40, 0.50, 0.12);
struct Camera {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    pos: vec3<f32>,
}

struct GrassParams {
    map_min: vec3<f32>,
    slot_shift: u32,
    scale_factors: vec3<f32>,
    nodes_per_tile_axis: u32,
    node_size: u32,
    tile_texels: u32,
    grass_range: f32,   // world distance where grass is fully gone
    fade_width: f32,    // distance over which blades thin out before grass_range
    blade_height: f32,
    blade_width: f32,
    bend: f32,
    jitter: f32,        // 0..1, fraction of a blade cell a blade may wander
    cell_subdiv: u32,   // S: mesh groups per cell axis, blades per cell = 32 * S^2
	lod_near: f32,
    lod_far: f32,
    _pad: u32,
}

struct SelectedNode {
    xz: u32,
    info: u32,
}

struct GrassNodes {
    groups_x: u32,
    groups_y: u32,
    groups_z: u32,
    nodes: array<SelectedNode>,
}

struct GrassBox {
	min: vec3<f32>,
	max: vec3<f32>,
}


fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand01(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}
fn bayer_cell(r: u32) -> vec2<u32> {
    let b0 = r & 1u;
    let b1 = (r >> 1u) & 1u;
    let b2 = (r >> 2u) & 1u;
    let b3 = (r >> 3u) & 1u;
    let b4 = (r >> 4u) & 1u;
    let b5 = (r >> 5u) & 1u;
    return vec2<u32>(
        ((b0 ^ b1) << 2u) | ((b2 ^ b3) << 1u) | (b4 ^ b5),
        (b0 << 2u) | (b2 << 1u) | b4,
    );
}

fn grass_box(q0: vec2<f32>, len: f32, h: vec4<f32>) -> GrassBox {
    let sf = params.scale_factors;
    let lo = min(min(h.x, h.y), min(h.z, h.w));
    let hi = max(max(h.x, h.y), max(h.z, h.w));
    let mn = params.map_min + vec3<f32>(q0.x, lo, q0.y) * sf;
    let mx = params.map_min + vec3<f32>(q0.x + len, hi, q0.y + len) * sf + vec3(0.0, params.blade_height, 0.0);
    return GrassBox(mn, mx);
}


fn load_cell_heights(node_coords: vec2<u32>, tile_slot: u32, cell_coords: vec2<u32>) -> vec4<f32> {
   	let slot_mask = (1u << params.slot_shift) - 1u;
    let slot_origin = vec2<u32>(tile_slot & slot_mask, tile_slot >> params.slot_shift) * params.tile_texels;
    let tex_coords: vec2<u32> =  ( slot_origin + GUTTER + (node_coords % params.nodes_per_tile_axis) * params.node_size ) + cell_coords;

	
 return vec4<f32>(
        f32(textureLoad(atlas, tex_coords, 0).r),
        f32(textureLoad(atlas, tex_coords + vec2<u32>(1u, 0u), 0).r),
        f32(textureLoad(atlas, tex_coords + vec2<u32>(0u, 1u), 0).r),
        f32(textureLoad(atlas, tex_coords + vec2<u32>(1u, 1u), 0).r),
    );
}

struct GrassVertex {
    @builtin(position) clip: vec4<f32>,
    @location(1) color: vec3<f32>,
}
struct GrassPrim {
    @builtin(triangle_indices) indices: vec3<u32>,
}
struct GrassOutput {
    @builtin(vertices) vertices: array<GrassVertex, GRASS_VERT_COUNT>,
    @builtin(primitives) primitives: array<GrassPrim, GRASS_PRIM_COUNT>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}
@group(0) @binding(0) var<uniform>       camera:   Camera;
@group(0) @binding(1) var<uniform>       params:   GrassParams;
@group(0) @binding(2) var<storage, read> grass:    GrassNodes;
@group(0) @binding(3) var                atlas:    texture_2d<u32>;

var<workgroup> mesh_out: GrassOutput;

@mesh(mesh_out)
@workgroup_size(128)
fn ms_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {

	let node = grass.nodes[wg.y];
	let node_coords = vec2<u32>(node.xz >> 16u, node.xz & 0xFFFFu);
	let quads_axis = params.cell_subdiv * params.node_size;
	let quad = vec2<u32>(wg.x % quads_axis, wg.x / quads_axis); // subdiv quad
	let node_cell_coords: vec2<u32> = quad  / params.cell_subdiv; // node cell made up of 4 samples
	let sub_coords = quad % params.cell_subdiv; // which sub division within the cell
	let cell_sample_tl = node_coords * params.node_size + node_cell_coords; // top left corner of the node cell in sample space


	let cell_heights: vec4<f32> = load_cell_heights(node_coords, node.info & 0xFFFFu, node_cell_coords);
	let h00 = cell_heights.x;
    let h10 = cell_heights.y;
    let h01 = cell_heights.z;
    let h11 = cell_heights.w;

	let inv_sub = 1.0 / f32(params.cell_subdiv);
	let box = grass_box(vec2<f32>(cell_sample_tl) + vec2<f32>(sub_coords) * inv_sub, inv_sub, cell_heights);

	let dist = distance(camera.pos, clamp(camera.pos, box.min, box.max));

	let density = saturate((params.grass_range - dist ) / params.fade_width);

	let keep = u32(ceil(density * f32(BLADE_COUNT)));

	let rows = select(select(2u, 3u, dist < params.lod_far), 4u, dist < params.lod_near);

	let verts_per_blade = rows * 2u;
	let prims_per_blade = (rows - 1u) * 2u;


	if ( li == 0u) {
		mesh_out.vertex_count = keep * verts_per_blade;
		mesh_out.primitive_count = keep * prims_per_blade;
	}

	let blade = li >> 2u;
	let row = li & 3u;
	if (blade >= keep || row >= rows) {
		return;
	}
  	let sf = params.scale_factors;
    let seed = pcg(cell_sample_tl.x * params.cell_subdiv + sub_coords.x) ^ pcg((cell_sample_tl.y * params.cell_subdiv + sub_coords.y) ^ 0x9E3779B9u);
    let r0 = pcg(seed ^ blade);
    let r1 = pcg(r0);
    let r2 = pcg(r1);
    let r3 = pcg(r2);

    let g = vec2<f32>(bayer_cell(blade));
    let jitter = (vec2<f32>(rand01(r0), rand01(r1)) - 0.5) * params.jitter;
    let uv = (vec2<f32>(sub_coords) + (g + 0.5 + jitter) * 0.125) * inv_sub; // 0..1 in cell
    var grad: vec2<f32>;

    var h0 = h00;

    if (((node_cell_coords.x + node_cell_coords.y) & 1u) == 0u) {
        // diagonal TR-BL: (TL, BL, TR) | (TR, BL, BR)
        if (uv.x + uv.y <= 1.0) {
            grad = vec2<f32>(h10 - h00, h01 - h00);
        } else {
            grad = vec2<f32>(h11 - h01, h11 - h10);
            h0 = h10 + h01 - h11;
        }
    } else {
        // diagonal TL-BR: (TL, BL, BR) | (TL, BR, TR)
        if (uv.x >= uv.y) {
            grad = vec2<f32>(h10 - h00, h11 - h10);
        } else {
            grad = vec2<f32>(h11 - h01, h01 - h00);
        }
    }

    let height_units = h0 + dot(grad, uv);
    let root = params.map_min + vec3<f32>(f32(cell_sample_tl.x) + uv.x, height_units, f32(cell_sample_tl.y) + uv.y) * sf;
    let normal = normalize(vec3<f32>(-grad.x * sf.y / sf.x, 1.0, -grad.y * sf.y / sf.z));
	
    // grow mostly vertical, leaning a little with the slope
    let up = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), normal, 0.3));
    let yaw = rand01(r2) * 6.2831853;
    let side = normalize(cross(up, vec3<f32>(sin(yaw), 0.0, cos(yaw)))) * (0.5 * params.blade_width);

	let bend_dir = normalize(cross(side, up));
	let bend_amount = params.bend * (0.5 + rand01(r1));

    let grow = saturate(density * f32(BLADE_COUNT) - f32(blade));
    let height = params.blade_height * (0.7 + 0.6 * rand01(r3)) * grow;

	let light = 0.35 + 0.65 * max(dot(normal, normalize(SUN)), 0.0);
	let tip = mix(TIP_A, TIP_B, rand01(r2));

	let height_frac = f32(row) / f32(rows - 1u);
	let center = root + up * (height * height_frac) + bend_dir * (bend_amount * height * height_frac * height_frac);

	let c = camera.view_proj * vec4<f32>(center, 1.0);
	let w = camera.view_proj * vec4<f32>(side * (1.0 - 0.85 * height_frac), 0.0);
	let color = mix(ROOT_COLOR, tip, height_frac) * light;

	let v_offset = blade * verts_per_blade + row * 2u;
	mesh_out.vertices[v_offset] = GrassVertex(c - w, color); // first vertex of this row
	mesh_out.vertices[v_offset + 1u] = GrassVertex(c + w, color); // second of this row

	if (row + 1u < rows) {
		let p = blade * prims_per_blade + row * 2u;
		mesh_out.primitives[p].indices = vec3<u32>(v_offset, v_offset + 1u, v_offset + 2u);
		mesh_out.primitives[p + 1u].indices = vec3<u32>(v_offset + 2u, v_offset + 1u, v_offset + 3u);
	}
}

@fragment
fn fs_main(in: GrassVertex) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
