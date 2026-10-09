
// definitions: 
// LOD space: one unit = one quad node
// tile space: one unit = one tile 
// sample space: one unit = one sample
// world space: one unit = one world unit


struct LODArgsIn {
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    count: u32,

}

struct LODArgsOut {
    wg_x: atomic<u32>,
    wg_y: u32,
    wg_z: u32,
    count: atomic<u32>,
}

struct RenderIndirectArgs {
    index_count: u32, 
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,

}

struct Camera {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    pos: vec3<f32>,
}

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct NodeWork {
    xz: u32, // IN LOD SPACE (one unit = one node)
    lod: u32,
}

struct SelectedNode {
    xz: u32, 
    info: u32, // lod | tile_slot
}
struct SelectedNodes {
    vertex_count:   u32,
    instance_count: atomic<u32>,   // still at offset 4
    first_vertex:   u32,
    first_instance: u32,
    nodes:          array<SelectedNode>,
}

struct LevelInfo {
    node_size: u32,
    lod_range: f32,
	_pad: vec2<u32>,
}

struct BakeValues {
    map_min: vec3<f32>,
    max_mip: u32,
    scale_factors: vec3<f32>, // calculated as scales to convert from sample space to world space
    nodes_per_tile_axis: u32, // must be even
    root_tiles: vec2<u32>, // always resident
    queue_capacity: u32,
    selected_capacity: u32,
    request_capacity: u32,
}

struct RequestList {
    count: atomic<u32>,
    tiles: array<u32>,
}

struct InQueue {
    args: LODArgsIn,
    queue: array<NodeWork>,
}
struct OutQueue {
    args: LODArgsOut,
    queue: array<NodeWork>,
}
const NOT_RESIDENT: u32 = 0xFFFFu;
const WG_SIZE: u32 = 64u;
const MAX_LODS = 16u;

@group(0) @binding(0) var<storage, read>       in_queue:   InQueue; // dispatch args for this dispatch  + scratch space
@group(0) @binding(1) var<storage, read_write> out_queue:  OutQueue;  // next level's args

@group(1) @binding(0) var<uniform>             camera:          Camera;

@group(2) @binding(0) var                      min_max_heights: texture_2d<u32>; // mipmap of height vals 
@group(2) @binding(1) var<uniform>             bake_values:     BakeValues;
@group(2) @binding(2) var<uniform>             levels:          array<LevelInfo, MAX_LODS>;

@group(3) @binding(0) var<storage, read_write> selected_nodes:  SelectedNodes;
@group(3) @binding(1) var                      residency:       texture_2d<u32>; // residency array for tiles
@group(3) @binding(2) var<storage, read_write> request_flags:   array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> requests:        RequestList;        


fn frustum_intersects(aabb: AABB) -> bool {
  for (var i = 0u; i < 6u; i++) {
        let p = camera.frustum_planes[i];
        let v = select(aabb.min, aabb.max, p.xyz >= vec3(0.0)); // corner furthest along n
        if (dot(p.xyz, v) + p.w < 0.0) {
            return false;
        }
    }
    return true;
}

fn get_tile_residency(mip: u32, t_coords: vec2<u32>) -> u32 {
    return textureLoad(residency, t_coords, mip).r;
}

fn tile_flat_index(mip: u32, t_coords: vec2<u32>) -> u32 {
    let mips_coarser = bake_values.max_mip - mip;
    let roots = bake_values.root_tiles.x * bake_values.root_tiles.y;
    let base_offset = roots * ((1u << (2u * mips_coarser)) - 1u) / 3u;
    let tiles_x = bake_values.root_tiles.x << mips_coarser;
    return base_offset + t_coords.y * tiles_x + t_coords.x;
}

fn request_tile(mip: u32, t_coords: vec2<u32>) {
    let tile_index = tile_flat_index(mip, t_coords);
    // set request flag, exit if already requested
    if (atomicExchange(&request_flags[tile_index], 1u) == 0u) {
        // increment request count
        let i = atomicAdd(&requests.count, 1u);
        // set request
        if (i < bake_values.request_capacity) { 
            requests.tiles[i] = tile_index; 
            }
    }
}

fn children_resident(lod: u32, x: u32, z: u32) -> bool {
    let t_coords = (vec2<u32>(x, z) * 2u) / bake_values.nodes_per_tile_axis;
    let residency = get_tile_residency(lod - 1u, t_coords);

    // request children's tile
    if residency == NOT_RESIDENT {
        request_tile(lod - 1u, t_coords);
        return false;
    }
    return true;
}


// using the frustum (just the camera distance), the lod, and the node aabb check whether the aabb interects
fn intersects_sphere(aabb: AABB, center: vec3<f32>, radius: f32) -> bool {
  let d = clamp(center, aabb.min, aabb.max) - center;
  return dot(d, d) <= radius * radius;
}

fn in_lod_range(lod: u32, aabb: AABB) -> bool {
    return intersects_sphere(aabb, camera.pos, levels[lod].lod_range);
}

fn get_aabb(lod: u32, minH: u32, maxH: u32, x: u32, z: u32) -> AABB {
    // get min and max with bit shifts
    let node_size = levels[lod].node_size;
    let s0 = vec2<u32>(x, z) * node_size; // LOD space -> sample space
    let s1 = s0 + vec2<u32>(node_size);
    let map_mins = bake_values.map_min;
    let sf = bake_values.scale_factors;

    // convert from LOD space to sample space to world space
    return AABB (
        vec3( 
            map_mins.x + f32(s0.x) * sf.x,
            map_mins.y + f32(minH) * sf.y,
            map_mins.z + f32(s0.y) * sf.z,
        ),
        vec3( 
            map_mins.x + f32(s1.x) * sf.x, 
            map_mins.y + f32(maxH) * sf.y, 
            map_mins.z + f32(s1.y) * sf.z
        )
    );
}

// for a node in this lod, either 
// 1. add to selected nodes
// 2. subdivide into 4 chldren, add to out queue
// NOTE this algorithm is liberal in the granularity it assigns
// as we do not allow for partial nodes. 
// every invocation already expects that this nodes aabb is contained in the current lod range (though not necessarily the frustum)
@compute @workgroup_size(64) 
fn select_nodes_for_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= min(in_queue.args.count, bake_values.queue_capacity)) {return;} // skip if this thread has no node to process
    let node_work = in_queue.queue[gid.x]; // the node for this thread
    let lod = node_work.lod;
    let x = (node_work.xz >> 16u) & 0xFFFF; // hi bits
    let z = (node_work.xz & 0xFFFF); // lo bits

    let mm = textureLoad(min_max_heights, vec2<u32>(x,z), lod).rg;

    let aabb = get_aabb(lod, mm.r, mm.g, x, z);

    if (!frustum_intersects(aabb)) {
        return;
    } 

    // if the next finest lod range also emcompasses this node
    // then we should include the child nodes in the node work for the next dispatch
    // also use this tile if the children nodes dont have residency
    if (lod > 0u && in_lod_range(lod - 1u, aabb) && children_resident(lod, x, z)) {
       let base = atomicAdd(&out_queue.args.count, 4u); // increase the count by 4, to handle 4 new children, store pre count
       if (base + 4u <= bake_values.queue_capacity) { // guard against queue overflow. If the queue is full, then we have to just draw the node as this level.
            for (var i = 0u; i < 4u; i++) {
                // bit magic to obtain the child LOD space coordinates
                let child_x = (x << 1u) | (i & 1u);
                let child_z = (z << 1u) | (i >> 1u);
                let cx_cz = ((child_x << 16u) | child_z);
                out_queue.queue[base + i] = NodeWork(cx_cz, lod - 1u);
            }
            // adjust the required workgroup size of the next dispatch based on the number of 
            // nodes that it needs to process
            atomicMax(&out_queue.args.wg_x, (base + 4u + WG_SIZE - 1u) / WG_SIZE);
            return;
       }
    }


    // this nodes tile is gauranteed resident, because the last compute
    // pass already checked if its children were resident before subdividing
    let t_coords = vec2<u32>(x, z) / bake_values.nodes_per_tile_axis;
    let tile_slot = get_tile_residency(lod, t_coords);


    // get the instance idx for the selected node, increment for next
    let i = atomicAdd(&draw_args.instance_count, 1u); 

    if (i >= bake_values.selected_capacity) {return;} // gaurd against selected node buf overflow

    selected_nodes[i] = SelectedNode(node_work.xz, (lod<<16u) | tile_slot); // lod | slot 

}

// runs before each level: the out queue was the previous level's in queue
@compute @workgroup_size(1)
fn reset_out_queue() {
    atomicStore(&out_queue.args.wg_x, 0u);
    out_queue.args.wg_y = 1u;
    out_queue.args.wg_z = 1u;
    atomicStore(&out_queue.args.count, 0u);
}

// runs after the last level: overflowing selects still bumped the count
@compute @workgroup_size(1)
fn clamp_selected() {
    let n = atomicLoad(&selected.instance_count);
    atomicStore(&selected.instance_count, min(n, bake_values.selected_capacity));
}

