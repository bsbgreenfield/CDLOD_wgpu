# CDLOD implementation plan


## BAKE TIME

given a heightmap dataset which is uniformly spaced along the x-z plane we need to generate a quad map


### quad map definition

### naive definition
A quad map is a list of nodes, where each node has a level of detail, which in turn correspods to a particular 3d bounding region
so each node will be (in theory) 
```rust
struct QuadNode {
    minHeight: u16,
    maxHeight: u16,
    center: (u16, u16),
    level: u8, 
}
```

### compressed
in practice however, the only values that the node itself will need to carry is min and max height as
LOD level is implicit in the  2d index of the node, and the center is extrapolated from its parent node.

The 3d axis aligned bounding box can be calulated from its width (implicit from its LOD level) and its height (maxH - minH). 
The node selelection process, compares the AABB against the camera distance as the selection criteria

so for example, with a world region extending from (x: -50, z: 50) in the top left and (x: 50, z: -50) in the bottom right (viewed from above)

nodes 1 - 4 are z-order subsets of node 0. So node 1 is the region bound by (-50, 50) - (0, 0),
node 2 is the region bound by (0, 50) - (50, 0) and so on.

As nodes are built, max and min height is calulated for that region (2d bounding square over the heightmap)

We store the quad map as a List of node lists, where the node list length is equal to 4^d (1, 4, 16, 64 so on) and d is the index within the outer list, indicating LOD depth
On the gpu, we can store this as a texture mipmap, where each quad level has a RGUInt32 value representing min | max height.


```rust
struct NodeGrid {
    nodes: Vec<QuadNode>,
    side_len: usize,
}

```

we can then calculate the AABB for a given node like so:
```rust


struct AABE {
    min: [f32;3],
    max: [f32;3]
}

struct MapDimensions {
    minX: f32,
    minY: f32,
    minZ: f32,
    sizeX: f32,
    sizeY: f32,
    sizeZ: f32,
}

struct HeightmapRasterSize {
   x: u32, 
   z: u32
}

// given the 2d size of the world, and the 2d size of the provided heightmap,
// calculate the scale factor to be applied to the AABBs of a node
// if the world x is 100, and the heightmap has 1000 samples, then each sample
// covers a tenth of a world unit
fn get_scale_factors(world_dim: &WorldDimensions, heightmap_dim: &HeightmapRasterSize) -> [f32;3] {
    let sx = world_dims.sizeX / (heightmap_dims.x - 1) as f32;
    let sz = world_dims.sizeZ / (heightmap_dims.z - 1) as f32;
    let sy = world_dims.sizeY / 65535.0;
    [
        sx, sy, sz
    ]
}

// using the scale factors calculated above, get the world space axis aligned bounding box
// of a given node. 
// if a node is placed at sample 10, and the scale factor is 1/10, that means the node location
// in world space is 1 (world unit)
// the node coordinates and node size are obtained through iteration through the node list
// where the size will be consistent for each node in an LOD level, and the coordinates can
// be extrapolated from the nodes position relative to its parent (bottom right, bottom left, ...)
fn get_AABE(
    node: &QuadNode, 
    node_coords: [u32;2], 
    node_size: u32, 
    world_dims: &MapDimensions, 
    heightmap_dims: &HeightmapRasterSize,
    scale_factors: [f32;3]
    ) -> AABB {
    AABE {
        min: [
            world_dims.minX + node_coords[0] as f32 * scale_factors[0],
            world_dims.minY + node.minH * scale_factors[1],
            world_dims.minZ + node_coords[1] as f32 * scale_factors[2],
        ],
        max: [
            world_dims.minX + ( node_coords[0] + node_size) as f32 * scale_factors[0],
            world_dims.minY + node.maxH * scale_factors[1],
            world_dims.minZ + ( node_coords[1] + node_size ) as f32 * scale_factors[2],
        ]
    }
}
```

#### quad map generation

The quad map is generated from the source data set by subdiving the source into four D times, 
where D is the number of levels of detail desired. For each subdivision, The min and max height is obtained 
for the section of samples contained within that subdivision. 

It should be noted that the position of the nodes that can be obtained through traversal of the quad map
is in **heightmap space** where one unit is one sample, as opposed to world space or any other space.
When converting from QuadNode -> world space AABB to perform the selection, such as the code above,
we are also converting from **heightmap space** into **world space**.



### DATA STORAGE AND TILE MIPMAPS

#### basic implementation
For a naive implementation, the entire source data could be made GPU resident,
and could be sampled at different resolutions depending on the lod of the selected node. 
This implementation, besides simplicity, has the added benefit of zero per frame throughput requirement for terrain rendering, 
all the terrain data is fully resident, and the selected nodes are calculated  per frame.

#### data streaming 
In practice, I want a balance between data throuput and data residency that better matches the requirement of an interactive game.
The key observation is that for much of the rendered area, we will not be using the full granularity of the source dataset. 
Indeed as the selected nodes get very far from the viewer, and thus have a very low level of detail, we will be sampling only small fraction of the dataset.
for LOD 5 for example, we would only sample once for every 2^5 (32) height value, if LOD 0 is taken to be 1:1 with the source. 

#### Tiles
We can take advantage of this using  a similar technique to geometry clipmapping, where we first bake a height mipmap pyramid from the source data.
Once we have the mipmapped height pyramid, in which the root is equal to the source (highest granularity) 
and the lowest descendant at level D is the source downsampled* by 2^D, we can subdivide each layer into "tiles", much the same way as the source dataset itself is subdivided into nodes.

Again, we have the mipmapped height data pyramid, and we can imagine it as an inverted physical pyramid where the peak contains the source (finest) data and the base contains the coarsest sampled data.
We now split up each layer such that they each contain the same number of *nodes* which we will call R, as the ratio of node/tile.  

#### The root tiles
The number of mipmap subdivisions is derived from the min_node_size and the nodes_per_tile_axis requested by the user. 
At LOD 0, all the nodes must necessarily span all the samples in the source dataset. 
The highest LOD is then calculated as being equal to the MIP level at which there is at least one tile with
"nodes_per_tile_axis" ^2 nodes contained in it. If the user also requested a mip level which is lower that
the coarsest mip for which this is true, then its clamped there.
Either way, there will be a max mip at which we can say the tile/tiles that span it are "root" tiles. Which 
have the useful property of being the ancestors of all other tiles. We ensure that these root tiles are always resident


#### source cropping
An invariant of this algorithm that must be upheld is that each mip level contains a whole number of tiles,
and that each node has exactly one tile as its data source. This requirement makes the code simpler and allows use of the
compute shader to request new tiles, because we can simultaneously check for child node data residency and request that tile 
to become resident if not in the same pass. The side effect of this is that the algorithm doesnt necessarily work for 
every shape of source data. Specifically, the source data must be subdivisable into a whole number of tiles.
Therefore, we crop the source data if this isnt possible

#### tile creation
Recall the inverted pyramid: to create our tile subdivisions we can iterate from base to peak.
At the base, the tile count = 1, and this layer contains the height samples spanned by the nodes of layer L such that each node in layer L has a side length of 1/sqrt(R).
R being the ratio of tile/node, and so also node count of this tile. 

Put in simpler terms, we just find the LOD wherein the the number of nodes contained is equal to R, and the nodes in that LOD are the nodes spanned by the base tile.

Next, at the layer above (layer 1), we split the layer into 4 tiles, such that each tile still contains R nodes, as the nodes themselves at this new layer have also subdivided into four.

This goes on until the peak of the pyramid, in which the number of tiles is equal to the number of nodes at the finest level of granularity divided by R.

Tiles are defined as containing T+1 X T+1 samples, rather than T X T, to provide deliberate row and column overlap between tiles, for morphs (discussed later)

Some properties of this system are as follows
1. The tile level, or mipmap level, selected for a given node at LOD L is min(L, maxMipLevel)
- This is a clamp on the LODOffset value, briefly mentioned above as deriving from the fact that we only create as many mip layers as required to reach the root tile.
So there may be 8 subdivision layers of quad nodes (quad LOD levels) but only 5 levels of the mipmap, so for nodes 8, 7, 6, and 5, the mipmap level is equal to 5.

2. Tile Sample Span per mip level is T * 2^mipLevel, or T << tileLevel. Where T is the number of samples in each tile axis
- it can be seen that as the mip level increases, meaning coarser, or as we approach the root tile, the span in **heightmap space** increases

3. the tile X offset within a  given mip level is equal to nodeX / tileSpan
- we can easily calculate the offset of a selected tile within by dividing the node offset, in heightmap space, by the tileSpan in heighmap space
This x offset can be thought of as the offset in **mip space**

4. tile stride is equal to 1 * 2^(nodeLevel - tileLevel)
- For the levels below LODOffset, in which the node level and tile level agree, the stride in sample units for the tile vs the node can be thought of as 1:1 with the source dataset.
For the higher levels, in which nodeLevel > tileLevel, the selected tile contains logical nodes which are wider in span than the actual selected node.
We account for this by sampling within the tile at a lower rate (coarser) than the nodeLevel would otherwise suggest, in the shader



#### data selection (To be continued)
Later, we will use this tile subdivided mipmap pyramid to fetch our data as a function of the selected nodes from the quad map.

In this fetch function, we have to first mark each tile (which is a single streamable unit) as either requested resident or not.
We iterate through the selected node list, and for each we find the highest granularity tile that contains it.
Then we bubble back up to the ancestors of the selected tile, marking each one as required resident, if it is not already so.
Note that this means that the root tile is *always* resident, and for each given tile we also include all of its ancestors.
The reason for this is that data streaming is asynchonous, and we cant guarantee ideal tile residency for every selected node, 
but with this scheme, and a logically mirrored eviction strategy, we CAN guarantee some ancestor tile for any given node (the root tile in the worst case).
This leads to some unused data (an extra 33% in the worst case), but allows us to avoid latency spikes while we wait for ideal tile residency. 

In the future we maybe could improve this with a reasonable prediction scheme, since it shouldnt be too hard to predict which tiles might be needed soon based on the velocity and 
direction of the camera. and we may be able to replace pessimestic ancestor loading with conservative predictive fine grained tile loading.


In practice, we avoid (or bypass) the issue of trying to render a node whose desired mip level tile is not resident by simply stopping node subdivision when we reach
a level for which there is no resident tile at the next level of detail. So in other words, we use the coarser tile than its distance from the camera would otherwise suggest.
This has the (slight) added benefit of reducing the work for the algorithm in the most throughput throttled times (i.e. fast moving camera)



*Note that "downsampling" must be done by sample decimation, rather than averaging, to uphold the morph invariant.*


## PER FRAME NODE Selection

In order to account for a moving camera, which can be expected to be the case for every frame, we must perform node selection every frame.

The goal of node selection is to provide a level of granularity in which the size of the triangles rendered on the terrain are roughly equivalent in screen space (larger triangles farther away)

Due to the nature of the quadtree, each successive LOD in the quad map is scaled by 2 such that a node at level 2 covers twice the **heightmap space** distance in each axis as the nodes at level 1.
Therfore, before performing selection, we calculate the LODRanges for each LOD, and put it in world space. 
                                              
                                            |offset|  | distance |  |scaled by lod level|
each LODRange for each level is calculated as near + ((far - near) * 1 >> LODLevel)

We can treat these LOD ranges as *radii* of concentric spheres extending from the camera.

Then the algorithm for node selection is  (pseudocode)

```rust
impl QuadNode {
    fn add_self(
             &self, 
             selected: &mut Vec<QuadNode>,
             node_index: usize,
             ranges: [f32; LODCount], 
             lod_level: usize, 
             frustum: &Frustum, 
             world_dims: &MapDimensions, 
             heightmap_dims: &HeightmapRasterSize, 
             scale_factors: [f32;3]) -> bool {
            let node_size = node_size_of(lod_level);
            let node_coords = get_node_coords(node_index, lod_level);
            let aabb =  get_AABE(&self,
                                 node_coords,
                                 node_size,
                                 world_dims,
                                 heightmap_dims,
                                 scale_factors);
            if !AABB::intersects_sphere(aabb, ranges[lod_level]) {
                return false;
            } 
            if !frustum::intersects(frustum) {
                return true
            }
            if lod_level == 0 {
                selected[node_index] = self.clone();
                return true;
            } else {
                if !AABB::intersects_sphere(aabb, ranges[lod_level - 1]) {
                    selected[node_index] = self.clone();
                } else {
                    for (child_index,  child ) in QuadMap<LODCount>::get_children(node_idex).iter() {
                        if !child.add_self(selected, child_index, ranges, lod_level-1, frustum, world_dims, heightmap_dims, scale_factors) {
                            addPartialNode(selected, self, child);
                        }
                    }
                }
            }
            return true
    }

}

```

The essence of the alorithm is that we populate a Vec<SelectedNode> by recursing through the quadtree from highest LOD to lowest, testing if the AABB of each node intesects the 
sphere about the camera at ranges[LODlevel]. If it does, and also is within the frustum, we recurse, and when we reach a point where the next lowest LOD will not intersect, we add the current node to the list



## rendering
The terrain is rendered using the list of selected nodes that was computed in a compute pass (see implementation section)
The selected node list is always GPU resident, and is in the form of a storage buffer of SelectedNode

where SelectedNode is

```wgsl
struct SelectedNode {
    xy: u32, // x | y, in terms of its LOD level, not world space
    meta: u32, // level(8b) | tile slot(16b) | flags(8b)
}
```
each thread in the dispatch will write to a unique slot within the storage buffer, which will be non deterministic due to the async behavior of the shader

The data stored in the selected node, obtained from the instance index, can then be used to sample the terrain texture 

The terrain texture itself is a single texture about 33 - 40% the size of the original source texture, populated with the data tiles mentioned above
Given that no single tile (aside from the root) is guaranteed to be resident at any given frame, we also keep a GPU resident residency table.
The residency table is a storage buffer that is indexed using the node's tile slot, contained in the meta field. 

```wgsl
 struct TileSlot {
    u: u32,  
    v: u32, 
 }
```


So for each node in selected nodes
1. read from the indirect buffer for the current instance to get the selected node idx
2. read from the selected node buffer to get the xz off and metadata
3. index in to the tileslot buffer to get the uv off of the selected tile
4. sample from the terrain texture using the nodes xz and the uv provided by the tileslot

note that the tileslot contained in the selected node 

## IMPLEMENTATION

### QUADMAP GEN
todo

### Node selection compute pass
One dispatch per LOD, and we can do indirect compute dispatches to dynamically decide workgroup sizes.

```wgsl

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

struct NodeWork {
    xz: u32, // IN LOD SPACE (one unit = one node)
    lod: u32,
}

struct SelectedNode {
    xz: u32, 
    meta: u32, // lod | tile_slot
}

struct LevelInfo {
    node_size: u32,
    lod_range: f32,
    nodes_per_axis: vec2<u32>,
}

struct BakeValues {
    map_min: vec3<f32>,
    max_mip: u32,
    map_size: vec3<u32>,
    tile_size: u32,
    heightmap_dims: vec2<u32>,
    root_tiles: vec2<u32>, // always resident
    scale_factors: vec3<f32>, // calculated as scales to convert from sample space to world space
    nodes_per_tile_axis: u32, // must be even
    min_node_size: u32,
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
@group(0) @binding(2) var<storage, read_write> out_queue:  OutQueue;  // next level's args

@group(1) @binding(0) var<storage, read_write> selected_nodes:  array<SelectedNode>;
@group(1) @binding(1) var<uniform>             frustum:         Frustum;
@group(1) @binding(2) var<uniform>             camera:          Camera;
@group(1) @binding(3) var<storage, read_write> draw_args:       RenderIndirectArgs;

@group(2) @binding(0) var                      min_max_heights: texture_2d<u32>; // mipmap of height vals 
@group(2) @binding(1) var<uniform>             bake_values:     BakeValues;
@group(2) @binding(2) var<uniform>             levels:          array<LevelInfo, MAX_LODS>;

@group(3) @binding(0) var                      residency:       texture_2d<u32>; // residency array for tiles
@group(3) @binding(1) var<storage, read_write> request_flags:   array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> requests:        RequestList;        


fn frustum_intersects(aabb: AABB) -> bool {
    // TODO: frustum code
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
```

### TODO:
1.  load terrain heightmap and diffuse
2.  create heightmap and diffuse mipmaps
3.  generate normal texture from heightmap, create mipmap
4.  create minmax height texture from heightmap, create mipmap
5.  create interface for selecting sections of the heightmap mipmap as "tiles"
6.  generate BakeValues struct from CDLOD settings and heighmap data
7.  generate LevelInfo per LOD level (node_size, LOD ranges, nodes per axis)
8.  create bind group layout for node select compute shader
9.  create bind groups/ buffers for all compute pass items
10. node select compute shader
11. process tile requests, update residency buffer, write tiles to a GPU terrain texture
12. render pass using selected nodes, tile slots
13. morphs
