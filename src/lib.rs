use std::num::NonZero;

use crate::tiles::TerrainSize;

mod bake;
mod compress;
mod mips;
mod tiles;
pub struct CDLODSettings {
    terrain_size: TerrainSize,
    /// the minimum number of samples in sample space of the source terrain texture
    /// that a quad node is allowed to span.
    /// if min_quad_node_size = 1, then each node at LOD 0 represent a single sample.
    /// lower values = more quad tree subdivisions and higher resolution terrains.
    min_quad_node_size: NonZero<u32>,

    /// the upper bound of the mip level for the terrain heightmap.
    /// The actual highest mip level may be lower, the source data cant fit a single
    /// tile at this level.
    /// A higher max_LOD value = fewer persitently resident root tiles, but
    /// requires a coarser crop of the source data
    max_LOD: NonZero<u32>,

    /// the number of quad nodes represented by a single streamable tile
    /// lower number = finer streaming granularity.
    /// this essentially balances memory footprint and througput valume and latency.
    /// MUST BE AN EVEN NUMBER, so that children of a node all land on the same tile
    nodes_per_tile_axis: NonZero<u32>,
}
