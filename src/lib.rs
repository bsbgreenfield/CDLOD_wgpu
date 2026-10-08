use std::{
    error::Error,
    num::NonZero,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    mips::{
        build_compressed_tile_files, build_height_mipmap_cpu, get_min_max_texture, min_max_levels,
    },
    stream::{GPUTerrainData, StreamConfig, TileStreamer},
    tile_selector::BakeValues,
    tiles::{TerrainSize, TileLayout, TileStore, WorldValues},
};

mod bake;
mod compress;
mod mips;
mod stream;
mod tile_selector;
mod tiles;

pub struct Terrain {
    tile_streamer: TileStreamer,
    gpu_terrain_data: GPUTerrainData,
    min_max_texture: wgpu::Texture,
    world: WorldValues,
}

impl Terrain {
    fn tile_select(&mut self) {

        //todo
    }
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera_pos: [f32; 3]) {
        self.tile_streamer.update(
            device,
            queue,
            camera_pos,
            &self.gpu_terrain_data,
            &self.world,
        );
    }
}

pub fn load_terrain(
    height_path: PathBuf,
    diffuse_path: PathBuf,
    tile_path: &Path,
    settings: CDLODSettings,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    stream_config: StreamConfig,
) -> Result<Terrain, Box<dyn Error>> {
    let heightmap = image::ImageReader::open(height_path)?.decode()?;
    let diffuse = image::ImageReader::open(diffuse_path)?.decode()?; // TODO: diffuse texture

    let tile_layout = TileLayout::new(&settings, heightmap.width(), heightmap.height());

    let world = WorldValues::new(&settings, &tile_layout);

    let request_capacity = stream_config.request_capacity;
    let max_inflight = stream_config.max_inflight;
    let height_mips = build_height_mipmap_cpu(&heightmap, &tile_layout);
    build_compressed_tile_files(&tile_layout, &height_mips, tile_path)?;
    let store = Arc::new(TileStore::open(tile_path)?);
    let gpu_terrain_data = GPUTerrainData::new(device, queue, &tile_layout, &stream_config);
    let streamer = TileStreamer::new(queue, &tile_layout, store, stream_config, &gpu_terrain_data);
    let (node_count, mm_levels) = min_max_levels(
        &height_mips,
        settings.min_quad_node_size.get() as usize,
        (tile_layout.max_mip + 1) as usize,
    );
    let min_max_texture = get_min_max_texture(node_count, mm_levels, device, queue);

    let bake_values = BakeValues {
        map_min: settings.map.map_min,
        max_mip: tile_layout.max_mip,
        root_tiles: tile_layout.root_tiles,
        scale_factors: world.scale_factors,
        nodes_per_tile_axis: tile_layout.nodes_per_tile_axis,
        queue_capacity: max_inflight as u32,
        selected_capacity: max_inflight as u32,
        request_capacity: request_capacity,
        _pad: [0, 0, 0],
    };

    Ok(Terrain {
        tile_streamer: streamer,
        gpu_terrain_data,
        min_max_texture,
        world,
    })
}

pub struct MapDimensions {
    // the world position of sample (0,0) at height val 0
    pub map_min: [f32; 3],
    size: [f32; 3],
}
pub struct CDLODSettings {
    map: MapDimensions,
    lod0_range: f32,

    pub terrain_size: TerrainSize,
    /// the minimum number of samples in sample space of the source terrain texture
    /// that a quad node is allowed to span.
    /// if min_quad_node_size = 1, then each node at LOD 0 represent a single sample.
    /// lower values = more quad tree subdivisions and higher resolution terrains.
    pub min_quad_node_size: NonZero<u32>,

    /// the upper bound of the mip level for the terrain heightmap.
    /// The actual highest mip level may be lower, the source data cant fit a single
    /// tile at this level.
    /// A higher max_LOD value = fewer persitently resident root tiles, but
    /// requires a coarser crop of the source data
    pub max_LOD: NonZero<u32>,

    /// the number of quad nodes represented by a single streamable tile
    /// lower number = finer streaming granularity.
    /// this essentially balances memory footprint and througput valume and latency.
    /// MUST BE AN EVEN NUMBER, so that children of a node all land on the same tile
    pub nodes_per_tile_axis: NonZero<u32>,
}
