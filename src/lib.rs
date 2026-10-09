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
    tile_selector::{BakeValues, TileSelector},
    tiles::{TerrainSize, TileLayout, TileStore, WorldValues},
};

mod bake;
mod compress;
mod mips;
mod stream;
mod tile_selector;
mod tiles;

pub struct ViewProjection {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub frustum_planes: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniformData {
    view_proj: [[f32; 4]; 4],
    frustum_planes: [[f32; 4]; 6],
    position: [f32; 3],
    _pad: f32,
}

struct ViewUniform {
    bg: wgpu::BindGroup,
    buf: wgpu::Buffer,
}

pub struct Terrain {
    tile_streamer: TileStreamer,
    gpu_terrain_data: GPUTerrainData,
    tile_selector: TileSelector,
    view_uniform: ViewUniform,
    world: WorldValues,
}

impl Terrain {
    //let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //    label: Some("camera buffer"),
    //    contents: bytemuck::bytes_of(view),
    //    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    //});
    fn update_view(&self, view: &ViewProjection, queue: &wgpu::Queue) {
        let data = ViewUniformData {
            view_proj: view.view_proj,
            frustum_planes: view.frustum_planes,
            position: view.position,
            _pad: 0.0,
        };

        queue.write_buffer(&self.view_uniform.buf, 0, bytemuck::bytes_of(&data));
    }
    fn tile_select(&mut self) {

        //todo
    }
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &ViewProjection) {
        self.tile_streamer.update(
            device,
            queue,
            view.position,
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
    let view_uniform = ViewUniform {
        bg: todo!(),
        buf: todo!(),
    };
    let tile_selector = TileSelector::new(
        bake_values,
        min_max_texture,
        device,
        queue,
        &settings,
        &tile_layout,
        &gpu_terrain_data,
        &world,
        &view_uniform,
    );

    Ok(Terrain {
        tile_streamer: streamer,
        gpu_terrain_data,
        world,
        tile_selector: todo!(),
        view_uniform: view_uniform,
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
