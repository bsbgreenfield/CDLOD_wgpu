use std::{
    error::Error,
    num::NonZero,
    path::{Path, PathBuf},
    sync::Arc,
};

use image::DynamicImage;
use wgpu::util::DeviceExt;

use crate::{
    mips::{
        HeightMips, build_compressed_tile_files, build_height_mipmap_cpu, get_min_max_texture,
        min_max_levels,
    },
    node_selector::{BakeValues, NodeSelector},
    stream::{GPUTerrainData, StreamConfig, TileStreamer},
    terrain_renderer::TerrainRenderer,
    tiles::{TileLayout, TileStore, WorldValues},
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainSize {
    T2,
    T4,
    T8,
    T16,
}

mod bake;
mod compress;
mod mips;
mod node_selector;
pub mod stream;
mod terrain_renderer;
mod tiles;

pub struct ViewProjection {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub frustum_planes: [[f32; 4]; 6],
}
impl ViewProjection {
    pub fn new(view_proj: [[f32; 4]; 4], position: [f32; 3]) -> Self {
        let m = &view_proj;
        let row = |r: usize| [m[0][r], m[1][r], m[2][r], m[3][r]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|i| a[i] + b[i]);
        let sub = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|i| a[i] - b[i]);
        let normalize = |p: [f32; 4]| {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            p.map(|c| c / l)
        };
        let frustum_planes = [
            add(r3, r0), // left
            sub(r3, r0), // right
            add(r3, r1), // bottom
            sub(r3, r1), // top
            r2,          // near (0..1 depth)
            sub(r3, r2), // far
        ]
        .map(normalize);
        Self {
            view_proj,
            position,
            frustum_planes,
        }
    }
    pub(crate) fn fov_and_aspect(&self) -> (f32, f32) {
        let m = &self.view_proj; // column-major: m[col][row]
        let row_len = |r: usize| (m[0][r] * m[0][r] + m[1][r] * m[1][r] + m[2][r] * m[2][r]).sqrt();
        let f = row_len(1);
        let fov_y = 2.0 * (1.0 / f).atan();
        let aspect = f / row_len(0);
        (fov_y, aspect)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniformData {
    view_proj: [[f32; 4]; 4],
    frustum_planes: [[f32; 4]; 6],
    position: [f32; 3],
    _pad: f32,
}

impl From<&ViewProjection> for ViewUniformData {
    fn from(v: &ViewProjection) -> Self {
        Self {
            view_proj: v.view_proj,
            frustum_planes: v.frustum_planes,
            position: v.position,
            _pad: 0.0,
        }
    }
}
struct ViewUniform {
    buf: wgpu::Buffer,
}
impl ViewUniform {
    fn new(device: &wgpu::Device, view_proj: &ViewProjection) -> Self {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view uniform"),
            contents: bytemuck::bytes_of(&ViewUniformData {
                view_proj: view_proj.view_proj,
                frustum_planes: view_proj.frustum_planes,
                position: view_proj.position,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self { buf }
    }
}

#[derive(Clone)]
pub struct HeightReader {
    tiles: Arc<TileStore>,
    tile_layout: TileLayout,
}

pub struct Terrain {
    tile_streamer: TileStreamer,
    gpu_terrain_data: GPUTerrainData,
    node_selector: NodeSelector,
    terrain_renderer: TerrainRenderer,
    view_uniform: ViewUniform,
    world: WorldValues,
}

impl Terrain {
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &ViewProjection) {
        queue.write_buffer(
            &self.view_uniform.buf,
            0,
            bytemuck::bytes_of(&ViewUniformData::from(view)),
        );
        self.tile_streamer.update(
            device,
            queue,
            view.position,
            &self.gpu_terrain_data,
            &self.world,
        );
        let mut encoder = device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
        self.node_selector.select_nodes(&mut encoder);
        self.tile_streamer
            .record_readback(&mut encoder, &self.gpu_terrain_data);
        queue.submit([encoder.finish()]);
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.terrain_renderer
            .render(pass, &self.node_selector.bind_groups.selected_nodes_buffer);
    }
}

pub struct TerrainInputs<'a> {
    pub diffuse_path: PathBuf,
    pub height_path: PathBuf,
    pub tile_path: &'a Path,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
}

pub fn create_tile_store(
    height_path: PathBuf,
    tile_path: &Path,
    settings: &CDLODSettings,
) -> Result<(HeightReader, HeightMips), Box<dyn Error>> {
    let heightmap = image::ImageReader::open(height_path.clone())?.decode()?;
    let tile_layout = TileLayout::new(&settings, heightmap.width(), heightmap.height());
    let height_mips = build_height_mipmap_cpu(&heightmap, &tile_layout);
    std::fs::create_dir_all(tile_path)?;
    let file_stem = &height_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("terrain");
    let tile_file = tile_path.join(format!("{file_stem}.cdlt"));
    if !tile_file.exists() {
        build_compressed_tile_files(&tile_layout, &height_mips, &tile_file)?;
    }
    let store = Arc::new(TileStore::open(&tile_file)?);
    Ok((
        HeightReader {
            tiles: store,
            tile_layout,
        },
        height_mips,
    ))
}

pub fn load_terrain(
    view_proj: &ViewProjection,
    settings: &CDLODSettings,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    stream_config: StreamConfig,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    height_reader: &HeightReader,
    height_mips: &HeightMips,
    diffuse_path: &PathBuf,
) -> Result<Terrain, Box<dyn Error>> {
    let diffuse = image::ImageReader::open(diffuse_path)?.decode()?; // TODO: diffuse texture
    let world = WorldValues::new(&settings, &height_reader.tile_layout);

    let request_capacity = stream_config.request_capacity;

    let gpu_terrain_data =
        GPUTerrainData::new(device, queue, &height_reader.tile_layout, &stream_config);
    let slots_per_axis = stream_config.slots_per_axis.clone();
    let tile_texels = height_reader.tile_layout.stored_tile_size();
    let streamer = TileStreamer::new(
        queue,
        &height_reader.tile_layout,
        height_reader.tiles.clone(),
        stream_config,
        &gpu_terrain_data,
    );
    let (node_count, mm_levels) = min_max_levels(
        &height_mips,
        settings.min_quad_node_size.get() as usize,
        (height_reader.tile_layout.max_mip + 1) as usize,
    );
    let min_max_texture = get_min_max_texture(node_count, mm_levels, device, queue);

    let k =
        settings.lod0_range / (settings.min_quad_node_size.get() as f32 * world.scale_factors[0]);
    let disc = std::f32::consts::PI * (k + 3.0 * std::f32::consts::SQRT_2).powi(2);

    let root_nodes = height_reader.tile_layout.root_tiles[0]
        * height_reader.tile_layout.root_tiles[1]
        * height_reader.tile_layout.nodes_per_tile_axis.pow(2);
    let queue_capacity = root_nodes.max((disc * 2.0) as u32).next_power_of_two();

    let bake_values = BakeValues {
        map_min: settings.map.map_min,
        max_mip: height_reader.tile_layout.max_mip,
        root_tiles: height_reader.tile_layout.root_tiles,
        scale_factors: world.scale_factors,
        nodes_per_tile_axis: height_reader.tile_layout.nodes_per_tile_axis,
        queue_capacity: queue_capacity,
        selected_capacity: tiles::selected_capacity(
            settings.lod0_range,
            settings.min_quad_node_size.get(),
            world.scale_factors[0],
            view_proj,
            height_reader.tile_layout.max_mip,
        ),
        request_capacity: request_capacity,
        _pad: [0, 0, 0],
    };
    let view_uniform = ViewUniform::new(device, view_proj);
    let node_selector = NodeSelector::new(
        bake_values,
        min_max_texture,
        device,
        &settings,
        &height_reader.tile_layout,
        &gpu_terrain_data,
        &world,
        &view_uniform,
    );

    let terrain_renderer = TerrainRenderer::new(
        device,
        queue,
        &bake_values,
        &settings,
        slots_per_axis,
        tile_texels,
        &node_selector,
        &view_uniform,
        diffuse,
        &gpu_terrain_data,
        color_format,
        depth_format,
    );

    Ok(Terrain {
        tile_streamer: streamer,
        terrain_renderer,
        gpu_terrain_data,
        world,
        node_selector,
        view_uniform: view_uniform,
    })
}

pub struct MapDimensions {
    // the world position of sample (0,0) at height val 0
    pub map_min: [f32; 3],
    pub size: [f32; 3],
}
pub struct CDLODSettings {
    pub map: MapDimensions,
    pub lod0_range: f32,

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
