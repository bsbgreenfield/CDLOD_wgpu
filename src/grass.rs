use wgpu::util::DeviceExt;

use crate::{CDLODSettings, ViewUniform, node_selector::BakeValues};

/// Grass look and density. Distances and sizes are in world units.
#[derive(Debug, Clone, Copy)]
pub struct GrassSettings {
    /// distance at which grass is fully gone
    pub range: f32,
    /// distance over which blades thin out before `range`
    pub fade_width: f32,
    pub blade_height: f32,
    pub blade_width: f32,
    pub bend: f32,
    /// 0..1, fraction of a blade slot a blade may wander
    pub jitter: f32,
    /// mesh groups per terrain cell axis; blades per cell = 16 * S^2
    pub cell_subdiv: u32,

    /// range at which the full level of detail for the blade is drawn (4 rows for now)
    pub lod_near: f32,

    // range at which the blade drops in detail. within its 3, and beyond its 2
    pub lod_far: f32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            range: 50.0,
            fade_width: 25.0,
            blade_height: 0.20,
            blade_width: 0.03,
            bend: 0.35,
            jitter: 0.9,
            cell_subdiv: 2,
            lod_near: 10.,
            lod_far: 25.,
        }
    }
}
pub(crate) fn groups_per_node(node_size: u32, cell_subdiv: u32) -> u32 {
    (node_size * cell_subdiv).pow(2)
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassParams {
    map_min: [f32; 3],
    slot_shift: u32,
    scale_factors: [f32; 3],
    nodes_per_tile_axis: u32,
    node_size: u32,
    tile_texels: u32,
    grass_range: f32,
    fade_width: f32,
    blade_height: f32,
    blade_width: f32,
    bend: f32,
    jitter: f32,
    cell_subdiv: u32,
    lod_near: f32,
    lod_far: f32,
    _pad: u32,
}

pub(crate) struct GrassRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: GrassParams,
    params_buffer: wgpu::Buffer,
    grass_nodes: wgpu::Buffer,
}
impl GrassRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        bake_values: &BakeValues,
        settings: &CDLODSettings,
        grass: &GrassSettings,
        slots_per_axis: u32,
        tile_texels: u32,
        view_uniform: &ViewUniform,
        grass_nodes: &wgpu::Buffer,
        atlas_view: &wgpu::TextureView,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let params = GrassParams {
            map_min: bake_values.map_min,
            slot_shift: slots_per_axis.ilog2(),
            scale_factors: bake_values.scale_factors,
            nodes_per_tile_axis: bake_values.nodes_per_tile_axis,
            node_size: settings.min_quad_node_size.get(),
            tile_texels,
            grass_range: grass.range,
            fade_width: grass.fade_width,
            blade_height: grass.blade_height,
            blade_width: grass.blade_width,
            bend: grass.bend,
            jitter: grass.jitter,
            cell_subdiv: grass.cell_subdiv,
            lod_near: grass.lod_near,
            lod_far: grass.lod_far,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grass shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grass.wgsl").into()),
        });

        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("grass"),
            layout: None,
            task: None,
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    // every output slot that is read gets written first
                    zero_initialize_workgroup_memory: false,
                },
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None, // blades are two-sided
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual), // reversed Z, as terrain
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview: None,
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_uniform.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grass_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            params,
            params_buffer,
            grass_nodes: grass_nodes.clone(),
        }
    }

    pub(crate) fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw_mesh_tasks_indirect(&self.grass_nodes, 0);
    }
}
