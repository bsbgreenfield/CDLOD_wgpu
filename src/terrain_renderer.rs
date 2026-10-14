use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::{
    CDLODSettings, ViewUniform,
    node_selector::{BakeValues, NodeSelector},
    stream::GPUTerrainData,
};

fn gen_vertex_swatch(node_size: usize) -> (Vec<[u16; 2]>, Vec<u16>) {
    let side = node_size + 1;
    let vertices: Vec<[u16; 2]> = (0..side)
        .flat_map(|y| (0..side).map(move |x| [x as u16, y as u16]))
        .collect();
    let mut indices: Vec<u16> = Vec::with_capacity(6 * node_size * node_size);

    // treat each vertex as the top left of a unique cell composed of four vertices
    // then split that cell into two triangles
    for y in 0..node_size {
        for x in 0..node_size {
            let top_left = (y * side + x) as u16;
            let top_right = top_left + 1;
            let bottom_left = top_left + side as u16;
            let bottom_right = bottom_left + 1;

            // alternate diagonals
            if (x + y) % 2 == 0 {
                indices.extend_from_slice(&[
                    top_left,
                    bottom_left,
                    top_right,
                    top_right,
                    bottom_left,
                    bottom_right,
                ]);
            } else {
                indices.extend_from_slice(&[
                    top_left,
                    bottom_left,
                    bottom_right,
                    top_left,
                    bottom_right,
                    top_right,
                ]);
            }
        }
    }

    (vertices, indices)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    map_min: [f32; 3],
    slot_shift: u32,
    scale_factors: [f32; 3],
    nodes_per_tile_axis: u32,
    texel_scale_factor: [f32; 2],
    node_size: u32,
    tile_texels: u32,
}

fn get_render_params_buf(
    device: &wgpu::Device,
    bake_values: &BakeValues,
    settings: &CDLODSettings,
    slots_per_axis: u32,
    tile_texels: u32,
) -> wgpu::Buffer {
    let params = RenderParams {
        map_min: bake_values.map_min,
        slot_shift: slots_per_axis.ilog2(),
        scale_factors: bake_values.scale_factors,
        nodes_per_tile_axis: bake_values.nodes_per_tile_axis,
        texel_scale_factor: [
            1. / settings.terrain_size.intervals() as f32,
            1. / settings.terrain_size.intervals() as f32,
        ],
        node_size: settings.min_quad_node_size.get(),
        tile_texels,
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("render params buf"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn create_diffuse_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    diffuse: &image::DynamicImage,
) -> wgpu::Texture {
    let mut level = diffuse.to_rgba8();
    let (w, h) = level.dimensions();
    let mip_count = w.max(h).ilog2() + 1;
    let mut bytes = Vec::with_capacity(level.as_raw().len() * 4 / 3 + 4);
    bytes.extend_from_slice(level.as_raw());
    for mip in 1..mip_count {
        level = image::imageops::resize(
            &level,
            (w >> mip).max(1),
            (h >> mip).max(1),
            image::imageops::FilterType::Triangle,
        );
        bytes.extend_from_slice(level.as_raw());
    }
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("terrain diffuse"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &bytes,
    )
}
fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let uniform = |binding, min_size: u64| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain render"),
        entries: &[
            uniform(0, 0),
            uniform(1, size_of::<RenderParams>() as u64),
            // selected nodes
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // height atlas
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // diffuse
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub(crate) struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    diffuse_image: image::DynamicImage,
    render_params: wgpu::Buffer,
}

impl TerrainRenderer {
    pub(crate) fn render(&self, pass: &mut wgpu::RenderPass<'_>, selected_nodes: &wgpu::Buffer) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed_indirect(selected_nodes, 0);
    }
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bake_values: &BakeValues,
        settings: &CDLODSettings,
        slots_per_axis: u32,
        tile_texels: u32,
        node_selector: &NodeSelector,
        view_uniform: &ViewUniform,
        diffuse: image::DynamicImage,
        gpu_terrain_data: &GPUTerrainData,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let diffuse_texture = create_diffuse_texture(device, queue, &diffuse);
        let diffuse_view = diffuse_texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain diffuse sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let layout = create_layout(device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain render pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain render"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<[u16; 2]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
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
            multiview_mask: None,
            cache: None,
        });
        let render_params =
            get_render_params_buf(device, bake_values, settings, slots_per_axis, tile_texels);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain render bg"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_uniform.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        offset: 0,
                        size: None,
                        buffer: &node_selector.bind_groups.selected_nodes_buffer,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&gpu_terrain_data.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            layout: &layout,
        });

        let (v, i) = gen_vertex_swatch(settings.min_quad_node_size.get() as usize);
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&i),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            pipeline,
            bind_group: bg,
            vertex_buffer: vb,
            index_buffer: ib,
            diffuse_image: diffuse,
            render_params,
        }
    }
}
