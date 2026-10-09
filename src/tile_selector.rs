#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub(crate) struct BakeValues {
    pub map_min: [f32; 3],
    pub max_mip: u32,
    pub scale_factors: [f32; 3],
    pub nodes_per_tile_axis: u32, // must be even
    pub root_tiles: [u32; 2],
    pub queue_capacity: u32,
    pub selected_capacity: u32,
    pub request_capacity: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub(crate) struct LODArgs {
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    count: u32,
}
#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub(crate) struct NodeWork {
    xz: u32,
    lod: u32,
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub(crate) struct LevelInfo {
    pub node_size: u32,
    pub lod_range: f32,
    _pad: [u32; 2],
}

pub(crate) struct TileSelector {
    shader: wgpu::ShaderModule,
    bake_values: BakeValues,
    node_queue_a: wgpu::Buffer,
    node_queue_b: wgpu::Buffer,
    min_max_heights: wgpu::Texture,
    min_max_tex_view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::PipelineLayout,
    bind_groups: TileSelectorBindGroups,
}

fn get_selected_nodes_buffer(device: &wgpu::Device, bake_values: &BakeValues) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selected nodes"),
        size: ((size_of::<wgpu::wgt::DrawIndexedIndirectArgs>() as usize)
            + (size_of::<NodeWork>() * bake_values.selected_capacity as usize))
            as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn get_node_work_queue(device: &wgpu::Device, bake_values: BakeValues) -> wgpu::Buffer {
    let size = (size_of::<LODArgs>()
        + bake_values.selected_capacity as usize * size_of::<NodeWork>()) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile select queue a"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT      // dispatch_workgroups_indirect reads wg_x/y/z at offset 0
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    let init = LODArgs {
        wg_x: 0,
        wg_y: 1,
        wg_z: 1,
        count: 0,
    };

    buf.slice(..size_of::<LODArgs>() as u64)
        .get_mapped_range_mut()
        .unwrap()
        .copy_from_slice(bytemuck::bytes_of(&init));
    buf.unmap();
    buf
}

fn get_bake_values_buffer(device: &wgpu::Device, bake_values: &BakeValues) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bake values uniform buffer"),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        contents: bytemuck::bytes_of(bake_values),
    })
}

fn get_levels_buffer(
    device: &wgpu::Device,
    layout: &TileLayout,
    world: &WorldValues,
    min_quad_node_size: u32,
) -> wgpu::Buffer {
    let lod_count = layout.max_mip as usize + 1;
    assert!(lod_count as u32 <= MAX_LODS);

    let mut levels = [LevelInfo::zeroed(); MAX_LODS as usize]; // unused tail stays zeroed
    for lod in 0..lod_count {
        levels[lod] = LevelInfo {
            node_size: min_quad_node_size << lod, // samples per node axis (sample space)
            lod_range: world.lod_ranges[lod],     // world-space radius
            _pad: [0; 2],
        };
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("levels"),
        contents: bytemuck::bytes_of(&levels), // 16 * 16 B = 256 B
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

impl TileSelector {
    pub(crate) fn new(
        bake_values: BakeValues,
        min_max_heights: wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &CDLODSettings,
        tile_layout: &TileLayout,
        gpu_terrain_data: &GPUTerrainData,
        world: &WorldValues,
        view_uniform: &ViewUniform,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile selection shader"),
            source: wgpu::ShaderSource::Wgsl("tile_selector".into()),
        });
        let layout = create_layouts(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile select pipeline layout"),
            bind_group_layouts: &[
                Some(&layout.queues),
                Some(&layout.camera),
                Some(&layout.bake),
                Some(&layout.state),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile selector"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("select_nodes_for_level"),
            compilation_options: Default::default(),
            cache: None,
        });
        let node_queue_a = get_node_work_queue(device, bake_values);
        let node_queue_b = get_node_work_queue(device, bake_values);

        let min_max_tex_view = min_max_heights.create_view(&wgpu::wgt::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Rg16Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(bake_values.max_mip + 1),
            base_array_layer: 0,
            array_layer_count: None,
        });
        let bind_groups = TileSelectorBindGroups::new(
            device,
            settings,
            tile_layout,
            &bake_values,
            gpu_terrain_data,
            world,
            &node_queue_a,
            &node_queue_b,
            view_uniform,
            &min_max_tex_view,
            &layout,
        );
        Self {
            shader,
            bake_values,
            node_queue_a,
            node_queue_b,
            min_max_heights,
            min_max_tex_view,
            bind_groups,
            pipeline,
            layout: pipeline_layout,
        }
    }
    pub(crate) fn select_tiles(&self, device: &wgpu::Device, queue: &wgpu::Queue) {}
}

use std::num::NonZeroU64;

use bytemuck::Zeroable;
use wgpu::{
    util::DeviceExt,
    wgc::device::{self, queue},
};

use crate::{
    CDLODSettings, ViewUniform, ViewUniformData,
    stream::GPUTerrainData,
    tiles::{MAX_LODS, TileLayout, WorldValues},
};

const CS: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: CS,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32, min_size: Option<u64>) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: CS,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: min_size.and_then(NonZeroU64::new),
        },
        count: None,
    }
}

// texture_2d<u32>, sampled via textureLoad with a mip level
fn uint_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: CS,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) struct TileSelectorLayouts {
    pub queues: wgpu::BindGroupLayout, // group 0
    pub camera: wgpu::BindGroupLayout, // group 1
    pub bake: wgpu::BindGroupLayout,   // group 2
    pub state: wgpu::BindGroupLayout,  // group 3
}
struct TileSelectorBindGroups {
    levels_buffer: wgpu::Buffer,
    bake_values_buffer: wgpu::Buffer,
    selected_nodes_buffer: wgpu::Buffer,
    queues: wgpu::BindGroup,
    camera: wgpu::BindGroup,
    bake: wgpu::BindGroup,
    state: wgpu::BindGroup,
}
impl TileSelectorBindGroups {
    fn new(
        device: &wgpu::Device,
        settings: &CDLODSettings,
        tile_layout: &TileLayout,
        bake_values: &BakeValues,
        gpu_terrain_data: &GPUTerrainData,
        world: &WorldValues,
        in_q: &wgpu::Buffer,
        out_q: &wgpu::Buffer,
        view_uniform: &ViewUniform,
        min_max_texture_view: &wgpu::TextureView,
        layouts: &TileSelectorLayouts,
    ) -> Self {
        let queues = Self::create_queues_bg(in_q, out_q, device, &layouts.queues);
        let camera = Self::create_camera_bg(view_uniform, &layouts.camera, device);

        let levels_buffer = get_levels_buffer(
            device,
            tile_layout,
            world,
            settings.min_quad_node_size.get(),
        );
        let bake_values_buffer = get_bake_values_buffer(device, bake_values);
        let bake = Self::create_bake_bg(
            min_max_texture_view,
            &bake_values_buffer,
            &levels_buffer,
            &layouts.bake,
            device,
        );
        let selected_nodes_buffer = get_selected_nodes_buffer(device, &bake_values);
        let state = Self::create_state_bg(
            &selected_nodes_buffer,
            &gpu_terrain_data.residency_view,
            &gpu_terrain_data.request_flags,
            &gpu_terrain_data.requests,
            &layouts.state,
            device,
        );

        Self {
            selected_nodes_buffer,
            levels_buffer,
            bake_values_buffer,
            queues,
            camera,
            bake,
            state,
        }
    }
    fn create_queues_bg(
        in_q: &wgpu::Buffer,
        out_q: &wgpu::Buffer,
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("queues bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: in_q,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: out_q,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        bg
    }

    fn create_camera_bg(
        view_uniform: &ViewUniform,
        layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> wgpu::BindGroup {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &view_uniform.buf,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        bg
    }

    fn create_bake_bg(
        min_max_texture_view: &wgpu::TextureView,
        bake_values_buffer: &wgpu::Buffer,
        levels_buffer: &wgpu::Buffer,
        layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> wgpu::BindGroup {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(min_max_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: bake_values_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: levels_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        bg
    }

    fn create_state_bg(
        selected_nodes_buffer: &wgpu::Buffer,
        residency_view: &wgpu::TextureView,
        request_flags_buffer: &wgpu::Buffer,
        request_list_buffer: &wgpu::Buffer,
        layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("state bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: selected_nodes_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(residency_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: request_flags_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: request_list_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

fn create_layouts(device: &wgpu::Device) -> TileSelectorLayouts {
    TileSelectorLayouts {
        // in_queue (ro), out_queue (rw). Binding 1 is unused, matching the shader.
        queues: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("queues"),
            entries: &[storage(0, true), storage(1, false)],
        }),
        camera: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[uniform(0, None)],
        }),
        // min_max_heights, bake_values, levels (16 * 16 B)
        bake: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake"),
            entries: &[
                uint_texture(0),
                uniform(1, Some(size_of::<BakeValues>() as u64)),
                uniform(2, Some(16 * 16)),
            ],
        }),
        // residency, request_flags, requests
        state: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shared state"),
            entries: &[
                storage(0, false),
                uint_texture(1),
                storage(2, false),
                storage(3, false),
            ],
        }),
    }
}
