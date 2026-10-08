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
    selected_nodes: wgpu::Buffer,
    min_max_heights: wgpu::Texture,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::PipelineLayout,
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
impl TileSelector {
    pub(crate) fn new(
        bake_values: BakeValues,
        min_max_heights: wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
                Some(&layout.selection),
                Some(&layout.bake),
                Some(&layout.residency),
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

        let selected_nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selected nodes buffer"),
            size: (size_of::<NodeWork>() * bake_values.selected_capacity as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            shader,
            bake_values,
            node_queue_a,
            node_queue_b,
            selected_nodes,
            min_max_heights,
            pipeline,
            layout: pipeline_layout,
        }
    }
    pub(crate) fn select_tiles(&self, device: &wgpu::Device, queue: &wgpu::Queue) {}
}

use std::num::NonZeroU64;

use wgpu::wgc::device::queue;

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

struct TileSelectorBindGroups {
    queues: wgpu::BindGroup,
    selected: wgpu::BindGroup,
    bake: wgpu::BindGroup,
    residency: wgpu::BindGroup,
}
impl TileSelectorBindGroups {
    fn get_queues_bg(
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
                    binding: 0,
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
}
pub(crate) struct TileSelectorLayouts {
    pub queues: wgpu::BindGroupLayout,    // group 0
    pub selection: wgpu::BindGroupLayout, // group 1
    pub bake: wgpu::BindGroupLayout,      // group 2
    pub residency: wgpu::BindGroupLayout, // group 3
}

fn create_layouts(device: &wgpu::Device) -> TileSelectorLayouts {
    let mk = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
    };
    TileSelectorLayouts {
        // in_queue (ro), out_queue (rw). Binding 1 is unused, matching the shader.
        queues: mk("ts queues", &[storage(0, true), storage(2, false)]),
        // selected_nodes, frustum, camera, draw_args
        selection: mk(
            "ts selection",
            &[
                storage(0, false),
                uniform(1, None),
                uniform(2, None),
                storage(3, false),
            ],
        ),
        // min_max_heights, bake_values, levels (16 * 16 B)
        bake: mk(
            "ts bake",
            &[
                uint_texture(0),
                uniform(1, Some(size_of::<BakeValues>() as u64)),
                uniform(2, Some(16 * 16)),
            ],
        ),
        // residency, request_flags, requests
        residency: mk(
            "ts residency",
            &[uint_texture(0), storage(1, false), storage(2, false)],
        ),
    }
}
