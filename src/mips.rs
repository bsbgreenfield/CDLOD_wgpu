use crate::{CDLODSettings, tiles::TileLayout};
use wgpu::util::DeviceExt;

pub(crate) struct MipChain {
    pub texture: wgpu::Texture,
    pub level_sizes: Vec<[u32; 2]>,
}

struct MipMapDescriptor<'a> {
    label: Option<&'a str>,
    format: wgpu::TextureFormat,
    view_formats: &'a [wgpu::TextureFormat],
    source_bytes: &'a [u8],
    stride: u32,
    width: u32,
    height: u32,
    count: u32,
}

fn build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    desc: MipMapDescriptor,
) -> MipChain {
    let level_sizes: Vec<[u32; 2]> = (0..desc.count)
        .map(|mip_level| {
            [
                ((desc.width - 1) >> mip_level) + 1,
                ((desc.height - 1) >> mip_level) + 1,
            ]
        })
        .collect();

    // padding so that each mip levels can hold an extra sample per axis
    let pad = 1 << (desc.count - 1);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: desc.label,
        size: wgpu::Extent3d {
            width: desc.width - 1 + pad,
            height: desc.height - 1 + pad,
            depth_or_array_layers: 1,
        },
        mip_level_count: desc.count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: desc.view_formats,
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        desc.source_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(desc.width * desc.stride),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: 1,
        },
    );

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: desc.label,
        layout: None,
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command encoder"),
    });

    for mip_level in 1..desc.count as usize {
        // width and height of the source
        let [src_w, src_h] = level_sizes[mip_level - 1];
        // width and height of the dest
        let [dest_w, dest_h] = level_sizes[mip_level];

        let params_bytes: Vec<u8> = [src_w, src_h, dest_w, dest_h]
            .iter()
            .flat_map(|val| val.to_le_bytes())
            .collect();

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mip params"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let src = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level as u32 - 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level as u32,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heightmap bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute mip pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dest_w.div_ceil(8), dest_h.div_ceil(8), 1);
    }
    queue.submit([encoder.finish()]);
    MipChain {
        texture,
        level_sizes,
    }
}

/// build the height mipmap that is the source dataset
/// for all data being streamed into the GPU.
/// Note that this function may crop the source data
/// in order to ensure that the algorithm can operate
/// using evenly sized streaming units (tiles)
fn build_height_mipmap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    heightmap: &image::DynamicImage,
    tile_layout: &TileLayout,
) -> MipChain {
    let [w, h] = tile_layout.cropped_size;
    let luma = image::imageops::crop_imm(&heightmap.to_luma16(), 0, 0, w, h).to_image();
    let bytes: Vec<u8> = luma
        .pixels()
        .flat_map(|p| (p.0[0] as u32).to_le_bytes())
        .collect();
    let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/mip_height.wgsl"));
    let mm = MipMapDescriptor {
        label: None,
        format: wgpu::TextureFormat::R32Uint,
        view_formats: &[],
        source_bytes: &bytes,
        stride: 4,
        width: luma.width(),
        height: luma.height(),
        count: tile_layout.max_mip + 1,
    };
    let mipchain = build(device, queue, &shader, mm);
    mipchain
}
