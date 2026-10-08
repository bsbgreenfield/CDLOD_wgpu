use std::{io::Write, path::Path};

use crate::tiles::{TileLayout, TileStore};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use wgpu::util::DeviceExt;
use zstd::zstd_safe::CParameter;

pub(crate) struct HeightMips {
    pub(crate) level_sizes: Vec<u32>,
    pub(crate) levels: Vec<Vec<u16>>,
}

impl HeightMips {
    fn row(&self, mip: u32, z: u32) -> &[u16] {
        let n = self.level_sizes[mip as usize] as usize;
        &self.levels[mip as usize][z as usize * n..][..n]
    }
}

pub(crate) fn min_max_levels(
    mips: &HeightMips,
    node_size: usize,
    lod_count: usize,
) -> (u32, Vec<Vec<[u16; 2]>>) {
    let level_size = mips.level_sizes[0] as usize;
    let heights = &mips.levels[0];
    let node_count = (level_size - 1) / node_size;
    let mut level0 = vec![[0u16; 2]; node_count * node_count];
    level0
        .par_chunks_mut(node_count)
        .enumerate()
        .for_each(|(nz, row)| {
            for (nx, mm) in row.iter_mut().enumerate() {
                let (mut min_h, mut max_h) = (u16::MAX, 0u16);
                let node_coords: (usize, usize) = (nx * node_size, nz * node_size);
                // for each sample within this node
                for z in node_coords.1..(node_coords.1 + node_size + 1) {
                    for &v in &heights[z * level_size + node_coords.0..][..(node_size + 1)] {
                        min_h = min_h.min(v);
                        max_h = max_h.max(v);
                    }
                }
                *mm = [min_h, max_h];
            }
        });

    let mut levels = vec![level0];
    for lod in 1..lod_count as usize {
        let prev = levels.last().unwrap();
        let prev_n = node_count >> (lod - 1);
        let node_count = node_count >> lod;
        let mut level = vec![[0u16; 2]; node_count * node_count];
        level
            .par_chunks_mut(node_count)
            .enumerate()
            .for_each(|(z, row)| {
                let r0 = &prev[2 * z * prev_n..][..prev_n];
                let r1 = &prev[(2 * z + 1) * prev_n..][..prev_n];
                for (x, mm) in row.iter_mut().enumerate() {
                    let c = [r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]];
                    *mm = [
                        c.iter().map(|m| m[0]).min().unwrap(),
                        c.iter().map(|m| m[1]).max().unwrap(),
                    ];
                }
            });
        levels.push(level);
    }
    (node_count as u32, levels)
}

pub(crate) fn get_min_max_texture(
    node_count: u32,
    levels: Vec<Vec<[u16; 2]>>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> wgpu::Texture {
    let mm_bytes: Vec<u8> = levels
        .iter()
        .flatten()
        .flat_map(|&[min, max]| [min.to_le_bytes(), max.to_le_bytes()])
        .flatten()
        .collect();

    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("minmax texture"),
            size: wgpu::Extent3d {
                width: node_count,
                height: node_count,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &mm_bytes,
    )
}

pub(crate) fn build_compressed_tile_files(
    tile_layout: &TileLayout,
    mips: &HeightMips,
    tile_path: &Path,
) -> std::io::Result<()> {
    let tile_sample_count = (tile_layout.tile_size + 1) as usize;

    let mut tile_ids: Vec<(u32, u32, u32)> = Vec::new();
    let mip_count = tile_layout.max_mip + 1;
    for mip in (0..mip_count).rev() {
        let [tx_count, tz_count] = tile_layout.tiles_per_axis(mip);
        for z in 0..tz_count {
            for x in 0..tx_count {
                tile_ids.push((mip, x, z));
            }
        }
    }

    let payloads: Vec<Vec<u8>> = tile_ids
        .par_iter()
        .map(|&(mip, tile_x, tile_z)| {
            let x0 = tile_x as usize * tile_layout.tile_size as usize;
            let mut tile =
                Vec::with_capacity((tile_layout.tile_size * tile_layout.tile_size) as usize);
            for j in 0..tile_sample_count as u32 {
                tile.extend_from_slice(
                    &mips.row(mip, tile_z * tile_layout.tile_size + j)[x0..x0 + tile_sample_count],
                );
            }
            let mut buf = Vec::new();
            crate::compress::encode_heights(&tile, tile_sample_count, &mut buf);
            buf
        })
        .collect();
    let samples: Vec<&[u8]> = payloads.iter().step_by(4).map(|p| p.as_slice()).collect();
    let dict = zstd::dict::from_samples(&samples, 64 * 1024)?;
    let mut compressor = zstd::bulk::Compressor::with_dictionary(19, &dict)?;
    compressor.set_parameter(CParameter::ChecksumFlag(false))?;
    compressor.set_parameter(CParameter::ContentSizeFlag(false))?;
    compressor.set_parameter(CParameter::DictIdFlag(false))?;

    let mut blob = Vec::new();
    let mut offsets = vec![0u64]; // keep track of blob offsets for file indexing
    for payload in payloads.iter() {
        blob.extend_from_slice(&compressor.compress(payload)?);
        offsets.push(blob.len() as u64);
    }

    //let non_dict_compressor = zstd::bulk::Compressor::new(19);
    //non_dict_compressor.set_parameter(CParameter::ChecksumFlag(false))?;
    //non_dict_compressor.set_parameter(CParameter::ContentSizeFlag(false))?;
    //let mut blob_nd = Vec::new();
    //for payload in payloads.iter() {
    //    blob_nd.extend_from_slice(&compressor.compress(payload)?);
    //}

    const TILE_FILE_MAGIC: [u8; 4] = *b"CDLT";
    const TILE_FILE_VERSION: u32 = 1;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(tile_path)?);
    writer.write_all(&TILE_FILE_MAGIC)?;

    for v in [
        TILE_FILE_VERSION,
        tile_sample_count as u32,
        payloads.len() as u32,
        dict.len() as u32,
    ] {
        writer.write_all(&v.to_le_bytes())?;
    }
    writer.write_all(&dict)?;

    for offset in &offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }
    writer.write_all(&blob)?;
    writer.flush()?;

    Ok(())
}

pub fn build_height_mipmap_cpu(
    heightmap: &image::DynamicImage,
    tile_layout: &TileLayout,
) -> HeightMips {
    let luma = heightmap.to_luma16();
    let n = tile_layout.size as usize;
    let (src_w, src_h) = (luma.width() as usize, luma.height() as usize);
    assert!(
        src_w + 1 >= n && src_h + 1 >= n,
        "source {src_w}x{src_h} is too small for terrain size {n}"
    );
    let src = luma.as_raw();

    let mut level0 = vec![0u16; n * n];
    level0.par_chunks_mut(n).enumerate().for_each(|(z, row)| {
        let src_row = &src[z.min(src_h - 1) * src_w..][..src_w];
        for (x, dst) in row.iter_mut().enumerate() {
            *dst = src_row[x.min(src_w - 1)];
        }
    });
    drop(luma);

    let mut levels = vec![level0];
    let mut level_sizes = vec![n as u32];
    for _ in 1..=tile_layout.max_mip {
        let prev = levels.last().unwrap();
        let prev_n = *level_sizes.last().unwrap() as usize;
        let n = (prev_n - 1) / 2 + 1;
        let mut level = vec![0u16; n * n];
        // point decimation: dst[z][x] = prev[2z][2x]
        level.par_chunks_mut(n).enumerate().for_each(|(z, row)| {
            let src_row = &prev[2 * z * prev_n..][..prev_n];
            for (dst, &h) in row.iter_mut().zip(src_row.iter().step_by(2)) {
                *dst = h;
            }
        });
        levels.push(level);
        level_sizes.push(n as u32);
    }

    HeightMips {
        level_sizes,
        levels,
    }
}

// ******* unused gpu code ***************************************
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
    let w = tile_layout.size;
    let luma = image::imageops::crop_imm(&heightmap.to_luma16(), 0, 0, w, w).to_image();

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
pub(crate) struct MipChain {
    pub texture: wgpu::Texture,
    pub level_sizes: Vec<[u32; 2]>,
}
