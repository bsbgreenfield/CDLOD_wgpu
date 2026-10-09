use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use wgpu::util::DeviceExt;

use crate::tiles::{TileLayout, TileStore, WorldValues};

const NOT_RESIDENT: u16 = 0xFFFF;

const READBACK_RING: usize = 3;

pub struct StreamConfig {
    pub slots_per_axis: u32,

    pub request_capacity: u32,

    pub uploads_per_frame: usize,

    pub max_inflight: usize,

    pub stale_frames: u32,
}

use std::sync::mpsc::{Receiver, Sender};
pub(crate) struct TileStreamer {
    layout: TileLayout,
    config: StreamConfig,
    frame: u32,

    free_staging: Vec<usize>, // free list

    mapped_tx: Sender<(usize, bool)>,
    mapped_rx: Receiver<(usize, bool)>,

    // CPU mirror of residency
    tile_slot: Vec<u16>,
    slot_tile: Vec<u32>,
    free_slots: Vec<u16>,
    /// root tiles live in slots 0..pinned and are never evicted
    pinned: usize,

    /// tile -> frame it was last requested. Removed once uploaded or dropped
    pending: HashMap<u32, u32>,
    /// pending tiles not yet sent to the decoder
    fifo: VecDeque<u32>,
    jobs: Sender<u32>,
    decoded: Receiver<(u32, Vec<u8>)>,
    /// decoded tiles waiting on the upload budget or a free slot
    ready: VecDeque<(u32, Vec<u8>)>,
    in_flight: usize,
}

pub(crate) struct GPUTerrainData {
    /// the gpu resident heightmap texture
    pub atlas: wgpu::Texture,
    pub atlas_view: wgpu::TextureView,

    /// tile -> slot map
    pub residency: wgpu::Texture,
    pub residency_view: wgpu::TextureView,

    /// residency markers for compute shader request dedup
    pub request_flags: wgpu::Buffer,

    pub requests: wgpu::Buffer,

    staging: Vec<wgpu::Buffer>,
}

impl GPUTerrainData {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &TileLayout,
        config: &StreamConfig,
    ) -> Self {
        let n = layout.tile_size + 1;
        let tile_count = layout.tile_count() as usize;

        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("height atlas"),
            size: wgpu::Extent3d {
                width: config.slots_per_axis * n,
                height: config.slots_per_axis * n,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // mip m is tiles_per_axis(m) wide, so the whole chain is exactly tile_count texels
        let [tiles_x, tiles_z] = layout.tiles_per_axis(0);
        let residency = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("tile residency"),
                size: wgpu::Extent3d {
                    width: tiles_x,
                    height: tiles_z,
                    depth_or_array_layers: 1,
                },
                mip_level_count: layout.max_mip + 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &vec![0xFF; tile_count * 2], // all NOT_RESIDENT
        );

        let request_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile request flags"),
            size: tile_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let request_bytes = 4 + config.request_capacity as u64 * 4;
        let requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile requests"),
            size: request_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = (0..READBACK_RING)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tile request readback"),
                    size: request_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("atlas text view"),
            format: Some(wgpu::TextureFormat::R16Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let residency_view = residency.create_view(&wgpu::TextureViewDescriptor {
            label: Some("res tex view"),
            format: Some(wgpu::TextureFormat::R16Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        Self {
            atlas_view,
            residency_view,
            atlas,
            residency,
            request_flags,
            requests,
            staging,
        }
    }
}

impl TileStreamer {
    pub fn new(
        queue: &wgpu::Queue,
        layout: &TileLayout,
        store: Arc<TileStore>,
        config: StreamConfig,
        gpu_data: &GPUTerrainData,
    ) -> Self {
        let tile_count = layout.tile_count() as usize;
        let slot_count = (config.slots_per_axis * config.slots_per_axis) as usize;
        assert!(
            slot_count < NOT_RESIDENT as usize,
            "slot index must fit in u16"
        );
        let pinned = (layout.root_tiles[0] * layout.root_tiles[1]) as usize;
        assert!(pinned < slot_count, "atlas can't hold the root tiles");
        let (mapped_tx, mapped_rx) = std::sync::mpsc::channel();

        // decoder thread: tile index in, upload-ready bytes out
        let (jobs, job_rx) = std::sync::mpsc::channel::<u32>();
        let (decoded_tx, decoded) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("tile decoder".into())
            .spawn(move || {
                let mut dec = store.decoder().expect("zstd decoder");
                let mut heights = vec![0u16; store.n * store.n];
                for tile in job_rx {
                    store
                        .load(&mut dec, tile, &mut heights)
                        .expect("corrupt tile file");
                    let bytes = heights.iter().flat_map(|h| h.to_le_bytes()).collect();
                    if decoded_tx.send((tile, bytes)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn tile decoder");

        let mut streamer = Self {
            layout: layout.clone(),
            config,
            frame: 0,
            free_staging: (0..READBACK_RING).collect(),
            mapped_tx,
            mapped_rx,
            tile_slot: vec![NOT_RESIDENT; tile_count],
            slot_tile: vec![u32::MAX; slot_count],
            free_slots: (pinned..slot_count).rev().map(|s| s as u16).collect(),
            pinned,
            pending: HashMap::new(),
            fifo: VecDeque::new(),
            jobs,
            decoded,
            ready: VecDeque::new(),
            in_flight: 0,
        };

        // roots are first in flat order: load them synchronously into slots 0..pinned
        for tile in 0..pinned as u32 {
            streamer.jobs.send(tile).unwrap();
        }
        for _ in 0..pinned {
            let (tile, bytes) = streamer.decoded.recv().expect("tile decoder died");
            streamer.upload(
                queue,
                tile,
                tile as u16,
                &bytes,
                &gpu_data.atlas,
                &gpu_data.residency,
            );
        }
        streamer
    }

    /// Call once per frame, before encoding. Everything it writes goes through the queue,
    /// so it lands before the next submit, i.e. before this frame's node selection.
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        gpu_data: &GPUTerrainData,
        world: &WorldValues,
    ) {
        self.frame += 1;
        let _ = device.poll(wgpu::PollType::Poll); // runs map callbacks of finished readbacks

        // for the pending request frames
        // map the requested tile to cpu memory and add to pending
        while let Ok((i, mapped)) = self.mapped_rx.try_recv() {
            if mapped {
                self.ingest(i, gpu_data);
            }
            // this buffer is free to be written to
            self.free_staging.push(i);
        }

        // FIFO -> decoder
        while self.in_flight < self.config.max_inflight {
            // get the next tile to process
            let Some(tile) = self.fifo.pop_front() else {
                break;
            };
            // check to see if the tile is still requested
            // if not, take off of pending
            if !self.wanted(tile) {
                self.pending.remove(&tile);
                continue;
            }

            // add in flight job
            self.jobs.send(tile).expect("tile decoder died");
            self.in_flight += 1;
        }

        // decoder -> GPU
        self.ready.extend(self.decoded.try_iter());
        let mut uploaded = 0;

        // loop through the received  tile bytes
        while uploaded < self.config.uploads_per_frame {
            let Some((tile, bytes)) = self.ready.pop_front() else {
                break;
            };
            if !self.wanted(tile) {
                self.pending.remove(&tile);
                self.in_flight -= 1;
                continue;
            }
            let Some(slot) = self.alloc_slot(queue, camera_pos, &gpu_data.residency, world) else {
                // every resident tile is still in range: retry next frame
                self.ready.push_front((tile, bytes));
                break;
            };
            self.upload(
                queue,
                tile,
                slot,
                &bytes,
                &gpu_data.atlas,
                &gpu_data.residency,
            );
            // tile data and residency are queued together, so this is "done"
            self.pending.remove(&tile);
            self.in_flight -= 1;
            uploaded += 1;
        }
    }

    /// Record after the node selection passes. Also resets the request buffers for next frame.
    pub fn record_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gpu_data: &GPUTerrainData,
    ) {
        // every staging buffer busy: skip, unserved tiles get re-requested next frame
        if let Some(i) = self.free_staging.pop() {
            let staging = &gpu_data.staging[i];
            encoder.copy_buffer_to_buffer(&gpu_data.requests, 0, staging, 0, staging.size());
            let tx = self.mapped_tx.clone();
            encoder.map_buffer_on_submit(staging, wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send((i, r.is_ok()));
            });
        }
        // the shader dedupes within a frame, `pending` dedupes across frames
        encoder.clear_buffer(&gpu_data.request_flags, 0, None);
        encoder.clear_buffer(&gpu_data.requests, 0, Some(4));
    }

    fn ingest(&mut self, i: usize, gpu_data: &GPUTerrainData) {
        let mut batch: Vec<u32> = {
            let view = gpu_data.staging[i]
                .get_mapped_range(..)
                .expect("readback not mapped");
            let count = u32::from_le_bytes(view[..4].try_into().unwrap())
                .min(self.config.request_capacity) as usize;
            view[4..][..count * 4]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        gpu_data.staging[i].unmap();

        // flat indices run coarse -> fine, so coarser tiles queue first
        batch.sort_unstable();
        for tile in batch {
            // uploaded since this readback was recorded
            if self.tile_slot[tile as usize] != NOT_RESIDENT {
                continue;
            }
            if self.pending.insert(tile, self.frame).is_none() {
                self.fifo.push_back(tile);
            }
        }
    }

    /// Still being requested, and still reachable: its parent may have been
    /// evicted while it was pending.
    fn wanted(&self, tile: u32) -> bool {
        let fresh = self
            .pending
            .get(&tile)
            .is_some_and(|&seen| self.frame - seen <= self.config.stale_frames);
        let (mip, [x, z]) = self.layout.tile_from_flat(tile);
        let parent_resident = mip == self.layout.max_mip
            || self.tile_slot[self.layout.tile_flat_index(mip + 1, [x / 2, z / 2]) as usize]
                != NOT_RESIDENT;
        fresh && parent_resident
    }

    // TODO: write a custom eviction strategy
    // This currently works by booting a slot that is not needed
    // whenever we have to allocate a new tile, but
    // really it should work based off of some scan that evicts nodes that are not
    // needed every x frames, iterating based off of the ratio of distance to camera
    // and lod level. i.e. (a super far away node that is at a low lod is likely to get booted)
    // this works for now though
    fn alloc_slot(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        residency: &wgpu::Texture,
        world: &WorldValues,
    ) -> Option<u16> {
        if let Some(slot) = self.free_slots.pop() {
            return Some(slot);
        }
        let slot = self.pick_victim(camera_pos, world)?;
        self.set_residency(
            queue,
            self.slot_tile[slot as usize],
            NOT_RESIDENT,
            residency,
        );
        Some(slot)
    }

    fn pick_victim(&self, cam: [f32; 3], world: &WorldValues) -> Option<u16> {
        (self.pinned..self.slot_tile.len())
            .map(|slot| {
                let (mip, [tx, tz]) = self.layout.tile_from_flat(self.slot_tile[slot]);
                let span = (self.layout.tile_size << mip) as f32;
                let dist = |axis: usize, t: u32| {
                    let lo = world.map_min[axis] + t as f32 * span * world.scale_factors[axis];
                    let hi = lo + span * world.scale_factors[axis];
                    (lo - cam[axis]).max(cam[axis] - hi).max(0.0)
                };
                let d = dist(0, tx).hypot(dist(2, tz));
                (slot as u16, d / world.lod_ranges[mip as usize])
            })
            .filter(|&(_, ratio)| ratio > 1.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(slot, _)| slot)
    }

    fn upload(
        &mut self,
        queue: &wgpu::Queue,
        tile: u32,
        slot: u16,
        bytes: &[u8],
        atlas: &wgpu::Texture,
        residency: &wgpu::Texture,
    ) {
        let n = self.layout.tile_size + 1;
        let s = slot as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: s % self.config.slots_per_axis * n,
                    y: s / self.config.slots_per_axis * n,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 2),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: 1,
            },
        );
        self.slot_tile[slot as usize] = tile;
        self.set_residency(queue, tile, slot, residency);
    }

    fn set_residency(
        &mut self,
        queue: &wgpu::Queue,
        tile: u32,
        slot: u16,
        residency: &wgpu::Texture,
    ) {
        self.tile_slot[tile as usize] = slot;
        let (mip, [x, y]) = self.layout.tile_from_flat(tile);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: residency,
                mip_level: mip,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &slot.to_le_bytes(),
            wgpu::TexelCopyBufferLayout::default(),
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
}
