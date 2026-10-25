use std::{collections::HashSet, sync::Arc};

use crate::tiles::{TileLayout, TileStore, WorldValues};

struct CPUTerrainData {
    lod_level: u32,
    distance_threshold: f32,
    grid_size: usize,
    grid: TileGrid,
    tile_layout: TileLayout,
    world: WorldValues,
    current_tiles: HashSet<usize>,
    pending_tile_set: Option<HashSet<usize>>,
    pending_top_left: Option<[u32; 2]>,
    prev_center: [f32; 2],
    request_tx: crossbeam::channel::Sender<usize>,
    decoded_rx: crossbeam::channel::Receiver<(usize, Vec<u16>)>,
    processing: bool,
    batch_count: usize,
}

#[derive(Debug, Default)]
struct TileGrid {
    data: Vec<Vec<u16>>,
    tiles: HashSet<usize>,
    grid_ids: Vec<usize>,
    top_left_coord: [u32; 2],
}

/// Upper bound on the samples `tiles_needed` can return for a square of side
/// `size` at `lod`, wherever the square is placed. Use it as scratch buffer capacity.
fn max_samples_needed(size: f32, layout: &TileLayout, scale_factors: [f32; 3], lod: u32) -> usize {
    let samples_per_tile = (layout.tile_size << lod) as f32;
    let [tiles_x, tiles_z] = layout.tiles_per_axis(lod);

    // Square side length measured in tiles, per axis
    let side_in_tiles_x = size / (samples_per_tile * scale_factors[0]);
    let side_in_tiles_z = size / (samples_per_tile * scale_factors[2]);

    // Worst case is the square straddling tile boundaries on both ends.
    // floor + 2 is always enough, even with float rounding near whole numbers.
    let max_tiles_x = (side_in_tiles_x.floor() as u32 + 2).min(tiles_x);
    let max_tiles_z = (side_in_tiles_z.floor() as u32 + 2).min(tiles_z);

    // Each stored tile has tile_size + 1 samples per axis (shared morph edge)
    let samples_per_stored_tile = ((layout.tile_size + 1) * (layout.tile_size + 1)) as usize;

    max_tiles_x as usize * max_tiles_z as usize * samples_per_stored_tile
}

impl CPUTerrainData {
    fn tiles_needed(&self) -> (HashSet<usize>, [u32; 2]) {
        let half_size = self.grid_size as f32 / 2.0;

        // How many LOD-0 samples one tile covers at this lod
        let samples_per_tile = (self.tile_layout.tile_size << self.lod_level) as f32;

        // World-space size of one tile on each axis
        let tile_world_x = samples_per_tile * self.world.scale_factors[0];
        let tile_world_z = samples_per_tile * self.world.scale_factors[2];

        // Square bounds, relative to the map's corner
        let min_x = self.prev_center[0] - half_size - self.world.map_min[0];
        let max_x = self.prev_center[0] + half_size - self.world.map_min[0];
        let min_z = self.prev_center[1] - half_size - self.world.map_min[2];
        let max_z = self.prev_center[1] + half_size - self.world.map_min[2];

        // world space to tile space, conservative
        let first_x = (min_x / tile_world_x).floor() as u32;
        let first_z = (min_z / tile_world_z).floor() as u32;
        let end_x = (max_x / tile_world_x).ceil() as u32;
        let end_z = (max_z / tile_world_z).ceil() as u32;

        // Clamp the high side to the map
        let [tiles_x, tiles_z] = self.tile_layout.tiles_per_axis(self.lod_level);
        let end_x = end_x.min(tiles_x);
        let end_z = end_z.min(tiles_z);

        let mut tiles = HashSet::new();
        for z in first_z..end_z {
            for x in first_x..end_x {
                tiles.insert(self.tile_layout.tile_flat_index(self.lod_level, [x, z]) as usize);
            }
        }
        (tiles, [first_x, first_z])
    }
    fn new(
        lod_level: u32,
        distance_threshold: f32,
        grid_size: usize,
        tile_layout: TileLayout,
        initial_center: [f32; 2],
        world: WorldValues,
        tile_store: Arc<TileStore>,
    ) -> Self {
        let (request_tx, request_rx) = crossbeam::channel::unbounded::<usize>();
        let (decoded_tx, decoded_rx) = crossbeam::channel::unbounded::<(usize, Vec<u16>)>();

        let layout = tile_layout.clone();
        std::thread::spawn(move || {
            let mut decoder = tile_store.decoder().expect("decoder");
            let tile_samples = ((layout.tile_size + 1) * (layout.tile_size + 1)) as usize;
            for tile_idx in request_rx {
                let mut scratch = vec![0u16; tile_samples];
                tile_store
                    .load(&mut decoder, tile_idx as u32, &mut scratch)
                    .expect("could not read tile file");

                if decoded_tx.send((tile_idx, scratch)).is_err() {
                    break;
                }
            }
        });

        Self {
            lod_level,
            distance_threshold,
            grid_size,
            grid: TileGrid::default(),
            tile_layout,
            world,
            current_tiles: HashSet::new(),
            pending_tile_set: None,
            pending_top_left: None,
            prev_center: initial_center,
            request_tx,
            decoded_rx,
            processing: false,
            batch_count: 0,
        }
    }

    fn update(&mut self, new_center: [f32; 2]) {
        let dist = ((new_center[0] - self.prev_center[0]).powi(2)
            + (new_center[1] - self.prev_center[1]).powi(2))
        .sqrt();

        if dist >= self.distance_threshold && !self.processing {
            self.prev_center = new_center;
            let (tiles, top_left) = self.tiles_needed();
            self.processing = true;
            let diff_iter = tiles.difference(&self.current_tiles);

            // shoot out the new request batch
            for new_tile in diff_iter {
                self.batch_count += 1;
                self.request_tx
                    .send(*new_tile)
                    .expect("shouldnt be limited");
            }
            self.pending_tile_set = Some(tiles);
            self.pending_top_left = Some(top_left);
        } else if self.processing && self.decoded_rx.len() == self.batch_count {
            //TODO: write the new data (along with the tiles that are still valid, into anew grid)

            self.processing = false;
            self.batch_count = 0;
        }
    }
}
