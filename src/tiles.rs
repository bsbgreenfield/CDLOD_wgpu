use crate::{CDLODSettings, mips::MipChain};

const MAX_LODS: u32 = 16;

pub(crate) struct TileLayout {
    /// The top mip level which equals the top quad LOD. Contains the root tiles
    pub max_mip: u32,
    /// The number of samples per tile axis in mip space.
    /// A tile actually contains tile_size + 1 for morphs
    pub tile_size: u32,
    pub nodes_per_tile_axis: u32,

    /// the number of root tiles at the max mip
    pub root_tiles: [u32; 2],

    /// the source size in samples after cropping
    pub cropped_size: [u32; 2],
}

impl TileLayout {
    pub fn new(settings: &CDLODSettings, width: u32, height: u32) -> Self {
        let node_size = settings.min_quad_node_size.get();
        let nodes_per_tile_axis = settings.nodes_per_tile_axis.get();

        assert!(
            nodes_per_tile_axis % 2 == 0,
            "nodes per tile axis must be an even number"
        );

        // tile size is constant in mip space
        let tile_size = node_size * nodes_per_tile_axis;

        // sample count for the tiles
        let intervals = (width - 1).min(height - 1);

        assert!(
            intervals >= tile_size,
            "source must span at least one tile. 
            Consider decreasing min_quad_node_size in the settings"
        );

        // the number of mips comes from the number of subdivisions it takes
        // to get from a single root tile, to the point where a tile is sampling from the source
        // dataset (necessarily mip 0), or the max_LOD specified by the user
        let max_mip = (intervals / tile_size).ilog2().min(settings.max_LOD.get());
        assert!(max_mip < MAX_LODS, "The source dataset is too large");

        // the "root tiles" are the tiles at the highest mip, and
        // are special because they are always gpu resident, and are ancestors of all
        // tiles. The normal case is a single root tile.
        let root_span = tile_size << max_mip; // sample span (of source) per root
        let root_tiles = [(width - 1) / root_span, (height - 1) / root_span];

        let layout = TileLayout {
            max_mip,
            tile_size,
            nodes_per_tile_axis,
            root_tiles,
            cropped_size: [root_tiles[0] * root_span + 1, root_tiles[1] * root_span + 1],
        };
        layout
    }

    fn tiles_per_axis(&self, mip: u32) -> [u32; 2] {
        let shift = self.max_mip - mip;
        let x = self.root_tiles[0] << shift;
        let z = self.root_tiles[1] << shift;
        [x, z]
    }
    pub fn tile_flat_index(&self, mip: u32, tile_coords: [u32; 2]) -> u32 {
        let mips_coarser = self.max_mip - mip;
        let roots = self.root_tiles[0] * self.root_tiles[1];
        // the offset is dependant on the number of root tiles in the highest mip
        let offset = roots * ((1 << (2 * mips_coarser)) - 1) / 3;
        offset + tile_coords[1] * self.tiles_per_axis(mip)[0] + tile_coords[0]
    }

    fn tile_count(&self) -> u32 {
        let roots = self.root_tiles[0] * self.root_tiles[1];
        roots * ((1 << (2 * (self.max_mip + 1))) - 1) / 3
    }
}

pub(crate) struct Tiles {
    sample_axis_count: u32,
    tiles: Vec<Tile>,
}

struct Tile {
    mip_level: u32,
    coords: (u32, u32), // mip space
}

pub(crate) fn create_tiles(layout: &TileLayout) -> Tiles {
    let tile_size = layout.tile_size;
    let mut tiles: Tiles = Tiles {
        sample_axis_count: tile_size + 1,
        tiles: Vec::with_capacity(layout.tile_count() as usize),
    };

    for mip_level in (0..(layout.max_mip + 1)).rev() {
        let [tiles_x, tiles_z] = layout.tiles_per_axis(mip_level);
        for zi in 0..tiles_z {
            for xi in 0..tiles_x {
                tiles.tiles.push(Tile {
                    mip_level,
                    coords: (xi * tile_size, zi * tile_size),
                });
            }
        }
    }
    tiles
}
