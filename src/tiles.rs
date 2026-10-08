use crate::{CDLODSettings, MapDimensions};

const MAX_LODS: u32 = 16;
const TILE_FILE_MAGIC: [u8; 4] = *b"CDLT";
const TILE_FILE_VERSION: u32 = 1;
const TILE_FILE_HEADER_LEN: usize = 20;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainSize {
    T2,
    T4,
    T8,
    T16,
}

impl TerrainSize {
    pub(crate) const fn intervals(self) -> u32 {
        match self {
            TerrainSize::T2 => 2048,
            TerrainSize::T4 => 4096,
            TerrainSize::T8 => 8192,
            TerrainSize::T16 => 16384,
        }
    }

    pub(crate) const fn samples(self) -> u32 {
        self.intervals() + 1
    }
}

#[derive(Debug, Clone)]
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
    pub size: u32,
}

pub(crate) struct WorldValues {
    pub map_min: [f32; 3],       // offset of height samples
    pub scale_factors: [f32; 3], // xyz scale factors (tile space -> world space)
    pub lod_ranges: Vec<f32>,
}

impl WorldValues {
    pub fn new(settings: &CDLODSettings, layout: &TileLayout) -> Self {
        let MapDimensions { map_min, size } = settings.map;
        let intervals = settings.terrain_size.intervals() as f32;
        let scale_factors = [
            size[0] / intervals,
            size[1] / u16::MAX as f32,
            size[2] / intervals,
        ];

        let lod_ranges = (0..layout.max_mip + 1)
            .map(|lod| settings.lod0_range * 2f32.powi(lod as i32))
            .collect();

        Self {
            map_min,
            scale_factors,
            lod_ranges,
        }
    }
}
pub(crate) struct TileStore {
    pub(crate) offsets: Vec<u64>,
    pub(crate) blob: memmap2::Mmap,
    pub(crate) ddict: zstd::dict::DecoderDictionary<'static>,
    pub(crate) n: usize,
}

pub(crate) struct TileDecoder<'a> {
    decompressor: zstd::bulk::Decompressor<'a>,
    scratch: Vec<u8>, // persistent allocation for moving data
}

impl TileStore {
    pub fn decoder(&self) -> std::io::Result<TileDecoder<'_>> {
        Ok(TileDecoder {
            decompressor: zstd::bulk::Decompressor::with_prepared_dictionary(&self.ddict)?,
            scratch: vec![0; self.n * self.n * 2],
        })
    }

    /// tile_index comes straight from RequestList.tiles[]
    pub(crate) fn load(
        &self,
        dec: &mut TileDecoder,
        tile_index: u32,
        out: &mut [u16],
    ) -> std::io::Result<()> {
        let i = tile_index as usize;
        let src = &self.blob[self.offsets[i] as usize..self.offsets[i + 1] as usize];
        dec.decompressor
            .decompress_to_buffer(src, dec.scratch.as_mut_slice())?;
        crate::compress::decode_heights(&dec.scratch, self.n, out);
        Ok(())
    }
    pub(crate) fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let bad = |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string());
        let file = std::fs::File::open(path)?;
        // SAFETY: the tile file must not be modified while it is mapped
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < TILE_FILE_HEADER_LEN || map[..4] != TILE_FILE_MAGIC {
            return Err(bad("not a tile file"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
        if u32_at(4) != TILE_FILE_VERSION {
            return Err(bad("unsupported tile file version"));
        }
        let [n, count, dict_len] = [8, 12, 16].map(|o| u32_at(o) as usize);

        let dict_end = TILE_FILE_HEADER_LEN + dict_len;
        let blob_start = dict_end + (count + 1) * 8;
        if map.len() < blob_start {
            return Err(bad("truncated tile file"));
        }
        let ddict = zstd::dict::DecoderDictionary::copy(&map[TILE_FILE_HEADER_LEN..dict_end]);
        // stored relative to the blob; rebase so load() can slice the whole map
        let offsets: Vec<u64> = map[dict_end..blob_start]
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) + blob_start as u64)
            .collect();
        if offsets[count] as usize > map.len() {
            return Err(bad("truncated tile file"));
        }
        Ok(Self {
            offsets,
            blob: map,
            ddict,
            n,
        })
    }
}

impl TileLayout {
    // given a tile index, get the tile
    pub(crate) fn tile_from_flat(&self, mut index: u32) -> (u32, [u32; 2]) {
        for mip in (0..self.max_mip + 1).rev() {
            let [x, z] = self.tiles_per_axis(mip);
            if index < x * z {
                return (mip, [index % x, index / x]);
            }
            index -= x * z
        }
        panic!("tile index out of range");
    }
    pub fn new(settings: &CDLODSettings, width: u32, height: u32) -> Self {
        let node_size = settings.min_quad_node_size.get();
        let nodes_per_tile_axis = settings.nodes_per_tile_axis.get();

        assert!(
            nodes_per_tile_axis % 2 == 0,
            "nodes per tile axis must be an even number"
        );

        // tile size is constant in mip space
        let tile_size = node_size * nodes_per_tile_axis;

        let intervals = settings.terrain_size.intervals();

        assert!(
            tile_size.is_power_of_two(),
            "min_quad_node_size * nodes_per_tile_axis must be a power of 2"
        );

        assert!(
            intervals >= tile_size,
            "source must span at least one tile. 
            Consider decreasing min_quad_node_size in the settings"
        );

        let tiles_per_axis = intervals / tile_size;
        let max_mip = tiles_per_axis.ilog2().min(settings.max_LOD.get());
        assert!(max_mip < MAX_LODS, " the source dataset is too large");

        let roots = tiles_per_axis >> max_mip;

        let layout = TileLayout {
            max_mip,
            tile_size,
            nodes_per_tile_axis,
            root_tiles: [roots, roots],
            size: settings.terrain_size.samples(),
        };
        layout
    }

    pub(crate) fn tiles_per_axis(&self, mip: u32) -> [u32; 2] {
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

    pub(crate) fn tile_count(&self) -> u32 {
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
