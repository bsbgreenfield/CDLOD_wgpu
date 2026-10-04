struct QuadNode {
    minH: u16,
    maxH: u16,
}

struct NodeGrid {
    nodes: Vec<QuadNode>,
    side_len: usize,
}

impl NodeGrid {
    pub fn new(side_len: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(side_len * side_len),
            side_len,
        }
    }

    pub fn get_min_max_h(&self, x: usize, z: usize) -> &QuadNode {
        return &self.nodes[(x + z * self.side_len) * 2];
    }
}
const LOD_COUNT: usize = 5;

type QuadMap = [NodeGrid; LOD_COUNT];

fn main() {
    println!("Hello, world!");
}
