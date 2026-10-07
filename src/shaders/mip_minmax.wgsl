
struct LevelInfo {
	node_size: u32, //node size in sample space
}


@group(0) @binding(0) var params: 
@group(0) @binding(1) var<storage, read_write> min_max_heights: array<atomic<vec2<u32>>>;








@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id): gid: vec3<32>) {
	
}

