use image::DynamicImage;
use std::{error::Error, num::NonZero, path::PathBuf};
use wgpu::{
    ComputePipelineDescriptor, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

fn load_terrain(
    height_path: PathBuf,
    diffuse_path: PathBuf,
) -> Result<(DynamicImage, DynamicImage), Box<dyn Error>> {
    let heightmap = image::ImageReader::open(height_path)?.decode()?;
    let diffuse = image::ImageReader::open(diffuse_path)?.decode()?;
    Ok((heightmap, diffuse))
}

fn main() {
    let (heightmap, diffuse) = load_terrain(
        PathBuf::from("./res/heights.png"),
        PathBuf::from("./res/diffuse.png"),
    )
    .expect("should load images");
}
