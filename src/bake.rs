use image::{ImageBuffer, Luma};

use crate::tiles::TerrainSize;

pub(crate) struct ConformedHeightMap {
    image: image::ImageBuffer<Luma<u16>, Vec<u16>>,
    footprint: [f32; 2],
}

pub(crate) fn conform_heightmap(
    source: &ImageBuffer<Luma<u16>, Vec<u16>>,
    size: TerrainSize,
) -> ConformedHeightMap {
    let (w, h) = source.dimensions();

    let target = size.samples();

    let square = w.max(h);

    let (image, scale) = if square > target {
        let resized = if w == h {
            image::imageops::resize(
                source,
                target,
                target,
                image::imageops::FilterType::Triangle,
            )
        } else {
            image::imageops::resize(
                &pad_edge(source, square, square),
                target,
                target,
                image::imageops::FilterType::Triangle,
            )
        };
        (resized, (target - 1) as f32 / (square - 1) as f32)
    } else {
        (pad_edge(source, target, target), 1.0)
    };

    let n = size.intervals() as f32;
    ConformedHeightMap {
        image,
        footprint: [(w - 1) as f32 * scale / n, (h - 1) as f32 * scale / n],
    }
}
/// Pad the diffuse by the same proportions as the heightmap so the two line up,
/// then resize it to `side` x `side` texels.
pub(crate) fn conform_diffuse(
    source: &image::RgbaImage,
    footprint: [f32; 2],
    side: u32,
) -> image::RgbaImage {
    let (w, h) = source.dimensions();
    let canvas_w = (w as f32 / footprint[0]).round() as u32;
    let canvas_h = (h as f32 / footprint[1]).round() as u32;
    image::imageops::resize(
        &pad_edge(source, canvas_w, canvas_h),
        side,
        side,
        image::imageops::FilterType::Triangle,
    )
}
fn pad_edge<P: image::Pixel + 'static>(
    img: &ImageBuffer<P, Vec<P::Subpixel>>,
    w: u32,
    h: u32,
) -> ImageBuffer<P, Vec<P::Subpixel>> {
    let (sw, sh) = img.dimensions();
    ImageBuffer::from_fn(w, h, |x, y| *img.get_pixel(x.min(sw - 1), y.min(sh - 1)))
}
