#[inline]
pub(crate) fn predict(t: &[u16], n: usize, x: usize, z: usize) -> u16 {
    match (x, z) {
        (0, 0) => 0,
        (_, 0) => t[x - 1],
        (0, _) => t[(z - 1) * n],
        _ => {
            let a = t[z * n + x - 1] as i32; // left
            let b = t[(z - 1) * n + x] as i32; // up
            let c = t[(z - 1) * n + x - 1] as i32; // up-left
            let p = if c >= a.max(b) {
                a.min(b)
            } else if c <= a.min(b) {
                a.max(b)
            } else {
                a + b - c
            };
            p as u16
        }
    }
}

// split the high and low bytes so they compress better
pub(crate) fn encode_heights(tile: &[u16], n: usize, out: &mut Vec<u8>) {
    let len = n * n;
    out.clear();
    out.resize(len * 2, 0);
    let (hi, lo) = out.split_at_mut(len);
    for z in 0..n {
        for x in 0..n {
            let i = z * n + x;
            let r = tile[i].wrapping_sub(predict(tile, n, x, z)) as i16;
            let zz = ((r << 1) ^ (r >> 15)) as u16; // zigzag
            hi[i] = (zz >> 8) as u8;
            lo[i] = zz as u8;
        }
    }
}

pub(crate) fn decode_heights(bytes: &[u8], n: usize, out: &mut [u16]) {
    let len = n * n;
    let (hi, lo) = bytes.split_at(len);
    for z in 0..n {
        for x in 0..n {
            let i = z * n + x;
            let zz = (hi[i] as u16) << 8 | lo[i] as u16;
            let r = ((zz >> 1) as i16) ^ -((zz & 1) as i16); // un-zigzag
            out[i] = predict(out, n, x, z).wrapping_add(r as u16);
        }
    }
}
