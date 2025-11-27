pub fn render_pix_sort_depth(
    pnt2pixcodepth: &[f32],
    pnt2rgb: &[f32],
    img_width: usize,
    img_data: &mut [[f32; 3]],
) -> anyhow::Result<()> {
    let img_height = img_data.len() / img_width;
    let num_pnt = pnt2pixcodepth.len() / 3;
    // make index of point sorting
    let idx2pnt = {
        let mut idx2pnt: Vec<usize> = (0..num_pnt).collect();
        idx2pnt.sort_by(|&idx0, &idx1| {
            let z0 = pnt2pixcodepth[idx0 * 3 + 2] + 1f32;
            let z1 = pnt2pixcodepth[idx1 * 3 + 2] + 1f32;
            z0.partial_cmp(&z1).unwrap()
        });
        idx2pnt
    };
    for &i_point in idx2pnt.iter() {
        let ndc_z = pnt2pixcodepth[i_point * 3 + 2];
        if ndc_z <= -1f32 || ndc_z >= 1f32 {
            continue;
        } // clipping
        let r0 = arrayref::array_ref![pnt2pixcodepth, i_point * 3, 2];
        let rgb = arrayref::array_ref![pnt2rgb, i_point * 3, 3];
        let ix = r0[0] as usize;
        let iy = r0[1] as usize;
        if ix >= img_width {
            continue;
        }
        if iy >= img_height {
            continue;
        }
        let ipix = iy * img_width + ix;
        img_data[ipix][0] = rgb[0];
        img_data[ipix][1] = rgb[1];
        img_data[ipix][2] = rgb[2];
    }
    Ok(())
}

pub fn render_gauss_sort_depth<Path>(
    pnt2pixxydepth: &[f32],
    pnt2pixconvinv: &[f32],
    pnt2pixaabb: &[f32],
    pnt2opacity: &[f32],
    pnt2rgb: &[f32],
    img_shape: (usize, usize),
    path: Path,
) -> anyhow::Result<()>
where
    Path: AsRef<std::path::Path>,
{
    let num_pnt = pnt2pixxydepth.len() / 3;
    let idx2pnt = {
        let mut idx2pnt: Vec<usize> = (0..num_pnt).collect();
        idx2pnt.sort_by(|&i, &j| {
            let zi = pnt2pixxydepth[i * 3 + 2] + 1f32;
            let zj = pnt2pixxydepth[j * 3 + 2] + 1f32;
            zi.partial_cmp(&zj).unwrap()
        });
        idx2pnt
    };
    // visualize as Gaussian without tile acceleration
    let mut img_data = vec![0f32; img_shape.1 * img_shape.0 * 3];
    for (ih, iw) in itertools::iproduct!(0..img_shape.1, 0..img_shape.0) {
        let t = [iw as f32 + 0.5, ih as f32 + 0.5];
        let mut alpha_sum = 0f32;
        let mut alpha_occu = 1f32;
        // draw front to back
        for idx in (0..idx2pnt.len()).rev() {
            let i_pnt = idx2pnt[idx];
            if pnt2pixxydepth[i_pnt * 3 + 2] <= -1f32 || pnt2pixxydepth[i_pnt * 3 + 2] >= 1f32 {
                continue;
            }
            let aabb = arrayref::array_ref![pnt2pixaabb, i_pnt * 4, 4];
            if !del_geo_core::aabb2::is_include_point2(aabb, &[t[0], t[1]]) {
                continue;
            }
            let pixxy = arrayref::array_ref![pnt2pixxydepth, i_pnt * 3, 2];
            let pixconvinv = arrayref::array_ref![pnt2pixconvinv, i_pnt * 3, 3];
            let rgb = arrayref::array_ref![pnt2rgb, i_pnt * 3, 3];
            let alpha = pnt2opacity[i_pnt];
            let t0 = [t[0] - pixxy[0], t[1] - pixxy[1]];
            let e = del_geo_core::mat2_sym::mult_vec_from_both_sides(pixconvinv, &t0, &t0);
            let e = (-0.5 * e).exp() * alpha;
            let e_out = alpha_occu * e;
            img_data[(ih * img_shape.0 + iw) * 3] += rgb[0] * e_out;
            img_data[(ih * img_shape.0 + iw) * 3 + 1] += rgb[1] * e_out;
            img_data[(ih * img_shape.0 + iw) * 3 + 2] += rgb[2] * e_out;
            alpha_occu *= 1f32 - e;
            alpha_sum += e_out;
            if alpha_sum > 0.999 {
                break;
            }
        }
    }
    del_canvas::write_png_from_float_image_rgb(path, &img_shape, &img_data)?;
    Ok(())
}
