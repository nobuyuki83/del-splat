/*
pub trait XyRgb {
    fn pos2_rgb(&self) -> ([f32; 2], [f32; 3]);
}

pub fn draw<POINT: XyRgb>(img_size: &(usize, usize), points: &[POINT]) -> Vec<f32> {
    let mut img_data = vec![0f32; img_size.1 * img_size.0 * 3];
    for point in points.iter() {
        //let x = (point.pos_ndc[0] + 1.0) * 0.5 * (img_size.0 as f32);
        //let y = (1.0 - point.pos_ndc[1]) * 0.5 * (img_size.1 as f32);
        let (scrn_pos, color) = point.pos2_rgb();
        let i_x = scrn_pos[0] as usize;
        let i_y = scrn_pos[1] as usize;
        img_data[(i_y * img_size.0 + i_x) * 3] = color[0];
        img_data[(i_y * img_size.0 + i_x) * 3 + 1] = color[1];
        img_data[(i_y * img_size.0 + i_x) * 3 + 2] = color[2];
    }
    img_data
}


pub trait NdcZ {
    fn ndc_z(&self) -> f32;
}

pub fn draw_pix_sort_z<S: XyRgb + NdcZ>(
    pnt2xyrgb: &[S],
    img_width: usize,
    img_data: &mut [[f32;3]],
) -> anyhow::Result<()>
{
    let img_height = img_data.len() / img_width;
    // make index of point sorting
    let idx2pnt = {
        let mut idx2pnt: Vec<usize> = (0..pnt2xyrgb.len()).collect();
        idx2pnt.sort_by(|&idx0, &idx1| {
            let z0 = pnt2xyrgb[idx0].ndc_z() + 1f32;
            let z1 = pnt2xyrgb[idx1].ndc_z() + 1f32;
            z0.partial_cmp(&z1).unwrap()
        });
        idx2pnt
    };
    for i_idx in 0..pnt2xyrgb.len() {
        let i_point = idx2pnt[i_idx];
        let ndc_z = pnt2xyrgb[i_point].ndc_z();
        if ndc_z <= -1f32 || ndc_z >= 1f32 {
            continue;
        } // clipping
        let (r0, rgb) = pnt2xyrgb[i_point].pos2_rgb();
        let ix = r0[0] as usize;
        let iy = r0[1] as usize;
        if ix >= img_width { continue; }
        if iy >= img_height { continue; }
        let ipix = iy * img_width + ix;
        img_data[ipix][0] = rgb[0];
        img_data[ipix][1] = rgb[1];
        img_data[ipix][2] = rgb[2];
    }
    Ok(())
}
 */

pub fn draw_pix_sort_z_(
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
