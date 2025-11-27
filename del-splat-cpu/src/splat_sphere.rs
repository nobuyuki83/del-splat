pub fn render_circle_sort_depth<Path>(
    pnt2pixxydepth: &[f32],
    pnt2pixrad: &[f32],
    pnt2rgb: &[f32],
    img_shape: (usize, usize),
    path: Path,
) -> anyhow::Result<()>
where
    Path: AsRef<std::path::Path>,
{
    let num_pnt = pnt2pixxydepth.len() / 3;
    // draw circles
    let idx2vtx = {
        let mut idx2vtx: Vec<usize> = (0..num_pnt).collect();
        idx2vtx.sort_by(|&idx0, &idx1| {
            let z0 = pnt2pixxydepth[idx0 * 3 + 2] + 1f32;
            let z1 = pnt2pixxydepth[idx1 * 3 + 2] + 1f32;
            z0.partial_cmp(&z1).unwrap()
        });
        idx2vtx
    };
    let mut img_data = vec![[0f32, 0f32, 0f32]; img_shape.0 * img_shape.1];
    #[allow(clippy::needless_range_loop)]
    for idx in 0..num_pnt {
        let i_pnt = idx2vtx[idx];
        let ndc_z = pnt2pixxydepth[i_pnt * 3 + 2];
        if ndc_z <= -1f32 || ndc_z >= 1f32 {
            continue;
        }
        let r0 = arrayref::array_ref![pnt2pixxydepth, i_pnt * 3, 2];
        let rad_pix = pnt2pixrad[i_pnt];
        let rgb = arrayref::array_ref![pnt2rgb, i_pnt * 3, 3];
        let pixs = del_canvas::rasterize::circle2::pixels_in_point(
            r0[0],
            r0[1],
            rad_pix,
            img_shape.0,
            img_shape.1,
        );
        for i_pix in pixs {
            img_data[i_pix][0] = rgb[0];
            img_data[i_pix][1] = rgb[1];
            img_data[i_pix][2] = rgb[2];
        }
    }
    use ::slice_of_array::SliceFlatExt; // for flat
    del_canvas::write_png_from_float_image_rgb(path, &img_shape, img_data.flat())?;
    Ok(())
}

pub fn project(
    pnt2xyz: &[f32],
    radius: f32,
    transform_world2ndc: &[f32; 16],
    img_shape: (usize, usize),
    pnt2pixxyndcz: &mut [f32],
    pnt2pixrad: &mut [f32],
) {
    let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    let num_pnt = pnt2xyz.len() / 3;
    for i_pnt in 0..num_pnt {
        let pos_world0 = arrayref::array_ref![pnt2xyz, i_pnt * 3, 3];
        let ndc0 =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, pos_world0)
                .unwrap();
        let pos_pix = del_geo_core::mat2x3_col_major::mult_vec3(
            &transform_ndc2pix,
            &[ndc0[0], ndc0[1], 1f32],
        );
        let rad_pix = {
            let dqdp =
                del_geo_core::mat4_col_major::jacobian_transform(transform_world2ndc, pos_world0);
            let dqdp = del_geo_core::mat3_col_major::try_inverse(&dqdp).unwrap();
            let dx = [dqdp[0], dqdp[1], dqdp[2]];
            let dy = [dqdp[3], dqdp[4], dqdp[5]];
            let rad_pix_x =
                (1.0 / del_geo_core::vec3::norm(&dx)) * 0.5 * img_shape.0 as f32 * radius;
            let rad_pxi_y =
                (1.0 / del_geo_core::vec3::norm(&dy)) * 0.5 * img_shape.1 as f32 * radius;
            0.5 * (rad_pix_x + rad_pxi_y)
        };
        pnt2pixxyndcz[i_pnt * 3 + 0] = pos_pix[0];
        pnt2pixxyndcz[i_pnt * 3 + 1] = pos_pix[1];
        pnt2pixxyndcz[i_pnt * 3 + 2] = ndc0[2];
        pnt2pixrad[i_pnt] = rad_pix;
    }
}
