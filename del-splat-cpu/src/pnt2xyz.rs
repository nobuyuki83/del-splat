pub fn save_image_pix<Path>(
    pnt2xyz: &[f32],
    pnt2rgb: &[f32],
    img_shape: (usize, usize),
    transform_world2ndc: &[f32; 16],
    path: Path,
) -> anyhow::Result<()>
where
    Path: AsRef<std::path::Path>,
{
    let num_pnt = pnt2xyz.len() / 3;
    let mut img_data = vec![[0f32, 0f32, 0f32]; img_shape.0 * img_shape.1]; // black
    let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    for i_pnt in 0..num_pnt {
        let xyz = arrayref::array_ref![pnt2xyz, i_pnt * 3, 3];
        let rgb = arrayref::array_ref![pnt2rgb, i_pnt * 3, 3];
        let q0 =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, xyz).unwrap();
        let r0 =
            del_geo_core::mat2x3_col_major::mult_vec3(&transform_ndc2pix, &[q0[0], q0[1], 1f32]);
        if r0[0] < 0f32 || r0[0] >= img_shape.0 as f32 {
            continue;
        }
        if r0[1] < 0f32 || r0[1] >= img_shape.1 as f32 {
            continue;
        }
        let ix = r0[0] as usize;
        let iy = r0[1] as usize;
        let ipix = iy * img_shape.0 + ix;
        img_data[ipix][0] = rgb[0];
        img_data[ipix][1] = rgb[1];
        img_data[ipix][2] = rgb[2];
    }
    use ::slice_of_array::SliceFlatExt; // for flat
    del_canvas::write_png_from_float_image_rgb(path, &img_shape, img_data.flat())?;
    Ok(())
}

fn world2pix(
    pos_world: &[f32; 3],
    transform_world2ndc: &[f32; 16],
    img_shape: (usize, usize),
) -> [f32; 6] {
    let mvp_grad =
        del_geo_core::mat4_col_major::jacobian_transform(&transform_world2ndc, pos_world);
    let ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    let world2pix = del_geo_core::mat2x3_col_major::mult_mat3_col_major(&ndc2pix, &mvp_grad);
    world2pix
}

pub fn project_gauss(
    pnt2xyz: &[f32],
    pnt2quat: &[f32],
    pnt2scale: &[f32],
    img_shape: (usize, usize),
    transform_world2ndc: &[f32; 16],
    pnt2pixxyndcz: &mut [f32],
    pnt2pixcovinv: &mut [f32],
    pnt2pixaabb: &mut [f32],
) {
    let num_pnt = pnt2xyz.len() / 3;
    assert_eq!(pnt2scale.len(), num_pnt * 3);
    assert_eq!(pnt2quat.len(), num_pnt * 4);
    assert_eq!(pnt2pixxyndcz.len(), num_pnt * 3);
    assert_eq!(pnt2pixcovinv.len(), num_pnt * 3);
    assert_eq!(pnt2pixaabb.len(), num_pnt * 4);
    let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    for i_pnt in 0..num_pnt {
        let xyz = arrayref::array_ref![&pnt2xyz, i_pnt * 3, 3];
        let quaternion = arrayref::array_ref![&pnt2quat, i_pnt * 4, 4];
        let scale = arrayref::array_ref![&pnt2scale, i_pnt * 3, 3];
        let ndc0 =
            del_geo_core::mat4_col_major::transform_homogeneous(transform_world2ndc, xyz).unwrap();
        let pos_pix =
            del_geo_core::mat2x3_col_major::mult_vec3(&transform_ndc2pix, &[ndc0[0], ndc0[1], 1.0]);
        let transform_world2pix = world2pix(xyz, &transform_world2ndc, img_shape);
        let (abc, _dabcdt) =
            del_geo_core::mat2_sym::wdw_projected_spd_mat3(&transform_world2pix, quaternion, scale);
        let pixcovinv = del_geo_core::mat2_sym::safe_inverse_preserve_positive_definiteness::<f32>(
            &abc, 1.0e-5f32,
        );
        let aabb = del_geo_core::mat2_sym::aabb2(&pixcovinv);
        let aabb = del_geo_core::aabb2::scale(&aabb, 3.0);
        let aabb = del_geo_core::aabb2::translate(&aabb, &pos_pix);
        pnt2pixxyndcz[i_pnt * 3 + 0] = pos_pix[0];
        pnt2pixxyndcz[i_pnt * 3 + 1] = pos_pix[1];
        pnt2pixxyndcz[i_pnt * 3 + 2] = ndc0[2];
        pnt2pixcovinv[i_pnt * 3 + 0] = pixcovinv[0];
        pnt2pixcovinv[i_pnt * 3 + 1] = pixcovinv[1];
        pnt2pixcovinv[i_pnt * 3 + 2] = pixcovinv[2];
        pnt2pixaabb[i_pnt * 4 + 0] = aabb[0];
        pnt2pixaabb[i_pnt * 4 + 1] = aabb[1];
        pnt2pixaabb[i_pnt * 4 + 2] = aabb[2];
        pnt2pixaabb[i_pnt * 4 + 3] = aabb[3];
    }
}
