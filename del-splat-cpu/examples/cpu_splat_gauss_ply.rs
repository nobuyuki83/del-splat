fn world2pix(
    pos_world: &[f32; 3],
    transform_world2ndc: &[f32; 16],
    img_shape: (usize, usize),
) -> [f32; 6] {
    let mvp_grad =
        del_geo_core::mat4_col_major::jacobian_transform(&transform_world2ndc, &pos_world);
    let ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    let world2pix = del_geo_core::mat2x3_col_major::mult_mat3_col_major(&ndc2pix, &mvp_grad);
    world2pix
}

fn main() -> anyhow::Result<()> {
    // let file_path = "C:/Users/nobuy/Downloads/ChilliPepperPlant.ply"; //"asset/dog.ply";
    let file_path = "asset/dog.ply";
    let (pnt2xyz, pnt2rgb, pnt2op, pnt2scale, pnt2quat) = {
        let (mut pnt2xyz, pnt2rgb, _pnt2sh, pnt2op, mut pnt2scale, pnt2quat) =
            del_splat_cpu::io_ply::read_3d_gauss_splat(file_path)?;
        let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&pnt2xyz, 0f32);
        let longest_edge = del_geo_core::aabb3::max_edge_size(&aabb3);
        let scale = 1.0 / longest_edge;
        let sqr_scale = scale * scale;
        let center = del_geo_core::aabb3::center(&aabb3);
        let num_pnt = pnt2xyz.len() / 3;
        for i_pnt in 0..num_pnt {
            pnt2xyz[i_pnt * 3 + 0] -= center[0];
            pnt2xyz[i_pnt * 3 + 1] -= center[1];
            pnt2xyz[i_pnt * 3 + 2] -= center[2];
            pnt2xyz[i_pnt * 3 + 0] *= scale;
            pnt2xyz[i_pnt * 3 + 1] *= scale;
            pnt2xyz[i_pnt * 3 + 2] *= scale;
            pnt2scale[i_pnt * 3 + 0] *= sqr_scale;
            pnt2scale[i_pnt * 3 + 1] *= sqr_scale;
            pnt2scale[i_pnt * 3 + 2] *= sqr_scale;
        }
        (pnt2xyz, pnt2rgb, pnt2op, pnt2scale, pnt2quat)
    };
    let img_shape = (600usize + 1, 1000usize + 1);
    let transform_world2ndc = {
        let cam_proj = del_geo_core::mat4_col_major::camera_perspective_blender(
            img_shape.0 as f32 / img_shape.1 as f32,
            50f32,
            0.1,
            2.0,
            true,
        );
        let cam_modelview = del_geo_core::mat4_col_major::camera_external_blender(
            &[0f32, 0f32, 2.0f32],
            0f32,
            0f32,
            0f32,
        );
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_proj, &cam_modelview)
    };
    let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    del_splat_cpu::splat_point3::draw_pix(
        &pnt2xyz,
        &pnt2rgb,
        img_shape,
        &transform_world2ndc,
        "target/del_canvas_cpu__splat_gauss__pix.png",
    )?;

    let num_pnt = pnt2xyz.len() / 3;
    let mut pnt2pixcodepth = vec![0f32; num_pnt * 3];
    let mut pnt2siginv = vec![0f32; num_pnt * 3];
    let mut pnt2aabb = vec![0f32; num_pnt * 4];
    for i_pnt in 0..num_pnt {
        let xyz = arrayref::array_ref![&pnt2xyz, i_pnt * 3, 3];
        let quaternion = arrayref::array_ref![&pnt2quat, i_pnt * 4, 4];
        let scale = arrayref::array_ref![&pnt2scale, i_pnt * 3, 3];
        let ndc0 =
            del_geo_core::mat4_col_major::transform_homogeneous(&transform_world2ndc, xyz).unwrap();
        let pos_pix =
            del_geo_core::mat2x3_col_major::mult_vec3(&transform_ndc2pix, &[ndc0[0], ndc0[1], 1.0]);
        let transform_world2pix = world2pix(xyz, &transform_world2ndc, img_shape);
        let (abc, _dabcdt) =
            del_geo_core::mat2_sym::wdw_projected_spd_mat3(&transform_world2pix, quaternion, scale);
        let sig_inv = del_geo_core::mat2_sym::safe_inverse_preserve_positive_definiteness::<f32>(
            &abc, 1.0e-5f32,
        );
        let aabb = del_geo_core::mat2_sym::aabb2(&sig_inv);
        let aabb = del_geo_core::aabb2::scale(&aabb, 3.0);
        let aabb = del_geo_core::aabb2::translate(&aabb, &pos_pix);
        pnt2pixcodepth[i_pnt * 3 + 0] = pos_pix[0];
        pnt2pixcodepth[i_pnt * 3 + 1] = pos_pix[1];
        pnt2pixcodepth[i_pnt * 3 + 2] = ndc0[2];
        pnt2siginv[i_pnt * 3 + 0] = sig_inv[0];
        pnt2siginv[i_pnt * 3 + 1] = sig_inv[1];
        pnt2siginv[i_pnt * 3 + 2] = sig_inv[2];
        pnt2aabb[i_pnt * 4 + 0] = aabb[0];
        pnt2aabb[i_pnt * 4 + 1] = aabb[1];
        pnt2aabb[i_pnt * 4 + 2] = aabb[2];
        pnt2aabb[i_pnt * 4 + 3] = aabb[3];
    }

    {
        println!("gaussian_naive");
        let now = std::time::Instant::now();
        del_splat_cpu::splat_gaussian2::rasterize_naive_(
            &pnt2pixcodepth,
            &pnt2siginv,
            &pnt2aabb,
            &pnt2op,
            &pnt2rgb,
            img_shape,
            "target/del_canvas_cpu__splat_gauss_ply__naive.png",
        )?;
        println!("   Elapsed gaussian_naive: {:.2?}", now.elapsed());
    }

    Ok(())
}
