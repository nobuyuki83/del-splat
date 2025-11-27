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

    del_splat_cpu::pnt2xyz::save_image_pix(
        &pnt2xyz,
        &pnt2rgb,
        img_shape,
        &transform_world2ndc,
        "target/del_canvas_cpu__splat_gauss__pix.png",
    )?;

    let num_pnt = pnt2xyz.len() / 3;
    let mut pnt2pixxyndcz = vec![0f32; num_pnt * 3];
    let mut pnt2pixcovinv = vec![0f32; num_pnt * 3];
    let mut pnt2pixaabb = vec![0f32; num_pnt * 4];
    del_splat_cpu::pnt2xyz::project_gauss(
        &pnt2xyz,
        &pnt2quat,
        &pnt2scale,
        img_shape,
        &transform_world2ndc,
        &mut pnt2pixxyndcz,
        &mut pnt2pixcovinv,
        &mut pnt2pixaabb,
    );

    {
        println!("gaussian_naive");
        let now = std::time::Instant::now();
        del_splat_cpu::pnt2pixxyndcz::render_gauss_sort_depth(
            &pnt2pixxyndcz,
            &pnt2pixcovinv,
            &pnt2pixaabb,
            &pnt2op,
            &pnt2rgb,
            img_shape,
            "target/del_canvas_cpu__splat_gauss_ply__naive.png",
        )?;
        println!("   Elapsed gaussian_naive: {:.2?}", now.elapsed());
    }

    Ok(())
}
