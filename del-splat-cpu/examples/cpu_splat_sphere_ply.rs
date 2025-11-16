use num_traits::cast::AsPrimitive;

fn main() -> anyhow::Result<()> {
    // let path = "/Users/nobuyuki/project/juice_box1.ply";
    let file_path = "asset/juice_box.ply";
    let (pnt2xyz, pnt2rgb) = del_splat_core::io_ply::read_xyzrgb::<_>(file_path)?;
    let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&pnt2xyz, 0f32);
    let aabb3: [f32; 6] = aabb3.map(|v| v.as_());
    let img_shape = (1600usize + 1, 960usize + 1);
    let transform_world2ndc = {
        let cam_proj = del_geo_core::mat4_col_major::camera_perspective_blender(
            img_shape.0 as f32 / img_shape.1 as f32,
            50f32,
            0.1,
            3.0,
            true,
        );
        let cam_modelview = del_geo_core::mat4_col_major::camera_external_blender(
            &[
                (aabb3[0] + aabb3[3]) * 0.5f32,
                (aabb3[1] + aabb3[4]) * 0.5f32,
                (aabb3[2] + aabb3[5]) * 0.5f32 + 1.4f32,
            ],
            0f32,
            0f32,
            0f32,
        );
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_proj, &cam_modelview)
    };
    let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix(img_shape);
    let radius = 0.0015;
    let (pnt2pixcodepth, pnt2pixrad) = {
        let num_pnt = pnt2xyz.len() / 3;
        let mut pnt2pixcodepth = vec![0f32; num_pnt * 3];
        let mut pnt2pixrad = vec![0f32; num_pnt];
        for i_pnt in 0..num_pnt {
            let pos_world0 = arrayref::array_ref![pnt2xyz, i_pnt * 3, 3];
            let ndc0 = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2ndc,
                &pos_world0,
            )
            .unwrap();
            let pos_pix = del_geo_core::mat2x3_col_major::mult_vec3(
                &transform_ndc2pix,
                &[ndc0[0], ndc0[1], 1f32],
            );
            let rad_pix = {
                let dqdp = del_geo_core::mat4_col_major::jacobian_transform(
                    &transform_world2ndc,
                    &pos_world0,
                );
                let dqdp = del_geo_core::mat3_col_major::try_inverse(&dqdp).unwrap();
                let dx = [dqdp[0], dqdp[1], dqdp[2]];
                let dy = [dqdp[3], dqdp[4], dqdp[5]];
                let rad_pix_x =
                    (1.0 / del_geo_core::vec3::norm(&dx)) * 0.5 * img_shape.0 as f32 * radius;
                let rad_pxi_y =
                    (1.0 / del_geo_core::vec3::norm(&dy)) * 0.5 * img_shape.1 as f32 * radius;
                0.5 * (rad_pix_x + rad_pxi_y)
            };
            pnt2pixcodepth[i_pnt * 3 + 0] = pos_pix[0];
            pnt2pixcodepth[i_pnt * 3 + 1] = pos_pix[1];
            pnt2pixcodepth[i_pnt * 3 + 2] = ndc0[2];
            pnt2pixrad[i_pnt] = rad_pix;
        }
        (pnt2pixcodepth, pnt2pixrad)
    };
    {
        let mut img_data = vec![[0f32; 3]; img_shape.0 * img_shape.1];
        del_splat_core::splat_point2::draw_pix_sort_z_(
            &pnt2pixcodepth,
            &pnt2rgb,
            img_shape.0,
            &mut img_data,
        )?;
        use ::slice_of_array::SliceFlatExt; // for flat
        del_canvas::write_png_from_float_image_rgb(
            "target/del_canvas_cpu__splat_sphere__pix_sort_z.png",
            &img_shape,
            img_data.flat(),
        )?;
    }
    {
        let now = std::time::Instant::now();
        del_splat_core::splat_circle::draw_sort_z_(
            &pnt2pixcodepth,
            &pnt2pixrad,
            &pnt2rgb,
            img_shape,
            "target/del_canvas_cpu__splat_sphere__sort_z.png",
        )?;
        println!(
            "   Elapsed splat circles without acceleration: {:.2?}",
            now.elapsed()
        );
    }

    // draw circles with tiles
    let now = std::time::Instant::now();
    const TILE_SIZE: usize = 16;
    let tile_shape = (
        img_shape.0 / TILE_SIZE + if img_shape.0 % TILE_SIZE == 0 { 0 } else { 1 },
        img_shape.1 / TILE_SIZE + if img_shape.0 % TILE_SIZE == 0 { 0 } else { 1 },
    );
    let (tile2ind, ind2pnt) =
        del_splat_core::tile_acceleration::hoge(&pnt2pixcodepth, &pnt2pixrad, img_shape, TILE_SIZE);
    //
    println!("   Elapsed tile2pnt: {:.2?}", now.elapsed());
    let now = std::time::Instant::now();
    let mut img_data = vec![[0f32, 0f32, 0f32]; img_shape.0 * img_shape.1];
    del_canvas::rasterize::aabb3::wireframe_dda(
        &mut img_data,
        img_shape,
        &transform_world2ndc,
        &aabb3,
        [1.0, 1.0, 1.0],
    );
    for (iw, ih) in itertools::iproduct!(0..img_shape.0, 0..img_shape.1) {
        let i_tile = (ih / TILE_SIZE) * tile_shape.0 + (iw / TILE_SIZE);
        let i_pix = ih * img_shape.0 + iw;
        for &i_pnt in &ind2pnt[tile2ind[i_tile]..tile2ind[i_tile + 1]] {
            // splat back to front
            let ndc_z = pnt2pixcodepth[i_pnt * 3 + 2];
            if ndc_z <= -1f32 || ndc_z >= 1f32 {
                continue;
            }
            let pos_pix = arrayref::array_ref![pnt2pixcodepth, i_pnt * 3, 2];
            let rad_pix = pnt2pixrad[i_pnt];
            let pos_pixel_center = [iw as f32 + 0.5f32, ih as f32 + 0.5f32];
            if del_geo_core::edge2::length(&pos_pix, &pos_pixel_center) > rad_pix {
                continue;
            }
            img_data[i_pix][0] = pnt2rgb[i_pnt * 3 + 0];
            img_data[i_pix][1] = pnt2rgb[i_pnt * 3 + 1];
            img_data[i_pix][2] = pnt2rgb[i_pnt * 3 + 2];
        }
    }
    use ::slice_of_array::SliceFlatExt; // for flat
    del_canvas::write_png_from_float_image_rgb(
        "target/del_canvas_cpu__splat_sphere__tile.png",
        &img_shape,
        (&img_data).flat(),
    )?;
    println!("   Elapsed gaussian_tile: {:.2?}", now.elapsed());
    Ok(())
}
