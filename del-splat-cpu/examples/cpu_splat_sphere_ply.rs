use num_traits::cast::AsPrimitive;

fn main() -> anyhow::Result<()> {
    // let path = "/Users/nobuyuki/project/juice_box1.ply";
    let file_path = "asset/juice_box.ply";
    let (pnt2xyz, pnt2rgb) = del_splat_cpu::io_ply::read_xyzrgb::<_>(file_path)?;
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
    let radius = 0.0015;
    let (pnt2pixxyndcz, pnt2pixrad) = {
        let num_pnt = pnt2xyz.len() / 3;
        let mut pnt2pixxyndcz = vec![0f32; num_pnt * 3];
        let mut pnt2pixrad = vec![0f32; num_pnt];
        del_splat_cpu::splat_sphere::project(
            &pnt2xyz,
            radius,
            &transform_world2ndc,
            img_shape,
            &mut pnt2pixxyndcz,
            &mut pnt2pixrad,
        );
        (pnt2pixxyndcz, pnt2pixrad)
    };
    dbg!(&pnt2pixxyndcz[0..100]);
    dbg!(&pnt2pixrad[0..100]);
    {
        let mut img_data = vec![[0f32; 3]; img_shape.0 * img_shape.1];
        del_splat_cpu::pnt2pixxyndcz::render_pix_sort_depth(
            &pnt2pixxyndcz,
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
        del_splat_cpu::splat_sphere::render_circle_sort_depth(
            &pnt2pixxyndcz,
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
    let (tile2ind, ind2pnt) = del_splat_cpu::tile_acceleration::tile2pnt_circle(
        &pnt2pixxyndcz,
        &pnt2pixrad,
        img_shape,
        TILE_SIZE,
    );
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
            let ndc_z = pnt2pixxyndcz[i_pnt * 3 + 2];
            if ndc_z <= -1f32 || ndc_z >= 1f32 {
                continue;
            }
            let pos_pix = arrayref::array_ref![pnt2pixxyndcz, i_pnt * 3, 2];
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
