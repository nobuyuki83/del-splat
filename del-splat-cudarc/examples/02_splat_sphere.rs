#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use del_cudarc_sys::cu;
    // let path = "/Users/nobuyuki/project/juice_box1.ply";
    let file_path = "asset/juice_box.ply";
    let (pnt2xyz, pnt2rgb) = del_splat_cpu::io_ply::read_xyzrgb::<_>(file_path)?;
    let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&pnt2xyz, 0f32);
    dbg!(aabb3);
    let img_shape = (2000usize + 1, 1200usize + 1);
    let transform_world2ndc = {
        let cam_proj = del_geo_core::mat4_col_major::camera_perspective_blender(
            img_shape.0 as f32 / img_shape.1 as f32,
            50f32,
            0.1,
            2.0,
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
        dbg!(cam_modelview);
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_proj, &cam_modelview)
    };
    dbg!(transform_world2ndc);
    let radius = 0.0015f32;
    //
    del_cudarc_sys::cache_func::clear();
    let (dev, _ctx) = del_cudarc_sys::init_cuda_and_make_context(0).unwrap();
    {
        let stream = del_cudarc_sys::create_stream_in_current_context().unwrap();
        //
        let num_pnt = pnt2xyz.len() / 3;
        use del_cudarc_sys::CuVec;
        let pnt2xyz_dev = CuVec::<f32>::from_slice(&pnt2xyz).unwrap();
        let pnt2rgb_dev = CuVec::<f32>::from_slice(&pnt2rgb).unwrap();
        let pnt2pixxydepth_dev = CuVec::<f32>::with_capacity(num_pnt * 3).unwrap();
        let pnt2pixrad_dev = CuVec::<f32>::with_capacity(num_pnt).unwrap();
        let transform_world2ndc_dev = CuVec::from_slice(&transform_world2ndc).unwrap();
        del_splat_cudarc::splat_circle::pnt2splat3_to_pnt2splat2(
            stream,
            &pnt2xyz_dev,
            &pnt2pixxydepth_dev,
            &pnt2pixrad_dev,
            &transform_world2ndc_dev,
            (img_shape.0 as u32, img_shape.1 as u32),
            radius,
        )?;
        {
            // draw pixels in cpu using the order computed in cpu
            let pnt2pixxyndcz = pnt2pixxydepth_dev.copy_to_host().unwrap();
            let idx2vtx = {
                let mut idx2vtx: Vec<usize> = (0..num_pnt).collect();
                idx2vtx.sort_by(|&idx0, &idx1| {
                    pnt2pixxyndcz[idx0 * 3 + 2]
                        .partial_cmp(&pnt2pixxyndcz[idx1 * 3 + 2])
                        .unwrap()
                });
                idx2vtx
            };
            let mut img_data = vec![[0f32, 0f32, 0f32]; img_shape.0 * img_shape.1];
            del_canvas::rasterize::aabb3::wireframe_dda(
                &mut img_data,
                img_shape,
                &transform_world2ndc,
                &aabb3,
                [1.0, 1.0, 1.0],
            );
            for i_idx in 0..num_pnt {
                let i_vtx = idx2vtx[i_idx];
                let pixxydepth = arrayref::array_ref![pnt2pixxyndcz, i_vtx * 3, 3];
                let ix = pixxydepth[0] as usize;
                let iy = pixxydepth[1] as usize;
                let ipix = iy * img_shape.0 + ix;
                img_data[ipix][0] = pnt2rgb[i_vtx * 3 + 0];
                img_data[ipix][1] = pnt2rgb[i_vtx * 3 + 1];
                img_data[ipix][2] = pnt2rgb[i_vtx * 3 + 2];
            }
            use ::slice_of_array::SliceFlatExt; // for flat
            del_canvas::write_png_from_float_image_rgb(
                "target/del_canvas_cuda__02_splat_sphere__pix.png",
                &img_shape,
                (&img_data).flat(),
            )?;
        } // end pixel
          // ---------------------------------------------------------
          // draw circles with tiles
        const TILE_SIZE: usize = 16;
        let tile_shape = (
            img_shape.0 / TILE_SIZE + if img_shape.0 % TILE_SIZE == 0 { 0 } else { 1 },
            img_shape.1 / TILE_SIZE + if img_shape.1 % TILE_SIZE == 0 { 0 } else { 1 },
        );
        let now = std::time::Instant::now();
        let (tile2idx_dev, idx2pnt_dev) = del_splat_cudarc::splat_circle::tile2idx_idx2pnt(
            stream,
            (tile_shape.0 as u32, tile_shape.1 as u32),
            &pnt2pixxydepth_dev,
            &pnt2pixrad_dev,
        )?;
        println!("tile2idx_idx2pnt: {:.2?}", now.elapsed());
        del_splat_cpu::tile_acceleration::check_tile2pnt_circle(
            &pnt2pixxydepth_dev.copy_to_host().unwrap(),
            &pnt2pixrad_dev.copy_to_host().unwrap(),
            tile_shape,
            TILE_SIZE,
            &tile2idx_dev.copy_to_host().unwrap(),
            &idx2pnt_dev.copy_to_host().unwrap(),
        );
        // --------------------------------------------------------
        {
            let pix2rgb_dev =
                CuVec::<f32>::with_capacity(img_shape.0 * img_shape.1 * 3).unwrap();
            let now = std::time::Instant::now();
            del_splat_cudarc::splat_circle::splat(
                stream,
                (img_shape.0 as u32, img_shape.1 as u32),
                &pix2rgb_dev,
                &pnt2pixxydepth_dev,
                &pnt2pixrad_dev,
                &pnt2rgb_dev,
                TILE_SIZE as u32,
                &tile2idx_dev,
                &idx2pnt_dev,
            )?;
            println!("splat: {:.2?}", now.elapsed());
            let pix2rgb = pix2rgb_dev.copy_to_host().unwrap();
            del_canvas::write_png_from_float_image_rgb(
                "target/del_canvas_cuda__02_splat_sphere__all_gpu.png",
                &img_shape,
                &pix2rgb,
            )?;
        }

        {
            // assert using cpu
            let idx2pnt = idx2pnt_dev.copy_to_host().unwrap();
            let pnt2pixxydepth = pnt2pixxydepth_dev.copy_to_host().unwrap();
            let pnt2pixrad = pnt2pixrad_dev.copy_to_host().unwrap();
            let pnt2rgb = pnt2rgb_dev.copy_to_host().unwrap();
            let tile2idx = tile2idx_dev.copy_to_host().unwrap();
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
                for &i_vtx in &idx2pnt[tile2idx[i_tile] as usize..tile2idx[i_tile + 1] as usize] {
                    let i_vtx = i_vtx as usize;
                    let p0 = arrayref::array_ref![pnt2pixxydepth, i_vtx * 3, 2];
                    let rad = pnt2pixrad[i_vtx];
                    let p1 = [iw as f32 + 0.5f32, ih as f32 + 0.5f32];
                    if del_geo_core::edge2::length(&p0, &p1) > rad {
                        continue;
                    }
                    img_data[i_pix][0] = pnt2rgb[i_vtx * 3 + 0];
                    img_data[i_pix][1] = pnt2rgb[i_vtx * 3 + 1];
                    img_data[i_pix][2] = pnt2rgb[i_vtx * 3 + 2];
                }
            }
            use ::slice_of_array::SliceFlatExt; // for flat
            del_canvas::write_png_from_float_image_rgb(
                "target/del_canvas_cuda__02_splat_sphere__tile_cpu.png",
                &img_shape,
                (&img_data).flat(),
            )?;
        }
        del_cudarc_sys::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    del_cudarc_sys::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
