#[cfg(feature = "cuda")]
use del_cudarc_sys::{cu, CuVec};

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let file_path = "asset/dog.ply";
    let pnt2gauss = del_splat_cpu::io_ply::read_3d_gauss_splat::<_>(file_path)?;
    let pnt2xyz = pnt2gauss.0;
    let pnt2rgb = pnt2gauss.1;
    let _pnt2sh = pnt2gauss.2;
    let pnt2opacity = pnt2gauss.3;
    let pnt2scale = pnt2gauss.4;
    let pnt2quat = pnt2gauss.5;
    // pnt2gs3.iter().enumerate().for_each(|(i_pnt, a)| { dbg!(i_pnt, a.xyz); } );
    let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&pnt2xyz, 0f32);
    let img_shape = (800usize + 1, 1200usize + 1);
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
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_proj, &cam_modelview)
    };
    // --------------------------
    // below: cuda code from here
    del_cudarc_sys::cache_func::clear();
    let (dev, _ctx) = del_cudarc_sys::init_cuda_and_make_context(0).unwrap();
    {
        let stream = del_cudarc_sys::create_stream_in_current_context().unwrap();
        //
        let transform_world2ndc_dev = CuVec::<f32>::from_slice(&transform_world2ndc).unwrap();
        let pnt2xyz_dev = CuVec::<f32>::from_slice(&pnt2xyz).unwrap();
        let pnt2scale_dev = CuVec::<f32>::from_slice(&pnt2scale).unwrap();
        let pnt2quat_dev = CuVec::<f32>::from_slice(&pnt2quat).unwrap();
        let pnt2opacity_dev = CuVec::<f32>::from_slice(&pnt2opacity).unwrap();
        let pnt2rgb_dev = CuVec::<f32>::from_slice(&pnt2rgb).unwrap();
        //
        let num_pnt = pnt2xyz.len() / 3;
        let pnt2pixxydepth_dev = CuVec::<f32>::with_capacity(num_pnt * 3).unwrap();
        let pnt2pixconvinv_dev = CuVec::<f32>::with_capacity(num_pnt * 3).unwrap();
        let pnt2pixaabb_dev = CuVec::<f32>::with_capacity(num_pnt * 4).unwrap();
        del_splat_cudarc::splat_gauss::pnt2splat3_to_pnt2splat2(
            stream,
            &pnt2xyz_dev,
            &pnt2scale_dev,
            &pnt2quat_dev,
            &pnt2pixxydepth_dev,
            &pnt2pixconvinv_dev,
            &pnt2pixaabb_dev,
            &transform_world2ndc_dev,
            (img_shape.0 as u32, img_shape.1 as u32),
        )?;

        let now = std::time::Instant::now();
        let tile_size = 16usize;
        let (tile2idx_dev, idx2pnt_dev) = del_splat_cudarc::splat_gauss::tile2idx_idx2pnt(
            stream,
            (img_shape.0 as u32, img_shape.1 as u32),
            tile_size as u32,
            &pnt2pixxydepth_dev,
            &pnt2pixaabb_dev,
        )
        .unwrap();
        println!(
            "   Elapsed rasterize on GPU with tile: {:.2?}",
            now.elapsed()
        );
        let (tile2idx_dev, idx2pnt_dev) = del_splat_cudarc::splat_gauss::tile2idx_idx2pnt(
            stream,
            (img_shape.0 as u32, img_shape.1 as u32),
            tile_size as u32,
            &pnt2pixxydepth_dev,
            &pnt2pixaabb_dev,
        )
        .unwrap();
        println!(
            "   Elapsed rasterize on GPU with tile: {:.2?}",
            now.elapsed()
        );

        {
            let now = std::time::Instant::now();
            let pix2rgb_dev = CuVec::<f32>::with_capacity(img_shape.0 * img_shape.1 * 3).unwrap();
            del_splat_cudarc::splat_gauss::rasterize_pnt2splat2(
                stream,
                (img_shape.0 as u32, img_shape.1 as u32),
                &pix2rgb_dev,
                &pnt2pixxydepth_dev,
                &pnt2pixconvinv_dev,
                &pnt2pixaabb_dev,
                &pnt2opacity_dev,
                &pnt2rgb_dev,
                tile_size as u32,
                &tile2idx_dev,
                &idx2pnt_dev,
            )?;
            println!(
                "   Elapsed rasterize on GPU with tile: {:.2?}",
                now.elapsed()
            );
            let pix2rgb = pix2rgb_dev.copy_to_host().unwrap();
            del_canvas::write_png_from_float_image_rgb(
                "target/del_canvas_cuda__03_splat_gauss.png",
                &img_shape,
                &pix2rgb,
            )?;
        }

        {
            // check tile2pnt
            let pnt2pixaabb = pnt2pixaabb_dev.copy_to_host().unwrap();
            let pnt2pixxydepth = pnt2pixxydepth_dev.copy_to_host().unwrap();
            let (tile2idx_cpu, idx2pnt_cpu) = del_splat_cpu::tile_acceleration::tile2pnt_gauss(
                &pnt2pixxydepth,
                &pnt2pixaabb,
                img_shape,
                tile_size,
            );
            {
                // assert tile2idx
                let tile2idx_gpu = tile2idx_dev.copy_to_host().unwrap();
                tile2idx_gpu
                    .iter()
                    .zip(tile2idx_cpu.iter())
                    .for_each(|(&a, &b)| {
                        // println!("{} {}", a,b);
                        assert_eq!(a, b);
                    });
            }
            {
                // assert idx2pnt
                let idx2pnt_gpu = idx2pnt_dev.copy_to_host().unwrap();
                idx2pnt_gpu
                    .iter()
                    .zip(idx2pnt_cpu.iter())
                    .for_each(|(&a, &b)| {
                        // println!("{} {} {} {}", a,b, pnt2splat2[a as usize].ndc_z, pnt2splat2[b as usize].ndc_z);
                        assert_eq!(a, b);
                    });
            }
        }

        {
            // cpu rendering from Vec<Splat2>
            let pnt2pixxydepth = pnt2pixxydepth_dev.copy_to_host().unwrap();
            let pnt2pixconvinv = pnt2pixconvinv_dev.copy_to_host().unwrap();
            let pnt2pixaabb = pnt2pixaabb_dev.copy_to_host().unwrap();
            println!("gaussian_naive without tile acceleration");
            let now = std::time::Instant::now();
            del_splat_cpu::pnt2pixxyndcz::render_gauss_sort_depth(
                &pnt2pixxydepth,
                &pnt2pixconvinv,
                &pnt2pixaabb,
                &pnt2opacity,
                &pnt2rgb,
                img_shape,
                "target/del_canvas_cuda__03_splat_gauss_test_splat3_to_splat2.png",
            )?;
            println!(
                "   Elapsed gaussian_naive from Vec<Splat2>: {:.2?}",
                now.elapsed()
            );
        }
        del_cudarc_sys::cuda_check!(cu::cuStreamDestroy_v2(stream)).unwrap();
    }
    del_cudarc_sys::cuda_check!(cu::cuDevicePrimaryCtxRelease_v2(dev)).unwrap();
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
