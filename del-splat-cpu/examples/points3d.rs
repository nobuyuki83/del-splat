use del_geo_core::mat2_col_major::Mat2ColMajor;
use del_geo_core::vec2::Vec2;
use rand::Rng;

/*
#[derive(Default, Debug)]
struct Point {
    // 3d information
    pub pos_world: [f32; 3],
    pub normal_world: [f32; 3],
    pub color: [f32; 3],
    pub sigma: nalgebra::Matrix3<f32>,

    // 2d information
    pub pos_ndc: [f32; 3],
    // pub quaternion: [f32;4]
    pub w: nalgebra::Matrix2<f32>,
    // screen position
    pub s: nalgebra::Vector2<f32>,
    // aabb
    pub aabb: [f32; 4],
}
 */

/*
impl del_splat_core::splat_point2::XyRgb for Point {
    fn pos2_rgb(&self) -> ([f32; 2], [f32; 3]) {
        ([self.s[0], self.s[1]], self.color)
    }
}
 */

fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz, _vtx2uv) = {
        let mut obj = del_msh_cpu::io_obj::WavefrontObj::<usize, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        obj.unified_xyz_uv_as_trimesh()
    };

    // define 3D points
    let (pnt2xyz, pnt2rgb, pnt2sigma) = {
        let num_pnt = 10000;
        // let mut points: Vec<Point> = vec![];
        let mut pnt2xyz = vec!();
        let mut pnt2nrm = vec!();
        let mut pnt2rgb = vec!();
        let mut pnt2conv = vec!();
        let cumsumarea = del_msh_cpu::trimesh::tri2cumsumarea(&tri2vtx, &vtx2xyz, 3);
        // let mut reng = rand::thread_rng();
        use rand::SeedableRng;
        let mut reng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        for _i in 0..num_pnt {
            let val01_a = reng.random::<f32>();
            let val01_b = reng.random::<f32>();
            let barycrd = del_msh_cpu::trimesh::sample_uniformly(&cumsumarea, val01_a, val01_b);
            let tri = del_msh_cpu::trimesh3::to_tri3(&tri2vtx, &vtx2xyz, barycrd.0);
            let pos_world = tri.position_from_barycentric_coordinates(barycrd.1, barycrd.2);
            let normal_world = del_geo_core::vec3::normalize(&tri.normal());
            let color = [
                normal_world[0] * 0.5 + 0.5,
                normal_world[1] * 0.5 + 0.5,
                normal_world[2] * 0.5 + 0.5,
            ];
            let rad = 0.03;
            use del_geo_core::mat3_col_major::Mat3ColMajor;
            let conv = del_geo_core::mat3_col_major::from_diagonal(&[rad * rad, rad*rad, rad*rad]);
            let un = del_geo_core::vec3::normalize(&normal_world);
            let tmp0 = del_geo_core::mat3_col_major::from_scaled_outer_product(rad * rad * 0.99, &un, &un);
            let conv = conv.sub(&tmp0);
            //
            pnt2xyz.extend_from_slice(&pos_world);
            pnt2nrm.extend_from_slice(&normal_world);
            pnt2rgb.extend_from_slice(&color);
            pnt2conv.extend_from_slice(&conv);
        }
        (pnt2xyz, pnt2rgb, pnt2conv)
    };

    const TILE_SIZE: usize = 16;
    let img_shape = (TILE_SIZE * 28, TILE_SIZE * 28);
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_shape.0 as f32 / img_shape.1 as f32,
        24f32,
        0.5,
        3.0,
        true,
    );
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.], 0., 0., 0.);
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);

    // transform points
    let pnt2ndc: Vec<f32> = pnt2xyz.chunks(3).flat_map(|xyz|
        del_geo_core::mat4_col_major::transform_homogeneous(
            &transform_world2ndc,
            arrayref::array_ref![xyz, 0, 3]
        ).unwrap()
    ).collect();
    let idx2pnt = {
        let num_pnt= pnt2ndc.len() / 3;
        let mut idx2pnt: Vec<usize> = (0..num_pnt).collect();
        idx2pnt.sort_by(|&i, &j| {
            let zi = pnt2ndc[i * 3 + 2] + 1f32;
            let zj = pnt2ndc[j * 3 + 2] + 1f32;
            zi.partial_cmp(&zj).unwrap()
        });
        idx2pnt
    };

    // projects points & covariance 2D
    let mut pnt2pixxydepth = vec!();
    let mut pnt2pixconvinv = vec!();
    let mut pnt2pixaabb = vec!();
    let num_pnt = pnt2xyz.len() / 3;
    for i_pnt in 0..num_pnt {
        // screen position
        let pixx = (pnt2ndc[i_pnt*3+0] + 1.0) * 0.5 * (img_shape.0 as f32);
        let pixy = (1.0 - pnt2ndc[i_pnt*3+1]) * 0.5 * (img_shape.1 as f32);
        pnt2pixxydepth.extend_from_slice(&[pixx, pixy, pnt2ndc[i_pnt*3+2]]);
        let w = del_geo_core::mat4_col_major::to_mat3_col_major_xyz(&cam_modelview);
        use del_geo_core::mat4_col_major::transform_homogeneous;
        let xyz = arrayref::array_ref![pnt2xyz, i_pnt*3, 3];
        let q = transform_homogeneous(&cam_modelview, &xyz).unwrap();
        let j: [f32;9] = del_geo_core::mat4_col_major::jacobian_transform(&cam_projection, &q); // 3x3
        let r = [ // 2x3
            0.5 * (img_shape.0 as f32),
            0.,
            0.,
            -0.5 * (img_shape.1 as f32),
            0.,
            0.,
        ];
        use del_geo_core::mat3_col_major::Mat3ColMajor;
        let jw = j.mult_mat_col_major(&w);
        let conv = arrayref::array_ref![pnt2sigma, i_pnt*9, 9];
        let l_conv = jw.mult_mat_col_major(conv).mult_mat_col_major(&jw.transpose());
        // let w0 = r * j * w * point.sigma * w.transpose() * j.transpose() * r.transpose();
        let tmp0 = del_geo_core::mat2x3_col_major::mult_mat3_col_major(&r, &l_conv);
        let rt = del_geo_core::mat2x3_col_major::transpose(&r);
        let pixconv = del_geo_core::mat2x3_col_major::mult_mat3x2_col_major(&tmp0, &rt);
        let pixconvinv = del_geo_core::mat2_col_major::try_inverse(&pixconv).unwrap();
        pnt2pixconvinv.extend_from_slice(&pixconvinv);
        // screen aabb
        {
            let pixaabb = del_geo_core::mat2_sym::aabb2(&[pixconvinv[0], pixconvinv[1], pixconvinv[3]]);
            let pixaabb = del_geo_core::aabb2::scale(&pixaabb, 3.0);
            let pixaabb = del_geo_core::aabb2::translate(&pixaabb, &[pixx, pixy]);
            pnt2pixaabb.extend_from_slice(&pixaabb);
        }
    }

    // visualize as points
    {
        /*
        let mut img_data = vec![0f32; img_size.1 * img_size.0 * 3];
        for point in points.iter() {
            let x = (point.pos_ndc[0] + 1.0) * 0.5 * (img_size.0 as f32);
            let y = (1.0 - point.pos_ndc[1]) * 0.5 * (img_size.1 as f32);
            let i_x = x as usize;
            let i_y = y as usize;
            img_data[(i_y * img_size.0 + i_x) * 3 + 0] = point.color[0];
            img_data[(i_y * img_size.0 + i_x) * 3 + 1] = point.color[1];
            img_data[(i_y * img_size.0 + i_x) * 3 + 2] = point.color[2];
        }
         */
        /*
        let img_data = del_splat_core::splat_point2::draw(&img_shape, &points);
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_pix.png",
            &img_shape,
            &img_data,
        )?;
         */
    }

    {
        // splatting Gaussian with tile-based acceleration
        println!("gaussian_tile");
        let now = std::time::Instant::now();
        let tile_shape: (usize, usize) = (img_shape.0 / TILE_SIZE, img_shape.1 / TILE_SIZE);
        let mut tile2gauss: Vec<Vec<usize>> = vec![vec!(); tile_shape.0 * tile_shape.1];
        for idx in (0..num_pnt).rev() {
            let i_pnt = idx2pnt[idx];
            let pixaabb = arrayref::array_ref![pnt2pixaabb, i_pnt * 4, 4];
            let tiles = del_geo_core::aabb2::overlapping_tiles(pixaabb, TILE_SIZE, tile_shape);
            for i_tile in tiles {
                tile2gauss[i_tile].push(i_pnt);
            }
        }
        let mut img_data = vec![0f32; img_shape.1 * img_shape.0 * 3];
        for ih in 0..img_shape.1 {
            for iw in 0..img_shape.0 {
                let t = [iw as f32 + 0.5, ih as f32 + 0.5];
                let mut alpha_sum = 0f32;
                let mut alpha_occu = 1f32;
                let i_tile = (ih / TILE_SIZE) * tile_shape.0 + (iw / TILE_SIZE);
                for &i_pnt in tile2gauss[i_tile].iter() {
                    // front to back
                    let pixaabb = arrayref::array_ref![pnt2pixaabb, i_pnt * 4, 4];
                    if !del_geo_core::aabb2::is_include_point2(pixaabb, &[t[0], t[1]]) {
                        continue;
                    }
                    let pixmu = arrayref::array_ref![pnt2pixxydepth, i_pnt*3, 2];
                    let pixconvinv = arrayref::array_ref![pnt2pixconvinv, i_pnt*4, 4];
                    let t0 = del_geo_core::vec2::sub(&t, pixmu);
                    use del_geo_core::mat2_col_major::Mat2ColMajor;
                    use del_geo_core::vec2::Vec2;
                    let e = pixconvinv.mult_vec(&t0).dot(&t0);
                    let e = (-0.5 * e).exp();
                    let e_out = alpha_occu * e;
                    let rgb = arrayref::array_ref![pnt2rgb, i_pnt*3, 3];
                    img_data[(ih * img_shape.0 + iw) * 3 + 0] += rgb[0] * e_out;
                    img_data[(ih * img_shape.0 + iw) * 3 + 1] += rgb[1] * e_out;
                    img_data[(ih * img_shape.0 + iw) * 3 + 2] += rgb[2] * e_out;
                    alpha_occu *= 1f32 - e;
                    alpha_sum += e_out;
                    if alpha_sum > 0.999 {
                        break;
                    }
                }
            }
        }

        /*
               let pix2rgb = |i_pix: usize| -> [f32;3] {
                   let ih = i_pix / img_size.0;
                   let iw = i_pix - ih * img_size.0;
                   let t = nalgebra::Vector2::<f32>::new(iw as f32 + 0.5, ih as f32 + 0.5);
                   let mut alpha_sum = 0f32;
                   let mut alpha_occu = 1f32;
                   let i_tile = (ih / 16) * tile_size.1 + (iw / 16);
                   let mut rgb = [0f32;3];
                   for &i_point in tile2gauss[i_tile].iter() {
                       let point = &points[i_point];
                       // front to back
                       if !del_geo_core::aabb2::is_inlcude_point(&point.aabb, &[t[0], t[1]]) { continue; }
                       let t0 = t - point.s;
                       let e = (t0.transpose() * point.w * t0).x;
                       let e = (-0.5 * e).exp();
                       let e_out = alpha_occu * e;
                       rgb[0] += point.color[0] * e_out;
                       rgb[1] += point.color[1] * e_out;
                       rgb[2] += point.color[2] * e_out;
                       alpha_occu *= 1f32 - e;
                       alpha_sum += e_out;
                       if alpha_sum > 0.999 { break; }
                   }
                   rgb
               };
               use rayon::iter::ParallelIterator;
               let img_data: Vec<f32> = (0..img_size.0 * img_size.1)
                   .into_par_iter()
                   .flat_map(pix2rgb)
                   .collect();
        */
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_gaussian_tile.png",
            &img_shape,
            &img_data,
        )?;
        println!("   Elapsed gaussian_tile: {:.2?}", now.elapsed());
    }

    {
        // visualize as Gaussian without tile acceleration
        println!("gaussian_naive");
        let now = std::time::Instant::now();
        let mut img_data = vec![0f32; img_shape.1 * img_shape.0 * 3];
        for ih in 0..img_shape.1 {
            for iw in 0..img_shape.0 {
                let t = [iw as f32 + 0.5, ih as f32 + 0.5];
                let mut alpha_sum = 0f32;
                let mut alpha_occu = 1f32;
                for idx in (0..num_pnt).rev() {
                    let i_pnt = idx2pnt[idx];
                    let pixaabb = arrayref::array_ref![pnt2pixaabb, i_pnt * 4, 4];
                    // front to back
                    if !del_geo_core::aabb2::is_include_point2(&pixaabb, &[t[0], t[1]]) {
                        continue;
                    }
                    let pixmu = arrayref::array_ref![pnt2pixxydepth, i_pnt*3, 2];
                    let pixconvinv = arrayref::array_ref![pnt2pixconvinv, i_pnt*4, 4];
                    let t0 = del_geo_core::vec2::sub(&t, pixmu);
                    let e = pixconvinv.mult_vec(&t0).dot(&t0);
                    let e = (-0.5 * e).exp();
                    let e_out = alpha_occu * e;
                    let rgb = arrayref::array_ref![pnt2rgb, i_pnt*3, 3];
                    img_data[(ih * img_shape.0 + iw) * 3 + 0] += rgb[0] * e_out;
                    img_data[(ih * img_shape.0 + iw) * 3 + 1] += rgb[1] * e_out;
                    img_data[(ih * img_shape.0 + iw) * 3 + 2] += rgb[2] * e_out;
                    alpha_occu *= 1f32 - e;
                    alpha_sum += e_out;
                    if alpha_sum > 0.999 {
                        break;
                    }
                }
            }
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_gaussian.png",
            &img_shape,
            &img_data,
        )?;
        println!("   Elapsed gaussian_naive: {:.2?}", now.elapsed());
    }

    {
        // visualize Gaussian as ellipsoid
        println!("gaussian_ellipse");
        let now = std::time::Instant::now();
        let mut img_data = vec![0f32; img_shape.1 * img_shape.0 * 3];
        for ih in 0..img_shape.1 {
            for iw in 0..img_shape.0 {
                let t = [iw as f32 + 0.5, ih as f32 + 0.5];
                for idx in (0..num_pnt).rev() {
                    let i_pnt = idx2pnt[idx];
                    let pixaabb = arrayref::array_ref![pnt2pixaabb, i_pnt * 4, 4];
                    if !del_geo_core::aabb2::is_include_point2(pixaabb, &[t[0], t[1]]) {
                        continue;
                    }
                    let pixmu = arrayref::array_ref![pnt2pixxydepth, i_pnt*3, 2];
                    let pixconvinv = arrayref::array_ref![pnt2pixconvinv, i_pnt*4, 4];
                    let t0 = del_geo_core::vec2::sub(&t, pixmu);
                    use del_geo_core::mat2_col_major::Mat2ColMajor;
                    use del_geo_core::vec2::Vec2;
                    let a = pixconvinv.mult_vec(&t0).dot(&t0);
                    if a > 1f32 {
                        continue;
                    }
                    let rgb = arrayref::array_ref![pnt2rgb, i_pnt*3, 3];
                    img_data[(ih * img_shape.0 + iw) * 3 + 0] = rgb[0];
                    img_data[(ih * img_shape.0 + iw) * 3 + 1] = rgb[1];
                    img_data[(ih * img_shape.0 + iw) * 3 + 2] = rgb[2];
                    break;
                }
            }
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_ellipse.png",
            &img_shape,
            &img_data,
        )?;
        println!("   Elapsed ellipse: {:.2?}", now.elapsed());
    }

    Ok(())
}
