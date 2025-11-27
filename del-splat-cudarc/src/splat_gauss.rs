// ---------------------------------
// below: global funcs

use del_cudarc_sys::{cu::CUstream, CuVec};

pub fn pnt2splat3_to_pnt2splat2(
    stream: CUstream,
    pnt2xyz: &CuVec<f32>,
    pnt2scale: &CuVec<f32>,
    pnt2quat: &CuVec<f32>,
    pnt2pixxydepth: &CuVec<f32>,
    pnt2pixconvinv: &CuVec<f32>,
    pnt2pixaabb: &CuVec<f32>,
    transform_world2ndc: &CuVec<f32>,
    img_shape: (u32, u32),
) -> anyhow::Result<()> {
    let num_pnt = pnt2xyz.n / 3;
    let fnc = del_cudarc_sys::cache_func::get_function_cached(
        "del_splat::splat_gauss",
        del_splat_cuda_kernels::get("splat_gauss").unwrap(),
        "splat3_to_splat2",
    )
    .unwrap();
    {
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_pnt as u32);
        builder.arg_dptr(pnt2pixxydepth.dptr);
        builder.arg_dptr(pnt2pixconvinv.dptr);
        builder.arg_dptr(pnt2pixaabb.dptr);
        builder.arg_dptr(pnt2xyz.dptr);
        builder.arg_dptr(pnt2quat.dptr);
        builder.arg_dptr(pnt2scale.dptr);
        builder.arg_dptr(transform_world2ndc.dptr);
        builder.arg_u32(img_shape.0);
        builder.arg_u32(img_shape.1);
        builder
            .launch_kernel(
                fnc,
                del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
            )
            .unwrap();
    }
    Ok(())
}

pub fn rasterize_pnt2splat2(
    stream: CUstream,
    img_shape: (u32, u32),
    pix2rgb_dev: &CuVec<f32>,
    pnt2pixxydepth_dev: &CuVec<f32>,
    pnt2pixconvinv_dev: &CuVec<f32>,
    pnt2pixaabb_dev: &CuVec<f32>,
    pnt2opacity_dev: &CuVec<f32>,
    pnt2rgb: &CuVec<f32>,
    tile_size: u32,
    tile2idx_dev: &CuVec<u32>,
    idx2pnt_dev: &CuVec<u32>,
) -> anyhow::Result<()> {
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
    );
    assert_eq!(tile2idx_dev.n, (tile_shape.0 * tile_shape.1 + 1) as usize);
    // gpu splat
    let cfg = del_cudarc_sys::LaunchConfig {
        grid_dim: (tile_shape.0 as u32, tile_shape.1 as u32, 1),
        block_dim: (tile_size as u32, tile_size as u32, 1),
        shared_mem_bytes: 0,
    };
    let func = del_cudarc_sys::cache_func::get_function_cached(
        "del_splat::splat_gauss",
        del_splat_cuda_kernels::get("splat_gauss").unwrap(),
        "rasterize_splat_using_tile",
    )
    .unwrap();
    {
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(img_shape.0);
        builder.arg_u32(img_shape.1);
        builder.arg_dptr(pix2rgb_dev.dptr);
        builder.arg_dptr(pnt2pixxydepth_dev.dptr);
        builder.arg_dptr(pnt2pixconvinv_dev.dptr);
        builder.arg_dptr(pnt2pixaabb_dev.dptr);
        builder.arg_dptr(pnt2opacity_dev.dptr);
        builder.arg_dptr(pnt2rgb.dptr);
        builder.arg_u32(tile_shape.0);
        builder.arg_u32(tile_shape.1);
        builder.arg_u32(tile_size);
        builder.arg_dptr(tile2idx_dev.dptr);
        builder.arg_dptr(idx2pnt_dev.dptr);
        builder.launch_kernel(func, cfg).unwrap();
    }
    Ok(())
}

pub fn tile2idx_idx2pnt(
    stream: CUstream,
    img_shape: (u32, u32),
    tile_size: u32,
    pnt2pixxydepth: &CuVec<f32>,
    pnt2pixaabb: &CuVec<f32>,
) -> anyhow::Result<(CuVec<u32>, CuVec<u32>)> {
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
    );
    let num_pnt = pnt2pixxydepth.n / 3;
    let num_tile = (tile_shape.0 * tile_shape.1) as usize;
    // count number of splat in the tile
    let (tile2idx_dev, pnt2ind_dev) = {
        let tile2idx_dev = CuVec::<u32>::alloc_zeros(num_tile + 1, stream).unwrap();
        let pnt2idx_dev = CuVec::<u32>::alloc_zeros(num_pnt + 1, stream).unwrap();
        let fnc = del_cudarc_sys::cache_func::get_function_cached(
            "del_splat::splat_gauss",
            del_splat_cuda_kernels::get("splat_gauss").unwrap(),
            "count_splat_in_tile",
        )
        .unwrap();
        {
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_dptr(pnt2pixaabb.dptr);
            builder.arg_dptr(tile2idx_dev.dptr);
            builder.arg_dptr(pnt2idx_dev.dptr);
            builder.arg_u32(tile_shape.0);
            builder.arg_u32(tile_shape.1);
            builder.arg_u32(16u32);
            builder
                .launch_kernel(
                    fnc,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
                )
                .unwrap();
        }
        (tile2idx_dev, pnt2idx_dev)
    };
    let tile2idx_dev = {
        let tmp = CuVec::<u32>::alloc_zeros(tile2idx_dev.n, stream).unwrap();
        del_cudarc_sys::cumsum::exclusive_scan(stream, &tile2idx_dev, &tmp);
        tmp
    };
    let pnt2idx_dev = {
        let tmp = CuVec::<u32>::alloc_zeros(pnt2ind_dev.n, stream).unwrap();
        del_cudarc_sys::cumsum::exclusive_scan(stream, &pnt2ind_dev, &tmp);
        tmp
    };
    let num_ind = pnt2idx_dev.last().unwrap();
    debug_assert_eq!(
        num_ind,
        tile2idx_dev
            .copy_to_host()
            .unwrap()
            .last()
            .unwrap()
            .to_owned()
    );
    let idx2pnt_dev = {
        let idx2tiledepth_dev = CuVec::<u64>::alloc_zeros(num_ind as usize, stream).unwrap();
        let idx2pnt_dev = CuVec::<u32>::alloc_zeros(num_ind as usize, stream).unwrap();
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_splat::splat_gauss",
            del_splat_cuda_kernels::get("splat_gauss").unwrap(),
            "fill_index_info",
        )
        .unwrap();
        {
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_dptr(pnt2pixxydepth.dptr);
            builder.arg_dptr(pnt2pixaabb.dptr);
            builder.arg_dptr(pnt2idx_dev.dptr);
            builder.arg_dptr(idx2tiledepth_dev.dptr);
            builder.arg_dptr(idx2pnt_dev.dptr);
            builder.arg_u32(tile_shape.0);
            builder.arg_u32(tile_shape.1);
            builder.arg_u32(16u32);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
                )
                .unwrap();
        }
        unsafe {
            del_cudarc_sys::sort_by_key_u64::radix_sort_by_key_u64_u32(
                stream,
                &idx2tiledepth_dev,
                &idx2pnt_dev,
            )
        }
        .unwrap();
        idx2pnt_dev
    };
    Ok((tile2idx_dev, idx2pnt_dev))
}
