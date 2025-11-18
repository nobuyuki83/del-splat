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

/*
pub fn rasterize_pnt2splat2(
    dev: &std::sync::Arc<cudarc::driver::CudaStream>,
    img_shape: (u32, u32),
    pix2rgb_dev: &mut cudarc::driver::CudaSlice<f32>,
    pnt2splat2_dev: &cudarc::driver::CudaSlice<Splat2>,
    tile_size: u32,
    tile2idx_dev: &cudarc::driver::CudaSlice<u32>,
    idx2pnt_dev: &cudarc::driver::CudaSlice<u32>,
) -> anyhow::Result<()> {
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
    );
    assert_eq!(
        tile2idx_dev.len(),
        (tile_shape.0 * tile_shape.1 + 1) as usize
    );
    // gpu splat
    let cfg = {
        cudarc::driver::LaunchConfig {
            grid_dim: (tile_shape.0 as u32, tile_shape.1 as u32, 1),
            block_dim: (tile_size as u32, tile_size as u32, 1),
            shared_mem_bytes: 0,
        }
    };
    let count_splat_in_tile = del_cudarc_safe::get_or_load_func(
        &dev.context(),
        "rasterize_splat_using_tile",
        del_splat_cudarc_kernel::SPLAT_GAUSS,
    )?;
    {
        let mut builder = dev.launch_builder(&count_splat_in_tile);
        let img_shape_0 = img_shape.0 as u32;
        let img_shape_1 = img_shape.1 as u32;
        let tile_shape_0 = tile_shape.0 as u32;
        let tile_shape_1 = tile_shape.1 as u32;
        builder.arg(&img_shape_0);
        builder.arg(&img_shape_1);
        builder.arg(pix2rgb_dev);
        builder.arg(&tile_shape_0);
        builder.arg(&tile_shape_1);
        builder.arg(&tile_size);
        builder.arg(tile2idx_dev);
        builder.arg(idx2pnt_dev);
        builder.arg(pnt2splat2_dev);
        unsafe { builder.launch(cfg) }?;
    }
    /*
    let param = (
        img_shape.0 as u32,
        img_shape.1 as u32,
        pix2rgb_dev,
        tile_shape.0 as u32,
        tile_shape.1 as u32,
        tile_size as u32,
        tile2idx_dev,
        idx2pnt_dev,
        pnt2splat2_dev,
    );
    use cudarc::driver::LaunchAsync;
    unsafe { count_splat_in_tile.launch(cfg, param) }?;
     */
    Ok(())
}

pub fn tile2idx_idx2pnt(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    img_shape: (u32, u32),
    tile_size: u32,
    pnt2splat_dev: &cudarc::driver::CudaSlice<Splat2>,
) -> anyhow::Result<(
    cudarc::driver::CudaSlice<u32>,
    cudarc::driver::CudaSlice<u32>,
)> {
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
    );
    let (tile2idx_dev, pnt2ind_dev) = {
        let num_pnt = pnt2splat_dev.len();
        let mut tile2idx_dev =
            stream.alloc_zeros::<u32>((tile_shape.0 * tile_shape.1 + 1) as usize)?;
        let mut pnt2idx_dev = stream.alloc_zeros::<u32>(num_pnt + 1)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_pnt as u32);
        let count_splat_in_tile = del_cudarc_safe::get_or_load_func(
            &stream.context(),
            "count_splat_in_tile",
            del_splat_cudarc_kernel::SPLAT_GAUSS,
        )?;
        {
            let mut builder = stream.launch_builder(&count_splat_in_tile);
            let pnt2splat_dev_len = pnt2splat_dev.len() as u32;
            let tile_shape_0 = tile_shape.0 as u32;
            let tile_shape_1 = tile_shape.1 as u32;
            builder.arg(&pnt2splat_dev_len);
            builder.arg(pnt2splat_dev);
            builder.arg(&mut tile2idx_dev);
            builder.arg(&mut pnt2idx_dev);
            builder.arg(&tile_shape_0);
            builder.arg(&tile_shape_1);
            builder.arg(&16u32);
            unsafe { builder.launch(cfg) }?;
        }
        /*
        let param = (
            pnt2splat_dev.len(),
            pnt2splat_dev,
            &mut tile2idx_dev,
            &mut pnt2idx_dev,
            tile_shape.0 as u32,
            tile_shape.1 as u32,
            16u32,
        );
        use cudarc::driver::LaunchAsync;
        unsafe { count_splat_in_tile.launch(cfg, param) }?;
         */
        (tile2idx_dev, pnt2idx_dev)
    };
    let tile2idx_dev = {
        let mut tmp = stream.alloc_zeros(tile2idx_dev.len())?;
        del_cudarc_safe::cumsum::sum_scan_blelloch(&stream, &mut tmp, &tile2idx_dev)?;
        tmp
    };
    let pnt2idx_dev = {
        let mut tmp = stream.alloc_zeros::<u32>(pnt2ind_dev.len())?;
        del_cudarc_safe::cumsum::sum_scan_blelloch(&stream, &mut tmp, &pnt2ind_dev)?;
        tmp
    };
    let num_ind = stream.memcpy_dtov(&pnt2idx_dev)?.last().unwrap().to_owned(); // todo: send only last element to cpu
    debug_assert_eq!(
        num_ind,
        stream
            .memcpy_dtov(&tile2idx_dev)?
            .last()
            .unwrap()
            .to_owned()
    );
    let idx2pnt_dev = {
        let mut idx2tiledepth_dev = stream.alloc_zeros::<u64>(num_ind as usize)?;
        let mut idx2pnt_dev = stream.alloc_zeros(num_ind as usize)?;
        let num_pnt = pnt2splat_dev.len();
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_pnt as u32);
        let count_splat_in_tile = del_cudarc_safe::get_or_load_func(
            &stream.context(),
            "fill_index_info",
            del_splat_cudarc_kernel::SPLAT_GAUSS,
        )?;
        {
            let mut builder = stream.launch_builder(&count_splat_in_tile);
            let pnt2splat_dev_len = pnt2splat_dev.len() as u32;
            let tile_shape_0 = tile_shape.0 as u32;
            let tile_shape_1 = tile_shape.1 as u32;
            builder.arg(&pnt2splat_dev_len);
            builder.arg(pnt2splat_dev);
            builder.arg(&pnt2idx_dev);
            builder.arg(&mut idx2tiledepth_dev);
            builder.arg(&mut idx2pnt_dev);
            builder.arg(&tile_shape_0);
            builder.arg(&tile_shape_1);
            builder.arg(&16u32);
            unsafe { builder.launch(cfg) }?;
        }
        /*
        let param = (
            pnt2splat_dev.len(),
            pnt2splat_dev,
            &pnt2idx_dev,
            &mut idx2tiledepth_dev,
            &mut idx2pnt_dev,
            tile_shape.0 as u32,
            tile_shape.1 as u32,
            16u32,
        );
        use cudarc::driver::LaunchAsync;
        unsafe { count_splat_in_tile.launch(cfg, param) }?;
         */
        del_cudarc_safe::sort_by_key_u64::radix_sort_by_key_u64(
            &stream,
            &mut idx2tiledepth_dev,
            &mut idx2pnt_dev,
        )?;
        idx2pnt_dev
    };
    Ok((tile2idx_dev, idx2pnt_dev))
}
 */
