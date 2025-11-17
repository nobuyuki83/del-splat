//use cudarc::driver::DeviceSlice;
use del_cudarc_sys::cu::CUstream;
use del_cudarc_sys::{cu, CuVec};
/*
#[derive(Clone, Default)]
#[repr(C)]
pub struct Splat3 {
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
}

impl del_msh_cpu::io_ply::XyzRgb for Splat3 {
    fn new(xyz: [f64; 3], rgb: [u8; 3]) -> Self {
        Splat3 {
            xyz: xyz.map(|v| v as f32),
            rgb,
        }
    }
}

impl del_msh_cpu::vtx2point::HasXyz<f32> for Splat3 {
    fn xyz(&self) -> &[f32; 3] {
        &self.xyz
    }
}
unsafe impl cudarc::driver::DeviceRepr for Splat3 {}

#[derive(Clone, Default, Debug)]
#[repr(C)]
pub struct Splat2 {
    pub ndc_z: f32,
    pub pos_pix: [f32; 2],
    pub rad_pix: f32,
    pub rgb: [f32; 3],
}

unsafe impl cudarc::driver::DeviceRepr for Splat2 {}
 */

pub fn tile2idx_idx2pnt(
    stream: CUstream,
    tile_shape: (u32, u32),
    pnt2pixxydepth: &CuVec<f32>,
    pnt2pixrad: &CuVec<f32>,
) -> anyhow::Result<(CuVec<u32>, CuVec<u32>)> {
    let num_pnt = pnt2pixxydepth.n / 3;
    let num_tile = tile_shape.0 * tile_shape.1;
    let (tile2idx_dev, pnt2ind_dev) = {
        let tile2idx = CuVec::<u32>::alloc_zeros(num_tile as usize + 1, stream).unwrap();
        let pnt2idx = CuVec::<u32>::alloc_zeros(num_pnt + 1, stream).unwrap();
        let fnc = del_cudarc_sys::cache_func::get_function_cached(
            "del_splat::splat_sphere",
            del_splat_cuda_kernels::get("splat_sphere").unwrap(),
            "count_splat_in_tile",
        )
        .unwrap();
        {
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_dptr(pnt2pixxydepth.dptr);
            builder.arg_dptr(pnt2pixrad.dptr);
            builder.arg_dptr(tile2idx.dptr);
            builder.arg_dptr(pnt2idx.dptr);
            builder.arg_u32(tile_shape.0);
            builder.arg_u32(tile_shape.1);
            builder.arg_u32(16u32);
            unsafe {
                builder.launch_kernel(
                    fnc,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
                )
            }
            .unwrap();
        }
        (tile2idx, pnt2idx)
    };
    let tile2idx_dev = {
        let mut tmp = CuVec::<u32>::alloc_zeros(tile2idx_dev.n, stream).unwrap();
        del_cudarc_sys::cumsum::exclusive_scan(stream, &tile2idx_dev, &mut tmp);
        tmp
    };
    let pnt2idx_dev = {
        let mut tmp = CuVec::<u32>::alloc_zeros(pnt2ind_dev.n, stream).unwrap();
        del_cudarc_sys::cumsum::exclusive_scan(stream, &pnt2ind_dev, &mut tmp);
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
        let mut idx2tiledepth = CuVec::<u64>::alloc_zeros(num_ind as usize, stream).unwrap();
        let mut idx2pnt = CuVec::<u32>::alloc_zeros(num_ind as usize, stream).unwrap();
        let fnc = del_cudarc_sys::cache_func::get_function_cached(
            "del_splat::splat_sphere",
            del_splat_cuda_kernels::get("splat_sphere").unwrap(),
            "fill_index_info",
        )
        .unwrap();
        {
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_dptr(pnt2pixxydepth.dptr);
            builder.arg_dptr(pnt2pixrad.dptr);
            builder.arg_dptr(pnt2idx_dev.dptr);
            builder.arg_dptr(idx2tiledepth.dptr);
            builder.arg_dptr(idx2pnt.dptr);
            builder.arg_u32(tile_shape.0);
            builder.arg_u32(tile_shape.1);
            builder.arg_u32(16u32);
            unsafe {
                builder.launch_kernel(
                    fnc,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
                )
            }
            .unwrap();
        }
        del_cudarc_sys::sort_by_key_u64::radix_sort_by_key_u64(
            stream,
            &mut idx2tiledepth,
            &mut idx2pnt,
        )?;
        idx2pnt
    };
    Ok((tile2idx_dev, idx2pnt_dev))
}

pub fn pnt2splat3_to_pnt2splat2(
    stream: cu::CUstream,
    pnt2xyz: &CuVec<f32>,
    pnt2pixxydepth: &CuVec<f32>,
    pnt2pixrad: &CuVec<f32>,
    transform_world2ndc: &CuVec<f32>,
    img_shape: (u32, u32),
    radius: f32,
) -> anyhow::Result<()> {
    let func = del_cudarc_sys::cache_func::get_function_cached(
        "del-splat::splat_sphere",
        del_splat_cuda_kernels::get("splat_sphere").unwrap(),
        "splat3_to_splat2",
    )
    .unwrap();
    let num_pnt = pnt2xyz.n / 3;
    {
        let mut builder = del_cudarc_sys::Builder::new(stream);
        let img_shape_0 = img_shape.0;
        let img_shape_1 = img_shape.1;
        builder.arg_u32(num_pnt as u32);
        builder.arg_dptr(pnt2xyz.dptr);
        builder.arg_dptr(pnt2pixxydepth.dptr);
        builder.arg_dptr(pnt2pixrad.dptr);
        builder.arg_dptr(transform_world2ndc.dptr);
        builder.arg_u32(img_shape_0);
        builder.arg_u32(img_shape_1);
        builder.arg_f32(radius);
        builder
            .launch_kernel(
                func,
                del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32),
            )
            .unwrap();
    }
    Ok(())
}

/*
pub fn splat(
    dev: &std::sync::Arc<cudarc::driver::CudaStream>,
    img_shape: (u32, u32),
    pix2rgb_dev: &mut cudarc::driver::CudaSlice<f32>,
    pnt2splat_dev: &cudarc::driver::CudaSlice<Splat2>,
    tile_size: u32,
    tile2idx_dev: &cudarc::driver::CudaSlice<u32>,
    idx2pnt_dev: &cudarc::driver::CudaSlice<u32>,
) -> anyhow::Result<()> {
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
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
        del_splat_cudarc_kernel::SPLAT_SPHERE,
    )?;
    {
        let mut builder = dev.launch_builder(&count_splat_in_tile);
        let img_shape_0 = img_shape.0 as u32;
        let img_shape_1 = img_shape.1 as u32;
        builder.arg(&img_shape_0);
        builder.arg(&img_shape_1);
        builder.arg(pix2rgb_dev);
        let tile_shape_0 = tile_shape.0 as u32;
        let tile_shape_1 = tile_shape.1 as u32;
        builder.arg(&tile_shape_0);
        builder.arg(&tile_shape_1);
        let tile_size = tile_size as u32;
        builder.arg(&tile_size);
        builder.arg(tile2idx_dev);
        builder.arg(idx2pnt_dev);
        builder.arg(pnt2splat_dev);
        unsafe { builder.launch(cfg) }?;
    }
    Ok(())
}
 */
