use cudarc::driver::DeviceSlice;
use del_cudarc_safe::cudarc;

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

pub fn tile2idx_idx2pnt(
    dev: &std::sync::Arc<cudarc::driver::CudaStream>,
    tile_shape: (u32, u32),
    pnt2splat_dev: &cudarc::driver::CudaSlice<Splat2>,
) -> anyhow::Result<(
    cudarc::driver::CudaSlice<u32>,
    cudarc::driver::CudaSlice<u32>,
)> {
    let (tile2idx_dev, pnt2ind_dev) = {
        let num_pnt = pnt2splat_dev.len();
        let mut tile2idx_dev =
            dev.alloc_zeros::<u32>((tile_shape.0 * tile_shape.1 + 1) as usize)?;
        let mut pnt2idx_dev = dev.alloc_zeros::<u32>(num_pnt + 1)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_pnt as u32);
        let count_splat_in_tile = del_cudarc_safe::get_or_load_func(
            &dev.context(),
            "count_splat_in_tile",
            del_splat_cudarc_kernel::SPLAT_SPHERE,
        )?;
        {
            use cudarc::driver::PushKernelArg;
            let mut builder = dev.launch_builder(&count_splat_in_tile);
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
        let mut tmp = dev.alloc_zeros(tile2idx_dev.len())?;
        del_cudarc_safe::cumsum::sum_scan_blelloch(&dev, &mut tmp, &tile2idx_dev)?;
        tmp
    };
    let pnt2idx_dev = {
        let mut tmp = dev.alloc_zeros::<u32>(pnt2ind_dev.len())?;
        del_cudarc_safe::cumsum::sum_scan_blelloch(&dev, &mut tmp, &pnt2ind_dev)?;
        tmp
    };
    let num_ind = dev.memcpy_dtov(&pnt2idx_dev)?.last().unwrap().to_owned(); // todo: send only last element to cpu
    debug_assert_eq!(
        num_ind,
        dev.memcpy_dtov(&tile2idx_dev)?
            .last()
            .unwrap()
            .to_owned()
    );
    let idx2pnt_dev = {
        let mut idx2tiledepth_dev = dev.alloc_zeros::<u64>(num_ind as usize)?;
        let mut idx2pnt_dev = dev.alloc_zeros(num_ind as usize)?;
        let num_pnt = pnt2splat_dev.len();
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_pnt as u32);
        let count_splat_in_tile = del_cudarc_safe::get_or_load_func(
            &dev.context(),
            "fill_index_info",
            del_splat_cudarc_kernel::SPLAT_SPHERE,
        )?;
        {
            use cudarc::driver::PushKernelArg;
            let mut builder = dev.launch_builder(&count_splat_in_tile);
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
            &dev,
            &mut idx2tiledepth_dev,
            &mut idx2pnt_dev,
        )?;
        idx2pnt_dev
    };
    Ok((tile2idx_dev, idx2pnt_dev))
}

pub fn pnt2splat3_to_pnt2splat2(
    dev: &std::sync::Arc<cudarc::driver::CudaStream>,
    pnt2splat3_dev: &cudarc::driver::CudaSlice<Splat3>,
    pnt2splat2_dev: &mut cudarc::driver::CudaSlice<Splat2>,
    transform_world2ndc_dev: &cudarc::driver::CudaSlice<f32>,
    img_shape: (u32, u32),
    radius: f32,
) -> anyhow::Result<()> {
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(pnt2splat3_dev.len() as u32);
    let xyzrgb_to_splat = del_cudarc_safe::get_or_load_func(
        &dev.context(),
        "splat3_to_splat2",
        del_splat_cudarc_kernel::SPLAT_SPHERE,
    )?;
    {
        use cudarc::driver::PushKernelArg;
        let mut builder = dev.launch_builder(&xyzrgb_to_splat);
        let img_shape_0 = img_shape.0  as u32;
        let img_shape_1 = img_shape.1  as u32;
        let pnt2splat3_dev_len = pnt2splat3_dev.len() as u32;
        builder.arg(&pnt2splat3_dev_len);
        builder.arg(pnt2splat2_dev);
        builder.arg(pnt2splat3_dev);
        builder.arg(transform_world2ndc_dev);
        builder.arg(&img_shape_0);
        builder.arg(&img_shape_1);
        builder.arg(&radius);
        unsafe { builder.launch(cfg) }?;
    }
    /*
    let param = (
        pnt2splat3_dev.len(),
        pnt2splat2_dev,
        pnt2splat3_dev,
        transform_world2ndc_dev,
        img_shape.0,
        img_shape.1,
        radius,
    );
    use cudarc::driver::LaunchAsync;
    unsafe { xyzrgb_to_splat. launch(cfg, param) }?;
     */
    Ok(())
}

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
        use cudarc::driver::PushKernelArg;
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
        pnt2splat_dev,
    );
    use cudarc::driver::LaunchAsync;
    unsafe { count_splat_in_tile.launch(cfg, param) }?;
     */
    Ok(())
}
