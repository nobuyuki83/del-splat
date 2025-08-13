use cudarc::driver::DeviceSlice;
use del_cudarc_safe::cudarc;
use del_cudarc_safe::cudarc::driver::PushKernelArg;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Splat3 {
    pub xyz: [f32; 3],
    // nrm: [f32; 3],
    pub rgb_dc: [f32; 3],
    pub rgb_sh: [f32; 45],
    pub opacity: f32,
    pub scale: [f32; 3],
    pub quaternion: [f32; 4],
}

impl del_msh_cpu::io_ply::GaussSplat3D for Splat3 {
    fn new(
        xyz: [f32; 3],
        rgb_dc: [f32; 3],
        rgb_sh: [f32; 45],
        opacity: f32,
        scale: [f32; 3],
        quaternion: [f32; 4],
    ) -> Self {
        Splat3 {
            xyz,
            rgb_dc,
            rgb_sh,
            opacity,
            scale,
            quaternion,
        }
    }
}

impl del_msh_cpu::vtx2point::HasXyz<f32> for Splat3 {
    fn xyz(&self) -> &[f32; 3] {
        &self.xyz
    }
}

unsafe impl cudarc::driver::DeviceRepr for Splat3 {}

// above: trait implementation for GSplat3
// ----------------------------

#[derive(Clone, Default)]
#[repr(C)]
pub struct Splat2 {
    pub pos_pix: [f32; 2],
    pub sig_inv: [f32; 3],
    pub aabb: [f32; 4],
    pub rgb: [f32; 3],
    pub alpha: f32,
    pub ndc_z: f32,
}

unsafe impl cudarc::driver::DeviceRepr for Splat2 {}

impl del_splat_core::splat_gaussian2::Splat2 for Splat2 {
    fn ndc_z(&self) -> f32 {
        self.ndc_z
    }
    fn aabb(&self) -> &[f32; 4] {
        &self.aabb
    }
    fn property(&self) -> (&[f32; 2], &[f32; 3], &[f32; 3], f32) {
        (&self.pos_pix, &self.sig_inv, &self.rgb, self.alpha)
    }
}

/*
impl del_canvas_cpu::tile_acceleration::Splat2 for Splat2{
    fn aabb(&self) -> [f32; 4] { self.aabb }
    fn ndc_z(&self) -> f32 { self.ndc_z }
}
 */

// ---------------------------------
// below: global funcs

pub fn pnt2splat3_to_pnt2splat2(
    dev: &std::sync::Arc<cudarc::driver::CudaStream>,
    pnt2gs3_dev: &cudarc::driver::CudaSlice<Splat3>,
    pnt2gs2_dev: &mut cudarc::driver::CudaSlice<Splat2>,
    transform_world2ndc_dev: &cudarc::driver::CudaSlice<f32>,
    img_shape: (u32, u32),
) -> anyhow::Result<()> {
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(pnt2gs3_dev.len() as u32);
    let splat3_to_splat2 = del_cudarc_safe::get_or_load_func(
        &dev.context(),
        "splat3_to_splat2",
        del_splat_cudarc_kernel::SPLAT_GAUSS,
    )?;
    {
        let mut builder = dev.launch_builder(&splat3_to_splat2);
        let pnt2gs3_dev_len = pnt2gs3_dev.len() as u32;
        let img_shape_0 = img_shape.0 as u32;
        let img_shape_1 = img_shape.1 as u32;
        builder.arg(&pnt2gs3_dev_len);
        builder.arg(pnt2gs2_dev);
        builder.arg(pnt2gs3_dev);
        builder.arg(transform_world2ndc_dev);
        builder.arg(&img_shape_0);
        builder.arg(&img_shape_1);
        unsafe { builder.launch(cfg) }?;
    }
    /*
    let param = (
        pnt2gs3_dev.len(),
        pnt2gs2_dev,
        pnt2gs3_dev,
        transform_world2ndc_dev,
        img_shape.0,
        img_shape.1,
    );
    use cudarc::driver::LaunchAsync;
    unsafe { splat3_to_splat2.launch(cfg, param) }?;
     */
    Ok(())
}

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
