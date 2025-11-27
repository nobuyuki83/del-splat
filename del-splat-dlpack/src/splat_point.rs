use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(point_splat_rasterize_pix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(point_splat_rasterize_pix_zbuffer, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn point_splat_rasterize_pix(
    _py: Python<'_>,
    pnt2pixxyndcz: &Bound<'_, PyAny>,
    pnt2rgb: &Bound<'_, PyAny>,
    pix2rgb: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pnt2pixxyndcz = del_dlpack::get_managed_tensor_from_pyany(pnt2pixxyndcz)?;
    let pnt2rgb = del_dlpack::get_managed_tensor_from_pyany(pnt2rgb)?;
    let pix2rgb = del_dlpack::get_managed_tensor_from_pyany(pix2rgb)?;
    let num_pnt = del_dlpack::get_shape_tensor(pnt2pixxyndcz, 0).unwrap();
    let device = pnt2pixxyndcz.ctx.device_type;
    let img_width = del_dlpack::get_shape_tensor(pix2rgb, 1).unwrap();
    //
    del_dlpack::check_2d_tensor::<f32>(pnt2pixxyndcz, num_pnt, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(pnt2rgb, num_pnt, 3, device).unwrap();
    del_dlpack::check_3d_tensor::<f32>(pix2rgb, -1, -1, 3, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let pnt2pixxyndcz =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pnt2pixxyndcz) }.unwrap();
            let pnt2rgb = unsafe { del_dlpack::slice_from_tensor::<f32>(pnt2rgb) }.unwrap();
            let pix2rgb = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pix2rgb) }.unwrap();
            use slice_of_array::SliceNestExt;
            let pix2rgb: &mut [[f32; 3]] = pix2rgb.nest_mut();
            del_splat_cpu::pnt2pixxyndcz::render_pix_sort_depth(pnt2pixxyndcz, pnt2rgb, img_width as usize, pix2rgb).unwrap();
        },
        _ => { todo!() }
    }
    Ok(())
}


#[pyo3::pyfunction]
pub fn point_splat_rasterize_pix_zbuffer(
    _py: Python<'_>,
    pnt2pixxyndcz: &Bound<'_, PyAny>,
    pnt2rgb: &Bound<'_, PyAny>,
    pix2rgb: &Bound<'_, PyAny>,
    pix2unitdepth: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pnt2pixxyndcz = del_dlpack::get_managed_tensor_from_pyany(pnt2pixxyndcz)?;
    let pnt2rgb = del_dlpack::get_managed_tensor_from_pyany(pnt2rgb)?;
    let pix2rgb = del_dlpack::get_managed_tensor_from_pyany(pix2rgb)?;
    let pix2unitdepth = del_dlpack::get_managed_tensor_from_pyany(pix2unitdepth)?;
    let num_pnt = del_dlpack::get_shape_tensor(pnt2pixxyndcz, 0).unwrap();
    let device = pnt2pixxyndcz.ctx.device_type;
    let img_width = del_dlpack::get_shape_tensor(pix2rgb, 1).unwrap();
    let img_height = del_dlpack::get_shape_tensor(pix2rgb, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<f32>(pnt2pixxyndcz, num_pnt, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(pnt2rgb, num_pnt, 3, device).unwrap();
    del_dlpack::check_3d_tensor::<f32>(pix2rgb, img_height, img_width, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(pix2unitdepth, img_height, img_width, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let pnt2pixxyndcz =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pnt2pixxyndcz) }.unwrap();
            let pnt2rgb = unsafe { del_dlpack::slice_from_tensor::<f32>(pnt2rgb) }.unwrap();
            let pix2rgb = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pix2rgb) }.unwrap();
            use slice_of_array::SliceNestExt;
            let pix2rgb: &mut [[f32; 3]] = pix2rgb.nest_mut();
            del_splat_cpu::pnt2pixxyndcz::render_pix_sort_depth(pnt2pixxyndcz, pnt2rgb, img_width as usize, pix2rgb).unwrap();
        },
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            {
                use del_cudarc_sys::{cu::CUdeviceptr, CuVec};
                let a = CuVec::<f32>::from_dptr(pix2unitdepth.data as CUdeviceptr, (img_width * img_height) as usize);
                del_cudarc_sys::array1d::fill_f32(stream,&a, f32::MAX);
            }
            //
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del-splat::splat_point",
                del_splat_cuda_kernels::get("splat_point").unwrap(),
                "rasterize_zbuffer").unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_data(&pnt2pixxyndcz.data);
            builder.arg_data(&pnt2rgb.data);
            builder.arg_data(&pix2unitdepth.data);
            builder.arg_data(&pix2rgb.data);
            builder.arg_u32(img_width as u32);
            builder.arg_u32(img_height as u32);
            builder.launch_kernel(func, del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32)).unwrap();
        },
        _ => { todo!() }
    }
    Ok(())
}