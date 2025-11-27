use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(circle_splat_from_project_spheres, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn circle_splat_from_project_spheres(
    _py: Python<'_>,
    pnt2xyz: &Bound<'_, PyAny>,
    radius: f32,
    transform_world2ndc: &Bound<'_, PyAny>,
    img_shape: (usize, usize),
    pnt2pixxyndcz: &Bound<'_, PyAny>,
    pnt2pixrad: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pnt2xyz = del_dlpack::get_managed_tensor_from_pyany(pnt2xyz)?;
    let transform_world2ndc = del_dlpack::get_managed_tensor_from_pyany(transform_world2ndc)?;
    let pnt2pixxyndcz = del_dlpack::get_managed_tensor_from_pyany(pnt2pixxyndcz)?;
    let pnt2pixrad = del_dlpack::get_managed_tensor_from_pyany(pnt2pixrad)?;
    let num_pnt = del_dlpack::get_shape_tensor(pnt2xyz, 0).unwrap();
    let device = pnt2xyz.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<f32>(pnt2xyz, num_pnt, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<f32>(transform_world2ndc, 16, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(pnt2pixxyndcz, num_pnt, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<f32>(pnt2pixrad, num_pnt, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let pnt2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(pnt2xyz) }.unwrap();
            let transform_world2ndc =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_world2ndc) }.unwrap();
            let transform_world2ndc = arrayref::array_ref![transform_world2ndc, 0, 16];
            let pnt2pixxyndcz =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pnt2pixxyndcz) }.unwrap();
            let pnt2pixrad = unsafe { del_dlpack::slice_from_tensor_mut::<f32>(pnt2pixrad) }.unwrap();
            del_splat_cpu::splat_sphere::project(
                pnt2xyz,
                radius,
                transform_world2ndc,
                img_shape,
                pnt2pixxyndcz,
                pnt2pixrad,
            );
        },
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let fnc = del_cudarc_sys::cache_func::get_function_cached(
                "del_splat::splat_sphere",
                del_splat_cuda_kernels::get("splat_sphere").unwrap(),
                "splat3_to_splat2",
            ).unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_pnt as u32);
            builder.arg_data(&pnt2xyz.data);
            builder.arg_data(&pnt2pixxyndcz.data);
            builder.arg_data(&pnt2pixrad.data);
            builder.arg_data(&transform_world2ndc.data);
            builder.arg_u32(img_shape.0 as u32);
            builder.arg_u32(img_shape.1 as u32);
            builder.arg_f32(radius);
            builder.launch_kernel(fnc, del_cudarc_sys::LaunchConfig::for_num_elems(num_pnt as u32)).unwrap();
        },
        _ => { todo!() }
    }
    Ok(())
}


