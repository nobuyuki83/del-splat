use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(project, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn project(
    _py: Python<'_>,
    pnt2xyz: &Bound<'_, PyAny>,
    radius: f32,
    transform_world2ndc: &Bound<'_, PyAny>,
    img_width: usize,
    img_height: usize,
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
                (img_width, img_height),
                pnt2pixxyndcz,
                pnt2pixrad,
            );
        },
        _ => { todo!() }
    }
    Ok(())
}


