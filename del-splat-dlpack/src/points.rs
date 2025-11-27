use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(points_load_ply, m)?)?;
    Ok(())
}

#[pyfunction]
fn points_load_ply(
    py: Python<'_>,
    path: String,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (pnt2xyz, pnt2rgb) = del_splat_cpu::io_ply::read_xyzrgb(path).unwrap();
     let pnt2xyz_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(pnt2xyz.len() as i64) / 3, 3], pnt2xyz);
    let pnt2rgb_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(pnt2rgb.len() as i64) / 3, 3], pnt2rgb);
    Ok((pnt2xyz_cap, pnt2rgb_cap))
}