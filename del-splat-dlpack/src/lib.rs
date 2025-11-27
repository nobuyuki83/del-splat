use pyo3::{types::PyModule, Bound, PyResult, Python};

mod splat_circle;
mod splat_point;
mod points;


#[pyo3::pymodule]
#[pyo3(name = "del_splat_dlpack")]
fn del_msh_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    splat_circle::add_functions(_py, m)?;
    splat_point::add_functions(_py, m)?;
    points::add_functions(_py, m)?;
    Ok(())
}
