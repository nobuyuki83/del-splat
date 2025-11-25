use pyo3::prelude::PyModuleMethods;
use pyo3::{types::PyModule, Bound, PyResult, Python};

mod splat_sphere;
mod points;

#[pyo3::pymodule]
#[pyo3(name = "del_splat_dlpack")]
fn del_msh_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    splat_sphere::add_functions(_py, m)?;
    Ok(())
}
