mod fastpd;
mod xgboost;

use ndarray_linalg::LeastSquaresSvd;
use numpy::PyArray1;
use numpy::{
    ndarray::{Array1, ArrayView1, ArrayView2},
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;

use crate::xgboost::extract_trees_from_xgboost;
use crate::xgboost::types::XGBoostTreeModel;

fn get_projection_beta(X: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Array1<f64> {
    let beta = X.least_squares(&y).unwrap();
    beta.solution
}

#[pyfunction]
#[pyo3(name = "get_projection_beta")]
fn get_projection_beta_py<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let X = X.as_array();
    let y = y.as_array();
    let beta = get_projection_beta(X, y);
    let py_array = beta.into_pyarray_bound(py);
    Ok(py_array)
}

/// Python-facing wrapper for XGBoost tree extraction.
///
/// This function accepts an XGBoost model or Booster and returns a list of
/// `XGBoostTreeModel` instances, one per tree in the ensemble.
#[pyfunction]
#[pyo3(name = "extract_trees_from_xgboost")]
fn extract_trees_from_xgboost_py(model: Bound<'_, PyAny>) -> PyResult<Vec<XGBoostTreeModel>> {
    extract_trees_from_xgboost(&model)
}

#[pymodule]
fn glex_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_projection_beta_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_trees_from_xgboost_py, m)?)?;
    m.add_class::<XGBoostTreeModel>()?;
    Ok(())
}
