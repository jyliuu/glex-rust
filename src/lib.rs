mod fastpd;
mod xgboost;

use pyo3::prelude::*;

use crate::xgboost::extract_trees_from_xgboost;
use crate::xgboost::types::XGBoostTreeModel;

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
    m.add_function(wrap_pyfunction!(extract_trees_from_xgboost_py, m)?)?;
    m.add_class::<XGBoostTreeModel>()?;
    Ok(())
}
