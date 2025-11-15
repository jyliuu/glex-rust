use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// Extracts JSON tree dumps from an XGBoost model or Booster.
///
/// # Arguments
/// * `py` - Python interpreter GIL token.
/// * `model` - Either an XGBoost model (XGBClassifier/XGBRegressor) or a Booster.
///
/// # Returns
/// Vector of JSON strings, one per tree in the ensemble.
///
/// # Errors
/// Returns `PyErr` if:
/// - Model is not an XGBoost model/Booster.
/// - Model is not fitted.
/// - `get_dump()` call fails.
/// - Result is not a list of strings.
pub fn get_booster_json_dumps(model: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    // Try to get booster from model (XGBClassifier/XGBRegressor have get_booster()).
    let booster = if model.hasattr("get_booster")? {
        model.call_method0("get_booster")?
    } else {
        // Assume it's already a Booster.
        model.clone()
    };

    // Validate it's actually a Booster by checking for get_dump method.
    if !booster.hasattr("get_dump")? {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected XGBoost model or Booster, but object has no 'get_dump' method",
        ));
    }

    // Call get_dump with JSON format using positional arguments:
    // get_dump(fmap: str = "", with_stats: bool = False, dump_format: str = "text")
    let args = ("", false, "json");
    let dumps_py = booster.call_method1("get_dump", args)?;

    // Extract as Vec<String>.
    let dumps: Vec<String> = dumps_py.extract()?;

    if dumps.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "XGBoost model has no trees (model may not be fitted)",
        ));
    }

    Ok(dumps)
}
