use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use serde_json::Value;

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

/// Extracts the base_score (intercept) from an XGBoost model or Booster.
///
/// The base_score is stored in the booster's config JSON under
/// `learner.learner_model_param.base_score`.
///
/// # Arguments
/// * `model` - Either an XGBoost model (XGBClassifier/XGBRegressor) or a Booster.
///
/// # Returns
/// The base_score as a float, or 0.0 if not found or if parsing fails.
///
/// # Errors
/// Returns `PyErr` if the model is not an XGBoost model/Booster.
pub fn get_booster_base_score(model: &Bound<'_, PyAny>) -> PyResult<f32> {
    // Try to get booster from model (XGBClassifier/XGBRegressor have get_booster()).
    let booster = if model.hasattr("get_booster")? {
        model.call_method0("get_booster")?
    } else {
        // Assume it's already a Booster.
        model.clone()
    };

    // Validate it's actually a Booster by checking for save_config method.
    if !booster.hasattr("save_config")? {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected XGBoost model or Booster, but object has no 'save_config' method",
        ));
    }

    // Get the config JSON
    let config_py = booster.call_method0("save_config")?;
    let config_str: String = config_py.extract()?;

    // Parse JSON to extract base_score
    // The base_score is stored as a string like "[1.0883814E1]" in the JSON
    let config: Value = serde_json::from_str(&config_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to parse booster config JSON: {}",
            e
        ))
    })?;

    // Navigate to base_score: learner.learner_model_param.base_score
    let base_score_str = config
        .get("learner")
        .and_then(|l| l.get("learner_model_param"))
        .and_then(|p| p.get("base_score"))
        .and_then(|v| v.as_str());

    if let Some(score_str) = base_score_str {
        // Parse the string (e.g., "[1.0883814E1]" -> 10.883814)
        let cleaned = score_str.trim_matches(|c| c == '[' || c == ']');
        cleaned.parse::<f32>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse base_score '{}': {}",
                cleaned, e
            ))
        })
    } else {
        // Default to 0.0 if base_score is not found
        Ok(0.0)
    }
}
