pub mod json_schema;
pub mod parser;
pub mod python_bridge;
pub mod tree_model;
pub mod types;

use pyo3::prelude::*;
use pyo3::Bound;

use crate::xgboost::parser::parse_json_tree;
use crate::xgboost::python_bridge::{get_booster_base_score, get_booster_json_dumps};
use crate::xgboost::types::XGBoostTreeModel;

/// Extracts all trees and base_score from an XGBoost model.
///
/// # Arguments
/// * `model` - XGBoost model (XGBClassifier, XGBRegressor) or Booster.
///
/// # Returns
/// Tuple of (trees, base_score) where:
/// - `trees`: Vector of `XGBoostTreeModel`, one per tree in the ensemble
/// - `base_score`: The intercept/base_score from the model config
///
/// # Errors
/// Returns `PyErr` if:
/// - Model is not an XGBoost model.
/// - Model is not fitted.
/// - JSON parsing fails for any tree.
/// - Tree structure validation fails.
pub fn extract_trees_from_xgboost(
    model: &Bound<'_, PyAny>,
) -> PyResult<(Vec<XGBoostTreeModel>, f64)> {
    let dumps = get_booster_json_dumps(model)?;
    let base_score = get_booster_base_score(model)?;
    let mut trees = Vec::with_capacity(dumps.len());

    for (tree_idx, dump) in dumps.into_iter().enumerate() {
        let xgb_tree = parse_json_tree(&dump).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to parse tree {}: {}",
                tree_idx, e
            ))
        })?;
        trees.push(XGBoostTreeModel { tree: xgb_tree });
    }

    Ok((trees, base_score))
}
