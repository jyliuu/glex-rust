mod fastpd;
mod xgboost;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::fastpd::augment_eval::FastPD;
use crate::xgboost::extract_trees_from_xgboost;
use crate::xgboost::types::XGBoostTreeModel;

/// Python-facing wrapper for XGBoost tree extraction.
///
/// This function accepts an XGBoost model or Booster and returns a list of
/// `XGBoostTreeModel` instances, one per tree in the ensemble.
/// Note: The base_score is extracted internally but not returned here.
/// It is automatically used when creating a FastPD instance.
#[pyfunction]
#[pyo3(name = "extract_trees_from_xgboost")]
fn extract_trees_from_xgboost_py(model: Bound<'_, PyAny>) -> PyResult<Vec<XGBoostTreeModel>> {
    let (trees, _base_score) = extract_trees_from_xgboost(&model)?;
    Ok(trees)
}

/// Python-facing wrapper for FastPD.
///
/// This class provides efficient computation of partial dependence functions
/// for tree-based models.
#[pyclass]
pub struct FastPDPy {
    fastpd: FastPD<XGBoostTreeModel>,
}

#[pymethods]
impl FastPDPy {
    /// Create a FastPD instance from an XGBoost model.
    ///
    /// This method extracts trees from the XGBoost model and augments them
    /// with the provided background samples.
    ///
    /// # Arguments
    /// * `model` - XGBoost model (Booster or XGBModel)
    /// * `background_samples` - Background samples for PD estimation
    ///     shape: (n_background, n_features)
    #[classmethod]
    fn from_xgboost(
        _cls: &Bound<'_, PyType>,
        _py: Python<'_>,
        model: Bound<'_, PyAny>,
        background_samples: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        // Extract trees and base_score from XGBoost model
        let (trees, base_score) = extract_trees_from_xgboost(&model)?;

        // Convert f64 array to f32 array
        let background_f64 = background_samples.as_array();
        let background_f32: ndarray::Array2<f32> = background_f64.mapv(|x| x as f32);

        // Create FastPD instance with intercept
        let fastpd = FastPD::new(trees, &background_f32.view(), base_score).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
        })?;

        Ok(Self { fastpd })
    }

    /// Compute PD function v_S(x_S) for a single feature subset.
    ///
    /// # Arguments
    /// * `evaluation_points` - Points at which to evaluate PD
    ///     shape: (n_evaluation_points, n_features)
    /// * `feature_subset` - Indices of features in subset S
    ///
    /// # Returns
    /// PD values at each evaluation point
    ///     shape: (n_evaluation_points,)
    fn pd_function<'a>(
        &mut self,
        py: Python<'a>,
        evaluation_points: PyReadonlyArray2<f64>,
        feature_subset: Vec<usize>,
    ) -> PyResult<Bound<'a, PyArray1<f32>>> {
        // Convert f64 array to f32 array
        let eval_f64 = evaluation_points.as_array();
        let eval_f32: ndarray::Array2<f32> = eval_f64.mapv(|x| x as f32);

        let result = self
            .fastpd
            .pd_function(&eval_f32.view(), &feature_subset)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
            })?;

        Ok(PyArray1::from_owned_array_bound(py, result))
    }

    /// Predicts the output for given input points by summing predictions from all trees.
    ///
    /// This is the standard ensemble prediction: sum of leaf values from all trees.
    ///
    /// # Arguments
    /// * `evaluation_points` - Points at which to predict
    ///     shape: (n_evaluation_points, n_features)
    ///
    /// # Returns
    /// Predictions at each evaluation point
    ///     shape: (n_evaluation_points,)
    fn predict<'a>(
        &self,
        py: Python<'a>,
        evaluation_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'a, PyArray1<f32>>> {
        // Convert f64 array to f32 array
        let eval_f64 = evaluation_points.as_array();
        let eval_f32: ndarray::Array2<f32> = eval_f64.mapv(|x| x as f32);

        let result = self.fastpd.predict(&eval_f32.view()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
        })?;

        Ok(PyArray1::from_owned_array_bound(py, result))
    }

    /// Clear all PD caches.
    ///
    /// This is useful when memory is a concern or when you want to ensure
    /// fresh computations for a new batch of evaluations.
    fn clear_caches(&mut self) {
        self.fastpd.clear_caches();
    }

    /// Returns the number of trees in the ensemble.
    fn num_trees(&self) -> usize {
        self.fastpd.num_trees()
    }

    /// Returns the number of background samples.
    fn n_background(&self) -> usize {
        self.fastpd.n_background()
    }

    /// Returns the number of features.
    fn n_features(&self) -> usize {
        self.fastpd.n_features()
    }
}

#[pymodule]
fn glex_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_trees_from_xgboost_py, m)?)?;
    m.add_class::<XGBoostTreeModel>()?;
    m.add_class::<FastPDPy>()?;
    Ok(())
}
