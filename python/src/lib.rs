// PyO3 bindings crate
// This crate contains all Python bindings and delegates to glex-core for computation

mod xgboost;

use glex_core::fastpd::parallel::ParallelSettings;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::xgboost::extract_trees_from_xgboost;
use crate::xgboost::types::XGBoostTreeModel;
use glex_core::FastPD;

/// Default number of evaluation points for default plotting
const DEFAULT_N_POINTS: usize = 1000;

/// Extract feature names from various sources using priority system.
///
/// Priority:
/// 1. Explicit feature_names parameter (handled in from_xgboost)
/// 2. background_samples.feature_names attribute (e.g., sklearn Bunch)
/// 3. pandas DataFrame.columns
/// 4. XGBoost model.feature_names_in_
/// 5. XGBoost booster.feature_names
/// 6. None (will use defaults)
fn extract_feature_names(
    background_samples: &Bound<'_, PyAny>,
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<String>>> {
    // Priority 2: Check if background_samples has feature_names attribute
    if let Ok(feature_names_attr) = background_samples.getattr("feature_names") {
        if !feature_names_attr.is_none() {
            if let Ok(names) = feature_names_attr.extract::<Vec<String>>() {
                return Ok(Some(names));
            }
        }
    }

    // Priority 3: Check if it's a pandas DataFrame
    if let Ok(pd) = py.import("pandas") {
        if let Ok(df_class) = pd.getattr("DataFrame") {
            if background_samples.is_instance(&df_class)? {
                if let Ok(columns) = background_samples.getattr("columns") {
                    if let Ok(names) = columns.extract::<Vec<String>>() {
                        return Ok(Some(names));
                    }
                }
            }
        }
    }

    // Priority 4: Try XGBoost model feature_names_in_
    if let Ok(feature_names_in) = model.getattr("feature_names_in_") {
        if !feature_names_in.is_none() {
            if let Ok(names) = feature_names_in.extract::<Vec<String>>() {
                return Ok(Some(names));
            }
        }
    }

    // Priority 5: Try Booster feature_names
    if let Ok(booster) = model.call_method0("get_booster") {
        if let Ok(feature_names) = booster.getattr("feature_names") {
            if !feature_names.is_none() {
                if let Ok(names) = feature_names.extract::<Vec<String>>() {
                    return Ok(Some(names));
                }
            }
        }
    }

    // Priority 6: None (will use defaults)
    Ok(None)
}

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
    feature_names: Option<Vec<String>>,
    background_samples: Array2<f32>, // Always stored (required, not optional)
    // Cache for components computed with DEFAULT_N_POINTS evaluation points
    cached_comp_values: Option<Array2<f32>>, // Shape: (DEFAULT_N_POINTS, n_subsets)
    cached_subsets: Option<Vec<Vec<usize>>>, // Parallel array: subset for each column in comp_values
    cached_eval_points: Option<Array2<f32>>, // Evaluation points used for cache (for reference)
    cached_max_order: Option<usize>,         // Maximum order of cached components
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
    ///     Can be numpy array, pandas DataFrame, or object with feature_names attribute
    /// * `feature_names` - Optional explicit feature names (list of strings)
    /// * `n_threads` - Number of threads to use for parallelization (default: 1)
    #[classmethod]
    #[pyo3(signature = (model, background_samples, feature_names = None, n_threads = 1))]
    fn from_xgboost(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        background_samples: Bound<'_, PyAny>,
        feature_names: Option<Vec<String>>,
        n_threads: usize,
    ) -> PyResult<Self> {
        // Extract feature names using priority system
        let extracted_feature_names = if feature_names.is_some() {
            feature_names
        } else {
            extract_feature_names(&background_samples, py, &model)?
        };

        // Convert background_samples to array
        // Try to get .data if it's a Bunch object, otherwise use directly
        let background_f64 = if let Ok(arr) = background_samples.cast::<PyArray2<f64>>() {
            // Convert PyArray2 to Array2 safely using to_owned_array
            arr.to_owned_array()
        } else {
            // Try to extract .data if it's a Bunch object
            let data_attr = background_samples.getattr("data")?;
            let arr = data_attr.cast::<PyArray2<f64>>()?;
            arr.to_owned_array()
        };

        // Convert f64 array to f32 array and store
        let background_f32: Array2<f32> = background_f64.mapv(|x| x as f32);

        // Extract trees and base_score from XGBoost model
        let (trees, base_score) = extract_trees_from_xgboost(&model)?;

        // Create FastPD instance with intercept
        let fastpd = FastPD::new(
            trees,
            &background_f32.view(),
            base_score,
            ParallelSettings::with_n_threads(n_threads),
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
        })?;

        Ok(Self {
            fastpd,
            feature_names: extracted_feature_names,
            background_samples: background_f32,
            cached_comp_values: None,
            cached_subsets: None,
            cached_eval_points: None,
            cached_max_order: None,
        })
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

        Ok(PyArray1::from_owned_array(py, result))
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

        Ok(PyArray1::from_owned_array(py, result))
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

    /// Get feature name by index.
    ///
    /// # Arguments
    /// * `feature_idx` - Feature index
    ///
    /// # Returns
    /// Feature name as string, or default name like "f0" if feature_names not available
    fn get_feature_name(&self, feature_idx: usize) -> String {
        if let Some(ref names) = self.feature_names {
            if feature_idx < names.len() {
                return names[feature_idx].clone();
            }
        }
        // Default name
        format!("Feature {}", feature_idx)
    }

    /// Get all feature names.
    ///
    /// # Returns
    /// Optional vector of feature names, or None if not available
    fn feature_names(&self) -> Option<Vec<String>> {
        self.feature_names.clone()
    }

    /// Get stored background samples.
    ///
    /// # Returns
    /// Background samples array (always available)
    fn get_background_samples<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray2<f32>> {
        PyArray2::from_owned_array(py, self.background_samples.clone())
    }

    /// Get cached components if available.
    ///
    /// # Returns
    /// Optional tuple of (comp_values, subsets, eval_points, max_order)
    fn get_cached_components<'a>(
        &self,
        py: Python<'a>,
    ) -> Option<(
        Bound<'a, PyArray2<f32>>,
        Vec<Vec<usize>>,
        Bound<'a, PyArray2<f32>>,
        usize,
    )> {
        if let (Some(ref comp_values), Some(ref subsets), Some(ref eval_points), Some(max_order)) = (
            &self.cached_comp_values,
            &self.cached_subsets,
            &self.cached_eval_points,
            self.cached_max_order,
        ) {
            Some((
                PyArray2::from_owned_array(py, comp_values.clone()),
                subsets.clone(),
                PyArray2::from_owned_array(py, eval_points.clone()),
                max_order,
            ))
        } else {
            None
        }
    }

    /// Extract component by subset from cache.
    ///
    /// # Arguments
    /// * `subset` - Feature subset to find
    ///
    /// # Returns
    /// Optional component values array if found
    fn extract_component_by_subset<'a>(
        &self,
        py: Python<'a>,
        subset: Vec<usize>,
    ) -> PyResult<Option<Bound<'a, PyArray1<f32>>>> {
        if let (Some(ref comp_values), Some(ref cached_subsets)) =
            (&self.cached_comp_values, &self.cached_subsets)
        {
            // Search for matching subset
            for (idx, cached_subset) in cached_subsets.iter().enumerate() {
                if cached_subset == &subset {
                    // Extract column
                    let col = comp_values.column(idx).to_owned();
                    return Ok(Some(PyArray1::from_owned_array(py, col)));
                }
            }
        }
        Ok(None)
    }

    /// Compute plain partial dependence surfaces v_S(x_S)
    /// for all subsets S with 1 <= |S| <= max_order.
    ///
    /// This function efficiently computes all PD surfaces up to a given order
    /// by batch-evaluating all subsets in a single pass through each tree.
    ///
    /// # Arguments
    /// * `evaluation_points` - Points at which to evaluate PD
    ///     shape: (n_evaluation_points, n_features)
    /// * `max_order` - Maximum interaction order (e.g., 1 for main effects, 2 for pairwise, etc.)
    ///
    /// # Returns
    /// A tuple `(pd_values, subsets)` where:
    /// - `pd_values`: 2D numpy array of shape (n_eval, n_subsets) with one column per subset S
    /// - `subsets`: List of lists, where each inner list contains the feature indices for a subset
    fn pd_functions_up_to_order<'a>(
        &mut self,
        py: Python<'a>,
        evaluation_points: PyReadonlyArray2<f64>,
        max_order: usize,
    ) -> PyResult<(Bound<'a, PyArray2<f32>>, Vec<Vec<usize>>)> {
        // Convert f64 array to f32 array
        let eval_f64 = evaluation_points.as_array();
        let eval_f32: ndarray::Array2<f32> = eval_f64.mapv(|x| x as f32);

        let (pd_values, subsets) = self
            .fastpd
            .pd_functions_up_to_order(&eval_f32.view(), max_order)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
            })?;

        // Convert FeatureSubset vector to Vec<Vec<usize>>
        let subsets_py: Vec<Vec<usize>> = subsets
            .into_iter()
            .map(|subset| subset.as_slice())
            .collect();

        Ok((PyArray2::from_owned_array(py, pd_values), subsets_py))
    }

    /// Compute functional decomposition components f_S(x_S)
    /// for all subsets S with 1 <= |S| <= max_order.
    ///
    /// This function computes the ANOVA functional decomposition components
    /// via inclusion–exclusion, sharing the same intermediate v_U(x) computation
    /// as `pd_functions_up_to_order`.
    ///
    /// # Arguments
    /// * `evaluation_points` - Points at which to evaluate
    ///     shape: (n_evaluation_points, n_features)
    /// * `max_order` - Maximum interaction order
    ///
    /// # Returns
    /// A tuple `(comp_values, subsets)` where:
    /// - `comp_values`: 2D numpy array of shape (n_eval, n_subsets) with one column per component f_S
    /// - `subsets`: List of lists, where each inner list contains the feature indices for a subset
    fn functional_decomp_up_to_order<'a>(
        &mut self,
        py: Python<'a>,
        evaluation_points: PyReadonlyArray2<f64>,
        max_order: usize,
    ) -> PyResult<(Bound<'a, PyArray2<f32>>, Vec<Vec<usize>>)> {
        // Convert f64 array to f32 array
        let eval_f64 = evaluation_points.as_array();
        let eval_f32: ndarray::Array2<f32> = eval_f64.mapv(|x| x as f32);

        let (comp_values, subsets) = self
            .fastpd
            .functional_decomp_up_to_order(&eval_f32.view(), max_order)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FastPD error: {}", e))
            })?;

        // Convert FeatureSubset vector to Vec<Vec<usize>>
        let subsets_py: Vec<Vec<usize>> = subsets
            .into_iter()
            .map(|subset| subset.as_slice())
            .collect();

        Ok((PyArray2::from_owned_array(py, comp_values), subsets_py))
    }
}

#[pymodule]
fn glex_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_trees_from_xgboost_py, m)?)?;
    m.add_class::<XGBoostTreeModel>()?;
    m.add_class::<FastPDPy>()?;
    Ok(())
}
