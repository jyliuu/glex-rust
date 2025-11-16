use ndarray::{Array1, ArrayView2};

use crate::fastpd::augment::augment_tree;
use crate::fastpd::augmented_tree::AugmentedTree;
use crate::fastpd::cache::PDCache;
use crate::fastpd::error::FastPDError;
use crate::fastpd::evaluate::evaluate_pd_function;
use crate::fastpd::tree::TreeModel;

/// High-level FastPD estimator for computing partial dependence functions.
///
/// This struct manages an ensemble of augmented trees and provides efficient
/// methods for computing PD functions, SHAP values, and functional decomposition.
#[derive(Debug)]
pub struct FastPD<T: TreeModel> {
    /// Augmented trees (one per tree in the ensemble)
    augmented_trees: Vec<AugmentedTree<T>>,
    /// Number of background samples used for augmentation
    n_background: usize,
    /// Number of features
    n_features: usize,
    /// PD caches (one per tree)
    /// Each cache stores (point_hash, U) -> v_U mappings
    /// Multiple S that map to same U can reuse the cached value
    caches: Vec<PDCache>,
    /// Intercept/base_score term (e.g., from XGBoost)
    /// This is added to predictions but not to PD functions
    intercept: f32,
}

impl<T: TreeModel> FastPD<T> {
    /// Creates a new FastPD instance from a collection of trees.
    ///
    /// This function augments all trees with the background samples, which is
    /// the expensive preprocessing step. Once created, the FastPD instance can
    /// efficiently evaluate PD functions for any feature subset.
    ///
    /// # Arguments
    /// * `trees` - Vector of trees to augment
    /// * `background_samples` - Background samples for PD estimation
    ///     shape: (n_background, n_features)
    /// * `intercept` - Intercept/base_score term to add to predictions (default: 0.0)
    ///
    /// # Returns
    /// `FastPD` instance with all trees augmented
    ///
    /// # Errors
    /// Returns `FastPDError` if augmentation fails for any tree
    pub fn new(
        trees: Vec<T>,
        background_samples: &ArrayView2<f32>,
        intercept: f32,
    ) -> Result<Self, FastPDError> {
        if background_samples.nrows() == 0 {
            return Err(FastPDError::EmptyBackground);
        }

        let n_features = background_samples.ncols();
        let n_background = background_samples.nrows();
        let n_trees = trees.len();

        let mut augmented_trees = Vec::with_capacity(n_trees);
        for tree in trees {
            let aug_tree = augment_tree(tree, background_samples)?;
            augmented_trees.push(aug_tree);
        }

        let mut caches = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            caches.push(PDCache::new());
        }

        Ok(Self {
            augmented_trees,
            n_background,
            n_features,
            caches,
            intercept,
        })
    }

    /// Computes the PD function v_S(x_S) for a single feature subset.
    ///
    /// This function evaluates the PD function at multiple evaluation points
    /// for a given feature subset S. The result is the sum of PD contributions
    /// from all trees in the ensemble.
    ///
    /// # Arguments
    /// * `evaluation_points` - Points at which to evaluate PD
    ///     shape: (n_evaluation_points, n_features)
    /// * `feature_subset` - Indices of features in subset S
    ///
    /// # Returns
    /// PD values at each evaluation point
    ///     shape: (n_evaluation_points,)
    ///
    /// # Errors
    /// Returns `FastPDError` if evaluation fails
    pub fn pd_function(
        &mut self,
        evaluation_points: &ArrayView2<f32>,
        feature_subset: &[usize],
    ) -> Result<Array1<f32>, FastPDError> {
        let n_eval = evaluation_points.nrows();
        let mut results = Vec::with_capacity(n_eval);

        for i in 0..n_eval {
            let point = evaluation_points.row(i);

            // Validate point dimension
            if point.len() != self.n_features {
                return Err(FastPDError::DimensionMismatch(point.len(), self.n_features));
            }

            // Sum contributions from all trees
            let mut total = 0.0;
            for (tree_idx, aug_tree) in self.augmented_trees.iter().enumerate() {
                // Use cache for this tree
                let cache = &mut self.caches[tree_idx];

                let value = evaluate_pd_function(aug_tree, &point, feature_subset, cache)?;
                total += value;
            }
            total += self.intercept;
            results.push(total);
        }

        Ok(Array1::from_vec(results))
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
    ///
    /// # Errors
    /// Returns `FastPDError` if evaluation fails (e.g., dimension mismatch)
    pub fn predict(&self, evaluation_points: &ArrayView2<f32>) -> Result<Array1<f32>, FastPDError> {
        let n_eval = evaluation_points.nrows();
        let mut results = Vec::with_capacity(n_eval);

        for i in 0..n_eval {
            let point = evaluation_points.row(i);

            // Validate point dimension
            if point.len() != self.n_features {
                return Err(FastPDError::DimensionMismatch(point.len(), self.n_features));
            }

            // Sum predictions from all trees
            let mut total = 0.0;
            for aug_tree in &self.augmented_trees {
                // Convert ArrayView1 to slice for prediction without allocating if possible.
                // Most ndarray arrays backing model inputs are contiguous, so `as_slice()`
                // should usually succeed and avoid per-evaluation allocations.
                let prediction = if let Some(slice) = point.as_slice() {
                    aug_tree.tree.predict(slice)
                } else {
                    let owned: Vec<f32> = point.iter().copied().collect();
                    aug_tree.tree.predict(&owned)
                };
                total += prediction;
            }
            // Add intercept/base_score
            total += self.intercept;
            results.push(total);
        }

        Ok(Array1::from_vec(results))
    }

    /// Clears all PD caches.
    ///
    /// This is useful when memory is a concern or when you want to ensure
    /// fresh computations for a new batch of evaluations.
    pub fn clear_caches(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }

    /// Returns the number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.augmented_trees.len()
    }

    /// Returns the number of background samples.
    pub fn n_background(&self) -> usize {
        self.n_background
    }

    /// Returns the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastpd::tree::{Tree, TreeNode};
    use ndarray::arr2;

    fn create_simple_tree() -> Tree {
        let nodes = vec![
            TreeNode {
                internal_idx: 0,
                feature: Some(0),
                threshold: Some(0.5),
                left: Some(1),
                right: Some(2),
                missing: Some(1),
                leaf_value: None,
            },
            TreeNode {
                internal_idx: 1,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(1.0),
            },
            TreeNode {
                internal_idx: 2,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(2.0),
            },
        ];
        Tree::new(nodes, 0)
    }

    #[test]
    fn test_fastpd_new() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let fastpd = FastPD::new(vec![tree], &background_view, 0.0).unwrap();
        assert_eq!(fastpd.num_trees(), 1);
        assert_eq!(fastpd.n_background(), 4);
        assert_eq!(fastpd.n_features(), 1);
    }

    #[test]
    fn test_fastpd_empty_background() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.0; 0]; 0]); // Empty array
        let background_view = background.view();

        let result = FastPD::new(vec![tree], &background_view, 0.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FastPDError::EmptyBackground));
    }

    #[test]
    fn test_fastpd_pd_function() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let mut fastpd = FastPD::new(vec![tree], &background_view, 0.0).unwrap();

        // Evaluate PD for S = {0} at point [0.3]
        let eval_points = arr2(&[[0.3]]);
        let eval_view = eval_points.view();
        let result = fastpd.pd_function(&eval_view, &[0]);

        assert!(result.is_ok());
        let values = result.unwrap();
        assert_eq!(values.len(), 1);
        // Value should be 1.0 (goes to left leaf)
        assert!((values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fastpd_pd_function_batch() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let mut fastpd = FastPD::new(vec![tree], &background_view, 0.0).unwrap();

        // Evaluate PD for S = {0} at multiple points
        let eval_points = arr2(&[[0.3], [0.7]]);
        let eval_view = eval_points.view();
        let result = fastpd.pd_function(&eval_view, &[0]);

        assert!(result.is_ok());
        let values = result.unwrap();
        assert_eq!(values.len(), 2);
        // Point [0.3] -> left leaf (1.0)
        assert!((values[0] - 1.0).abs() < 1e-10);
        // Point [0.7] -> right leaf (2.0)
        assert!((values[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fastpd_clear_caches() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let mut fastpd = FastPD::new(vec![tree], &background_view, 0.0).unwrap();

        // Evaluate to populate cache
        let eval_points = arr2(&[[0.3]]);
        let eval_view = eval_points.view();
        let _ = fastpd.pd_function(&eval_view, &[0]).unwrap();

        // Caches should not be empty
        assert!(!fastpd.caches[0].is_empty());

        // Clear caches
        fastpd.clear_caches();

        // Caches should be empty
        assert!(fastpd.caches[0].is_empty());
    }
}
