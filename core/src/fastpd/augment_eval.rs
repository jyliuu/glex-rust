use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::fastpd::augment::{augment_tree_rayon, augment_tree_seq};
use crate::fastpd::augmented_tree::AugmentedTree;
use crate::fastpd::cache::PDCache;
use crate::fastpd::error::FastPDError;
use crate::fastpd::evaluate::{
    evaluate_pd_batch_for_subsets_rayon, evaluate_pd_batch_for_subsets_seq,
    evaluate_pd_function_rayon, evaluate_pd_function_seq, functional_component_from_u_matrix,
    pd_from_u_matrix,
};
use crate::fastpd::parallel::ParallelSettings;
use crate::fastpd::tree::TreeModel;
use crate::fastpd::types::FeatureSubset;

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
    /// Parallelization settings used for both augmentation and evaluation
    parallel: ParallelSettings,
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
    /// * `parallel` - Parallelization settings (default: sequential)
    ///
    /// # Parallelization
    /// - If `parallel.n_threads <= 1`: Fully sequential execution, no Rayon overhead
    /// - If `parallel.n_threads > 1`: Parallel execution across trees and within tree recursion
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
        parallel: ParallelSettings,
    ) -> Result<Self, FastPDError> {
        if background_samples.nrows() == 0 {
            return Err(FastPDError::EmptyBackground);
        }

        let n_features = background_samples.ncols();
        let n_background = background_samples.nrows();
        let n_trees = trees.len();

        let augmented_trees = if !parallel.is_parallel() {
            // Sequential path: no Rayon anywhere
            let mut augmented_trees = Vec::with_capacity(n_trees);
            for tree in trees {
                let aug_tree = augment_tree_seq(tree, background_samples)?;
                augmented_trees.push(aug_tree);
            }
            augmented_trees
        } else {
            // Parallel path: create thread pool and parallelize
            let n_threads = parallel.n_threads;
            let pool = ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| FastPDError::ThreadPoolError(e.to_string()))?;

            // All Rayon work happens inside this install scope
            pool.install(|| {
                trees
                    .into_par_iter()
                    .map(|tree| augment_tree_rayon(tree, background_samples))
                    .collect::<Result<Vec<_>, _>>()
            })?
        };

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
            parallel,
        })
    }

    /// Legacy constructor: uses sequential execution by default.
    ///
    /// This method is kept for backward compatibility. New code should use
    /// `new()` with explicit `ParallelSettings`.
    pub fn new_sequential(
        trees: Vec<T>,
        background_samples: &ArrayView2<f32>,
        intercept: f32,
    ) -> Result<Self, FastPDError> {
        Self::new(
            trees,
            background_samples,
            intercept,
            ParallelSettings::sequential(),
        )
    }

    /// Computes the PD function v_S(x_S) for a single feature subset.
    ///
    /// This function evaluates the PD function at multiple evaluation points
    /// for a given feature subset S. The result is the sum of PD contributions
    /// from all trees in the ensemble.
    ///
    /// Uses the parallelization settings stored in the FastPD instance.
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

        // Construct FeatureSubset for S once per batch of evaluation points
        let subset_s = FeatureSubset::from_slice(feature_subset);

        let results = if !self.parallel.is_parallel() {
            // Sequential path: no Rayon anywhere
            let mut results = Vec::with_capacity(n_eval);
            for i in 0..n_eval {
                let point = evaluation_points.row(i);
                let mut total = 0.0;
                for (tree_idx, aug_tree) in self.augmented_trees.iter().enumerate() {
                    let cache = &mut self.caches[tree_idx];
                    let value = evaluate_pd_function_seq(aug_tree, &point, &subset_s, cache)?;
                    total += value;
                }
                total += self.intercept;
                results.push(total);
            }
            results
        } else {
            // Parallel path: create thread pool and parallelize across trees
            let n_threads = self.parallel.n_threads;
            let pool = ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| FastPDError::ThreadPoolError(e.to_string()))?;

            // Parallelize across trees: each thread processes all evaluation points for one tree
            // Use Mutex to protect mutable caches during parallel tree evaluation
            use std::sync::Mutex;
            let caches_mutex: Vec<Mutex<&mut PDCache>> =
                self.caches.iter_mut().map(|c| Mutex::new(c)).collect();

            let tree_results: Result<Vec<Vec<f32>>, FastPDError> = pool.install(|| {
                self.augmented_trees
                    .par_iter()
                    .enumerate()
                    .map(|(tree_idx, aug_tree)| {
                        // For this tree, evaluate all evaluation points
                        let mut tree_values = Vec::with_capacity(n_eval);
                        for i in 0..n_eval {
                            let point = evaluation_points.row(i);
                            let mut cache_guard = caches_mutex[tree_idx].lock().unwrap();
                            let value = evaluate_pd_function_rayon(
                                aug_tree,
                                &point,
                                &subset_s,
                                &mut *cache_guard,
                            )?;
                            tree_values.push(value);
                        }
                        Ok(tree_values)
                    })
                    .collect()
            });

            // Sum across trees for each evaluation point
            let tree_results = tree_results?;
            let mut results = Vec::with_capacity(n_eval);
            for i in 0..n_eval {
                let mut total = self.intercept;
                for tree_values in &tree_results {
                    total += tree_values[i];
                }
                results.push(total);
            }
            results
        };

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
    /// - `pd_values`: Matrix of shape [n_eval, n_subsets] with one column per subset S
    /// - `subsets`: Vector of FeatureSubset corresponding to each column
    ///
    /// # Errors
    /// Returns `FastPDError` if evaluation fails
    pub fn pd_functions_up_to_order(
        &mut self,
        evaluation_points: &ArrayView2<f32>,
        max_order: usize,
    ) -> Result<(Array2<f32>, Vec<FeatureSubset>), FastPDError> {
        let n_eval = evaluation_points.nrows();

        // 1. Union of encountered features across trees
        let mut all_encountered = FeatureSubset::empty();
        for aug_tree in &self.augmented_trees {
            let feats_vec = aug_tree.all_tree_features.as_slice();
            all_encountered = all_encountered.union(&feats_vec);
        }

        // 2. Build U = all subsets of all_encountered with |U| <= max_order
        let subsets_u = all_encountered.subsets_up_to_order(max_order);
        let n_subsets_u = subsets_u.len();

        // 3. Aggregate v_U(x) across trees
        let mat_u = if !self.parallel.is_parallel() {
            // Sequential path: no Rayon anywhere
            let mut mat_u = Array2::<f32>::zeros((n_eval, n_subsets_u));
            for aug_tree in &self.augmented_trees {
                let tree_mat =
                    evaluate_pd_batch_for_subsets_seq(aug_tree, evaluation_points, &subsets_u)?;
                mat_u = &mat_u + &tree_mat;
            }
            mat_u
        } else {
            // Parallel path: create thread pool and parallelize across trees
            let n_threads = self.parallel.n_threads;
            let pool = ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| FastPDError::ThreadPoolError(e.to_string()))?;

            pool.install(|| {
                self.augmented_trees
                    .par_iter()
                    .map(|aug_tree| {
                        evaluate_pd_batch_for_subsets_rayon(aug_tree, evaluation_points, &subsets_u)
                    })
                    .try_reduce(
                        || Array2::<f32>::zeros((n_eval, n_subsets_u)),
                        |mut a, b| {
                            a = &a + &b;
                            Ok(a)
                        },
                    )
            })?
        };

        // 4. Extract v_S(x_S) for all S with 1 <= |S| <= max_order
        // Precompute which subsets we need to extract
        let target_subsets: Vec<FeatureSubset> = all_encountered
            .subsets_up_to_order(max_order)
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();

        // Preallocate result matrix
        let n_target_subsets = target_subsets.len();
        let mut pd_mat = Array2::<f32>::zeros((n_eval, n_target_subsets));
        let mut pd_subsets = Vec::with_capacity(n_target_subsets);

        for (col_idx, s) in target_subsets.iter().enumerate() {
            if let Some(col) = pd_from_u_matrix(&mat_u, &subsets_u, s) {
                pd_mat.column_mut(col_idx).assign(&col);
                pd_subsets.push(s.clone());
            }
        }

        // Add intercept to all columns
        for mut col in pd_mat.columns_mut() {
            for val in col.iter_mut() {
                *val += self.intercept;
            }
        }

        Ok((pd_mat, pd_subsets))
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
    /// - `comp_values`: Matrix of shape [n_eval, n_subsets] with one column per component f_S
    /// - `subsets`: Vector of FeatureSubset corresponding to each column
    ///
    /// # Errors
    /// Returns `FastPDError` if evaluation fails
    pub fn functional_decomp_up_to_order(
        &mut self,
        evaluation_points: &ArrayView2<f32>,
        max_order: usize,
    ) -> Result<(Array2<f32>, Vec<FeatureSubset>), FastPDError> {
        let n_eval = evaluation_points.nrows();

        // 1. Union of encountered features across trees
        let mut all_encountered = FeatureSubset::empty();
        for aug_tree in &self.augmented_trees {
            let feats_vec = aug_tree.all_tree_features.as_slice();
            all_encountered = all_encountered.union(&feats_vec);
        }

        // 2. Build U = all subsets of all_encountered with |U| <= max_order
        let subsets_u = all_encountered.subsets_up_to_order(max_order);
        let n_subsets_u = subsets_u.len();

        // 3. Aggregate v_U(x) across trees
        // evaluate_pd_batch_for_subsets populates v_∅ with each tree's expected value
        // NOTE: Do NOT add intercept to v_∅ here - it will be added to f_∅ after inclusion-exclusion
        let mat_u = if !self.parallel.is_parallel() {
            // Sequential path: no Rayon anywhere
            let mut mat_u = Array2::<f32>::zeros((n_eval, n_subsets_u));
            for aug_tree in &self.augmented_trees {
                let tree_mat =
                    evaluate_pd_batch_for_subsets_seq(aug_tree, evaluation_points, &subsets_u)?;
                mat_u = &mat_u + &tree_mat;
            }
            mat_u
        } else {
            // Parallel path: create thread pool and parallelize across trees
            let n_threads = self.parallel.n_threads;
            let pool = ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| FastPDError::ThreadPoolError(e.to_string()))?;

            pool.install(|| {
                self.augmented_trees
                    .par_iter()
                    .map(|aug_tree| {
                        evaluate_pd_batch_for_subsets_rayon(aug_tree, evaluation_points, &subsets_u)
                    })
                    .try_reduce(
                        || Array2::<f32>::zeros((n_eval, n_subsets_u)),
                        |mut a, b| {
                            a = &a + &b;
                            Ok(a)
                        },
                    )
            })?
        };

        // 4. Derive f_S(x_S) for all S with 0 <= |S| <= max_order via inclusion–exclusion
        let target_subsets: Vec<FeatureSubset> = all_encountered
            .subsets_up_to_order(max_order)
            .into_iter()
            .collect();

        let n_target_subsets = target_subsets.len();
        let mut comp_mat = Array2::<f32>::zeros((n_eval, n_target_subsets));
        let mut comp_subsets = Vec::with_capacity(n_target_subsets);

        let empty_subset = FeatureSubset::empty();
        for (col_idx, s) in target_subsets.iter().enumerate() {
            let mut col = functional_component_from_u_matrix(&mat_u, &subsets_u, s);

            // For empty subset, add intercept to f_∅ after inclusion-exclusion
            // This ensures intercept is only in f_∅ and doesn't get cancelled in other components
            if s == &empty_subset {
                col += self.intercept;
            }

            comp_mat.column_mut(col_idx).assign(&col);
            comp_subsets.push(s.clone());
        }

        Ok((comp_mat, comp_subsets))
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

        let fastpd = FastPD::new_sequential(vec![tree], &background_view, 0.0).unwrap();
        assert_eq!(fastpd.num_trees(), 1);
        assert_eq!(fastpd.n_background(), 4);
        assert_eq!(fastpd.n_features(), 1);
    }

    #[test]
    fn test_fastpd_empty_background() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.0; 0]; 0]); // Empty array
        let background_view = background.view();

        let result = FastPD::new_sequential(vec![tree], &background_view, 0.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FastPDError::EmptyBackground));
    }

    #[test]
    fn test_fastpd_pd_function() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let mut fastpd = FastPD::new_sequential(vec![tree], &background_view, 0.0).unwrap();

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

        let mut fastpd = FastPD::new_sequential(vec![tree], &background_view, 0.0).unwrap();

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

        let mut fastpd = FastPD::new_sequential(vec![tree], &background_view, 0.0).unwrap();

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

    /// Create a tree with 2 features for testing multiple subsets
    fn create_two_feature_tree() -> Tree {
        // Tree structure:
        // Root splits on feature 0 at 0.5
        //   Left child splits on feature 1 at 0.5 -> leaf value 1.0
        //   Right child splits on feature 1 at 0.5 -> leaf value 2.0
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
                feature: Some(1),
                threshold: Some(0.5),
                left: Some(3),
                right: Some(4),
                missing: Some(3),
                leaf_value: None,
            },
            TreeNode {
                internal_idx: 2,
                feature: Some(1),
                threshold: Some(0.5),
                left: Some(5),
                right: Some(6),
                missing: Some(5),
                leaf_value: None,
            },
            TreeNode {
                internal_idx: 3,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(1.0),
            },
            TreeNode {
                internal_idx: 4,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(2.0),
            },
            TreeNode {
                internal_idx: 5,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(3.0),
            },
            TreeNode {
                internal_idx: 6,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(4.0),
            },
        ];
        Tree::new(nodes, 0)
    }

    #[test]
    fn test_pd_functions_up_to_order_matches_individual_calls() {
        let tree = create_two_feature_tree();
        // Create background data with 2 features
        let background = arr2(&[
            [0.3, 0.3],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.7, 0.7],
            [0.2, 0.2],
            [0.8, 0.8],
        ]);
        let background_view = background.view();

        let mut fastpd = FastPD::new_sequential(vec![tree], &background_view, 1.0).unwrap();

        // Create multiple evaluation points
        let eval_points = arr2(&[[0.3, 0.3], [0.7, 0.7], [0.4, 0.6]]);
        let eval_view = eval_points.view();

        // Test with max_order = 2 (should include {0}, {1}, {0,1})
        let max_order = 2;
        let (batch_pd_values, batch_subsets) = fastpd
            .pd_functions_up_to_order(&eval_view, max_order)
            .unwrap();

        // Verify we got the expected number of subsets
        // For 2 features with max_order=2, we should get:
        // - {0} (order 1)
        // - {1} (order 1)
        // - {0,1} (order 2)
        assert_eq!(batch_subsets.len(), 3);
        assert_eq!(batch_pd_values.nrows(), 3); // 3 evaluation points
        assert_eq!(batch_pd_values.ncols(), 3); // 3 subsets

        // Now compare with individual pd_function calls
        for (subset_idx, subset) in batch_subsets.iter().enumerate() {
            let subset_features = subset.as_slice();

            for (point_idx, point_row) in eval_points.rows().into_iter().enumerate() {
                // Call pd_function for this single point and subset
                let single_point = point_row.to_owned().insert_axis(ndarray::Axis(0));
                let single_point_view = single_point.view();
                let individual_result = fastpd
                    .pd_function(&single_point_view, &subset_features)
                    .unwrap();

                // Compare with batch result
                let batch_value = batch_pd_values[[point_idx, subset_idx]];
                let individual_value = individual_result[0];

                assert!(
                    (batch_value - individual_value).abs() < 1e-6,
                    "Mismatch for subset {:?} at point {}: batch={}, individual={}",
                    subset_features,
                    point_idx,
                    batch_value,
                    individual_value
                );
            }
        }
    }

    #[test]
    fn test_functional_decomp_sum_equals_predictions() {
        let tree = create_two_feature_tree();
        // Create background data with 2 features
        let background = arr2(&[
            [0.3, 0.3],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.7, 0.7],
            [0.2, 0.2],
            [0.8, 0.8],
        ]);
        let background_view = background.view();

        let mut fastpd = FastPD::new_sequential(vec![tree], &background_view, 0.0).unwrap();

        // Create multiple evaluation points
        let eval_points = arr2(&[[0.3, 0.3], [0.7, 0.7], [0.4, 0.6], [0.1, 0.9]]);
        let eval_view = eval_points.view();

        // Compute functional decomposition up to order 2
        let max_order = 2;
        let (comp_mat, comp_subsets) = fastpd
            .functional_decomp_up_to_order(&eval_view, max_order)
            .unwrap();

        // Compute predictions for the same evaluation points
        let predictions = fastpd.predict(&eval_view).unwrap();

        // Verify that sum of all functional components equals predictions
        let n_eval = eval_points.nrows();
        assert_eq!(comp_mat.nrows(), n_eval);
        assert_eq!(predictions.len(), n_eval);

        for point_idx in 0..n_eval {
            // Sum all functional components for this point
            let component_sum: f32 = comp_mat.row(point_idx).sum();

            // Get the prediction for this point
            let prediction = predictions[point_idx];

            assert!(
                (component_sum - prediction).abs() < 1e-6,
                "Mismatch at point {}: sum of components={}, prediction={}",
                point_idx,
                component_sum,
                prediction
            );
        }

        // Also verify we have the expected components
        // Should have: {0}, {1}, {0,1}, and ∅ (empty subset)
        assert_eq!(comp_subsets.len(), 4);
        assert_eq!(comp_mat.ncols(), 4);
    }
}
