use std::collections::HashSet;

use ndarray::ArrayView1;

use crate::fastpd::error::FastPDError;
use crate::fastpd::tree::{FeatureIndex, TreeModel};
use crate::fastpd::types::FeatureSubset;

use super::augmented_tree::AugmentedTree;
use super::cache::PDCache;

/// Evaluates the partial dependence function v_S(x_S) using Algorithm 2 from the FastPD paper.
///
/// This function computes the partial dependence value for a given evaluation point
/// and feature subset using the pre-computed augmented tree data.
///
/// # Arguments
/// * `augmented_tree` - The augmented tree containing path features and path data
/// * `evaluation_point` - The evaluation point x (1D array)
/// * `feature_subset` - The feature subset S (indices of features to compute PD for)
/// * `cache` - Optional cache for storing computed values (can reuse across different S)
///
/// # Returns
/// The partial dependence value v_S(x_S)
///
/// # Errors
/// Returns `FastPDError` if:
/// - Evaluation point dimension doesn't match tree features
/// - Missing path features or path data for a leaf
/// - Missing observation set for a feature subset
pub fn evaluate_pd_function<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_point: &ArrayView1<f64>,
    feature_subset: &[FeatureIndex],
    cache: &mut PDCache,
) -> Result<f64, FastPDError> {
    // Note: We validate the evaluation point dimension during tree traversal
    // when we check if feature indices are within bounds

    // Create FeatureSubset from input
    let subset_s = FeatureSubset::new(feature_subset.to_vec());

    // Check cache first for S
    if let Some(cached) = cache.get(evaluation_point, &subset_s) {
        return Ok(cached);
    }

    // Compute U = S ∩ (union of all T_j)
    // Collect all features that appear in any path
    let mut all_tree_features = HashSet::new();
    for path_features in augmented_tree.path_features.values() {
        for &feature in path_features {
            all_tree_features.insert(feature);
        }
    }
    let mut all_tree_features_vec: Vec<FeatureIndex> = all_tree_features.into_iter().collect();
    all_tree_features_vec.sort_unstable();

    let u = subset_s.intersect(&all_tree_features_vec);

    // Check cache for U (different S may map to same U)
    if let Some(cached) = cache.get(evaluation_point, &u) {
        // Cache the result for S as well
        cache.insert(evaluation_point, subset_s, cached);
        return Ok(cached);
    }

    // Traverse tree to compute v_U(x_U)
    let tree = &augmented_tree.tree;
    let value = evaluate_recursive(tree, tree.root(), evaluation_point, &u, augmented_tree)?;

    // Cache result for both U and S
    cache.insert(evaluation_point, u.clone(), value);
    cache.insert(evaluation_point, subset_s, value);

    Ok(value)
}

/// Recursive helper function for tree evaluation (implements function G from Algorithm 2).
///
/// This function traverses the tree and computes the partial dependence contribution
/// at each node, summing contributions from leaves where x_U would land.
fn evaluate_recursive<T: TreeModel>(
    tree: &T,
    node_id: usize,
    point: &ArrayView1<f64>,
    feature_subset: &FeatureSubset,
    augmented_tree: &AugmentedTree<T>,
) -> Result<f64, FastPDError> {
    if tree.is_leaf(node_id) {
        // Get U_j = U ∩ T_j
        let path_features = augmented_tree
            .path_features
            .get(&node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing path features".into()))?;
        let u_j = feature_subset.intersect(path_features);

        // Get D_{U_j} from P_j
        let path_data = augmented_tree
            .path_data
            .get(&node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing path data".into()))?;
        let shared_obs_set = path_data
            .get(&u_j)
            .ok_or(FastPDError::MissingObservationSet)?;

        // Compute empirical probability: |D_{U_j}| / n_b
        let prob = shared_obs_set.len() as f64 / augmented_tree.n_background as f64;

        // Get leaf value
        let leaf_value = tree
            .leaf_value(node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing value".into()))?;

        return Ok(leaf_value * prob);
    }

    // Get node information
    let feature = tree
        .node_feature(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Internal node missing feature".into()))?;
    let threshold = tree
        .node_threshold(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Internal node missing threshold".into()))?;
    let left_child = tree
        .left_child(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Missing left child".into()))?;
    let right_child = tree
        .right_child(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Missing right child".into()))?;

    // Validate feature index
    if feature >= point.len() {
        return Err(FastPDError::InvalidFeature(feature, point.len()));
    }

    if feature_subset.contains(feature) {
        // d_j ∈ U: follow path based on x[d_j]
        let feature_value = point[feature];
        if feature_value <= threshold {
            evaluate_recursive(tree, left_child, point, feature_subset, augmented_tree)
        } else {
            evaluate_recursive(tree, right_child, point, feature_subset, augmented_tree)
        }
    } else {
        // d_j ∉ U: sum contributions from both children
        let left_val = evaluate_recursive(tree, left_child, point, feature_subset, augmented_tree)?;
        let right_val =
            evaluate_recursive(tree, right_child, point, feature_subset, augmented_tree)?;
        Ok(left_val + right_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastpd::augment::augment_tree;
    use crate::fastpd::tree::{Tree, TreeNode};
    use ndarray::arr1;

    // Helper: Create a simple 2-level tree for testing
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
    fn test_evaluate_pd_simple_tree() {
        let tree = create_simple_tree();

        // Create background data: 4 samples
        // Sample 0: [0.3] -> goes to left (leaf 1, value 1.0)
        // Sample 1: [0.7] -> goes to right (leaf 2, value 2.0)
        // Sample 2: [0.2] -> goes to left (leaf 1, value 1.0)
        // Sample 3: [0.8] -> goes to right (leaf 2, value 2.0)
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        // Augment the tree
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        // Evaluate PD for S = {0} at point x = [0.3]
        // Since 0.3 <= 0.5, we go to left child (leaf 1)
        // U = {0} ∩ {0} = {0}
        // At leaf 1, U_j = {0} ∩ {0} = {0}
        // We use D_{{0}} from P_1, which contains all 4 samples (when feature 0 is in S, all samples go to both children)
        // So prob = 4/4 = 1.0
        // Value = 1.0 * 1.0 = 1.0
        let mut cache = PDCache::new();
        let point = arr1(&[0.3]);
        let subset = vec![0];
        let value = evaluate_pd_function(&aug_tree, &point.view(), &subset, &mut cache).unwrap();
        assert!((value - 1.0).abs() < 1e-10);

        // Evaluate PD for S = {0} at point x = [0.7]
        // Since 0.7 > 0.5, we go to right child (leaf 2)
        // U = {0} ∩ {0} = {0}
        // At leaf 2, U_j = {0} ∩ {0} = {0}
        // We use D_{{0}} from P_2, which contains all 4 samples
        // So prob = 4/4 = 1.0
        // Value = 2.0 * 1.0 = 2.0
        let point2 = arr1(&[0.7]);
        let value2 = evaluate_pd_function(&aug_tree, &point2.view(), &subset, &mut cache).unwrap();
        assert!((value2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_pd_empty_subset() {
        let tree = create_simple_tree();
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        // Evaluate PD for S = {} (empty subset)
        // This means we sum contributions from all leaves
        // Leaf 1: 1.0 * (2/4) = 0.5
        // Leaf 2: 2.0 * (2/4) = 1.0
        // Total: 1.5
        let mut cache = PDCache::new();
        let point = arr1(&[0.5]);
        let subset = vec![];
        let value = evaluate_pd_function(&aug_tree, &point.view(), &subset, &mut cache).unwrap();
        assert!((value - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_pd_caching() {
        let tree = create_simple_tree();
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        let mut cache = PDCache::new();
        let point = arr1(&[0.3]);

        // First evaluation
        let subset1 = vec![0];
        let value1 = evaluate_pd_function(&aug_tree, &point.view(), &subset1, &mut cache).unwrap();
        // When S = {0}, U = {0} ∩ {0} = {0}, so U = S
        // We cache for both U and S, but they're the same key, so cache has 1 entry
        assert!(!cache.is_empty());

        // Second evaluation with same S should use cache
        let value2 = evaluate_pd_function(&aug_tree, &point.view(), &subset1, &mut cache).unwrap();
        assert_eq!(value1, value2);

        // Test with different S that maps to same U
        // S = {0, 99} where 99 is not in tree, so U = {0} ∩ {0} = {0}
        let subset2 = vec![0, 99];
        let value3 = evaluate_pd_function(&aug_tree, &point.view(), &subset2, &mut cache).unwrap();
        // Should use cached value for U = {0}
        assert_eq!(value1, value3);
        // Cache should now have 2 entries: one for {0} and one for {0, 99}
        assert!(cache.len() >= 2);
    }

    #[test]
    fn test_evaluate_pd_matches_empirical() {
        // Test that FastPD evaluation matches the empirical PD estimator:
        // v_S(x_S) = (1/n_b) * sum_{i=1}^{n_b} m(x_S, X^{(i)}_{\overline S})

        let tree = create_simple_tree();

        // Hardcoded background samples (generated with np.random.seed(42))
        let background = ndarray::arr2(&[
            [3.745401, 9.507143],
            [7.319939, 5.986585],
            [1.560186, 1.559945],
            [0.580836, 8.661761],
            [6.011150, 7.080726],
            [0.205845, 9.699099],
            [8.324426, 2.123391],
            [1.818250, 1.834045],
            [3.042422, 5.247564],
            [4.319450, 2.912291],
        ]);
        let background_view = background.view();
        let n_b = background.nrows();

        // Augment the tree
        let aug_tree = augment_tree(tree.clone(), &background_view).unwrap();

        // Test case 1: S = {0}, x = [0.3, 0.0] (x_1 doesn't matter for S={0})
        // Empirical PD: (1/10) * sum_{i=1}^{10} m([0.3, X^{(i)}_1])
        let evaluation_point = arr1(&[0.3, 0.0]);
        let subset = vec![0];

        // Compute FastPD estimate
        let mut cache = PDCache::new();
        let fastpd_value =
            evaluate_pd_function(&aug_tree, &evaluation_point.view(), &subset, &mut cache).unwrap();

        // Compute empirical PD manually
        let mut empirical_sum = 0.0;
        for i in 0..n_b {
            // Create synthetic sample: [x_0, X^{(i)}_1]
            let synthetic = [evaluation_point[0], background[[i, 1]]];
            // Traverse tree to get prediction
            let prediction = predict_tree(&tree, &synthetic);
            empirical_sum += prediction;
        }
        let empirical_value = empirical_sum / n_b as f64;

        // They should match (within floating point tolerance)
        assert!(
            (fastpd_value - empirical_value).abs() < 1e-10,
            "FastPD value {} does not match empirical PD {}",
            fastpd_value,
            empirical_value
        );

        // Test case 2: S = {} (empty subset), x = [0.5, 0.0]
        // Empirical PD: (1/10) * sum_{i=1}^{10} m([X^{(i)}_0, X^{(i)}_1]) = average of all predictions
        let evaluation_point2 = arr1(&[0.5, 0.0]);
        let subset2 = vec![];

        let fastpd_value2 =
            evaluate_pd_function(&aug_tree, &evaluation_point2.view(), &subset2, &mut cache)
                .unwrap();

        let mut empirical_sum2 = 0.0;
        for i in 0..n_b {
            // For S = {}, we use the full background sample
            let synthetic = [background[[i, 0]], background[[i, 1]]];
            let prediction = predict_tree(&tree, &synthetic);
            empirical_sum2 += prediction;
        }
        let empirical_value2 = empirical_sum2 / n_b as f64;

        assert!(
            (fastpd_value2 - empirical_value2).abs() < 1e-10,
            "FastPD value {} does not match empirical PD {} for empty subset",
            fastpd_value2,
            empirical_value2
        );
    }

    // Helper function to predict using a tree (for computing empirical PD)
    fn predict_tree(tree: &Tree, x: &[f64]) -> f64 {
        let mut node_id = tree.root();
        loop {
            if tree.is_leaf(node_id) {
                return tree.leaf_value(node_id).unwrap();
            }
            let feature = tree.node_feature(node_id).unwrap();
            let threshold = tree.node_threshold(node_id).unwrap();
            if x[feature] <= threshold {
                node_id = tree.left_child(node_id).unwrap();
            } else {
                node_id = tree.right_child(node_id).unwrap();
            }
        }
    }
}
