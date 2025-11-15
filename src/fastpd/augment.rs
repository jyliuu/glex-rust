use std::collections::HashMap;
use std::sync::Arc;

use ndarray::ArrayView2;

use crate::fastpd::error::FastPDError;
use crate::fastpd::tree::TreeModel;
use crate::fastpd::types::{
    FeatureSubset, ObservationSet, PathData, PathFeatures, SharedObservationSet,
};

use super::augmented_tree::AugmentedTree;

/// Augments a tree with background samples according to Algorithm 1 from the FastPD paper.
///
/// This function implements the augmentation phase of FastPD, which precomputes
/// observation sets D_S for each feature subset S and each leaf j.
///
/// # Arguments
/// * `tree` - The tree model to augment
/// * `background_samples` - Background data array (n_samples, n_features)
///
/// # Returns
/// An `AugmentedTree` containing path features T_j and path data P_j for each leaf j.
///
/// # Errors
/// Returns `FastPDError` if:
/// - Background samples are empty
/// - Tree structure is invalid
/// - Feature indices are out of bounds
pub fn augment_tree<T: TreeModel>(
    tree: T,
    background_samples: &ArrayView2<f32>,
) -> Result<AugmentedTree<T>, FastPDError> {
    let n_background = background_samples.nrows();
    if n_background == 0 {
        return Err(FastPDError::EmptyBackground);
    }

    let n_features = background_samples.ncols();
    let mut path_features = HashMap::new();
    let mut path_data = HashMap::new();

    // Initialize: all observations in D_∅
    // Wrap in Arc to allow sharing during recursion
    let all_indices: Vec<usize> = (0..n_background).collect();
    let initial_set: SharedObservationSet = Arc::new(ObservationSet::Indices(all_indices));
    let mut initial_path_data = PathData::new();
    initial_path_data.insert(FeatureSubset::empty(), initial_set);

    // Recursive augmentation starting from root
    // T = ∅ (empty path features), P = {(∅, D_{n_b})}
    augment_recursive(
        &tree,
        tree.root(),
        &mut path_features,
        &mut path_data,
        Vec::new(),        // T (path features so far)
        initial_path_data, // P (path data so far)
        background_samples,
        n_features,
    )?;

    Ok(AugmentedTree::new(
        tree,
        n_background,
        path_features,
        path_data,
    ))
}

/// Recursive helper function for tree augmentation.
///
/// This implements the RECURSE function from Algorithm 1 in the paper.
fn augment_recursive<T: TreeModel>(
    tree: &T,
    node_id: usize,
    path_features: &mut HashMap<usize, PathFeatures>,
    path_data: &mut HashMap<usize, PathData>,
    current_path_features: PathFeatures, // T
    current_path_data: PathData,         // P
    background_samples: &ArrayView2<f32>,
    n_features: usize,
) -> Result<(), FastPDError> {
    if tree.is_leaf(node_id) {
        // Store T_j and P_j for this leaf
        path_features.insert(node_id, current_path_features);
        path_data.insert(node_id, current_path_data);
        return Ok(());
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
    if feature >= n_features {
        return Err(FastPDError::InvalidFeature(feature, n_features));
    }

    // Build T_new: add d_j if it's not already in T
    let mut new_path_features = current_path_features;
    let feature_in_path = new_path_features.contains(&feature);
    if !feature_in_path {
        new_path_features.push(feature);
        new_path_features.sort_unstable();
        new_path_features.dedup();
    }

    // Collect original (S, D_S) pairs before processing (needed for S ∪ {d_j} case)
    let original_path_data: Vec<(FeatureSubset, SharedObservationSet)> =
        current_path_data.into_iter().collect();

    // Build P_yes and P_no for children
    let mut path_data_yes = PathData::new();
    let mut path_data_no = PathData::new();

    // Process each (S, D_S) in P
    for (subset_s, shared_obs_set) in original_path_data.iter() {
        if subset_s.contains(feature) {
            // d_j ∈ S: split D_S to both children
            // Arc::clone is cheap (just increments reference count)
            // No need to clone the underlying Vec<usize>!
            path_data_yes.insert(subset_s.clone(), Arc::clone(shared_obs_set));
            path_data_no.insert(subset_s.clone(), Arc::clone(shared_obs_set));
        } else {
            // d_j ∉ S: split D_S based on threshold in a single pass
            // This creates NEW observation sets (filtered subsets)
            // Use const generic with T::COMPARISON for compile-time monomorphism
            let (filtered_yes, filtered_no) = shared_obs_set.as_ref().split_by_threshold(
                background_samples,
                feature,
                threshold,
                T::COMPARISON,
            );
            path_data_yes.insert(subset_s.clone(), Arc::new(filtered_yes));
            path_data_no.insert(subset_s.clone(), Arc::new(filtered_no));
        }
    }

    // If d_j is a new feature, add (S ∪ {d_j}, D_S) to both children
    // Use the ORIGINAL D_S (before filtering) for the union case
    if !feature_in_path {
        for (subset_s, original_obs_set) in original_path_data {
            // Create S ∪ {d_j}
            let mut union_features = subset_s.as_slice().to_vec();
            union_features.push(feature);
            let union_subset = FeatureSubset::new(union_features);

            // Add (S ∪ {d_j}, D_S) to both children using original D_S
            // Arc::clone is cheap - we're sharing the same observation set
            path_data_yes.insert(union_subset.clone(), Arc::clone(&original_obs_set));
            path_data_no.insert(union_subset, Arc::clone(&original_obs_set));
        }
    }

    // Recurse to children
    augment_recursive(
        tree,
        left_child,
        path_features,
        path_data,
        new_path_features.clone(),
        path_data_yes,
        background_samples,
        n_features,
    )?;
    augment_recursive(
        tree,
        right_child,
        path_features,
        path_data,
        new_path_features,
        path_data_no,
        background_samples,
        n_features,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastpd::tree::{Tree, TreeNode};
    use ndarray::arr2;

    // Helper: Create a simple 2-level tree for testing
    // Root splits on feature 0 at threshold 0.5
    // Left child (leaf 1): value 1.0
    // Right child (leaf 2): value 2.0
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
    fn test_augment_simple_tree() {
        let tree = create_simple_tree();

        // Create background data: 4 samples
        // Sample 0: [0.3] -> goes to left (leaf 1)
        // Sample 1: [0.7] -> goes to right (leaf 2)
        // Sample 2: [0.2] -> goes to left (leaf 1)
        // Sample 3: [0.8] -> goes to right (leaf 2)
        let background = arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        let aug_tree = augment_tree(tree, &background_view).unwrap();

        // Check that we have 2 leaves
        assert_eq!(aug_tree.num_leaves(), 2);

        // Check leaf 1 (left child)
        let path_features_1 = aug_tree.get_path_features(1).unwrap();
        assert_eq!(path_features_1, &vec![0]); // Feature 0 is on the path

        let path_data_1 = aug_tree.get_path_data(1).unwrap();
        // Should have D_∅ (all samples that reach leaf 1 after filtering)
        let empty_subset = FeatureSubset::empty();
        assert!(path_data_1.contains_key(&empty_subset));
        let d_empty_1 = path_data_1[&empty_subset].as_indices();
        // After filtering by feature 0 <= 0.5, we should have samples 0 and 2
        assert_eq!(d_empty_1.len(), 2);
        assert!(d_empty_1.contains(&0));
        assert!(d_empty_1.contains(&2));

        // Should also have D_{0} (when feature 0 is in S)
        let subset_0 = FeatureSubset::new(vec![0]);
        assert!(path_data_1.contains_key(&subset_0));
        let d_0_1 = path_data_1[&subset_0].as_indices();
        // When feature 0 is in S, all samples that reach this node go to both children
        // So D_{0} should contain all 4 samples
        assert_eq!(d_0_1.len(), 4);

        // Check leaf 2 (right child)
        let path_features_2 = aug_tree.get_path_features(2).unwrap();
        assert_eq!(path_features_2, &vec![0]);

        let path_data_2 = aug_tree.get_path_data(2).unwrap();
        assert!(path_data_2.contains_key(&empty_subset));
        let d_empty_2 = path_data_2[&empty_subset].as_indices();
        // After filtering by feature 0 > 0.5, we should have samples 1 and 3
        assert_eq!(d_empty_2.len(), 2);
        assert!(d_empty_2.contains(&1));
        assert!(d_empty_2.contains(&3));

        assert!(path_data_2.contains_key(&subset_0));
        let d_0_2 = path_data_2[&subset_0].as_indices();
        assert_eq!(d_0_2.len(), 4);
    }

    #[test]
    fn test_augment_empty_background() {
        let tree = create_simple_tree();
        let background = arr2(&[[0.0]; 0]); // Empty array
        let background_view = background.view();

        let result = augment_tree(tree, &background_view);
        assert!(result.is_err());
        if let Err(FastPDError::EmptyBackground) = result {
            // Correct error type
        } else {
            panic!("Expected EmptyBackground error");
        }
    }

    #[test]
    fn test_augment_single_node_tree() {
        // Tree with just a root leaf
        let nodes = vec![TreeNode {
            internal_idx: 0,
            feature: None,
            threshold: None,
            left: None,
            right: None,
            missing: None,
            leaf_value: Some(1.0),
        }];
        let tree = Tree::new(nodes, 0);

        let background = arr2(&[[0.5], [0.6], [0.7]]);
        let background_view = background.view();

        let aug_tree = augment_tree(tree, &background_view).unwrap();

        assert_eq!(aug_tree.num_leaves(), 1);
        let path_features = aug_tree.get_path_features(0).unwrap();
        assert!(path_features.is_empty()); // No features on path to root leaf

        let path_data = aug_tree.get_path_data(0).unwrap();
        let empty_subset = FeatureSubset::empty();
        assert!(path_data.contains_key(&empty_subset));
        // All 3 samples should be in D_∅
        assert_eq!(path_data[&empty_subset].len(), 3);
    }
}
