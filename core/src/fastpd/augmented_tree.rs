use std::collections::HashMap;

use crate::fastpd::tree::TreeModel;
use crate::fastpd::types::{FeatureSubset, PathData};

/// Augmented tree storing path features and path data for each leaf.
///
/// This structure stores the results of the FastPD augmentation algorithm (Algorithm 1).
/// For each leaf j in the tree, it stores:
/// - `T_j`: The path features (features encountered on the path from root to leaf j)
/// - `P_j`: The path data (maps feature subset S -> observation set D_S)
///
/// # Type Parameters
/// * `T` - The tree model type implementing `TreeModel`
///
/// # Memory Optimization
/// The tree itself can be owned or borrowed - `Arc` is not necessary for the tree structure.
/// `Arc` is only used for `ObservationSet` sharing within `PathData` to avoid cloning
/// large vectors during augmentation.
#[derive(Debug)]
pub struct AugmentedTree<T: TreeModel> {
    /// The original tree model.
    pub tree: T,
    /// Maps leaf node ID -> path features T_j (sorted, deduplicated feature indices).
    pub path_features: HashMap<usize, FeatureSubset>,
    /// Maps leaf node ID -> path data P_j.
    pub path_data: HashMap<usize, PathData>,
    /// Number of background samples used for augmentation.
    pub n_background: usize,
    /// Union of all path features across leaves: U_T = ⋃_j T_j.
    ///
    /// This is precomputed once during augmentation so that evaluation of
    /// v_S(x_S) does not need to recompute the union of all path features
    /// for every query.
    pub all_tree_features: FeatureSubset,
}

impl<T: TreeModel> AugmentedTree<T> {
    /// Creates a new augmented tree with the given path features and path data.
    ///
    /// # Arguments
    /// * `tree` - The tree model
    /// * `n_background` - Number of background samples
    /// * `path_features` - Path features T_j for each leaf j
    /// * `path_data` - Path data P_j for each leaf j
    /// * `all_tree_features` - Union of all path features across leaves
    pub fn new(
        tree: T,
        n_background: usize,
        path_features: HashMap<usize, FeatureSubset>,
        path_data: HashMap<usize, PathData>,
        all_tree_features: FeatureSubset,
    ) -> Self {
        Self {
            tree,
            path_features,
            path_data,
            n_background,
            all_tree_features,
        }
    }

    /// Returns the number of leaves in the augmented tree.
    pub fn num_leaves(&self) -> usize {
        self.path_features.len()
    }

    /// Gets the path features for a given leaf node.
    ///
    /// # Arguments
    /// * `leaf_id` - The leaf node ID
    ///
    /// # Returns
    /// A reference to the path features, or `None` if the leaf doesn't exist.
    pub fn get_path_features(&self, leaf_id: usize) -> Option<&FeatureSubset> {
        self.path_features.get(&leaf_id)
    }

    /// Gets the path data for a given leaf node.
    ///
    /// # Arguments
    /// * `leaf_id` - The leaf node ID
    ///
    /// # Returns
    /// A reference to the path data, or `None` if the leaf doesn't exist.
    pub fn get_path_data(&self, leaf_id: usize) -> Option<&PathData> {
        self.path_data.get(&leaf_id)
    }

    /// Sets the path features for a given leaf node.
    pub fn set_path_features(&mut self, leaf_id: usize, features: FeatureSubset) {
        self.path_features.insert(leaf_id, features);
    }

    /// Sets the path data for a given leaf node.
    pub fn set_path_data(&mut self, leaf_id: usize, data: PathData) {
        self.path_data.insert(leaf_id, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastpd::tree::{Tree, TreeNode};
    use crate::fastpd::types::{FeatureSubset, ObservationSet, SharedObservationSet};
    use std::sync::Arc;

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
    fn test_augmented_tree_new() {
        let tree = create_simple_tree();
        let aug_tree = AugmentedTree::new(
            tree,
            100,
            HashMap::new(),
            HashMap::new(),
            FeatureSubset::empty(),
        );

        assert_eq!(aug_tree.n_background, 100);
        assert_eq!(aug_tree.num_leaves(), 0);
    }

    #[test]
    fn test_augmented_tree_path_features() {
        let tree = create_simple_tree();
        let mut path_features = HashMap::new();
        let features = FeatureSubset::new(vec![0]);
        path_features.insert(1, features.clone());
        let aug_tree =
            AugmentedTree::new(tree, 100, path_features, HashMap::new(), features.clone());

        assert_eq!(aug_tree.get_path_features(1), Some(&features));
        assert_eq!(aug_tree.num_leaves(), 1);
    }

    #[test]
    fn test_augmented_tree_path_data() {
        let tree = create_simple_tree();
        let mut path_data_map = HashMap::new();
        let mut path_data = HashMap::new();
        let empty_subset = FeatureSubset::empty();
        let obs_set: SharedObservationSet = Arc::new(ObservationSet::all(100));
        path_data.insert(empty_subset.clone(), obs_set.clone());
        path_data_map.insert(1, path_data.clone());
        let aug_tree = AugmentedTree::new(
            tree,
            100,
            HashMap::new(),
            path_data_map,
            FeatureSubset::empty(),
        );

        let retrieved = aug_tree.get_path_data(1);
        assert!(retrieved.is_some());
        let retrieved_data = retrieved.unwrap();
        assert_eq!(retrieved_data.len(), 1);
        assert!(retrieved_data.contains_key(&empty_subset));
        assert_eq!(retrieved_data[&empty_subset].len(), obs_set.len());
    }
}
