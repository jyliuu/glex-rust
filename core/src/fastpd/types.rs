use std::collections::HashMap;
use std::sync::Arc;

use ndarray::ArrayView2;

use crate::fastpd::tree::{FeatureIndex, Threshold};

/// Index of a sample in the background dataset.
pub type SampleIndex = usize;

/// Sorted feature subset for use as HashMap key.
///
/// Feature subsets are stored as sorted, deduplicated vectors of feature indices.
/// This allows efficient set operations (intersection, membership) and use as HashMap keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureSubset(pub Vec<FeatureIndex>);

impl FeatureSubset {
    /// Creates a new feature subset from a vector of feature indices.
    /// The indices are sorted and deduplicated.
    pub fn new(mut features: Vec<FeatureIndex>) -> Self {
        features.sort_unstable();
        features.dedup();
        Self(features)
    }

    /// Creates an empty feature subset.
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Checks if the subset contains a given feature.
    pub fn contains(&self, feature: FeatureIndex) -> bool {
        self.0.binary_search(&feature).is_ok()
    }

    /// Returns the intersection of this subset with another set of features.
    /// The result is a new sorted, deduplicated feature subset.
    pub fn intersect(&self, other: &[FeatureIndex]) -> Self {
        let mut res = Vec::new();
        let mut i = 0;
        let mut j = 0;
        let mut other_sorted = other.to_vec();
        other_sorted.sort_unstable();
        other_sorted.dedup();

        while i < self.0.len() && j < other_sorted.len() {
            match self.0[i].cmp(&other_sorted[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    res.push(self.0[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        FeatureSubset(res)
    }

    /// Returns the union of this subset with another set of features.
    /// The result is a new sorted, deduplicated feature subset.
    pub fn union(&self, other: &[FeatureIndex]) -> Self {
        let mut res = self.0.clone();
        res.extend_from_slice(other);
        res.sort_unstable();
        res.dedup();
        FeatureSubset(res)
    }

    /// Returns the number of features in the subset.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if the subset is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a reference to the underlying sorted vector.
    pub fn as_slice(&self) -> &[FeatureIndex] {
        &self.0
    }
}

/// Observation set storing indices of background samples.
///
/// This enum allows for different representations:
/// - `Indices`: Stores a vector of sample indices (current implementation)
/// - Future: `BitSet` variant for memory efficiency with large n_b
///
/// Observation sets are wrapped in `Arc` to allow sharing during augmentation
/// without expensive cloning of the underlying data.
#[derive(Debug, Clone)]
pub enum ObservationSet {
    /// Stores sample indices as a vector.
    Indices(Vec<SampleIndex>),
    // Future: BitSet(BitVec),
}

impl ObservationSet {
    /// Creates a new observation set from a vector of sample indices.
    pub fn new(indices: Vec<SampleIndex>) -> Self {
        Self::Indices(indices)
    }

    /// Creates an observation set containing all indices from 0 to n-1.
    pub fn all(n: usize) -> Self {
        Self::Indices((0..n).collect())
    }

    /// Returns the number of observations in the set.
    pub fn len(&self) -> usize {
        match self {
            ObservationSet::Indices(v) => v.len(),
        }
    }

    /// Checks if the observation set is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            ObservationSet::Indices(v) => v.is_empty(),
        }
    }

    /// Returns a reference to the underlying indices vector.
    pub fn as_indices(&self) -> &[SampleIndex] {
        match self {
            ObservationSet::Indices(v) => v,
        }
    }
    /// Splits observations by threshold into two sets in a single pass.
    ///
    /// # Arguments
    /// * `data` - Background data array (n_samples, n_features)
    /// * `feature` - Feature index to filter on
    /// * `threshold` - Threshold value
    /// * `strict` - If true, use strict comparison (< for left, >= for right); if false, use weak comparison (<= for left, > for right)
    ///
    /// # Returns
    /// A tuple `(left_set, right_set)` where:
    /// - `left_set` contains indices where the feature value satisfies the left child condition
    /// - `right_set` contains indices where the feature value satisfies the right child condition
    pub fn split_by_threshold(
        &self,
        data: &ArrayView2<f32>,
        feature: FeatureIndex,
        threshold: Threshold,
        strict: bool,
    ) -> (Self, Self) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        match self {
            ObservationSet::Indices(indices) => {
                for &idx in indices {
                    let val = data[[idx, feature]];
                    let go_left = if strict {
                        val < threshold
                    } else {
                        val <= threshold
                    };
                    if go_left {
                        left.push(idx);
                    } else {
                        right.push(idx);
                    }
                }
            }
        }
        (
            ObservationSet::Indices(left),
            ObservationSet::Indices(right),
        )
    }
}

/// Shared observation set using Arc for efficient sharing during augmentation.
///
/// When `d_j ∈ S` during augmentation, we can `Arc::clone` (cheap pointer copy)
/// instead of cloning the entire `Vec<usize>` (expensive for large n_b).
pub type SharedObservationSet = Arc<ObservationSet>;

/// Path data P_j: maps feature subset S -> shared observation set D_S.
///
/// For each leaf j, this stores the observation sets D_S for each feature subset S
/// encountered on the path from root to leaf j.
pub type PathData = HashMap<FeatureSubset, SharedObservationSet>;

/// Path features T_j: features encountered on path to leaf j.
///
/// This is a vector of feature indices in the order they appear on the path
/// from root to leaf j.
pub type PathFeatures = Vec<FeatureIndex>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_subset_new() {
        let subset = FeatureSubset::new(vec![3, 1, 2, 1, 3]);
        assert_eq!(subset.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_feature_subset_empty() {
        let subset = FeatureSubset::empty();
        assert!(subset.is_empty());
        assert_eq!(subset.len(), 0);
    }

    #[test]
    fn test_feature_subset_contains() {
        let subset = FeatureSubset::new(vec![1, 3, 5]);
        assert!(subset.contains(1));
        assert!(subset.contains(3));
        assert!(subset.contains(5));
        assert!(!subset.contains(2));
        assert!(!subset.contains(0));
    }

    #[test]
    fn test_feature_subset_intersect() {
        let subset1 = FeatureSubset::new(vec![1, 2, 3, 5]);
        let subset2 = vec![2, 3, 4, 5];
        let intersection = subset1.intersect(&subset2);
        assert_eq!(intersection.as_slice(), &[2, 3, 5]);
    }

    #[test]
    fn test_feature_subset_union() {
        let subset1 = FeatureSubset::new(vec![1, 2, 3]);
        let subset2 = vec![3, 4, 5];
        let union = subset1.union(&subset2);
        assert_eq!(union.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_observation_set_all() {
        let obs_set = ObservationSet::all(5);
        assert_eq!(obs_set.len(), 5);
        assert_eq!(obs_set.as_indices(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_observation_set_filter_by_threshold() {
        use ndarray::arr2;

        // Create a simple 3x2 data array
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let data_view = data.view();

        // Start with all indices
        let obs_set = ObservationSet::all(3);

        let (filtered_yes, filtered_no) = obs_set.split_by_threshold(&data_view, 0, 3.0, true);
        assert_eq!(filtered_yes.len(), 1);
        assert_eq!(filtered_yes.as_indices(), &[0]);
        assert_eq!(filtered_no.len(), 2);
        assert_eq!(filtered_no.as_indices(), &[1, 2]);
    }

    #[test]
    fn test_shared_observation_set_arc_clone() {
        let obs_set = ObservationSet::all(100);
        let shared1: SharedObservationSet = Arc::new(obs_set);
        let shared2 = Arc::clone(&shared1);

        // Arc::clone is cheap (just increments reference count)
        assert_eq!(shared1.len(), shared2.len());
        assert_eq!(shared1.as_indices(), shared2.as_indices());

        // Both point to the same underlying data
        assert!(Arc::ptr_eq(&shared1, &shared2));
    }
}
