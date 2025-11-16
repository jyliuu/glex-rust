use std::collections::HashMap;
use std::sync::Arc;

use ndarray::ArrayView2;

use crate::fastpd::tree::{FeatureIndex, Threshold};

/// Number of bits per chunk in the bitmask
const BITS_PER_CHUNK: usize = 64;

/// Index of a sample in the background dataset.
pub type SampleIndex = usize;

/// Feature subset represented as a bitmask for efficient operations.
///
/// Optimized for small feature sets (< 64 features) using a single u64,
/// with fallback to Vec<u64> for larger sets. This avoids Vec overhead
/// for the common case while supporting up to 10,000+ features.
#[derive(Debug, Clone)]
pub enum FeatureSubset {
    /// Small subset: single u64 for features 0-63 (most common case)
    Small(u64),
    /// Large subset: Vec<u64> for features >= 64
    Large(Vec<u64>),
}

impl FeatureSubset {
    /// Creates an empty feature subset.
    pub fn empty() -> Self {
        Self::Small(0)
    }

    /// Creates a new feature subset from a vector of feature indices.
    /// The indices are deduplicated automatically (bitmask naturally handles this).
    pub fn new(features: Vec<FeatureIndex>) -> Self {
        Self::from_slice(&features)
    }

    /// Creates a new feature subset from a slice of feature indices.
    /// This avoids an intermediate Vec allocation when the caller already has a slice.
    pub fn from_slice(features: &[FeatureIndex]) -> Self {
        if features.is_empty() {
            return Self::empty();
        }

        // Find max feature to determine required mask size
        let max_feature = features.iter().copied().max().unwrap_or(0);

        // Optimize for common case: single u64 for features < 64
        if max_feature < BITS_PER_CHUNK {
            let mut mask = 0u64;
            for &feature in features {
                mask |= 1u64 << feature;
            }
            return Self::Small(mask);
        }

        // Large case: use Vec<u64>
        let n_chunks = (max_feature / BITS_PER_CHUNK) + 1;
        let mut mask = vec![0u64; n_chunks];

        // Set bits for each feature
        for &feature in features {
            let chunk_idx = feature / BITS_PER_CHUNK;
            let bit_pos = feature % BITS_PER_CHUNK;
            mask[chunk_idx] |= 1u64 << bit_pos;
        }

        Self::Large(mask)
    }

    /// Checks if the subset contains a given feature.
    /// O(1) via bitwise operation.
    pub fn contains(&self, feature: FeatureIndex) -> bool {
        match self {
            Self::Small(mask) => feature < BITS_PER_CHUNK && (*mask & (1u64 << feature)) != 0,
            Self::Large(mask) => {
                let chunk_idx = feature / BITS_PER_CHUNK;
                let bit_pos = feature % BITS_PER_CHUNK;
                chunk_idx < mask.len() && (mask[chunk_idx] & (1u64 << bit_pos)) != 0
            }
        }
    }

    /// Returns a new subset with the given feature added.
    /// More efficient than converting to Vec and back.
    pub fn with_feature(&self, feature: FeatureIndex) -> Self {
        match self {
            Self::Small(mask) => {
                if feature < BITS_PER_CHUNK {
                    Self::Small(mask | (1u64 << feature))
                } else {
                    // Need to convert to Large
                    let mut new_mask = vec![*mask];
                    let chunk_idx = feature / BITS_PER_CHUNK;
                    let bit_pos = feature % BITS_PER_CHUNK;
                    if chunk_idx >= new_mask.len() {
                        new_mask.resize(chunk_idx + 1, 0);
                    }
                    new_mask[chunk_idx] |= 1u64 << bit_pos;
                    Self::Large(new_mask)
                }
            }
            Self::Large(mask) => {
                let chunk_idx = feature / BITS_PER_CHUNK;
                let bit_pos = feature % BITS_PER_CHUNK;
                let mut new_mask = mask.clone();
                if chunk_idx >= new_mask.len() {
                    new_mask.resize(chunk_idx + 1, 0);
                }
                new_mask[chunk_idx] |= 1u64 << bit_pos;
                // Convert to Small if result fits in single u64
                if new_mask.len() == 1 {
                    Self::Small(new_mask[0])
                } else {
                    Self::Large(new_mask)
                }
            }
        }
    }

    /// Returns the intersection of this subset with another FeatureSubset.
    /// Uses bitwise AND for efficient computation.
    pub fn intersect_with(&self, other: &FeatureSubset) -> Self {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Self::Small(a & b),
            (Self::Small(a), Self::Large(b)) => {
                if b.is_empty() {
                    Self::Small(0)
                } else {
                    Self::Small(a & b[0])
                }
            }
            (Self::Large(a), Self::Small(b)) => {
                if a.is_empty() {
                    Self::Small(0)
                } else {
                    Self::Small(a[0] & b)
                }
            }
            (Self::Large(a), Self::Large(b)) => {
                let common_size = a.len().min(b.len());
                let mut result = vec![0u64; common_size];
                for i in 0..common_size {
                    result[i] = a[i] & b[i];
                }
                Self::Large(result)
            }
        }
    }

    /// Computes the intersection of this subset with another `FeatureSubset`,
    /// writing the result into `out` to allow buffer reuse and reduce allocations.
    ///
    /// This is primarily intended for hot paths where a single scratch subset
    /// can be reused many times (e.g., during tree evaluation).
    pub fn intersect_with_into(&self, other: &FeatureSubset, out: &mut FeatureSubset) {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => {
                *out = Self::Small(a & b);
            }
            (Self::Small(a), Self::Large(b)) => {
                let result = if b.is_empty() { 0 } else { a & b[0] };
                *out = Self::Small(result);
            }
            (Self::Large(a), Self::Small(b)) => {
                let result = if a.is_empty() { 0 } else { a[0] & b };
                *out = Self::Small(result);
            }
            (Self::Large(a), Self::Large(b)) => {
                let common_size = a.len().min(b.len());
                if common_size == 0 {
                    *out = Self::Small(0);
                    return;
                }

                match out {
                    Self::Large(ref mut vec) => {
                        if vec.len() < common_size {
                            vec.resize(common_size, 0);
                        }
                        for (dst, (&lhs, &rhs)) in
                            vec.iter_mut().zip(a.iter().zip(b.iter())).take(common_size)
                        {
                            *dst = lhs & rhs;
                        }
                    }
                    Self::Small(_) => {
                        let mut vec = Vec::with_capacity(common_size);
                        vec.extend(
                            a.iter()
                                .zip(b.iter())
                                .take(common_size)
                                .map(|(&lhs, &rhs)| lhs & rhs),
                        );
                        *out = Self::Large(vec);
                    }
                }
            }
        }
    }

    /// Returns the intersection of this subset with another set of features.
    /// Uses bitwise AND for efficient computation.
    pub fn intersect(&self, other: &[FeatureIndex]) -> Self {
        if other.is_empty() {
            return Self::empty();
        }

        let other_subset = Self::new(other.to_vec());
        self.intersect_with(&other_subset)
    }

    /// Returns the union of this subset with another set of features.
    /// Uses bitwise OR for efficient computation.
    pub fn union(&self, other: &[FeatureIndex]) -> Self {
        if other.is_empty() {
            return self.clone();
        }

        let other_subset = Self::new(other.to_vec());
        match (self, &other_subset) {
            (Self::Small(a), Self::Small(b)) => Self::Small(a | b),
            (Self::Small(a), Self::Large(b)) => {
                let mut result = b.clone();
                if !result.is_empty() {
                    result[0] |= a;
                } else {
                    result.push(*a);
                }
                Self::Large(result)
            }
            (Self::Large(a), Self::Small(b)) => {
                let mut result = a.clone();
                if !result.is_empty() {
                    result[0] |= b;
                } else {
                    result.push(*b);
                }
                Self::Large(result)
            }
            (Self::Large(a), Self::Large(b)) => {
                let max_size = a.len().max(b.len());
                let mut result = vec![0u64; max_size];
                for i in 0..a.len() {
                    result[i] = a[i];
                }
                for i in 0..b.len() {
                    result[i] |= b[i];
                }
                Self::Large(result)
            }
        }
    }

    /// Returns the number of features in the subset.
    /// Counts set bits (popcount).
    pub fn len(&self) -> usize {
        match self {
            Self::Small(mask) => mask.count_ones() as usize,
            Self::Large(mask) => mask.iter().map(|chunk| chunk.count_ones() as usize).sum(),
        }
    }

    /// Checks if the subset is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Small(mask) => *mask == 0,
            Self::Large(mask) => mask.iter().all(|&chunk| chunk == 0),
        }
    }

    /// Returns a sorted vector of feature indices.
    /// Converts bitmask back to Vec<FeatureIndex> for compatibility.
    pub fn as_slice(&self) -> Vec<FeatureIndex> {
        let mut features = Vec::new();
        match self {
            Self::Small(mask) => {
                if *mask == 0 {
                    return features;
                }
                for bit_pos in 0..BITS_PER_CHUNK {
                    if (*mask & (1u64 << bit_pos)) != 0 {
                        features.push(bit_pos);
                    }
                }
            }
            Self::Large(mask) => {
                for (chunk_idx, &chunk) in mask.iter().enumerate() {
                    if chunk == 0 {
                        continue;
                    }
                    for bit_pos in 0..BITS_PER_CHUNK {
                        if (chunk & (1u64 << bit_pos)) != 0 {
                            features.push(chunk_idx * BITS_PER_CHUNK + bit_pos);
                        }
                    }
                }
            }
        }
        features
    }
}

impl PartialEq for FeatureSubset {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => a == b,
            (Self::Small(a), Self::Large(b)) => {
                b.len() == 1 && b[0] == *a && b.iter().skip(1).all(|&x| x == 0)
            }
            (Self::Large(a), Self::Small(b)) => {
                a.len() == 1 && a[0] == *b && a.iter().skip(1).all(|&x| x == 0)
            }
            (Self::Large(a), Self::Large(b)) => {
                let max_len = a.len().max(b.len());
                for i in 0..max_len {
                    let a_chunk = a.get(i).copied().unwrap_or(0);
                    let b_chunk = b.get(i).copied().unwrap_or(0);
                    if a_chunk != b_chunk {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl Eq for FeatureSubset {}

impl std::hash::Hash for FeatureSubset {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Small(mask) => {
                mask.hash(state);
                // Tag to distinguish from Large with single chunk
                0u8.hash(state);
            }
            Self::Large(mask) => {
                for &chunk in mask {
                    chunk.hash(state);
                }
                mask.len().hash(state);
                // Tag to distinguish from Small
                1u8.hash(state);
            }
        }
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
