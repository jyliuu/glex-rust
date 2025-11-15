use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use ndarray::ArrayView1;

use crate::fastpd::types::FeatureSubset;

/// Cache for computed PD function values.
///
/// This cache stores previously computed partial dependence values to avoid
/// redundant computations. The key is a combination of:
/// - A hash of the evaluation point coordinates
/// - The feature subset (U or S)
///
/// Different feature subsets S may map to the same U = S ∩ (union of T_j),
/// so caching at the U level allows reuse across different S queries.
#[derive(Debug)]
pub struct PDCache {
    cache: HashMap<(u64, FeatureSubset), f32>,
}

impl PDCache {
    /// Creates a new empty PD cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Gets a cached PD value if it exists.
    ///
    /// # Arguments
    /// * `point` - The evaluation point
    /// * `subset` - The feature subset (S or U)
    ///
    /// # Returns
    /// The cached value if found, `None` otherwise.
    pub fn get(&self, point: &ArrayView1<f32>, subset: &FeatureSubset) -> Option<f32> {
        let hash = Self::hash_point(point);
        self.cache.get(&(hash, subset.clone())).copied()
    }

    /// Inserts a PD value into the cache.
    ///
    /// # Arguments
    /// * `point` - The evaluation point
    /// * `subset` - The feature subset (S or U)
    /// * `value` - The computed PD value
    pub fn insert(&mut self, point: &ArrayView1<f32>, subset: FeatureSubset, value: f32) {
        let hash = Self::hash_point(point);
        self.cache.insert((hash, subset), value);
    }

    /// Clears all cached values.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Checks if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Hashes an evaluation point to a u64 for use as a cache key.
    ///
    /// This uses a simple hash of the point's coordinates. For better performance
    /// with many cache lookups, consider using `fxhash` or similar.
    fn hash_point(point: &ArrayView1<f32>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for &x in point.iter() {
            x.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl Default for PDCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_pd_cache_new() {
        let cache = PDCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_pd_cache_insert_get() {
        let mut cache = PDCache::new();
        let point = arr1(&[1.0, 2.0, 3.0]);
        let subset = FeatureSubset::new(vec![0, 1]);
        let value = 42.0;

        cache.insert(&point.view(), subset.clone(), value);
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&point.view(), &subset);
        assert_eq!(retrieved, Some(value));
    }

    #[test]
    fn test_pd_cache_missing() {
        let cache = PDCache::new();
        let point = arr1(&[1.0, 2.0]);
        let subset = FeatureSubset::new(vec![0]);

        let retrieved = cache.get(&point.view(), &subset);
        assert_eq!(retrieved, None);
    }

    #[test]
    fn test_pd_cache_clear() {
        let mut cache = PDCache::new();
        let point = arr1(&[1.0]);
        let subset = FeatureSubset::new(vec![0]);

        cache.insert(&point.view(), subset.clone(), 1.0);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.get(&point.view(), &subset), None);
    }
}
