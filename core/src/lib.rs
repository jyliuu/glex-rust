// Pure Rust FastPD core crate
// This crate contains all algorithmic logic and is independent of PyO3

pub mod fastpd;
pub mod xgboost;

// Re-export commonly used types
pub use fastpd::augment_eval::FastPD;
pub use fastpd::error::{FastPDError, TreeExtractionError};
pub use fastpd::tree::{Tree, TreeModel, TreeNode};
pub use xgboost::parse_json_tree;

/// Generate all possible subsets of [0, 1, ..., p-1].
///
/// Returns all 2^p subsets, including the empty set.
/// Subsets are returned as sorted vectors of feature indices.
///
/// # Arguments
/// * `p` - Number of features (generates subsets of [0, 1, ..., p-1])
///
/// # Returns
/// A list of all subsets, where each subset is a sorted list of feature indices.
///
/// # Example
/// ```
/// all_subsets(3)
/// [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
/// ```
pub fn all_subsets(p: usize) -> Vec<Vec<usize>> {
    let mut subsets = Vec::new();

    // Generate all subsets using bit manipulation
    // For each number from 0 to 2^p - 1, interpret its binary representation
    // as indicating which elements are in the subset
    for mask in 0..(1usize << p) {
        let mut subset = Vec::new();
        for i in 0..p {
            if mask & (1 << i) != 0 {
                subset.push(i);
            }
        }
        subsets.push(subset);
    }

    subsets
}
