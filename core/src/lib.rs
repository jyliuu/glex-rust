// Pure Rust FastPD core crate
// This crate contains all algorithmic logic and is independent of PyO3

pub mod fastpd;
pub mod xgboost;

// Re-export commonly used types
pub use fastpd::augment_eval::FastPD;
pub use fastpd::error::{FastPDError, TreeExtractionError};
pub use fastpd::tree::{Tree, TreeModel, TreeNode};
pub use xgboost::parse_json_tree;
