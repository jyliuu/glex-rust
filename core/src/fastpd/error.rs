use thiserror::Error;

/// Errors that can occur during extraction of tree models from backend-specific
/// formats (e.g., XGBoost JSON dumps).
#[derive(Debug, Error)]
pub enum TreeExtractionError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Tree has no nodes")]
    EmptyTree,

    #[error("Root node not found in tree")]
    NoRootNode,

    #[error("Duplicate node ID found: {0}")]
    DuplicateNodeId(u32),

    #[error("Root index {0} is out of bounds (tree has {1} nodes)")]
    InvalidRoot(usize, usize),

    #[error("Node {0} is unreachable from root")]
    UnreachableNode(usize),

    #[error("Cycle detected at node {0}")]
    CycleDetected(usize),

    #[error("Node {0} has invalid child reference {1}")]
    InvalidChild(usize, usize),

    #[error("Node {0} has both leaf and internal node properties")]
    MixedNodeType(usize),
}

/// Errors that can occur during FastPD computation.
#[derive(Debug, Error)]
pub enum FastPDError {
    #[error("Invalid tree structure: {0}")]
    InvalidTree(String),

    #[error("Feature index {0} out of bounds (max: {1})")]
    InvalidFeature(usize, usize),

    #[error("Array shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Missing observation set for feature subset")]
    MissingObservationSet,

    #[error("Background samples must have at least one row")]
    EmptyBackground,

    #[error("Evaluation point dimension {0} does not match tree features {1}")]
    DimensionMismatch(usize, usize),

    #[error("No leaves found in tree")]
    NoLeaves,

    #[error("Internal error: {0}")]
    Internal(String),
}
