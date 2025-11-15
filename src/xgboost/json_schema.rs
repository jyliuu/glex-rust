use serde::Deserialize;

/// XGBoost JSON tree structure as returned by `Booster.get_dump(dump_format="json")`.
///
/// XGBoost returns a single root `JsonNode` (not a wrapper object with treeid/nodes).
/// The tree structure is recursive via the `children` array.
pub type JsonTree = JsonNode;

/// XGBoost JSON node structure.
///
/// Note: XGBoost JSON uses `nodeid` (which can be any u32, typically 0, 1, 2, ...)
/// rather than sequential indices. We map these to internal indices during parsing.
///
/// The structure is recursive: internal nodes have a `children` array with typically 2 elements
/// (yes and no branches), while leaf nodes have a `leaf` value.
#[derive(Debug, Deserialize, Clone)]
pub struct JsonNode {
    /// Node ID as stored in XGBoost (typically 0 for root, but not guaranteed sequential).
    pub nodeid: u32,
    /// Depth of the node (optional, for debugging).
    #[serde(default)]
    pub depth: Option<u32>,

    // Internal node fields
    /// Feature name for splitting (e.g., "f0", "f1") - None for leaves.
    /// This is a string that needs to be parsed to extract the feature index.
    #[serde(default)]
    pub split: Option<String>,
    /// Threshold value for the split (None for leaves).
    #[serde(default)]
    pub split_condition: Option<f32>,
    /// Node ID of "yes" child (x <= threshold) - deprecated in favor of children array.
    #[serde(default)]
    pub yes: Option<u32>,
    /// Node ID of "no" child (x > threshold) - deprecated in favor of children array.
    #[serde(default)]
    pub no: Option<u32>,
    /// Node ID for missing value routing (optional).
    #[serde(default)]
    pub missing: Option<u32>,
    /// Children nodes (for internal nodes). Typically has 2 elements: [yes_child, no_child].
    #[serde(default)]
    pub children: Option<Vec<JsonNode>>,

    // Leaf node fields
    /// Leaf value (None for internal nodes).
    #[serde(default)]
    pub leaf: Option<f32>,
}
