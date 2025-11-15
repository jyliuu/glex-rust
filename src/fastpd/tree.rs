use crate::fastpd::error::TreeExtractionError;

/// Type aliases for tree model components used by FastPD.
pub type FeatureIndex = usize;
pub type Threshold = f32;
pub type LeafValue = f32;

/// Unified representation of a tree node.
/// This is the common structure used by all tree-based models (XGBoost, LightGBM, etc.).
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Internal index (0..N-1) in the nodes vector.
    pub internal_idx: usize,
    /// Feature index for splitting (None for leaves).
    pub feature: Option<usize>,
    /// Threshold value for splitting (None for leaves).
    pub threshold: Option<f32>,
    /// Left child index (for "yes" branch).
    /// The comparison semantics (strict < vs weak <=) are determined by the TreeModel implementation.
    pub left: Option<usize>,
    /// Right child index (for "no" branch).
    /// The comparison semantics (strict >= vs weak >) are determined by the TreeModel implementation.
    pub right: Option<usize>,
    /// Missing value routing: where to send samples with missing feature value.
    /// If None, missing values follow the default (usually left/right based on model).
    pub missing: Option<usize>,
    /// Leaf value (Some for leaves, None for internal nodes).
    pub leaf_value: Option<f32>,
}

/// Unified representation of a complete tree.
/// This is the common structure used by all tree-based models.
#[derive(Debug, Clone)]
pub struct Tree {
    /// All nodes in the tree, indexed by `internal_idx`.
    pub nodes: Vec<TreeNode>,
    /// Root node index (typically 0).
    pub root: usize,
}

impl Tree {
    /// Creates a new tree with the given nodes and root.
    pub fn new(nodes: Vec<TreeNode>, root: usize) -> Self {
        Self { nodes, root }
    }

    /// Validates the tree structure:
    /// - Root exists and is valid.
    /// - No cycles.
    /// - All nodes are reachable from root.
    /// - Child indices are valid.
    /// - Nodes are either leaves or internal (not both).
    pub fn validate(&self) -> Result<(), TreeExtractionError> {
        if self.nodes.is_empty() {
            return Err(TreeExtractionError::EmptyTree);
        }
        if self.root >= self.nodes.len() {
            return Err(TreeExtractionError::InvalidRoot(
                self.root,
                self.nodes.len(),
            ));
        }

        // Check for cycles and unreachable nodes using DFS.
        let mut visited = vec![false; self.nodes.len()];
        let mut in_stack = vec![false; self.nodes.len()];

        self.validate_dfs(self.root, &mut visited, &mut in_stack)?;

        // Check all nodes are reachable.
        for (idx, was_visited) in visited.iter().enumerate() {
            if !was_visited {
                return Err(TreeExtractionError::UnreachableNode(idx));
            }
        }

        // Validate node consistency.
        for node in &self.nodes {
            let is_leaf = node.leaf_value.is_some();
            let is_internal = node.feature.is_some() || node.left.is_some() || node.right.is_some();

            if is_leaf && is_internal {
                return Err(TreeExtractionError::MixedNodeType(node.internal_idx));
            }

            if let Some(left) = node.left {
                if left >= self.nodes.len() {
                    return Err(TreeExtractionError::InvalidChild(node.internal_idx, left));
                }
            }
            if let Some(right) = node.right {
                if right >= self.nodes.len() {
                    return Err(TreeExtractionError::InvalidChild(node.internal_idx, right));
                }
            }
            if let Some(missing) = node.missing {
                if missing >= self.nodes.len() {
                    return Err(TreeExtractionError::InvalidChild(
                        node.internal_idx,
                        missing,
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_dfs(
        &self,
        node_id: usize,
        visited: &mut [bool],
        in_stack: &mut [bool],
    ) -> Result<(), TreeExtractionError> {
        if in_stack[node_id] {
            return Err(TreeExtractionError::CycleDetected(node_id));
        }
        if visited[node_id] {
            return Ok(());
        }

        in_stack[node_id] = true;
        visited[node_id] = true;

        let node = &self.nodes[node_id];
        if let Some(left) = node.left {
            self.validate_dfs(left, visited, in_stack)?;
        }
        if let Some(right) = node.right {
            self.validate_dfs(right, visited, in_stack)?;
        }
        if let Some(missing) = node.missing {
            self.validate_dfs(missing, visited, in_stack)?;
        }

        in_stack[node_id] = false;
        Ok(())
    }
}

/// Trait for tree-based models that FastPD can operate on.
///
/// This trait abstracts over different tree implementations (XGBoost,
/// LightGBM, etc.) to allow model-agnostic FastPD computation.
///
/// # Node IDs
/// Node IDs are indices into the tree's internal representation. The root node
/// is always accessible via `root()`, and all valid node IDs are in
/// `[0, num_nodes())`.
///
/// # Leaf vs Internal Nodes
/// - Internal nodes: have `feature` and `threshold`, and `left_child`/`right_child`.
/// - Leaf nodes: have `leaf_value`, no children, and `is_leaf()` returns `true`.
///
/// # Missing Values
/// For models that support missing value routing (e.g., XGBoost), the trait does not
/// expose this directly. Instead, implementations should route missing values according
/// to their model's semantics when traversing (e.g., XGBoost uses a `missing` field).
pub trait TreeModel: Send + Sync {
    /// Comparison operator mode: true for strict (<), false for weak (<=)
    ///
    /// This determines how the tree splits at thresholds:
    /// - `true` (strict): `feature_value < threshold` goes to left child (XGBoost behavior)
    /// - `false` (weak): `feature_value <= threshold` goes to left child
    const COMPARISON: bool;

    /// Returns the feature index used for splitting at this node, or `None` if leaf.
    fn node_feature(&self, node_id: usize) -> Option<FeatureIndex>;

    /// Returns the threshold value for splitting, or `None` if leaf.
    fn node_threshold(&self, node_id: usize) -> Option<Threshold>;

    /// Returns the left child node ID, or `None` if leaf or invalid.
    /// For strict comparison (<), this is the "yes" branch when `feature_value < threshold`.
    /// For weak comparison (<=), this is the "yes" branch when `feature_value <= threshold`.
    fn left_child(&self, node_id: usize) -> Option<usize>;

    /// Returns the right child node ID, or `None` if leaf or invalid.
    /// For strict comparison (<), this is the "no" branch when `feature_value >= threshold`.
    /// For weak comparison (<=), this is the "no" branch when `feature_value > threshold`.
    fn right_child(&self, node_id: usize) -> Option<usize>;

    /// Returns `true` if the node is a leaf (has a leaf value).
    fn is_leaf(&self, node_id: usize) -> bool;

    /// Returns the leaf value, or `None` if not a leaf or invalid node.
    fn leaf_value(&self, node_id: usize) -> Option<LeafValue>;

    /// Returns the root node ID (typically 0).
    fn root(&self) -> usize;

    /// Returns the total number of nodes in the tree.
    fn num_nodes(&self) -> usize;

    /// Predicts the output for a given input point by traversing the tree.
    ///
    /// Uses the comparison semantics defined by `Self::COMPARISON`:
    /// - If `COMPARISON = true`: `feature_value < threshold` goes left
    /// - If `COMPARISON = false`: `feature_value <= threshold` goes left
    ///
    /// # Arguments
    /// * `x` - Input point (slice of feature values)
    ///
    /// # Returns
    /// The leaf value reached by traversing the tree with the given input.
    ///
    /// # Panics
    /// Panics if the tree structure is invalid or if feature indices are out of bounds.
    fn predict(&self, x: &[f32]) -> f32 {
        let mut node_id = self.root();
        loop {
            if self.is_leaf(node_id) {
                return self.leaf_value(node_id).unwrap();
            }
            let feature = self.node_feature(node_id).unwrap();
            let threshold = self.node_threshold(node_id).unwrap();
            let feature_value = x[feature];

            let go_left = if Self::COMPARISON {
                feature_value < threshold
            } else {
                feature_value <= threshold
            };

            if go_left {
                node_id = self.left_child(node_id).unwrap();
            } else {
                node_id = self.right_child(node_id).unwrap();
            }
        }
    }
}

/// Implement TreeModel for Tree directly - this is the unified representation.
/// Uses strict comparison (<) to match XGBoost behavior.
impl TreeModel for Tree {
    const COMPARISON: bool = true; // XGBoost-style behavior
    fn node_feature(&self, node_id: usize) -> Option<FeatureIndex> {
        self.nodes.get(node_id)?.feature
    }

    fn node_threshold(&self, node_id: usize) -> Option<Threshold> {
        self.nodes.get(node_id)?.threshold
    }

    fn left_child(&self, node_id: usize) -> Option<usize> {
        self.nodes.get(node_id)?.left
    }

    fn right_child(&self, node_id: usize) -> Option<usize> {
        self.nodes.get(node_id)?.right
    }

    fn is_leaf(&self, node_id: usize) -> bool {
        self.nodes.get(node_id).and_then(|n| n.leaf_value).is_some()
    }

    fn leaf_value(&self, node_id: usize) -> Option<LeafValue> {
        self.nodes.get(node_id)?.leaf_value
    }

    fn root(&self) -> usize {
        self.root
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}
