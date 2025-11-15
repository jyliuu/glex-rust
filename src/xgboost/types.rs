use crate::fastpd::tree::Tree;
use pyo3::prelude::*;

/// Python-facing wrapper around the unified `Tree` for XGBoost models.
///
/// This is a thin wrapper that provides Python bindings and library-specific
/// metadata. The actual tree structure is the unified `Tree` type from `fastpd::tree`.
#[derive(Debug, Clone)]
#[pyclass]
pub struct XGBoostTreeModel {
    pub(crate) tree: Tree,
}

impl XGBoostTreeModel {
    fn format_tree_recursive(&self, node_id: usize, depth: usize, output: &mut String) {
        let indent = "  ".repeat(depth);
        let node = &self.tree.nodes[node_id];

        if let Some(leaf_value) = node.leaf_value {
            // Leaf node
            output.push_str(&format!("{}Leaf: {:.6}\n", indent, leaf_value));
        } else {
            // Internal node
            let feature = node.feature.unwrap_or(0);
            let threshold = node.threshold.unwrap_or(0.0);
            output.push_str(&format!(
                "{}Feature {} <= {:.6}\n",
                indent, feature, threshold
            ));

            // Left child
            if let Some(left) = node.left {
                output.push_str(&format!("{}  -> Yes:\n", indent));
                self.format_tree_recursive(left, depth + 2, output);
            }

            // Right child
            if let Some(right) = node.right {
                output.push_str(&format!("{}  -> No:\n", indent));
                self.format_tree_recursive(right, depth + 2, output);
            }
        }
    }
}

#[pymethods]
impl XGBoostTreeModel {
    /// Returns a string representation of the tree structure.
    fn __repr__(&self) -> String {
        format!(
            "XGBoostTreeModel(nodes={}, source=xgboost)",
            self.tree.nodes.len()
        )
    }

    /// Returns a formatted string showing the tree structure.
    fn format_tree(&self) -> String {
        let mut output = String::new();
        self.format_tree_recursive(self.tree.root, 0, &mut output);
        output
    }

    /// Returns the number of nodes in the tree.
    fn num_nodes(&self) -> usize {
        self.tree.nodes.len()
    }

    /// Returns the root node index.
    fn root(&self) -> usize {
        self.tree.root
    }
}
