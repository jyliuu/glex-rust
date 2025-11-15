use crate::fastpd::tree::TreeModel;
use crate::xgboost::types::XGBoostTreeModel;

/// XGBoostTreeModel implements TreeModel by delegating to the unified Tree.
/// Since Tree already implements TreeModel, we can just delegate to it.
/// Uses strict comparison (<) to match XGBoost behavior.
impl TreeModel for XGBoostTreeModel {
    const COMPARISON: bool = true; // XGBoost uses strict comparison
    fn node_feature(&self, node_id: usize) -> Option<usize> {
        self.tree.node_feature(node_id)
    }

    fn node_threshold(&self, node_id: usize) -> Option<f64> {
        self.tree.node_threshold(node_id)
    }

    fn left_child(&self, node_id: usize) -> Option<usize> {
        self.tree.left_child(node_id)
    }

    fn right_child(&self, node_id: usize) -> Option<usize> {
        self.tree.right_child(node_id)
    }

    fn is_leaf(&self, node_id: usize) -> bool {
        self.tree.is_leaf(node_id)
    }

    fn leaf_value(&self, node_id: usize) -> Option<f64> {
        self.tree.leaf_value(node_id)
    }

    fn root(&self) -> usize {
        self.tree.root()
    }

    fn num_nodes(&self) -> usize {
        self.tree.num_nodes()
    }
}
