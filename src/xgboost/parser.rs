use std::collections::HashMap;

use crate::fastpd::error::TreeExtractionError;
use crate::fastpd::tree::{Tree, TreeNode};
use crate::xgboost::json_schema::JsonNode;

/// Parses a feature name string (e.g., "f0", "f1") into a feature index.
fn parse_feature_index(split: &str) -> Result<usize, TreeExtractionError> {
    if let Some(stripped) = split.strip_prefix('f') {
        stripped.parse::<usize>().map_err(|_| {
            TreeExtractionError::InvalidJson(format!("Invalid feature name: {}", split))
        })
    } else {
        Err(TreeExtractionError::InvalidJson(format!(
            "Feature name must start with 'f': {}",
            split
        )))
    }
}

/// Flattens the recursive JSON tree structure into a flat list of nodes.
/// Also builds a mapping from nodeid to the internal index.
/// Returns the internal index of the processed node.
fn flatten_tree(
    json_node: &JsonNode,
    nodes: &mut Vec<TreeNode>,
    id_to_idx: &mut HashMap<u32, usize>,
) -> Result<usize, TreeExtractionError> {
    let nodeid = json_node.nodeid;

    // Check for duplicate node IDs.
    if let Some(&existing_idx) = id_to_idx.get(&nodeid) {
        return Ok(existing_idx); // Already processed, return existing index
    }

    // Reserve the internal index by pushing a placeholder node first.
    // This ensures that children get correct indices when they're processed.
    let internal_idx = nodes.len();
    id_to_idx.insert(nodeid, internal_idx);

    // Push a placeholder node that we'll update after processing children.
    nodes.push(TreeNode {
        internal_idx,
        feature: None,
        threshold: None,
        left: None,
        right: None,
        missing: None,
        leaf_value: None,
    });

    // Parse feature index from split string (e.g., "f0" -> 0).
    let feature = json_node
        .split
        .as_ref()
        .map(|s| parse_feature_index(s))
        .transpose()?;

    let threshold = json_node.split_condition;
    let leaf_value = json_node.leaf;

    // Process children recursively (if using children array).
    // Use yes/no fields to determine which child is left (yes) vs right (no).
    // This matches the approach in pltreeshap: https://github.com/schufa-innovationlab/pltreeshap
    let (left, right) = if let Some(children) = &json_node.children {
        let yes_nodeid = json_node.yes;
        let no_nodeid = json_node.no;

        let mut left_idx = None;
        let mut right_idx = None;

        for child in children {
            let child_idx = flatten_tree(child, nodes, id_to_idx)?;
            if Some(child.nodeid) == yes_nodeid {
                left_idx = Some(child_idx);
            } else if Some(child.nodeid) == no_nodeid {
                right_idx = Some(child_idx);
            }
        }

        (left_idx, right_idx)
    } else {
        // Fall back to yes/no fields (legacy format).
        // In this case, we need to find the nodes by ID, which is more complex.
        // For now, we'll set them to None and handle them later if needed.
        (None, None)
    };

    // Handle missing value routing.
    // The missing field points to a nodeid that should already be processed
    // as one of the children. We just need to look it up in the id_to_idx map.
    let missing = json_node
        .missing
        .and_then(|missing_id| id_to_idx.get(&missing_id).copied());

    // Update the placeholder node with the actual values.
    nodes[internal_idx] = TreeNode {
        internal_idx,
        feature,
        threshold,
        left,
        right,
        missing,
        leaf_value,
    };

    Ok(internal_idx)
}

/// Parses an XGBoost JSON tree dump into a unified `Tree`.
///
/// # Arguments
/// * `json_str` - JSON string from `Booster.get_dump(dump_format="json")[i]`.
///
/// # Returns
/// Parsed and validated `Tree` (unified representation).
///
/// # Errors
/// Returns `TreeExtractionError` if:
/// - JSON is malformed.
/// - Root node is missing.
/// - Duplicate node IDs found.
/// - Tree structure is invalid (cycles, invalid references, mixed node types).
pub fn parse_json_tree(json_str: &str) -> Result<Tree, TreeExtractionError> {
    // Parse JSON root node.
    let json_root: JsonNode = serde_json::from_str(json_str)
        .map_err(|e| TreeExtractionError::InvalidJson(e.to_string()))?;

    // Flatten the recursive structure into a flat list.
    let mut nodes = Vec::new();
    let mut id_to_idx = HashMap::new();
    let root = flatten_tree(&json_root, &mut nodes, &mut id_to_idx)?;

    if nodes.is_empty() {
        return Err(TreeExtractionError::EmptyTree);
    }

    let tree = Tree::new(nodes, root);

    // Validate tree structure.
    tree.validate()?;

    Ok(tree)
}
