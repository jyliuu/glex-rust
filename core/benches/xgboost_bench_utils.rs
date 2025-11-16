use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use glex_core::{parse_json_tree, Tree};

/// Dataset stems used in all XGBoost/FastPD benchmarks.
pub const SYNTHETIC_LOW_DIM_STEMS: [&str; 3] = [
    "synthetic_n200_p2",
    "synthetic_n500_p3",
    "synthetic_n1000_p5",
];

pub const SYNTHETIC_STEMS: [&str; 5] = [
    "synthetic_n200_p2",
    "synthetic_n500_p3",
    "synthetic_n1000_p5",
    "synthetic_n2000_p7",
    "synthetic_n5000_p10",
];

pub const CALIFORNIA_STEM: &str = "california_housing";

/// Resolve the workspace root (parent of the `core` crate).
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core has a parent directory")
        .to_path_buf()
}

/// Load feature matrix X (all columns except last) from a CSV file.
///
/// Assumes header row and that the last column is the target `y`.
pub fn load_features_from_csv(path: &Path) -> Array2<f32> {
    let mut reader = csv::Reader::from_path(path)
        .unwrap_or_else(|e| panic!("Failed to open CSV {}: {e}", path.display()));

    let mut rows: Vec<Vec<f32>> = Vec::new();
    for record in reader
        .records()
        .map(|r| r.unwrap_or_else(|e| panic!("CSV read error in {}: {e}", path.display())))
    {
        if record.is_empty() {
            continue;
        }
        let n_cols = record.len();
        // Drop last column (y)
        let mut row = Vec::with_capacity(n_cols.saturating_sub(1));
        for field in record.iter().take(n_cols.saturating_sub(1)) {
            let value: f32 = field.parse().unwrap_or_else(|e| {
                panic!("Failed to parse '{}' in {}: {e}", field, path.display())
            });
            row.push(value);
        }
        rows.push(row);
    }

    let n_samples = rows.len();
    let n_features = rows
        .first()
        .map(|r| r.len())
        .unwrap_or_else(|| panic!("No rows found in CSV {}", path.display()));

    let mut array = Array2::<f32>::zeros((n_samples, n_features));
    for (i, row) in rows.into_iter().enumerate() {
        assert_eq!(
            row.len(),
            n_features,
            "Inconsistent row length in {} at row {}",
            path.display(),
            i
        );
        for (j, value) in row.into_iter().enumerate() {
            array[(i, j)] = value;
        }
    }

    array
}

/// Compute some simple statistics about a tree: (num_nodes, num_leaves, max_depth, num_features_used).
pub fn tree_stats(tree: &Tree) -> (usize, usize, usize, usize) {
    use std::collections::HashSet;

    let num_nodes = tree.nodes.len();
    let num_leaves = tree.nodes.iter().filter(|n| n.leaf_value.is_some()).count();

    // DFS to compute max depth
    fn dfs_depth(tree: &Tree, node_id: usize, depth: usize) -> usize {
        let node = &tree.nodes[node_id];
        let mut max_d = depth;
        if let Some(left) = node.left {
            max_d = max_d.max(dfs_depth(tree, left, depth + 1));
        }
        if let Some(right) = node.right {
            max_d = max_d.max(dfs_depth(tree, right, depth + 1));
        }
        max_d
    }
    let max_depth = dfs_depth(tree, tree.root, 0);

    // Unique feature indices used in splits
    let mut features = HashSet::new();
    for node in &tree.nodes {
        if let Some(f) = node.feature {
            features.insert(f);
        }
    }
    let num_features_used = features.len();

    (num_nodes, num_leaves, max_depth, num_features_used)
}

/// Load XGBoost trees from a JSON dump produced by `Booster.get_dump(dump_format="json")`.
///
/// The file is expected to contain a JSON array of strings, one JSON object per tree.
pub fn load_xgboost_trees(path: &Path) -> Vec<Tree> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read model {}: {e}", path.display()));
    let dump: Vec<String> = serde_json::from_str(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse JSON dump {}: {e}", path.display()));

    dump.into_iter()
        .enumerate()
        .map(|(idx, tree_json)| {
            let tree = parse_json_tree(&tree_json).unwrap_or_else(|e| {
                panic!(
                    "Failed to parse XGBoost tree {} from {}: {e}",
                    idx,
                    path.display()
                )
            });
            tree
        })
        .collect()
}
