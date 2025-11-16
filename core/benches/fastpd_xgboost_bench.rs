use std::fs;
use std::path::{Path, PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};

use glex_core::{parse_json_tree, FastPD, Tree};

/// Resolve the workspace root (parent of the `core` crate).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core has a parent directory")
        .to_path_buf()
}

/// Load feature matrix X (all columns except last) from a CSV file.
///
/// Assumes header row and that the last column is the target `y`.
fn load_features_from_csv(path: &Path) -> Array2<f32> {
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

/// Compute per-feature median to obtain a single "median datapoint".
fn median_point(data: &Array2<f32>) -> Array1<f32> {
    let n_features = data.ncols();
    let mut medians = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let mut col: Vec<f32> = data.column(j).iter().copied().collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let m = col.len();
        let median = if m == 0 {
            0.0
        } else if m % 2 == 1 {
            col[m / 2]
        } else {
            0.5 * (col[m / 2 - 1] + col[m / 2])
        };
        medians.push(median);
    }

    Array1::from(medians)
}

/// Compute some simple statistics about a tree: (num_nodes, num_leaves, max_depth, num_features_used).
fn tree_stats(tree: &Tree) -> (usize, usize, usize, usize) {
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
fn load_xgboost_trees(path: &Path) -> Vec<Tree> {
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

            let (num_nodes, num_leaves, max_depth, num_features_used) = tree_stats(&tree);
            println!(
                "Loaded tree {}: nodes={}, leaves={}, max_depth={}, features_used={}",
                idx,
                num_nodes,
                num_leaves,
                max_depth,
                num_features_used
            );

            tree
        })
        .collect()
}

fn benchmark_fastpd_for_dataset(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    dataset_stem: &str,
) {
    let root = workspace_root();
    let data_dir = root.join("data");
    let datasets_dir = data_dir.join("datasets");
    let models_dir = data_dir.join("models");

    let csv_path = datasets_dir.join(format!("{dataset_stem}.csv"));
    let model_path = models_dir.join(format!("{dataset_stem}_xgb.json"));

    let background = load_features_from_csv(&csv_path);
    let eval_point = median_point(&background);

    let trees = load_xgboost_trees(&model_path);
    let n_features = background.ncols();
    let all_features: Vec<usize> = (0..n_features).collect();

    // Use the whole dataset as background for augmentation
    let background_view = background.view();
    let mut fastpd = FastPD::new(trees, &background_view, 0.0).expect("Failed to construct FastPD");

    // Single evaluation point (1, n_features)
    let mut eval_points = Array2::<f32>::zeros((1, n_features));
    eval_points.row_mut(0).assign(&eval_point);
    let eval_view = eval_points.view();

    group.bench_function(BenchmarkId::from_parameter(dataset_stem), |b| {
        b.iter(|| {
            let result = fastpd
                .pd_function(&eval_view, &all_features)
                .expect("FastPD evaluation failed");
            black_box(result);
        });
    });
}

fn fastpd_xgboost_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastpd_xgboost_median_point");

    // Synthetic datasets of increasing dimensionality
    let synthetic_stems = [
        "synthetic_n200_p2",
        "synthetic_n500_p3",
        "synthetic_n1000_p5",
        "synthetic_n2000_p7",
        "synthetic_n5000_p10",
    ];

    for stem in &synthetic_stems {
        benchmark_fastpd_for_dataset(&mut group, stem);
    }

    // California housing dataset
    benchmark_fastpd_for_dataset(&mut group, "california_housing");

    group.finish();
}

criterion_group!(benches, fastpd_xgboost_benchmarks);
criterion_main!(benches);
