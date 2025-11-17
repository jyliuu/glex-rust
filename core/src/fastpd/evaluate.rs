use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::fastpd::error::FastPDError;
use crate::fastpd::parallel::{Joiner, ParallelSettings, RayonJoin, SeqJoin};
use crate::fastpd::tree::TreeModel;
use crate::fastpd::types::FeatureSubset;

use super::augmented_tree::AugmentedTree;

/// Evaluates v_U(x_U) for multiple subsets U and multiple evaluation points sequentially.
///
/// This is a thin wrapper around `evaluate_pd_batch_for_subsets_impl::<T, SeqJoin>`.
pub fn evaluate_pd_batch_for_subsets_seq<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
) -> Result<Array2<f32>, FastPDError> {
    evaluate_pd_batch_for_subsets_impl::<T, SeqJoin>(augmented_tree, evaluation_points, subsets_u)
}

/// Evaluates v_U(x_U) for multiple subsets U and multiple evaluation points in parallel.
///
/// This is a thin wrapper around `evaluate_pd_batch_for_subsets_impl::<T, RayonJoin>`.
/// **Important**: This function must be called from within a Rayon thread pool
/// (via `ThreadPool::install()`) for proper parallel execution.
pub fn evaluate_pd_batch_for_subsets_rayon<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
) -> Result<Array2<f32>, FastPDError> {
    evaluate_pd_batch_for_subsets_impl::<T, RayonJoin>(augmented_tree, evaluation_points, subsets_u)
}

/// Legacy function: delegates to `evaluate_pd_batch_for_subsets_seq` for backward compatibility.
pub fn evaluate_pd_batch_for_subsets<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
) -> Result<Array2<f32>, FastPDError> {
    evaluate_pd_batch_for_subsets_seq(augmented_tree, evaluation_points, subsets_u)
}

/// Internal implementation of batch evaluation using the Joiner trait.
///
/// This function implements the Rcpp `recurseMarginalizeSBitmask` pattern,
/// evaluating all evaluation points and all feature subsets in a single tree traversal.
fn evaluate_pd_batch_for_subsets_impl<T, J>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
) -> Result<Array2<f32>, FastPDError>
where
    T: TreeModel,
    J: Joiner,
{
    let n_eval = evaluation_points.nrows();
    let n_subsets = subsets_u.len();
    let mut out = Array2::<f32>::zeros((n_eval, n_subsets));

    // Find empty subset index (if present) to set it directly to expected value
    let empty_subset = FeatureSubset::empty();
    let empty_idx = subsets_u.iter().position(|u| u == &empty_subset);

    evaluate_batch_recursive::<T, J>(
        augmented_tree,
        augmented_tree.tree.root(),
        evaluation_points,
        subsets_u,
        &mut out,
    )?;

    // Set empty subset column directly to expected value (constant across all evaluation points)
    if let Some(idx) = empty_idx {
        let mut empty_col = out.column_mut(idx);
        empty_col.fill(augmented_tree.expected_value);
    }

    Ok(out)
}

/// Aggregates v_U(x) across multiple augmented trees.
///
/// This is a pure function that computes the sum of v_U(x) contributions
/// from all trees in the ensemble. It handles both sequential and parallel
/// execution based on the provided `ParallelSettings`.
///
/// # Arguments
/// * `augmented_trees` - Slice of augmented trees to aggregate
/// * `evaluation_points` - Evaluation points, shape: [n_eval, n_features]
/// * `subsets_u` - Vector of feature subsets U to evaluate
/// * `parallel` - Parallelization settings
///
/// # Returns
/// Matrix of shape [n_eval, n_subsets] containing aggregated v_U(x) values
///
/// # Errors
/// Returns `FastPDError` if evaluation fails
pub fn aggregate_v_u_across_trees<T: TreeModel>(
    augmented_trees: &[AugmentedTree<T>],
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
    parallel: ParallelSettings,
) -> Result<Array2<f32>, FastPDError> {
    let n_eval = evaluation_points.nrows();
    let n_subsets_u = subsets_u.len();

    if !parallel.is_parallel() {
        // Sequential path: no Rayon anywhere
        let mut mat_u = Array2::<f32>::zeros((n_eval, n_subsets_u));
        for aug_tree in augmented_trees {
            let tree_mat =
                evaluate_pd_batch_for_subsets_seq(aug_tree, evaluation_points, subsets_u)?;
            mat_u = &mat_u + &tree_mat;
        }
        Ok(mat_u)
    } else {
        // Parallel path: create thread pool and parallelize across trees
        let n_threads = parallel.n_threads;
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| FastPDError::ThreadPoolError(e.to_string()))?;

        pool.install(|| {
            augmented_trees
                .par_iter()
                .map(|aug_tree| {
                    evaluate_pd_batch_for_subsets_rayon(aug_tree, evaluation_points, subsets_u)
                })
                .try_reduce(
                    || Array2::<f32>::zeros((n_eval, n_subsets_u)),
                    |mut a, b| {
                        a = &a + &b;
                        Ok(a)
                    },
                )
        })
    }
}

/// Recursive helper function for batch evaluation.
///
/// This function mirrors the Rcpp `recurseMarginalizeSBitmask` implementation,
/// processing all evaluation points and all subsets in a single traversal.
fn evaluate_batch_recursive<T, J>(
    augmented_tree: &AugmentedTree<T>,
    node_id: usize,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
    out: &mut Array2<f32>,
) -> Result<(), FastPDError>
where
    T: TreeModel,
    J: Joiner,
{
    let tree = &augmented_tree.tree;

    if tree.is_leaf(node_id) {
        let path_features = augmented_tree
            .path_features
            .get(&node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing path features".into()))?;

        let n_eval = evaluation_points.nrows();
        let mut scratch_subset = FeatureSubset::empty();

        for (j, u) in subsets_u.iter().enumerate() {
            // U_j = U ∩ T_j
            u.intersect_with_into(path_features, &mut scratch_subset);

            let expectation = augmented_tree
                .precomputed_expectation(node_id, &scratch_subset)
                .ok_or(FastPDError::MissingObservationSet)?;

            let mut col = out.column_mut(j);
            for i in 0..n_eval {
                col[i] += expectation;
            }
        }
        return Ok(());
    }

    // Internal node: recurse on children, then merge
    let feature = tree
        .node_feature(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Internal node missing feature".into()))?;
    let threshold = tree
        .node_threshold(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Internal node missing threshold".into()))?;
    let left_child = tree
        .left_child(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Missing left child".into()))?;
    let right_child = tree
        .right_child(node_id)
        .ok_or_else(|| FastPDError::InvalidTree("Missing right child".into()))?;

    if feature >= evaluation_points.ncols() {
        return Err(FastPDError::InvalidFeature(
            feature,
            evaluation_points.ncols(),
        ));
    }

    let n_eval = evaluation_points.nrows();
    let n_subsets = subsets_u.len();
    let mut mat_yes = Array2::<f32>::zeros((n_eval, n_subsets));
    let mut mat_no = Array2::<f32>::zeros((n_eval, n_subsets));

    // Use Joiner trait to parallelize recursive calls to both children
    // The child evaluations are independent; only the combination logic differs afterward
    let (left_result, right_result) = J::join(
        || {
            evaluate_batch_recursive::<T, J>(
                augmented_tree,
                left_child,
                evaluation_points,
                subsets_u,
                &mut mat_yes,
            )
        },
        || {
            evaluate_batch_recursive::<T, J>(
                augmented_tree,
                right_child,
                evaluation_points,
                subsets_u,
                &mut mat_no,
            )
        },
    );
    left_result?;
    right_result?;

    let feature_col = evaluation_points.column(feature);
    for (j, u) in subsets_u.iter().enumerate() {
        let feature_in_u = u.contains(feature);
        let col_yes = mat_yes.column(j);
        let col_no = mat_no.column(j);
        let mut col_out = out.column_mut(j);

        if !feature_in_u {
            for i in 0..n_eval {
                col_out[i] += col_yes[i] + col_no[i];
            }
        } else if T::COMPARISON {
            for i in 0..n_eval {
                col_out[i] += if feature_col[i] < threshold {
                    col_yes[i]
                } else {
                    col_no[i]
                };
            }
        } else {
            for i in 0..n_eval {
                col_out[i] += if feature_col[i] <= threshold {
                    col_yes[i]
                } else {
                    col_no[i]
                };
            }
        }
    }

    Ok(())
}

/// Given v_U(x) = E[m(X) | X_U = x_U] for all U in `subsets_u`,
/// extract the plain partial dependence surface v_S(x_S) by
/// selecting the column corresponding to U = S (when present).
///
/// # Arguments
/// * `mat_u` - Matrix of shape [n_eval, n_subsets_u] containing v_U(x) values
/// * `subsets_u` - Vector of feature subsets U corresponding to columns
/// * `s` - Target feature subset S
///
/// # Returns
/// Vector of v_S(x_S) values, or None if S is not in subsets_u
pub fn pd_from_u_matrix(
    mat_u: &Array2<f32>,
    subsets_u: &[FeatureSubset],
    s: &FeatureSubset,
) -> Option<Array1<f32>> {
    subsets_u
        .iter()
        .position(|u| u == s)
        .map(|idx| mat_u.column(idx).to_owned())
}

/// Given v_U(x) = E[m(X) | X_U = x_U] for all U in `subsets_u`,
/// compute the functional component f_S(x_S) for a single target
/// subset S via inclusion–exclusion, as in the GLEX ANOVA decomposition:
///
///   f_S(x_S) = sum_{V ⊆ S} (-1)^{|S|-|V|} E[m(X) | X_V = x_V]
///
/// # Arguments
/// * `mat_u` - Matrix of shape [n_eval, n_subsets_u] containing v_U(x) values
/// * `subsets_u` - Vector of feature subsets U corresponding to columns
/// * `s` - Target feature subset S
///
/// # Returns
/// Vector of f_S(x_S) values
pub fn functional_component_from_u_matrix(
    mat_u: &Array2<f32>,
    subsets_u: &[FeatureSubset],
    s: &FeatureSubset,
) -> Array1<f32> {
    let n_eval = mat_u.nrows();
    let mut result = Array1::<f32>::zeros(n_eval);

    // Build lookup map for efficiency (avoids O(|U|) search per V)
    let u_to_idx: rustc_hash::FxHashMap<FeatureSubset, usize> = subsets_u
        .iter()
        .enumerate()
        .map(|(i, u)| (u.clone(), i))
        .collect();

    // Enumerate all V ⊆ S and combine v_V(x) with ±1 coefficients
    for v in s.all_subsets() {
        if let Some(&idx) = u_to_idx.get(&v) {
            let sign = if (s.len() - v.len()).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let col_v = mat_u.column(idx);
            for i in 0..n_eval {
                result[i] += sign * col_v[i];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastpd::augment::augment_tree;
    use crate::fastpd::tree::{Tree, TreeNode};

    // Helper: Create a simple 2-level tree for testing
    fn create_simple_tree() -> Tree {
        let nodes = vec![
            TreeNode {
                internal_idx: 0,
                feature: Some(0),
                threshold: Some(0.5),
                left: Some(1),
                right: Some(2),
                missing: Some(1),
                leaf_value: None,
            },
            TreeNode {
                internal_idx: 1,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(1.0),
            },
            TreeNode {
                internal_idx: 2,
                feature: None,
                threshold: None,
                left: None,
                right: None,
                missing: None,
                leaf_value: Some(2.0),
            },
        ];
        Tree::new(nodes, 0)
    }

    #[test]
    fn test_evaluate_pd_simple_tree() {
        let tree = create_simple_tree();

        // Create background data: 4 samples
        // Sample 0: [0.3] -> goes to left (leaf 1, value 1.0)
        // Sample 1: [0.7] -> goes to right (leaf 2, value 2.0)
        // Sample 2: [0.2] -> goes to left (leaf 1, value 1.0)
        // Sample 3: [0.8] -> goes to right (leaf 2, value 2.0)
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();

        // Augment the tree
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        // Evaluate PD for S = {0} at points [0.3] and [0.7]
        // Since 0.3 <= 0.5, we go to left child (leaf 1) -> value = 1.0
        // Since 0.7 > 0.5, we go to right child (leaf 2) -> value = 2.0
        let points = ndarray::arr2(&[[0.3], [0.7]]);
        let subset = FeatureSubset::from_slice(&[0]);
        let subsets_u = vec![subset];
        let mat = evaluate_pd_batch_for_subsets_seq(&aug_tree, &points.view(), &subsets_u).unwrap();
        assert!((mat[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((mat[[1, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_pd_empty_subset() {
        let tree = create_simple_tree();
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        // Evaluate PD for S = {} (empty subset)
        // This means we sum contributions from all leaves
        // Leaf 1: 1.0 * (2/4) = 0.5
        // Leaf 2: 2.0 * (2/4) = 1.0
        // Total: 1.5
        let point = ndarray::arr2(&[[0.5]]);
        let subset = FeatureSubset::empty();
        let subsets_u = vec![subset];
        let mat = evaluate_pd_batch_for_subsets_seq(&aug_tree, &point.view(), &subsets_u).unwrap();
        assert!((mat[[0, 0]] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_pd_matches_empirical() {
        // Test that FastPD evaluation matches the empirical PD estimator:
        // v_S(x_S) = (1/n_b) * sum_{i=1}^{n_b} m(x_S, X^{(i)}_{\overline S})

        let tree = create_simple_tree();

        // Hardcoded background samples (generated with np.random.seed(42))
        let background = ndarray::arr2(&[
            [3.745401, 9.507143],
            [7.319939, 5.986585],
            [1.560186, 1.559945],
            [0.580836, 8.661761],
            [6.011_15, 7.080726],
            [0.205845, 9.699099],
            [8.324426, 2.123391],
            [1.818_25, 1.834045],
            [3.042422, 5.247564],
            [4.319_45, 2.912291],
        ]);
        let background_view = background.view();
        let n_b = background.nrows();

        // Augment the tree
        let aug_tree = augment_tree(tree.clone(), &background_view).unwrap();

        // Test case 1: S = {0}, x = [0.3, 0.0] (x_1 doesn't matter for S={0})
        // Empirical PD: (1/10) * sum_{i=1}^{10} m([0.3, X^{(i)}_1])
        let evaluation_point = ndarray::arr2(&[[0.3, 0.0]]);
        let subset = FeatureSubset::from_slice(&[0]);
        let subsets_u = vec![subset.clone()];

        // Compute FastPD estimate
        let mat =
            evaluate_pd_batch_for_subsets_seq(&aug_tree, &evaluation_point.view(), &subsets_u)
                .unwrap();
        let fastpd_value = mat[[0, 0]];

        // Compute empirical PD manually
        let mut empirical_sum = 0.0;
        for i in 0..n_b {
            // Create synthetic sample: [x_0, X^{(i)}_1]
            let synthetic = [evaluation_point[[0, 0]], background[[i, 1]]];
            // Traverse tree to get prediction
            let prediction = tree.predict(&synthetic);
            empirical_sum += prediction;
        }
        let empirical_value = empirical_sum / n_b as f32;

        // They should match (within floating point tolerance)
        assert!(
            (fastpd_value - empirical_value).abs() < 1e-10,
            "FastPD value {} does not match empirical PD {}",
            fastpd_value,
            empirical_value
        );

        // Test case 2: S = {} (empty subset), x = [0.5, 0.0]
        // Empirical PD: (1/10) * sum_{i=1}^{10} m([X^{(i)}_0, X^{(i)}_1]) = average of all predictions
        let evaluation_point2 = ndarray::arr2(&[[0.5, 0.0]]);
        let subset2 = FeatureSubset::empty();
        let subsets_u2 = vec![subset2.clone()];

        let mat2 =
            evaluate_pd_batch_for_subsets_seq(&aug_tree, &evaluation_point2.view(), &subsets_u2)
                .unwrap();
        let fastpd_value2 = mat2[[0, 0]];

        let mut empirical_sum2 = 0.0;
        for i in 0..n_b {
            // For S = {}, we use the full background sample
            let synthetic = [background[[i, 0]], background[[i, 1]]];
            let prediction = tree.predict(&synthetic);
            empirical_sum2 += prediction;
        }
        let empirical_value2 = empirical_sum2 / n_b as f32;

        assert!(
            (fastpd_value2 - empirical_value2).abs() < 1e-10,
            "FastPD value {} does not match empirical PD {} for empty subset",
            fastpd_value2,
            empirical_value2
        );
    }
}
