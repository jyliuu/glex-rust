use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::fastpd::error::FastPDError;
use crate::fastpd::parallel::{Joiner, RayonJoin, SeqJoin};
use crate::fastpd::tree::TreeModel;
use crate::fastpd::types::FeatureSubset;

use super::augmented_tree::AugmentedTree;
use super::cache::PDCache;

/// Evaluates the partial dependence function v_S(x_S) sequentially.
///
/// This is a thin wrapper around `evaluate_recursive::<T, SeqJoin>`.
pub fn evaluate_pd_function_seq<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_point: &ArrayView1<f32>,
    feature_subset: &FeatureSubset,
    cache: &mut PDCache,
) -> Result<f32, FastPDError> {
    evaluate_pd_function_impl::<T, SeqJoin>(augmented_tree, evaluation_point, feature_subset, cache)
}

/// Evaluates the partial dependence function v_S(x_S) in parallel.
///
/// This is a thin wrapper around `evaluate_recursive::<T, RayonJoin>`.
/// **Important**: This function must be called from within a Rayon thread pool
/// (via `ThreadPool::install()`) for proper parallel execution.
pub fn evaluate_pd_function_rayon<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_point: &ArrayView1<f32>,
    feature_subset: &FeatureSubset,
    cache: &mut PDCache,
) -> Result<f32, FastPDError> {
    evaluate_pd_function_impl::<T, RayonJoin>(
        augmented_tree,
        evaluation_point,
        feature_subset,
        cache,
    )
}

/// Legacy function: delegates to `evaluate_pd_function_seq` for backward compatibility.
pub fn evaluate_pd_function<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_point: &ArrayView1<f32>,
    feature_subset: &FeatureSubset,
    cache: &mut PDCache,
) -> Result<f32, FastPDError> {
    evaluate_pd_function_seq(augmented_tree, evaluation_point, feature_subset, cache)
}

/// Internal implementation of PD evaluation using the Joiner trait.
fn evaluate_pd_function_impl<T, J>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_point: &ArrayView1<f32>,
    feature_subset: &FeatureSubset,
    cache: &mut PDCache,
) -> Result<f32, FastPDError>
where
    T: TreeModel,
    J: Joiner,
{
    // Note: We validate the evaluation point dimension during tree traversal
    // when we check if feature indices are within bounds

    // Use provided FeatureSubset directly; callers are responsible for constructing it
    let subset_s = feature_subset;

    // Check cache first for S
    if let Some(cached) = cache.get(evaluation_point, subset_s) {
        return Ok(cached);
    }

    // Compute U = S ∩ (union of all T_j) using the precomputed union of
    // path features stored in the augmented tree.
    let u = subset_s.intersect_with(&augmented_tree.all_tree_features);

    // Check cache for U (different S may map to same U)
    if let Some(cached) = cache.get(evaluation_point, &u) {
        // Cache the result for S as well
        cache.insert(evaluation_point, subset_s.clone(), cached);
        return Ok(cached);
    }

    // Traverse tree to compute v_U(x_U)
    let tree = &augmented_tree.tree;
    // Scratch subset reused across recursive calls to avoid per-leaf allocations
    let mut scratch_subset = FeatureSubset::empty();
    let value = evaluate_recursive::<T, J>(
        tree,
        tree.root(),
        evaluation_point,
        &u,
        augmented_tree,
        &mut scratch_subset,
    )?;

    // Cache result for both U and S
    cache.insert(evaluation_point, u, value);
    cache.insert(evaluation_point, subset_s.clone(), value);

    Ok(value)
}

/// Recursive helper function for tree evaluation (implements function G from Algorithm 2).
///
/// This function traverses the tree and computes the partial dependence contribution
/// at each node, summing contributions from leaves where x_U would land.
fn evaluate_recursive<T, J>(
    tree: &T,
    node_id: usize,
    point: &ArrayView1<f32>,
    feature_subset: &FeatureSubset,
    augmented_tree: &AugmentedTree<T>,
    scratch_subset: &mut FeatureSubset,
) -> Result<f32, FastPDError>
where
    T: TreeModel,
    J: Joiner,
{
    if tree.is_leaf(node_id) {
        // Get U_j = U ∩ T_j
        let path_features = augmented_tree
            .path_features
            .get(&node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing path features".into()))?;
        // Compute U_j into a reusable scratch subset to avoid per-leaf allocations
        feature_subset.intersect_with_into(path_features, scratch_subset);

        // Get precomputed probability for U_j
        let prob = augmented_tree
            .precomputed_prob(node_id, scratch_subset)
            .ok_or(FastPDError::MissingObservationSet)?;

        // Get leaf value
        let leaf_value = tree
            .leaf_value(node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing value".into()))?;

        return Ok(leaf_value * prob);
    }

    // Get node information
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

    // Validate feature index
    if feature >= point.len() {
        return Err(FastPDError::InvalidFeature(feature, point.len()));
    }

    if feature_subset.contains(feature) {
        // d_j ∈ U: follow path based on x[d_j]
        let feature_value = point[feature];
        let go_left = if T::COMPARISON {
            feature_value < threshold
        } else {
            feature_value <= threshold
        };
        if go_left {
            evaluate_recursive::<T, J>(
                tree,
                left_child,
                point,
                feature_subset,
                augmented_tree,
                scratch_subset,
            )
        } else {
            evaluate_recursive::<T, J>(
                tree,
                right_child,
                point,
                feature_subset,
                augmented_tree,
                scratch_subset,
            )
        }
    } else {
        // d_j ∉ U: sum contributions from both children using Joiner trait
        // Note: We need separate scratch subsets for left and right to avoid conflicts
        let mut scratch_left = FeatureSubset::empty();
        let mut scratch_right = FeatureSubset::empty();
        let (left_val, right_val) = J::join(
            || {
                evaluate_recursive::<T, J>(
                    tree,
                    left_child,
                    point,
                    feature_subset,
                    augmented_tree,
                    &mut scratch_left,
                )
            },
            || {
                evaluate_recursive::<T, J>(
                    tree,
                    right_child,
                    point,
                    feature_subset,
                    augmented_tree,
                    &mut scratch_right,
                )
            },
        );
        Ok(left_val? + right_val?)
    }
}

/// Evaluate v_U(x_U) for multiple subsets U and multiple evaluation points
/// in one pass through a single augmented tree.
///
/// This function implements the Rcpp `recurseMarginalizeSBitmask` pattern,
/// evaluating all evaluation points and all feature subsets in a single tree traversal.
///
/// # Arguments
/// * `augmented_tree` - The augmented tree to evaluate
/// * `evaluation_points` - Evaluation points, shape: [n_eval, n_features]
/// * `subsets_u` - Vector of feature subsets U to evaluate
///
/// # Returns
/// Matrix of shape [n_eval, n_subsets] where entry (i, j) is v_{U[j]}(x[i])
pub fn evaluate_pd_batch_for_subsets<T: TreeModel>(
    augmented_tree: &AugmentedTree<T>,
    evaluation_points: &ArrayView2<f32>,
    subsets_u: &[FeatureSubset],
) -> Result<Array2<f32>, FastPDError> {
    let n_eval = evaluation_points.nrows();
    let n_subsets = subsets_u.len();
    let mut out = Array2::<f32>::zeros((n_eval, n_subsets));
    evaluate_batch_recursive::<T, SeqJoin>(
        augmented_tree,
        augmented_tree.tree.root(),
        evaluation_points,
        subsets_u,
        &mut out,
    )?;
    Ok(out)
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
        let leaf_value = tree
            .leaf_value(node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing value".into()))?;

        let path_features = augmented_tree
            .path_features
            .get(&node_id)
            .ok_or_else(|| FastPDError::InvalidTree("Leaf missing path features".into()))?;

        let n_eval = evaluation_points.nrows();
        let mut scratch_subset = FeatureSubset::empty();

        for (j, u) in subsets_u.iter().enumerate() {
            // U_j = U ∩ T_j
            u.intersect_with_into(path_features, &mut scratch_subset);

            let prob = augmented_tree
                .precomputed_prob(node_id, &scratch_subset)
                .ok_or(FastPDError::MissingObservationSet)?;

            let val = leaf_value * prob;
            let mut col = out.column_mut(j);
            for i in 0..n_eval {
                col[i] += val;
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

    evaluate_batch_recursive::<T, J>(
        augmented_tree,
        left_child,
        evaluation_points,
        subsets_u,
        &mut mat_yes,
    )?;
    evaluate_batch_recursive::<T, J>(
        augmented_tree,
        right_child,
        evaluation_points,
        subsets_u,
        &mut mat_no,
    )?;

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
        } else {
            if T::COMPARISON {
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
    if let Some(idx) = subsets_u.iter().position(|u| u == s) {
        Some(mat_u.column(idx).to_owned())
    } else {
        None
    }
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
            let sign = if (s.len() - v.len()) % 2 == 0 {
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
    use ndarray::arr1;

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

        // Evaluate PD for S = {0} at point x = [0.3]
        // Since 0.3 <= 0.5, we go to left child (leaf 1)
        // U = {0} ∩ {0} = {0}
        // At leaf 1, U_j = {0} ∩ {0} = {0}
        // We use D_{{0}} from P_1, which contains all 4 samples (when feature 0 is in S, all samples go to both children)
        // So prob = 4/4 = 1.0
        // Value = 1.0 * 1.0 = 1.0
        let mut cache = PDCache::new();
        let point = arr1(&[0.3]);
        let subset = FeatureSubset::from_slice(&[0]);
        let value = evaluate_pd_function(&aug_tree, &point.view(), &subset, &mut cache).unwrap();
        assert!((value - 1.0).abs() < 1e-10);

        // Evaluate PD for S = {0} at point x = [0.7]
        // Since 0.7 > 0.5, we go to right child (leaf 2)
        // U = {0} ∩ {0} = {0}
        // At leaf 2, U_j = {0} ∩ {0} = {0}
        // We use D_{{0}} from P_2, which contains all 4 samples
        // So prob = 4/4 = 1.0
        // Value = 2.0 * 1.0 = 2.0
        let point2 = arr1(&[0.7]);
        let value2 = evaluate_pd_function(&aug_tree, &point2.view(), &subset, &mut cache).unwrap();
        assert!((value2 - 2.0).abs() < 1e-10);
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
        let mut cache = PDCache::new();
        let point = arr1(&[0.5]);
        let subset = FeatureSubset::empty();
        let value = evaluate_pd_function(&aug_tree, &point.view(), &subset, &mut cache).unwrap();
        assert!((value - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_pd_caching() {
        let tree = create_simple_tree();
        let background = ndarray::arr2(&[[0.3], [0.7], [0.2], [0.8]]);
        let background_view = background.view();
        let aug_tree = augment_tree(tree, &background_view).unwrap();

        let mut cache = PDCache::new();
        let point = arr1(&[0.3]);

        // First evaluation
        let subset1 = FeatureSubset::from_slice(&[0]);
        let value1 = evaluate_pd_function(&aug_tree, &point.view(), &subset1, &mut cache).unwrap();
        // When S = {0}, U = {0} ∩ {0} = {0}, so U = S
        // We cache for both U and S, but they're the same key, so cache has 1 entry
        assert!(!cache.is_empty());

        // Second evaluation with same S should use cache
        let value2 = evaluate_pd_function(&aug_tree, &point.view(), &subset1, &mut cache).unwrap();
        assert_eq!(value1, value2);

        // Test with different S that maps to same U
        // S = {0, 99} where 99 is not in tree, so U = {0} ∩ {0} = {0}
        let subset2 = FeatureSubset::from_slice(&[0, 99]);
        let value3 = evaluate_pd_function(&aug_tree, &point.view(), &subset2, &mut cache).unwrap();
        // Should use cached value for U = {0}
        assert_eq!(value1, value3);
        // Cache should now have 2 entries: one for {0} and one for {0, 99}
        assert!(cache.len() >= 2);
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
        let evaluation_point = arr1(&[0.3, 0.0]);
        let subset = FeatureSubset::from_slice(&[0]);

        // Compute FastPD estimate
        let mut cache = PDCache::new();
        let fastpd_value =
            evaluate_pd_function(&aug_tree, &evaluation_point.view(), &subset, &mut cache).unwrap();

        // Compute empirical PD manually
        let mut empirical_sum = 0.0;
        for i in 0..n_b {
            // Create synthetic sample: [x_0, X^{(i)}_1]
            let synthetic = [evaluation_point[0], background[[i, 1]]];
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
        let evaluation_point2 = arr1(&[0.5, 0.0]);
        let subset2 = FeatureSubset::empty();

        let fastpd_value2 =
            evaluate_pd_function(&aug_tree, &evaluation_point2.view(), &subset2, &mut cache)
                .unwrap();

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
