"""
Tests for FastPD functionality with XGBoost models.
"""

import glex_rust
import numpy as np
import pytest

import xgboost as xgb


def test_fastpd_empirical_consistency():
    """Test that FastPD matches empirical PD computation for simple cases.

    Empirical PD is computed as the average of model predictions over background samples.
    FastPD should match this exactly (within floating point precision).
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 1.0

    # Use more trees and deeper trees to ensure good fit
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Test point
    eval_point = np.array([[5.0, 3.0]])

    # Compute PD for feature 0 using FastPD
    pd_fastpd = fastpd.pd_function(evaluation_points=eval_point, feature_subset=[0])[0]

    # Compute empirical PD manually: average of model([5.0, X[i, 1]]) for all i
    # Now that precision is fixed, we can use model.predict() directly
    empirical_pd = np.mean(
        [model.predict(np.array([[5.0, X[i, 1]]]))[0] for i in range(n_samples)]
    )

    # FastPD should match empirical PD within reasonable tolerance
    # For f32 precision, we expect differences < 1e-5
    diff = abs(pd_fastpd - empirical_pd)
    assert diff < 1e-5, (
        f"FastPD ({pd_fastpd:.10f}) should match empirical PD ({empirical_pd:.10f}), "
        f"but difference is {diff:.10e}"
    )


def test_fastpd_single_tree_empirical_consistency():
    """Check empirical consistency for a *single* XGBoost tree.

    This is closer in spirit to the Rust unit test
    `fastpd::evaluate::tests::test_evaluate_pd_matches_empirical`, but uses a
    real XGBoost tree extracted via the Python bridge.

    For a single tree, FastPD should match empirical PD very closely.
    """
    np.random.seed(32)
    n_samples = 200
    n_features = 2

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 1.0

    # Force a single tree so we can compare against empirical PD more directly.
    model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=2,
        learning_rate=1.0,
        random_state=42,
    )
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Test point
    eval_point = np.array([[5.0, 3.0]])

    # Compute PD for feature 0 using FastPD
    pd_fastpd = fastpd.pd_function(evaluation_points=eval_point, feature_subset=[0])[0]

    # Compute empirical PD manually: average of model([5.0, X[i, 1]]) for all i
    # Now that precision is fixed, we can use model.predict() directly
    empirical_pd = np.mean(
        [model.predict(np.array([[5.0, X[i, 1]]]))[0] for i in range(n_samples)]
    )

    # FastPD should match empirical PD within reasonable tolerance
    # For f32 precision with single tree, we expect differences < 1e-5
    diff = abs(pd_fastpd - empirical_pd)
    assert diff < 1e-5, (
        f"FastPD ({pd_fastpd:.10f}) should match empirical PD ({empirical_pd:.10f}), "
        f"but difference is {diff:.10e}"
    )


def test_fastpd_batch_evaluation():
    """Test batch evaluation with multiple points."""
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 1.0

    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Create multiple evaluation points
    n_eval = 5
    eval_points = np.random.rand(n_eval, n_features) * 10

    # Compute PD for all points at once
    pd_values = fastpd.pd_function(evaluation_points=eval_points, feature_subset=[0])

    assert pd_values.shape == (n_eval,)

    # Compute PD for each point individually and compare
    for i in range(n_eval):
        pd_single = fastpd.pd_function(
            evaluation_points=eval_points[i : i + 1], feature_subset=[0]
        )[0]
        assert abs(pd_values[i] - pd_single) < 1e-10, (
            f"Batch result {pd_values[i]} should match single result {pd_single}"
        )


def test_pd_functions_up_to_order():
    """Test that pd_functions_up_to_order matches individual pd_function calls."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + 1.0

    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Create multiple evaluation points
    n_eval = 5
    eval_points = np.random.rand(n_eval, n_features) * 10

    # Test with max_order = 2 (should include {0}, {1}, {2}, {0,1}, {0,2}, {1,2})
    max_order = 2
    pd_values_batch, subsets_batch = fastpd.pd_functions_up_to_order(
        evaluation_points=eval_points, max_order=max_order
    )
    # Verify shape
    assert pd_values_batch.shape[0] == n_eval, "Number of rows should match n_eval"
    assert pd_values_batch.shape[1] == len(subsets_batch), (
        "Number of columns should match number of subsets"
    )

    # Verify we got the expected subsets (order 1 and 2)
    subset_sizes = [len(s) for s in subsets_batch]
    assert all(1 <= size <= max_order for size in subset_sizes), (
        "All subsets should have size <= max_order"
    )
    assert all(size > 0 for size in subset_sizes), "No empty subsets should be included"

    # Compare with individual pd_function calls
    for subset_idx, subset in enumerate(subsets_batch):
        for point_idx in range(n_eval):
            # Call pd_function for this single point and subset
            pd_single = fastpd.pd_function(
                evaluation_points=eval_points[point_idx : point_idx + 1],
                feature_subset=subset,
            )[0]

            # Compare with batch result
            pd_batch = pd_values_batch[point_idx, subset_idx]
            diff = abs(pd_batch - pd_single)
            assert diff < 1e-5, (
                f"Mismatch for subset {subset} at point {point_idx}: "
                f"batch={pd_batch:.10f}, individual={pd_single:.10f}, diff={diff:.10e}"
            )


def test_functional_decomp_up_to_order():
    """Test that functional_decomp_up_to_order works correctly."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 3

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + 1.0

    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Create multiple evaluation points
    eval_points = X
    n_eval = eval_points.shape[0]

    # Test with max_order = n_features to get complete decomposition
    # For functional decomposition to sum to predictions, we need ALL subsets
    max_order = n_features
    comp_values, subsets = fastpd.functional_decomp_up_to_order(
        evaluation_points=eval_points, max_order=max_order
    )

    # Verify shape
    assert comp_values.shape[0] == n_eval, "Number of rows should match n_eval"
    assert comp_values.shape[1] == len(subsets), (
        "Number of columns should match number of subsets"
    )

    # Verify we got the expected subsets
    subset_sizes = [len(s) for s in subsets]
    assert all(0 <= size <= max_order for size in subset_sizes), (
        "All subsets should have size <= max_order"
    )

    # For functional decomposition, the sum of all components should equal the prediction
    # This includes f_∅ (empty subset) which contains the expected value/intercept
    predictions = fastpd.predict(evaluation_points=eval_points)
    for point_idx in range(n_eval):
        sum_components = np.sum(comp_values[point_idx, :])
        prediction = predictions[point_idx]
        assert np.isclose(
            sum_components,
            prediction,
            rtol=1e-5,
            atol=1e-6,
        ), (
            f"Mismatch at point {point_idx}: sum of components={sum_components:.10f}, "
            f"prediction={prediction:.10f}, diff={abs(sum_components - prediction):.10e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
