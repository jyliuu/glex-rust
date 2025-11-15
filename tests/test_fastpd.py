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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
