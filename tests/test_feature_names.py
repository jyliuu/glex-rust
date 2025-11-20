"""
Test feature name extraction from various sources.
"""

import glex_rust
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

import xgboost as xgb


def test_feature_names_from_sklearn_bunch():
    """Test that feature names are extracted from sklearn Bunch object."""
    # Load California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Fit XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    # Create FastPD instance - should extract feature_names from model or Bunch
    # Note: We pass the Bunch object so it can extract feature_names from data.feature_names
    # The implementation will extract .data from the Bunch object
    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=data,  # Pass Bunch object, not just X
        n_threads=1,
    )

    # Check that feature names were extracted
    feature_names = fastpd.feature_names()
    assert feature_names is not None, (
        "Feature names should be extracted from sklearn Bunch"
    )
    assert len(feature_names) == X.shape[1], (
        f"Expected {X.shape[1]} feature names, got {len(feature_names)}"
    )

    # Check that feature names match expected California Housing feature names
    expected_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    assert feature_names == expected_names, (
        f"Expected {expected_names}, got {feature_names}"
    )

    # Test get_feature_name() method
    assert fastpd.get_feature_name(0) == "MedInc"
    assert fastpd.get_feature_name(1) == "HouseAge"
    assert fastpd.get_feature_name(7) == "Longitude"

    # Test that background_samples are stored
    background_stored = fastpd.get_background_samples()
    assert background_stored.shape == X.shape, (
        "Background samples should be stored with correct shape"
    )


def test_feature_names_explicit():
    """Test that explicitly provided feature names are used."""
    # Generate simple data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    # Fit XGBoost model
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create FastPD with explicit feature names
    custom_names = ["feature_A", "feature_B", "feature_C"]
    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=X,
        feature_names=custom_names,
        n_threads=1,
    )

    # Check that explicit feature names are used
    feature_names = fastpd.feature_names()
    assert feature_names == custom_names, (
        f"Expected {custom_names}, got {feature_names}"
    )

    # Test get_feature_name()
    assert fastpd.get_feature_name(0) == "feature_A"
    assert fastpd.get_feature_name(1) == "feature_B"
    assert fastpd.get_feature_name(2) == "feature_C"


def test_feature_names_defaults():
    """Test that default feature names are used when none are available."""
    # Generate simple data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    # Fit XGBoost model
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create FastPD without feature names (should use defaults)
    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=X,
        n_threads=1,
    )

    # Check that default feature names are used
    feature_names = fastpd.feature_names()
    if feature_names is None:
        # If None, get_feature_name should still work with defaults
        assert fastpd.get_feature_name(0) == "f0"
        assert fastpd.get_feature_name(1) == "f1"
        assert fastpd.get_feature_name(2) == "f2"
    else:
        # Or it might have extracted from model
        assert len(feature_names) == 3


def test_get_feature_name_out_of_bounds():
    """Test get_feature_name() with out-of-bounds index."""
    # Generate simple data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.random.rand(100)

    # Fit XGBoost model
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create FastPD
    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=X,
        n_threads=1,
    )

    # Should return default name for out-of-bounds index
    assert fastpd.get_feature_name(10) == "Feature 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
