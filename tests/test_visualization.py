"""
Test visualization functions for functional decomposition components.
"""

import glex_rust
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

import xgboost as xgb

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def simple_fastpd():
    """Create a simple FastPD instance for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 3) * 10
    y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=X,
        feature_names=["feature_A", "feature_B", "feature_C"],
        n_threads=1,
    )
    return fastpd


@pytest.fixture
def california_housing_fastpd():
    """Create FastPD instance with California Housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)

    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=data,  # Pass Bunch to extract feature_names
        n_threads=1,
    )
    return fastpd


class TestPlot1DComponents:
    """Test plot_1d_components method."""

    def test_plot_all_1d_components(self, simple_fastpd):
        """Test plotting all 1D components (default behavior)."""
        fig = simple_fastpd.plot_1d_components()
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_single_feature_by_index(self, simple_fastpd):
        """Test plotting single feature by index."""
        fig = simple_fastpd.plot_1d_components(features=0)
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_single_feature_by_name(self, simple_fastpd):
        """Test plotting single feature by name."""
        fig = simple_fastpd.plot_1d_components(features="feature_A")
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_multiple_features_by_index(self, simple_fastpd):
        """Test plotting multiple features by index."""
        fig = simple_fastpd.plot_1d_components(features=[0, 1])
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_multiple_features_by_name(self, simple_fastpd):
        """Test plotting multiple features by name."""
        fig = simple_fastpd.plot_1d_components(features=["feature_A", "feature_B"])
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_with_custom_evaluation_points(self, simple_fastpd):
        """Test plotting with custom evaluation points."""
        # Create custom evaluation points
        n_points = 50
        eval_points = np.random.rand(n_points, 3) * 10
        fig = simple_fastpd.plot_1d_components(
            features=0, evaluation_points=eval_points
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_figsize(self, simple_fastpd):
        """Test plotting with custom figure size."""
        fig = simple_fastpd.plot_1d_components(figsize=(10, 8))
        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)

    def test_plot_with_custom_n_cols(self, simple_fastpd):
        """Test plotting with custom number of columns."""
        fig = simple_fastpd.plot_1d_components(n_cols=2)
        assert fig is not None
        plt.close(fig)

    def test_plot_california_housing_all_features(self, california_housing_fastpd):
        """Test plotting all 1D components for California Housing dataset."""
        fig = california_housing_fastpd.plot_1d_components()
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_california_housing_single_feature(self, california_housing_fastpd):
        """Test plotting single feature for California Housing dataset."""
        fig = california_housing_fastpd.plot_1d_components(features="MedInc")
        assert fig is not None
        plt.close(fig)

    def test_plot_invalid_feature_name(self, simple_fastpd):
        """Test that invalid feature name raises error."""
        with pytest.raises(ValueError, match="not found"):
            simple_fastpd.plot_1d_components(features="invalid_feature")

    def test_plot_invalid_feature_index(self, simple_fastpd):
        """Test that invalid feature index raises error."""
        with pytest.raises(IndexError):
            simple_fastpd.plot_1d_components(features=100)

    def test_plot_caching(self, simple_fastpd):
        """Test that components are cached after first plot."""
        # First plot should compute and cache
        fig1 = simple_fastpd.plot_1d_components(features=0)
        plt.close(fig1)

        # Check that cache exists
        cached = simple_fastpd.get_cached_components()
        assert cached is not None
        comp_values, subsets, eval_points, max_order = cached
        assert max_order >= 1

        # Second plot should use cache (no error means it worked)
        fig2 = simple_fastpd.plot_1d_components(features=0)
        plt.close(fig2)


class TestPlot2DInteractions:
    """Test plot_2d_interactions method."""

    def test_plot_all_2d_interactions(self, simple_fastpd):
        """Test plotting all 2D interactions (default behavior)."""
        fig = simple_fastpd.plot_2d_interactions()
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_single_interaction_by_index(self, simple_fastpd):
        """Test plotting single interaction pair by indices."""
        fig = simple_fastpd.plot_2d_interactions(features=(0, 1))
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_single_interaction_by_name(self, simple_fastpd):
        """Test plotting single interaction pair by names."""
        fig = simple_fastpd.plot_2d_interactions(
            features=("feature_A", "feature_B")
        )
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_multiple_interactions_by_index(self, simple_fastpd):
        """Test plotting multiple interaction pairs by indices."""
        fig = simple_fastpd.plot_2d_interactions(features=[(0, 1), (0, 2)])
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_multiple_interactions_by_name(self, simple_fastpd):
        """Test plotting multiple interaction pairs by names."""
        fig = simple_fastpd.plot_2d_interactions(
            features=[("feature_A", "feature_B"), ("feature_A", "feature_C")]
        )
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_with_custom_evaluation_points(self, simple_fastpd):
        """Test plotting with custom evaluation points."""
        # Create custom evaluation points for 2D interaction
        n_points = 20
        eval_points = np.random.rand(n_points, 3) * 10
        fig = simple_fastpd.plot_2d_interactions(
            features=(0, 1), evaluation_points=eval_points
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_cmap(self, simple_fastpd):
        """Test plotting with custom colormap."""
        fig = simple_fastpd.plot_2d_interactions(
            features=(0, 1), cmap="viridis"
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_alpha(self, simple_fastpd):
        """Test plotting with custom alpha."""
        fig = simple_fastpd.plot_2d_interactions(
            features=(0, 1), alpha=0.8
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_california_housing_all_interactions(
        self, california_housing_fastpd
    ):
        """Test plotting all 2D interactions for California Housing dataset."""
        fig = california_housing_fastpd.plot_2d_interactions()
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_california_housing_single_interaction(
        self, california_housing_fastpd
    ):
        """Test plotting single interaction for California Housing dataset."""
        fig = california_housing_fastpd.plot_2d_interactions(
            features=("MedInc", "HouseAge")
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_invalid_feature_name(self, simple_fastpd):
        """Test that invalid feature name raises error."""
        with pytest.raises(ValueError, match="not found"):
            simple_fastpd.plot_2d_interactions(
                features=("invalid_feature", "feature_A")
            )

    def test_plot_invalid_feature_index(self, simple_fastpd):
        """Test that invalid feature index raises error."""
        with pytest.raises(IndexError):
            simple_fastpd.plot_2d_interactions(features=(100, 0))

    def test_plot_caching(self, simple_fastpd):
        """Test that components are cached after first plot."""
        # First plot should compute and cache
        fig1 = simple_fastpd.plot_2d_interactions(features=(0, 1))
        plt.close(fig1)

        # Check that cache exists with max_order >= 2
        cached = simple_fastpd.get_cached_components()
        assert cached is not None
        comp_values, subsets, eval_points, max_order = cached
        assert max_order >= 2

        # Second plot should use cache (no error means it worked)
        fig2 = simple_fastpd.plot_2d_interactions(features=(0, 1))
        plt.close(fig2)


class TestVisualizationHelpers:
    """Test helper functions used by visualization methods."""

    def test_resolve_feature_by_index(self, simple_fastpd):
        """Test _resolve_feature with integer index."""
        from glex_rust.visualization import _resolve_feature

        idx = _resolve_feature(simple_fastpd, 0)
        assert idx == 0

    def test_resolve_feature_by_name(self, simple_fastpd):
        """Test _resolve_feature with string name."""
        from glex_rust.visualization import _resolve_feature

        idx = _resolve_feature(simple_fastpd, "feature_A")
        assert idx == 0

    def test_resolve_feature_invalid_name(self, simple_fastpd):
        """Test _resolve_feature with invalid name."""
        from glex_rust.visualization import _resolve_feature

        with pytest.raises(ValueError):
            _resolve_feature(simple_fastpd, "invalid")

    def test_generate_evaluation_points_1d(self, simple_fastpd):
        """Test _generate_default_evaluation_points_1d."""
        from glex_rust.visualization import (
            DEFAULT_N_POINTS,
            _generate_default_evaluation_points_1d,
        )

        eval_points = _generate_default_evaluation_points_1d(simple_fastpd)
        n_features = simple_fastpd.n_features()
        expected_shape = (DEFAULT_N_POINTS * n_features, n_features)
        assert eval_points.shape == expected_shape

    def test_generate_evaluation_points_2d(self, simple_fastpd):
        """Test _generate_default_evaluation_points_2d."""
        from glex_rust.visualization import (
            DEFAULT_N_POINTS,
            _generate_default_evaluation_points_2d,
        )

        eval_points = _generate_default_evaluation_points_2d(simple_fastpd)
        n_features = simple_fastpd.n_features()
        n_pairs = n_features * (n_features - 1) // 2
        expected_shape = (n_pairs * (DEFAULT_N_POINTS ** 2), n_features)
        assert eval_points.shape == expected_shape

    def test_extract_component_by_subset(self, simple_fastpd):
        """Test _extract_component_by_subset."""
        from glex_rust.visualization import (
            DEFAULT_N_POINTS,
            _extract_component_by_subset,
        )

        # Ensure components are cached
        simple_fastpd.plot_1d_components(features=0)
        cached = simple_fastpd.get_cached_components()
        assert cached is not None
        comp_values, subsets, _, _ = cached

        # Extract component for feature 0
        comp = _extract_component_by_subset(comp_values, subsets, [0])
        assert comp.shape[0] == DEFAULT_N_POINTS
        assert len(comp.shape) == 1

    def test_extract_component_by_subset_not_found(self, simple_fastpd):
        """Test _extract_component_by_subset with non-existent subset."""
        from glex_rust.visualization import _extract_component_by_subset

        # Ensure components are cached
        simple_fastpd.plot_1d_components(features=0)
        cached = simple_fastpd.get_cached_components()
        assert cached is not None
        comp_values, subsets, _, _ = cached

        # Try to extract non-existent subset
        with pytest.raises(ValueError, match="not found"):
            _extract_component_by_subset(comp_values, subsets, [999])


class TestVisualizationIntegration:
    """Integration tests for visualization workflow."""

    def test_plot_1d_then_2d_uses_cache(self, simple_fastpd):
        """Test that plotting 1D then 2D properly updates cache."""
        # Plot 1D (caches up to order 1)
        fig1 = simple_fastpd.plot_1d_components(features=0)
        plt.close(fig1)

        cached1 = simple_fastpd.get_cached_components()
        assert cached1 is not None
        _, _, _, max_order1 = cached1
        assert max_order1 >= 1

        # Plot 2D (should extend cache to order 2)
        fig2 = simple_fastpd.plot_2d_interactions(features=(0, 1))
        plt.close(fig2)

        cached2 = simple_fastpd.get_cached_components()
        assert cached2 is not None
        _, _, _, max_order2 = cached2
        assert max_order2 >= 2

    def test_custom_evaluation_points_bypass_cache(self, simple_fastpd):
        """Test that custom evaluation points bypass cache."""
        # Plot with default (caches)
        fig1 = simple_fastpd.plot_1d_components(features=0)
        plt.close(fig1)

        # Plot with custom evaluation points (should not use cache)
        custom_eval = np.random.rand(50, 3) * 10
        fig2 = simple_fastpd.plot_1d_components(
            features=0, evaluation_points=custom_eval
        )
        plt.close(fig2)

        # Cache should still exist from first plot
        cached = simple_fastpd.get_cached_components()
        assert cached is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

