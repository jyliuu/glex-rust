"""
Test for California Housing dataset with univariate PD function extraction and plotting.
"""

import glex_rust
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

import xgboost as xgb


def test_california_housing_univariate_pd():
    """Test univariate PD functions for California Housing dataset.

    This test:
    1. Loads California Housing data from sklearn
    2. Fits XGBoost with optimized hyperparameters (1348 trees, max_depth=4)
    3. Extracts ALL univariate PD functions using FastPD
    4. Plots them using matplotlib
    """
    # Load California Housing dataset
    print("\nLoading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {data.feature_names}")

    # Fit XGBoost model with specified parameters
    print("\nFitting XGBoost model with optimized hyperparameters (max_depth=4)...")
    model = xgb.XGBRegressor(
        n_estimators=1348,
        max_depth=7,
        learning_rate=0.02661632003068007,
        colsample_bylevel=0.6195479434577076,
        gamma=182.10035437102098,
        reg_alpha=229.39185842258482,
        reg_lambda=0.7996764605377253,
        subsample=0.9094355871840993,
        random_state=42,
    )
    model.fit(X, y)
    print("Model fitted successfully!")

    # Create FastPD instance using the training data as background samples
    print("\nCreating FastPD instance...")
    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X, n_threads=1)
    print(
        f"FastPD created: {fastpd.num_trees()} trees, "
        f"{fastpd.n_background()} background samples, "
        f"{fastpd.n_features()} features"
    )

    # Extract univariate PD functions for all features
    n_features = X.shape[1]
    feature_names = data.feature_names
    n_background = X.shape[0]

    # Use background samples as evaluation points
    print("\nComputing univariate PD functions using batch FastPD...")
    print(f"Using {n_background} background samples as evaluation points")

    # Compute all univariate PD functions in a single batch call
    print("  Computing all univariate PD functions (max_order=1) in batch...")
    pd_values_batch, subsets_batch = fastpd.pd_functions_up_to_order(
        evaluation_points=X, max_order=1
    )

    # Extract PD functions for each subset returned from batch
    pd_functions = []
    eval_ranges = []
    feature_indices = []  # Track which feature each PD function corresponds to
    for subset_idx, subset in enumerate(subsets_batch):
        if len(subset) == 1:  # Only univariate subsets
            feature_idx = subset[0]
            pd_functions.append(pd_values_batch[:, subset_idx])
            eval_ranges.append(X[:, feature_idx])
            feature_indices.append(feature_idx)
            print(
                f"  Feature {feature_idx} ({feature_names[feature_idx]}): PD range [{pd_functions[-1].min():.4f}, {pd_functions[-1].max():.4f}]"
            )

    # Plot all univariate PD functions
    print(f"\nPlotting {len(pd_functions)} univariate PD functions...")
    n_cols = 3
    n_rows = (len(pd_functions) + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if len(pd_functions) > 1 else [axes]

    for pd_idx in range(len(pd_functions)):
        ax = axes[pd_idx]
        # Sort by feature value for better visualization
        sort_idx = np.argsort(eval_ranges[pd_idx])
        sorted_eval = eval_ranges[pd_idx][sort_idx]
        sorted_pd = pd_functions[pd_idx][sort_idx]

        # Get feature index and name
        feature_idx = feature_indices[pd_idx]
        feature_name = feature_names[feature_idx]

        ax.plot(
            sorted_eval,
            sorted_pd,
            linewidth=2,
            alpha=0.8,
        )
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel("Partial Dependence", fontsize=10)
        ax.set_title(f"PD Function: {feature_name}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(pd_functions), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save the plot
    output_path = "california_housing_univariate_pd.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Display the plot (optional, comment out if running in headless environment)
    # plt.show()
    plt.close()

    # Verify we have PD functions (may be fewer than n_features if some features aren't in trees)
    assert len(pd_functions) > 0, "Expected at least one PD function"

    # Verify each PD function has the expected number of points
    for idx, pd_func in enumerate(pd_functions):
        assert len(pd_func) == n_background, (
            f"PD function {idx} should have {n_background} points, got {len(pd_func)}"
        )

    print(
        f"\n✓ Successfully computed and plotted {len(pd_functions)} univariate PD functions!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
