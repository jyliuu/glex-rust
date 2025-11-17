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
    """Test univariate functional decomposition components for California Housing dataset.

    This test:
    1. Loads California Housing data from sklearn
    2. Fits XGBoost with optimized hyperparameters (1348 trees, max_depth=4)
    3. Extracts ALL univariate functional decomposition components using FastPD
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

    # Extract univariate functional decomposition components for all features
    feature_names = data.feature_names
    n_background = X.shape[0]

    # Use background samples as evaluation points
    print(
        "\nComputing univariate functional decomposition components using batch FastPD..."
    )
    print(f"Using {n_background} background samples as evaluation points")

    # Compute all univariate functional decomposition components in a single batch call
    print("  Computing all univariate functional components (max_order=1) in batch...")
    comp_values_batch, subsets_batch = fastpd.functional_decomp_up_to_order(
        evaluation_points=X, max_order=1
    )

    # Extract functional components for each subset returned from batch
    comp_functions = []
    eval_ranges = []
    feature_indices = []  # Track which feature each component corresponds to
    for subset_idx, subset in enumerate(subsets_batch):
        if len(subset) == 1:  # Only univariate subsets
            feature_idx = subset[0]
            comp_functions.append(comp_values_batch[:, subset_idx])
            eval_ranges.append(X[:, feature_idx])
            feature_indices.append(feature_idx)
            print(
                f"  Feature {feature_idx} ({feature_names[feature_idx]}): Component range [{comp_functions[-1].min():.4f}, {comp_functions[-1].max():.4f}]"
            )

    # Plot all univariate functional components
    print(f"\nPlotting {len(comp_functions)} univariate functional components...")
    n_cols = 3
    n_rows = (len(comp_functions) + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if len(comp_functions) > 1 else [axes]

    for comp_idx in range(len(comp_functions)):
        ax = axes[comp_idx]
        # Sort by feature value for better visualization
        sort_idx = np.argsort(eval_ranges[comp_idx])
        sorted_eval = eval_ranges[comp_idx][sort_idx]
        sorted_comp = comp_functions[comp_idx][sort_idx]

        # Get feature index and name
        feature_idx = feature_indices[comp_idx]
        feature_name = feature_names[feature_idx]

        ax.plot(
            sorted_eval,
            sorted_comp,
            linewidth=2,
            alpha=0.8,
        )
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel("Functional Component", fontsize=10)
        ax.set_title(f"Functional Component: f_{{{feature_name}}}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5
        )  # Add zero line

    # Hide unused subplots
    for idx in range(len(comp_functions), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save the plot
    output_path = "california_housing_univariate_pd.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Display the plot (optional, comment out if running in headless environment)
    # plt.show()
    plt.close()

    # Verify we have functional components (may be fewer than n_features if some features aren't in trees)
    assert len(comp_functions) > 0, "Expected at least one functional component"

    # Verify each functional component has the expected number of points
    for idx, comp_func in enumerate(comp_functions):
        assert len(comp_func) == n_background, (
            f"Functional component {idx} should have {n_background} points, got {len(comp_func)}"
        )

    print(
        f"\n✓ Successfully computed and plotted {len(comp_functions)} univariate functional components!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
