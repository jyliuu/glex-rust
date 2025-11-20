"""
Example script demonstrating 1D functional decomposition component visualization.

This script shows how to:
1. Create a FastPD instance from an XGBoost model
2. Plot 1D functional components
3. Display the plots interactively
"""

import glex_rust
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import xgboost as xgb

# Set matplotlib to use interactive backend (shows plots)
plt.ion()  # Turn on interactive mode


def main():
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature names: {data.feature_names}")

    # Fit XGBoost model
    print("\nFitting XGBoost model...")
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

    # Create FastPD instance
    print("\nCreating FastPD instance...")
    fastpd = glex_rust.FastPD.from_xgboost(
        model,
        background_samples=data,  # Pass Bunch to extract feature_names
        n_threads=1,
    )
    print(
        f"FastPD created: {fastpd.num_trees()} trees, "
        f"{fastpd.n_background()} background samples, "
        f"{fastpd.n_features()} features"
    )

    # Example 1: Plot all 1D components
    print("\n" + "=" * 60)
    print("Example 1: Plotting all 1D functional components")
    print("=" * 60)
    fig1 = fastpd.plot_1d_components()
    plt.show(block=False)  # Show without blocking
    input("Press Enter to continue to next plot...")
    plt.close(fig1)

    # Example 2: Plot specific 1D component by name
    print("\n" + "=" * 60)
    print("Example 2: Plotting single 1D component (MedInc)")
    print("=" * 60)
    fig2 = fastpd.plot_1d_components(features="MedInc", figsize=(12, 8))
    plt.show(block=False)
    input("Press Enter to continue to next plot...")
    plt.close(fig2)

    # Example 3: Plot multiple 1D components
    print("\n" + "=" * 60)
    print("Example 3: Plotting multiple 1D components")
    print("=" * 60)
    fig3 = fastpd.plot_1d_components(
        features=["MedInc", "HouseAge", "AveRooms"], figsize=(15, 5)
    )
    plt.show(block=False)
    input("Press Enter to finish...")
    plt.close(fig3)

    print("\n" + "=" * 60)
    print("All 1D visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

