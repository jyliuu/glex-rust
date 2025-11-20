"""
Example script demonstrating 2D interaction component visualization.

This script shows how to:
1. Create a FastPD instance from an XGBoost model
2. Plot 2D interaction components
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

    # Example 1: Plot all 2D interactions
    print("\n" + "=" * 60)
    print("Example 1: Plotting all 2D interaction components")
    print("=" * 60)
    print("(This may take a moment to compute...)")
    fig1 = fastpd.plot_2d_interactions(figsize=(20, 20), n_cols=3)
    plt.show(block=False)
    input("Press Enter to continue to next plot...")
    plt.close(fig1)

    # Example 2: Plot specific 2D interaction
    print("\n" + "=" * 60)
    print("Example 2: Plotting specific 2D interaction (Longitude vs Latitude)")
    print("=" * 60)
    fig2 = fastpd.plot_2d_interactions(
        features=("Longitude", "Latitude"), figsize=(10, 8), cmap="plasma"
    )
    plt.show(block=False)
    input("Press Enter to continue to next plot...")
    plt.close(fig2)

    # Example 3: Plot multiple 2D interactions
    print("\n" + "=" * 60)
    print("Example 3: Plotting multiple 2D interactions")
    print("=" * 60)
    fig3 = fastpd.plot_2d_interactions(
        features=[("MedInc", "HouseAge"), ("AveRooms", "AveBedrms")],
        figsize=(15, 6),
        n_cols=2,
        cmap="plasma",
    )
    plt.show(block=False)
    input("Press Enter to finish...")
    plt.close(fig3)

    print("\n" + "=" * 60)
    print("All 2D interaction visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

