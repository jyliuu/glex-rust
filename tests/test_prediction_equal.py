"""
Test to compare fastpd.predict vs model.predict sample by sample.
"""

import glex_rust
import numpy as np
import pytest

import xgboost as xgb


def test_prediction_equal():
    """Compare fastpd.predict vs model.predict for 300 samples."""
    # Generate simulated data with 2 covariates
    np.random.seed(1)
    n_samples = 200
    n_features = 2

    X = np.random.rand(n_samples, n_features) * 10
    y = X[:, 0] + X[:, 1] + 1.0  # Simple linear relationship

    # Fit XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=2,
        learning_rate=1.0,
        random_state=42,
    )
    model.fit(X, y)

    # Create FastPD instance (now accepts float64, converts internally)
    fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

    # Generate 300 test samples
    np.random.seed(42)  # Different seed for test samples
    n_test = 300
    X_test = np.random.rand(n_test, n_features) * 10

    # Get predictions from both methods (now accepts float64, converts internally)
    fastpd_predictions = fastpd.predict(X_test)
    xgb_predictions = model.predict(X_test)

    # Compare sample by sample
    discrepancies = []
    for i in range(n_test):
        fastpd_pred = fastpd_predictions[i]
        xgb_pred = xgb_predictions[i]
        diff = abs(fastpd_pred - xgb_pred)

        if diff > 1e-6:  # Allow for small floating point differences
            discrepancies.append(
                {
                    "sample": i,
                    "fastpd": fastpd_pred,
                    "xgb": xgb_pred,
                    "diff": diff,
                    "X": X_test[i],
                }
            )

    # Print discrepancies
    if discrepancies:
        print(f"\nFound {len(discrepancies)} discrepancies out of {n_test} samples:")
        print(
            f"{'Sample':<8} {'FastPD':<15} {'XGBoost':<15} {'Difference':<15} {'X values':<20}"
        )
        print("-" * 80)
        for disc in discrepancies:
            # Format X values with full precision
            x_str = np.array2string(
                disc["X"], separator=" ", precision=17, suppress_small=False
            )
            print(
                f"{disc['sample']:<8} "
                f"{disc['fastpd']:<15.10f} "
                f"{disc['xgb']:<15.10f} "
                f"{disc['diff']:<15.10e} "
                f"{x_str}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
