"""
Generate datasets and train XGBoost models for Rust benchmarks.

This script:
1. Generates synthetic datasets with specified parameters
2. Loads the California housing dataset
3. Trains XGBoost models with standard parameters
4. Saves model dumps as JSON (with base_score) for Rust parsing
5. Saves feature matrices as CSV
6. Saves metadata as JSON
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"

# Create output directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
SYNTHETIC_DATASETS = [
    {"name": "synthetic_p2_n100", "n_features": 2, "n_samples": 100},
    {"name": "synthetic_p3_n500", "n_features": 3, "n_samples": 500},
    {"name": "synthetic_p5_n1000", "n_features": 5, "n_samples": 1000},
    {"name": "synthetic_p5_n3000", "n_features": 5, "n_samples": 3000},
    {"name": "synthetic_p7_n5000", "n_features": 7, "n_samples": 5000},
    {"name": "synthetic_p10_n10000", "n_features": 10, "n_samples": 10000},
]

# XGBoost training parameters
XGBOOST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
}


def generate_synthetic_dataset(n_features: int, n_samples: int, seed: int = 42):
    """Generate a synthetic regression dataset."""
    rng = np.random.default_rng(seed)

    # Generate random features (uniform distribution in [0, 10])
    X = rng.uniform(0, 10, size=(n_samples, n_features))

    # Generate target as a non-linear function of features
    # Use a combination of features to make it interesting for tree models
    y = (
        np.sum(X[:, : min(3, n_features)] ** 2, axis=1)
        + 2 * np.prod(X[:, : min(2, n_features)], axis=1)
        + rng.normal(0, 0.1, size=n_samples)
    )

    return X, y


def load_california_housing():
    """Load the California housing dataset."""
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # The dataset has 8 features and 20640 samples
    return X, y


def train_xgboost_model(X, y, **kwargs):
    """Train an XGBoost model with standard parameters."""
    params = {**XGBOOST_PARAMS, **kwargs}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def save_dumps(model, dataset_name: str):
    """Save XGBoost JSON dumps with base_score for Rust parsing."""
    booster = model.get_booster()

    # Get JSON dumps (for Rust parsing with parse_json_tree)
    tree_dumps = booster.get_dump(dump_format="json")
    base_score_str = booster.save_config()

    config = json.loads(base_score_str)
    base_score = float(
        config["learner"]["learner_model_param"]["base_score"].strip("[]")
    )

    # Save dumps and base_score for Rust
    dumps_path = MODELS_DIR / f"{dataset_name}_dumps.json"
    dumps_data = {
        "base_score": base_score,
        "tree_dumps": tree_dumps,
    }
    with open(dumps_path, "w") as f:
        json.dump(dumps_data, f, indent=2)
    print(f"  Saved JSON dumps to {dumps_path}")


def save_dataset(
    X, y, dataset_name: str, n_features: int, n_samples: int, seed: int = None
):
    """Save dataset as CSV and metadata as JSON."""
    dataset_dir = DATASETS_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrix as CSV
    X_csv_path = dataset_dir / "X.csv"
    np.savetxt(X_csv_path, X, delimiter=",", fmt="%.10f")
    print(f"  Saved features to {X_csv_path}")

    # Save metadata as JSON
    metadata = {
        "name": dataset_name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_estimators": XGBOOST_PARAMS["n_estimators"],
        "max_depth": XGBOOST_PARAMS["max_depth"],
        "learning_rate": XGBOOST_PARAMS["learning_rate"],
    }

    if seed is not None:
        metadata["seed"] = int(seed)

    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")


def process_dataset(
    X, y, dataset_name: str, n_features: int, n_samples: int, seed: int = None
):
    """Process a single dataset: train model and save all outputs."""
    print(f"Processing {dataset_name}...")
    print(f"  Shape: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Train model
    print("  Training XGBoost model...")
    model = train_xgboost_model(X, y)

    # Save dumps
    save_dumps(model, dataset_name)

    # Save dataset
    save_dataset(X, y, dataset_name, n_features, n_samples, seed)

    print(f"  ✓ Completed {dataset_name}\n")


def main():
    """Generate all datasets and train models."""
    print("Generating datasets and training models for Rust benchmarks...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models output: {MODELS_DIR}")
    print(f"Datasets output: {DATASETS_DIR}")
    print()

    # Process synthetic datasets
    print("Generating synthetic datasets...")
    for config in SYNTHETIC_DATASETS:
        X, y = generate_synthetic_dataset(
            n_features=config["n_features"],
            n_samples=config["n_samples"],
            seed=42,
        )
        process_dataset(
            X,
            y,
            dataset_name=config["name"],
            n_features=config["n_features"],
            n_samples=config["n_samples"],
            seed=42,
        )

    # Process California housing dataset
    print("Loading California housing dataset...")
    X, y = load_california_housing()
    n_samples, n_features = X.shape
    process_dataset(
        X,
        y,
        dataset_name="california_housing_p8_n20640",
        n_features=n_features,
        n_samples=n_samples,
        seed=None,  # Real dataset, no seed
    )

    print("=" * 60)
    print("Generation complete!")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Datasets saved to: {DATASETS_DIR}")
    print(f"\nGenerated {len(SYNTHETIC_DATASETS)} synthetic datasets + 1 real dataset")


if __name__ == "__main__":
    main()
