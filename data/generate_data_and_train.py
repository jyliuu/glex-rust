import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing

import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"


def ensure_dirs() -> None:
    """Ensure that data directories exist."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(
    n: int = 1000, p: int = 5, random_state: int = 0
) -> None:
    """Generate a synthetic regression dataset and train an XGBoost model.

    The CSV will be saved as datasets/synthetic_n{n}_p{p}.csv with columns X0..X{p-1},y.
    The model will be saved as models/synthetic_n{n}_p{p}_xgb.json using the raw XGBoost dump.
    """
    ensure_dirs()

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))

    # Simple linear model with some non-linearity and noise
    beta = rng.normal(size=p)
    linear = X @ beta
    y = linear + 0.1 * rng.normal(size=n)

    # Save dataset
    csv_path = DATASETS_DIR / f"synthetic_n{n}_p{p}.csv"
    header = ",".join([f"X{i}" for i in range(p)] + ["y"])
    data = np.column_stack([X, y])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    # Train XGBoost model
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.1,
        "base_score": 0.0,
    }
    num_boost_round = 50
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

    # Save raw JSON dump (one JSON string per tree) for Rust parser
    model_path = MODELS_DIR / f"synthetic_n{n}_p{p}_xgb.json"
    dump = booster.get_dump(dump_format="json")
    with model_path.open("w") as f:
        json.dump(dump, f)

    print(f"Wrote synthetic data to {csv_path}")
    print(f"Wrote synthetic XGBoost model to {model_path}")


def generate_california_housing(random_state: int = 0) -> None:
    """Fetch the California housing dataset, save it, and train an XGBoost model.

    - CSV: datasets/california_housing.csv
    - Model: models/california_housing_xgb.json
    """
    ensure_dirs()

    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()

    # Ensure target column is named 'y' for consistency
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "y"})
    elif "target" in df.columns:
        df = df.rename(columns={"target": "y"})

    csv_path = DATASETS_DIR / "california_housing.csv"
    df.to_csv(csv_path, index=False)

    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.1,
        "base_score": 0.0,
    }
    num_boost_round = 50
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

    # Save raw JSON dump (one JSON string per tree) for Rust parser
    model_path = MODELS_DIR / "california_housing_xgb.json"
    dump = booster.get_dump(dump_format="json")
    with model_path.open("w") as f:
        json.dump(dump, f)

    print(f"Wrote California housing data to {csv_path}")
    print(f"Wrote California housing XGBoost model to {model_path}")


def main() -> None:
    # Generate a range of synthetic datasets with varying (n, p).
    synthetic_configs = [
        (200, 2),
        (500, 3),
        (1000, 5),
        (2000, 7),
        (5000, 10),
    ]

    for n, p in synthetic_configs:
        generate_synthetic_dataset(n=n, p=p)

    generate_california_housing()


if __name__ == "__main__":
    main()
