# glex-rust

![CI](https://github.com/jyliuu/glex-rust/actions/workflows/ci.yml/badge.svg)
![Version](https://img.shields.io/github/v/release/jyliuu/glex-rust?label=version)

A high-performance Rust implementation of the FastPD algorithm from Liu et al. (2025) for estimating Partial Dependence (PD) functions from tree-based models, with Python bindings.

## Overview

`glex-rust` provides efficient computation of Partial Dependence functions for tree-based machine learning models. It implements the FastPD algorithm from the paper "Fast Estimation of Partial Dependence Functions using Trees", achieving $O(2^{D+F}(n_e + n_b))$ complexity where:
- $D$ is the tree depth
- $F$ is the number of features
- $n_e$ is the number of evaluation points
- $n_b$ is the number of background samples

**Currently, only XGBoost models are supported.** Support for additional tree-based models (e.g., LightGBM, scikit-learn trees) is planned for future releases.

## Installation

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Python 3.12 or higher
- pip (maturin will be installed automatically as a build dependency when using `pip install`)

### Building from Source

1. Clone the repository:
```bash
git clone git@github.com:jyliuu/glex-rust.git
cd glex-rust
```

2. Install the package:
```bash
pip install .
```

For development/editable install:
```bash
pip install -e .
```

Alternatively, you can use `maturin` directly:
```bash
# Install maturin first
pip install maturin

# Build and install in development mode
maturin develop

# For optimized release builds
maturin develop --release
```

### Installing Test Dependencies

To run the test suite:
```bash
pip install -e ".[test]"
```

## Usage

### Basic Example

```python
import numpy as np
import xgboost as xgb
import glex_rust

# Generate sample data
np.random.seed(42)
X = np.random.rand(200, 2) * 10
y = X[:, 0] + X[:, 1] + 1.0

# Fit an XGBoost model
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X, y)

# Create FastPD instance with background samples
fastpd = glex_rust.FastPD.from_xgboost(model, background_samples=X)

# Compute PD function for feature 0
n_eval = 10
eval_points = np.random.rand(n_eval, 2) * 10  # Random features x1 and x2 in [0, 10)
pd_values = fastpd.pd_function(
    evaluation_points=eval_points,
    feature_subset=[0]
)

print(f"PD values: {pd_values}")
```

### API Reference

#### `FastPD.from_xgboost(model, background_samples)`

Create a FastPD instance from an XGBoost model.

**Parameters:**
- `model`: XGBoost model (XGBRegressor, XGBClassifier, or Booster)
- `background_samples`: NumPy array of shape `(n_background, n_features)` used for PD estimation

**Returns:** A `FastPD` instance

#### `fastpd.pd_function(evaluation_points, feature_subset)`

Compute the Partial Dependence function for a given feature subset.

**Parameters:**
- `evaluation_points`: NumPy array of shape `(n_evaluation_points, n_features)` - points at which to evaluate PD
- `feature_subset`: List of feature indices (e.g., `[0]` for single feature, `[0, 1]` for interaction)

**Returns:** NumPy array of shape `(n_evaluation_points,)` containing PD values

#### `fastpd.predict(evaluation_points)`

Predict model output for given input points.

**Parameters:**
- `evaluation_points`: NumPy array of shape `(n_points, n_features)`

**Returns:** NumPy array of shape `(n_points,)` containing predictions

#### `fastpd.clear_caches()`

Clear all internal caches. Useful for memory management when processing multiple batches.

#### Utility Methods

- `fastpd.num_trees()`: Returns the number of trees in the ensemble
- `fastpd.n_background()`: Returns the number of background samples
- `fastpd.n_features()`: Returns the number of features

### Tree Extraction

You can also extract trees directly from XGBoost models:

```python
import glex_rust

trees = glex_rust.extract_trees_from_xgboost(model)
print(f"Extracted {len(trees)} trees")

# Inspect tree structure
for i, tree in enumerate(trees):
    print(f"Tree {i}: {tree.num_nodes()} nodes")
    print(tree.format_tree())
```

## Development

### Running Tests

Run Rust unit tests:
```bash
cargo test
```

Run Python integration tests:
```bash
pytest tests/
```

Run both:
```bash
cargo test && pytest tests/
```

### Building for Distribution

Build a wheel:
```bash
maturin build
```

Build optimized release wheel:
```bash
maturin build --release
```

## Algorithm Details

The FastPD algorithm consists of two main phases:

1. **Augmentation (Algorithm 1)**: Preprocesses trees with background samples, storing observation sets for each leaf and feature subset combination.

2. **Evaluation (Algorithm 2)**: Efficiently evaluates PD functions at query points by traversing augmented trees and aggregating leaf values.

The implementation achieves linear scaling with the number of evaluation points and background samples, while handling the exponential complexity in tree depth and feature subsets through efficient caching and data structures.

## Dependencies

### Rust Dependencies
- `pyo3`: Python bindings
- `numpy`: NumPy array integration
- `ndarray`: N-dimensional arrays
- `ndarray-linalg`: Linear algebra operations
- `serde` / `serde_json`: JSON parsing for XGBoost models
- `thiserror`: Error handling

### Python Dependencies
- Python >= 3.8
- NumPy
- XGBoost (for model training)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite the FastPD paper:

```bibtex
@InProceedings{pmlr-v267-liu25bm,
  title = 	 {Fast Estimation of Partial Dependence Functions using Trees},
  author =       {Liu, Jinyang and Steensgaard, Tessa and Wright, Marvin N. and Pfister, Niklas and Hiabu, Munir},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {39496--39534},
  year = 	 {2025},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25bm/liu25bm.pdf},
  url = 	 {https://proceedings.mlr.press/v267/liu25bm.html},
}
```
