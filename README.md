# glex-rust

![CI](https://github.com/jyliuu/glex-rust/actions/workflows/ci.yml/badge.svg)
![Version](https://img.shields.io/github/v/release/jyliuu/glex-rust?label=version)

A high-performance Rust implementation of the FastPD algorithm from Liu et al. (2025) for estimating Partial Dependence (PD) functions and functional decomposition components for tree-based models, with Python bindings. This package is a Python port of the original [`glex`](https://github.com/PlantedML/glex) package written in R.

## Overview

`glex-rust` provides efficient computation of Partial Dependence functions and functional decomposition components for tree-based machine learning models. It implements the FastPD algorithm from the paper "Fast Estimation of Partial Dependence Functions using Trees", achieving $O(2^{D+F}(n_e + n_b))$ complexity where:
- $D$ is the tree depth
- $F$ is the number of features
- $n_e$ is the number of evaluation points
- $n_b$ is the number of background samples

The library supports:
- **Partial Dependence (PD) functions**: Compute PD surfaces for individual feature subsets or batch compute all PD functions up to a given interaction order
- **Functional decomposition**: Compute ANOVA-style functional decomposition components that decompose model predictions into main effects, interactions, and an intercept term

**Currently, only XGBoost regression models are supported.** Support for additional tree-based models (e.g., LightGBM, scikit-learn trees) is planned for future releases.

## Installation

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Python 3.12 or higher
- pip (maturin will be installed automatically as a build dependency when using `pip install`)
### Quick Install

```bash
pip install git+https://github.com/jyliuu/glex-rust.git
```

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

For development, you can use `maturin` to build and install the package:
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

### Batch PD Functions

Compute all PD functions up to a given interaction order efficiently in a single batch call:

```python
# Compute all PD functions up to order 2 (main effects and pairwise interactions)
pd_values, subsets = fastpd.pd_functions_up_to_order(
    evaluation_points=eval_points,
    max_order=2
)

# pd_values shape: (n_eval, n_subsets)
# subsets: list of feature subsets, e.g., [[0], [1], [2], [0, 1], [0, 2], [1, 2]]

# Access PD values for a specific subset
for subset_idx, subset in enumerate(subsets):
    print(f"Subset {subset}: {pd_values[:, subset_idx]}")
```

### Functional Decomposition

Compute functional decomposition components that decompose model predictions into interpretable parts:

```python
# Compute functional decomposition up to order 2
comp_values, subsets = fastpd.functional_decomp_up_to_order(
    evaluation_points=eval_points,
    max_order=2
)

# comp_values shape: (n_eval, n_subsets)
# subsets includes the empty subset [] for the intercept

# Functional decomposition components sum to predictions
predictions = fastpd.predict(evaluation_points=eval_points)
for i in range(len(eval_points)):
    sum_components = np.sum(comp_values[i, :])
    assert np.isclose(sum_components, predictions[i], rtol=1e-5)
    
# Extract main effects (univariate components)
for subset_idx, subset in enumerate(subsets):
    if len(subset) == 1:  # Main effect
        feature_idx = subset[0]
        main_effect = comp_values[:, subset_idx]
        print(f"Main effect for feature {feature_idx}: {main_effect}")
```

### API Reference

#### `FastPD.from_xgboost(model, background_samples, n_threads=1)`

Create a FastPD instance from an XGBoost model.

**Parameters:**
- `model`: XGBoost model (XGBRegressor, XGBClassifier, or Booster)
- `background_samples`: NumPy array of shape `(n_background, n_features)` used for PD estimation
- `n_threads`: Number of threads to use for parallelization (default: 1)

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

#### `fastpd.pd_functions_up_to_order(evaluation_points, max_order)`

Compute all partial dependence surfaces v_S(x_S) for all subsets S with 1 ≤ |S| ≤ max_order in a single batch call.

**Parameters:**
- `evaluation_points`: NumPy array of shape `(n_evaluation_points, n_features)` - points at which to evaluate PD
- `max_order`: Maximum interaction order (e.g., 1 for main effects, 2 for pairwise interactions, etc.)

**Returns:** A tuple `(pd_values, subsets)` where:
- `pd_values`: 2D NumPy array of shape `(n_eval, n_subsets)` with one column per subset S
- `subsets`: List of lists, where each inner list contains the feature indices for a subset

**Example:**
```python
pd_values, subsets = fastpd.pd_functions_up_to_order(eval_points, max_order=2)
# subsets might be: [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
```

#### `fastpd.functional_decomp_up_to_order(evaluation_points, max_order)`

Compute functional decomposition components f_S(x_S) for all subsets S with 0 ≤ |S| ≤ max_order. These components decompose model predictions via ANOVA-style functional decomposition using inclusion-exclusion.

**Parameters:**
- `evaluation_points`: NumPy array of shape `(n_evaluation_points, n_features)` - points at which to evaluate
- `max_order`: Maximum interaction order (use `n_features` for complete decomposition)

**Returns:** A tuple `(comp_values, subsets)` where:
- `comp_values`: 2D NumPy array of shape `(n_eval, n_subsets)` with one column per component f_S
- `subsets`: List of lists, where each inner list contains the feature indices for a subset (includes empty subset `[]` for intercept)

**Key Properties:**
- The sum of all components equals the model prediction: `sum(comp_values[i, :]) == predict(eval_points[i])`
- The empty subset `[]` contains the intercept term E[f(X)]
- Components are orthogonal in the sense of functional ANOVA decomposition

**Example:**
```python
comp_values, subsets = fastpd.functional_decomp_up_to_order(eval_points, max_order=3)
# subsets includes [] for intercept, [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]
```

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

This repository is now a Cargo workspace with two crates:

- `core/` – pure Rust FastPD core (`glex-core` crate)
- `python/` – PyO3 bindings crate used for the Python package

#### Rust tests
Run all Rust tests:
```bash
cargo test -p glex-core
```

#### Python integration tests

Run the Python tests from the repository root:
```bash
pytest tests/
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

### Partial Dependence vs Functional Decomposition

- **Partial Dependence (PD) functions** `v_S(x_S)`: The average prediction when features in subset S are fixed to values x_S and all other features vary over the background distribution. PD functions can overlap and don't necessarily sum to predictions.

- **Functional Decomposition components** `f_S(x_S)`: Orthogonal components that decompose the model prediction via ANOVA-style functional decomposition. These components satisfy:
  - `f(x) = f_∅ + Σ_i f_i(x_i) + Σ_{i<j} f_{ij}(x_i, x_j) + ...`
  - Where `f_∅` is the intercept (expected value) and higher-order terms capture interactions.

## Dependencies

### Rust Dependencies

The workspace contains two Rust crates:

- **`glex-core` (core crate)**:
  - `ndarray`: N-dimensional arrays
  - `serde` / `serde_json`: JSON parsing for XGBoost models
  - `thiserror`: Error handling

- **`glex-rust` (Python bindings crate in `python/`)**:
  - `pyo3`: Python bindings
  - `numpy`: NumPy array integration
  - `ndarray`: used for array conversions
  - `serde_json`: used by the Python bridge to parse booster config

### Python Dependencies
- Python >= 3.12
- NumPy

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
