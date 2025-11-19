# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-18

### Added
- Functional decomposition support: compute functional decomposition up to a given order
- New methods: `pd_functions_up_to_order` and `functional_decomp_up_to_order` for computing partial dependence functions and functional decomposition components
- New batch evaluation method: `evaluate_pd_batch_for_subsets` for efficient batch partial dependence computation
- Python type stubs for better IDE support and type checking

### Changed
- Refactored to use new batch/matrix approach for improved performance and code organization
- Added parallelization support for batch PD function and functional decomposition computation
- Store leaf expectations directly on augmented trees for faster access

## [0.2.1] - 2025-11-17

### Changed
- Added new parallelization option for both augmentation and evaluation using rayon.
- Enhance performance by using more efficient data structures such as FxHashMap and bitsets.
- Added benchmark modules to measure performance

## [0.2.0] - 2025-11-16

### Changed
- Refactored the project into a Cargo workspace with two crates:
  - `glex-core`: pure Rust FastPD core and XGBoost JSON parsing without any PyO3 dependency.
  - `glex-rust` (python): PyO3 bindings crate used by maturin, depending on `glex-core`.
- Updated `pyproject.toml` to build the Python extension via the new `python` crate.
- Removed the legacy monolithic `src/` crate in favor of the workspace layout.
- Preserved the Python public API (`glex_rust` package and its symbols) while changing the internal structure.

## [0.1.1] - 2025-11-15

### Fixed
- Fixed precision discrepancies between FastPD and XGBoost predictions by using `f32` internally to match XGBoost's internal precision
- Python API now accepts `float64` numpy arrays and automatically converts them to `float32` internally, eliminating the need for manual type conversion

### Changed
- Internal precision changed from `f64` to `f32` throughout the codebase to match XGBoost's internal representation
- All threshold and leaf value types now use `f32` for exact precision matching with XGBoost

## [0.1.0] - 2025-11-15

### Added
- Initial release
- FastPD implementation for efficient partial dependence computation
- XGBoost tree extraction and parsing
- Python bindings for FastPD functionality
- Support for tree-based model prediction and PD function evaluation
