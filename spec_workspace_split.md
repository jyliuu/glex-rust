## Workspace Refactor: Split Rust Core and PyO3 Bindings

## Goal
**Clear Problem Statement**: Refactor the existing mixed Rust/PyO3 crate into a clean two-crate Cargo workspace: a pure Rust `core` crate containing all algorithmic and XGBoost/PD logic, and a `python` crate containing only PyO3 bindings and Python-facing glue. The Python package should continue to expose the same public API (`glex_rust`) while building via `maturin` against the new `python` crate. Success means:
- **Rust**: `cargo build --workspace` and `cargo test --workspace` succeed.
- **Python**: `maturin develop` succeeds and existing tests under `tests/` pass without API-breaking changes.
- **Structure**: Root is a Cargo workspace, with `core/` and `python/` crates as members, and a repo-level `pyproject.toml` configured for the new layout.

## AI Agent Analysis

**Smart Context Gathering:**
- [x] Use `codebase_search` to find similar implementations and patterns
- [x] Read relevant source files completely to understand existing architecture
- [x] Identify performance-critical paths and optimization opportunities
- [x] Map error handling strategies and failure modes

Current architecture (from `Cargo.toml`, `src/lib.rs`, and `src/xgboost/*`):
- **Single crate**: `glex-rust` is currently a single `cdylib` crate that mixes PyO3 bindings and core logic.
- **Bindings in `src/lib.rs`**:
  - `#[pymodule] fn glex_rust(...)` registers:
    - `extract_trees_from_xgboost_py` (wraps `xgboost::extract_trees_from_xgboost`)
    - `all_subsets`
    - `XGBoostTreeModel`
    - `FastPDPy` (wrapper around `FastPD<XGBoostTreeModel>`)
  - `FastPDPy` exposes methods:
    - `from_xgboost`, `pd_function`, `predict`, `clear_caches`, `num_trees`, `n_background`, `n_features`.
- **Core logic already somewhat separated**:
  - `fastpd` module implements PD core algorithms, independent of PyO3.
  - `xgboost` module implements tree parsing, JSON bridge, and `XGBoostTreeModel`. It uses PyO3 types (`PyAny`, `PyResult`, `Python`) in `python_bridge.rs` and `extract_trees_from_xgboost`.
- **Python package layout**:
  - `pyproject.toml` configures `maturin` with `python-source = "python"` and a Rust library named `glex_rust`.
  - `python/glex_rust/__init__.py` re-exports `FastPDPy`, `all_subsets`, `extract_trees_from_xgboost` from `.glex_rust` and sets `FastPD = FastPDPy`, with `__all__` exposing the same.

Key observations:
- Most algorithmic logic (`fastpd`, `xgboost::tree_model`, `xgboost::types`, `xgboost::parser`) is pure Rust and good candidates for `core`.
- PyO3-specific code exists in:
  - `src/lib.rs`: `#[pymodule]`, `#[pyclass]` on `FastPDPy`, `#[pyfunction]`s, and direct use of `numpy`/`PyArray`, `PyReadonlyArray2`.
  - `src/xgboost/python_bridge.rs`: uses `Bound<PyAny>`, attribute access, `PyErr`.
  - `src/xgboost/mod.rs`: `extract_trees_from_xgboost` uses `PyAny`, `PyResult`, and functions from `python_bridge`.
- To get a **clean core crate**, we should remove PyO3 types from `core` (or at least isolate them) and represent interaction with Python in the `python` crate.

**Strategic Design Decisions:**
- **Option A: Core crate completely pure Rust (no `pyo3` dependency)**  
  - Move all algorithmic and XGBoost parsing to `core`, but retain any functions that require `PyAny`, `PyErr`, or direct Python method calls inside the `python` crate.  
  - Provide a `core` API that:
    - Represents XGBoost trees via Rust types, not Python objects.
    - Accepts generic data structures (`ndarray`, slices) and returns plain Rust types (`Vec`, `ndarray`, etc.).
  - Re-implement `extract_trees_from_xgboost` logic in the `python` crate as a thin PyO3 layer that:
    - Calls Python methods on the model (`get_booster`, `get_dump`, `save_config`).
    - Converts results to strings/JSON and passes them into `core` parsing functions.
  - **Pros**: Clean separation, `core` usable from non-Python environments, minimized PyO3 footprint.  
  - **Cons**: Requires refactoring functions like `extract_trees_from_xgboost` and peeling off their PyO3 layers.

- **Option B: Core crate allows minimal PyO3 (keeps `PyAny` types)**  
  - `core` still depends on `pyo3` (but not `numpy`), using `PyAny` and `PyResult` for functions manipulating Python XGBoost models.
  - Heavy binding code (`#[pymodule]`, `#[pyclass]`, `#[pyfunction]`, `numpy` integration) moves into `python`.
  - `core` exposes a mixed API, partly Rust-only (FastPD computations) and partly PyO3-aware (XGBoost extractors).
  - **Pros**: Smaller refactor; implementation change is mostly modularization.  
  - **Cons**: `core` is not fully independent of Python, reducing portability and conceptual cleanliness.

- **Chosen approach: Hybrid toward Option A but minimal change (leaning toward pure Rust core)**  
  - Since `fastpd` is already pure Rust and `xgboost::parser`, `json_schema`, `types`, `tree_model` are JSON/Rust-only, they belong entirely in `core`.
  - For XGBoost integration:
    - Move all JSON parsing and model representation to `core`.
    - Move direct Python interaction (`get_dump`, `save_config`, base_score extraction) into the `python` crate as PyO3 helpers.
    - `core` will expose functions that operate on JSON strings and plain data, not Python objects.
  - For PD computations:
    - `FastPD<T>` and its methods belong to `core`.
    - The `FastPDPy` wrapper and `numpy` conversions live only in the `python` crate.
  - This yields a **clean `core` API** while keeping changes local to the binding layer.

## Design Update: pure FastPD core, XGBoost model in Python crate

- `core` will contain only the generic FastPD logic (`fastpd` module, including `Tree`, `TreeModel`, augmentation, caches, PD algorithms) and any JSON helpers that operate purely on Rust types.
- `XGBoostTreeModel` stays in the Python bindings crate as a `#[pyclass]` that wraps `core`’s `Tree` and implements `TreeModel`; all PyO3 and Python/XGBoost-interop code (`python_bridge`, `extract_trees_from_xgboost`, `FastPDPy`, numpy conversions) lives in the `python` crate.
- The implementation steps below should be interpreted with this adjustment: when they say “move XGBoost types into core”, it instead means “move only generic parsing helpers into core and keep XGBoost-specific, Python-facing wrappers in the `python` crate`.

## AI Agent Implementation Plan

**Chain-of-Thought Execution:**

### 1) Introduce Cargo workspace and crate skeletons — Foundation
   - **AI Actions**: 
     - Create root-level workspace `Cargo.toml` listing `core` and `python` as members.
     - Extract current package metadata from existing `Cargo.toml` into new crate manifests.
     - Create `core/Cargo.toml`, `core/src/lib.rs`, `python/Cargo.toml`, and `python/src/lib.rs` skeletons.
   - **Changes**:
     - New `Cargo.toml` at repository root with `[workspace]` and `members = ["core", "python"]`.
     - Move current `[package]` metadata from old `Cargo.toml` to `python/Cargo.toml`.
     - Create `core` crate with a new package name (e.g., `glex_core`) and dependencies (`ndarray`, `ndarray-linalg`, `serde`, `serde_json`, `thiserror`, etc.).
     - Configure `python/Cargo.toml` as a `cdylib` with `pyo3` and `numpy` dependencies, and add dependency on `core` via `path = "../core"`.
   - **Dependencies**:
     - Requires understanding of all current Rust dependencies to split them between `core` and `python`.
   - **Testing**:
     - `cargo build --workspace` to validate the new workspace structure compiles.
   - **Commit**:
     - Suggested message: `refactor: introduce core and python crates in workspace`
   - **Progress**: NOT BEGUN!

### 2) Move pure Rust core logic into `core` crate — Core algorithms
   - **AI Actions**:
     - Relocate the following modules into `core/src/`:
       - `fastpd` (all: `augment_eval.rs`, `augment.rs`, `augmented_tree.rs`, `cache.rs`, `error.rs`, `evaluate.rs`, `tree.rs`, `types.rs`, `mod.rs`).
       - `xgboost` submodules that are pure Rust: `json_schema.rs`, `parser.rs`, `tree_model.rs`, `types.rs`.
     - Create a new `core/src/lib.rs` that:
       - Re-exports the core types and functions needed by the bindings (e.g., `FastPD`, `XGBoostTreeModel`, tree-parsing functions or structs).
       - Ensures no `pyo3` or `numpy` dependencies in this crate.
   - **Changes**:
     - Replace `crate::fastpd::...` and `crate::xgboost::...` references with `glex_core::...` (or the chosen `core` crate name) in the `python` crate.
     - Ensure `core` has appropriate public modules and `pub use` statements to avoid leaking internal paths.
   - **Dependencies**:
     - Step 1 must be completed; `core` crate must be recognized by workspace.
   - **Testing**:
     - `cargo test --workspace` (or at least `cargo test -p glex_core`).
   - **Commit**:
     - Suggested message: `refactor(core): move fastpd and xgboost core logic into separate crate`
   - **Progress**: NOT BEGUN!

### 3) Rebuild PyO3 binding crate `python` — Python-facing layer
   - **AI Actions**:
     - Implement `python/src/lib.rs` as the new PyO3 binding module containing:
       - `#[pymodule] fn glex_rust(...)` to register functions/classes.
       - `#[pyclass]` wrapper `FastPDPy` holding a `glex_core::FastPD<glex_core::XGBoostTreeModel>` (or via type aliases).
       - `#[pyfunction]` wrappers for:
         - `extract_trees_from_xgboost` (renamed to `extract_trees_from_xgboost_py`) that:
           - Interacts with the Python XGBoost model to obtain JSON dumps and base_score.
           - Delegates JSON parsing and tree construction to `core` (via helper functions).
         - `all_subsets`, implemented by calling a pure Rust function in `core`.
     - Move or re-implement `python_bridge` logic in `python` crate:
       - Keep `get_booster_json_dumps` and `get_booster_base_score` in `python/src/xgboost/python_bridge.rs` (or similar) because they require `PyAny`.
       - Introduce a small Rust-only API in `core` to parse JSON into `XGBoostTreeModel`, and wire `python` crate to call it.
   - **Changes**:
     - The previous `src/lib.rs` will be effectively replaced by:
       - `core/src/lib.rs` (Rust-only).
       - `python/src/lib.rs` (PyO3).
     - All PyO3 attributes and `numpy` imports move into `python`.
     - `python` crate depends on `glex_core` for core computation.
   - **Dependencies**:
     - Steps 1 & 2 must be complete; `core` API exported.
   - **Testing**:
     - `cargo test --workspace` focusing on `python` crate when it has unit tests.
     - `maturin develop` followed by `pytest`.
   - **Commit**:
     - Suggested message: `refactor(python): move PyO3 bindings into dedicated crate`
   - **Progress**: NOT BEGUN!

### 4) Update `pyproject.toml` for maturin + new crate layout — Packaging
   - **AI Actions**:
     - Modify repo-level `pyproject.toml` to:
       - Set `tool.maturin.manifest-path = "python/Cargo.toml"`.
       - Set `tool.maturin.bindings = "pyo3"`.
       - Set `tool.maturin.module-name = "glex_rust.glex_rust"` (or similar), ensuring consistency with existing extension name.
       - Keep existing project metadata (name `glex-rust`, version, authors, classifiers).
     - Remove any `features = ["pyo3/extension-module"]` settings that were tied to the old single-crate layout if they become redundant.
   - **Changes**:
     - `pyproject.toml` `[tool.maturin]` section updated from:
       - `python-source = "python"` and `features = ["pyo3/extension-module"]`
     - To:
       - `manifest-path = "python/Cargo.toml"`, `bindings = "pyo3"`, `module-name = "glex_rust.glex_rust"`.
       - Optionally keep `python-source = "python"` if still desirable (for pure Python sources).
   - **Dependencies**:
     - `python` crate must be in place and buildable.
   - **Testing**:
     - Run `maturin develop` and ensure extension installs with the expected module name.
   - **Commit**:
     - Suggested message: `chore: update pyproject for new python crate layout`
   - **Progress**: NOT BEGUN!

### 5) Ensure Python package API compatibility — Import layer
   - **AI Actions**:
     - Verify that the compiled extension module name remains `glex_rust.glex_rust` so that `python/glex_rust/__init__.py` continues to work with:
       - `from .glex_rust import FastPDPy, all_subsets, extract_trees_from_xgboost`
     - If module naming changes (e.g. to `glex_rust._rust` or `glex_rust._glex`), update:
       - `__init__.py` imports to match.
       - `tool.maturin.module-name` to align.
     - Preserve the public API:
       - Keep `FastPD = FastPDPy` alias.
       - Maintain `__all__ = ["extract_trees_from_xgboost", "all_subsets", "FastPD", "FastPDPy"]`.
   - **Changes**:
     - Potentially adjust import line in `python/glex_rust/__init__.py` to whichever module name we finalize:
       - Example: `from ._rust import FastPDPy, all_subsets, extract_trees_from_xgboost`.
   - **Dependencies**:
     - `python` crate and `pyproject` must be configured to produce the correct extension module.
   - **Testing**:
     - `python -c "import glex_rust; help(glex_rust)"` to ensure objects are present.
     - Existing tests in `tests/` should import `glex_rust` as before.
   - **Commit**:
     - Suggested message: `fix(python): ensure package API remains backward compatible`
   - **Progress**: NOT BEGUN!

### 6) Fix imports, paths, and references across Rust crates — Wiring
   - **AI Actions**:
     - Update all Rust `use` paths and module paths to respect the new crate separation:
       - Replace `crate::fastpd::...` and `crate::xgboost::...` with `glex_core::fastpd::...` or re-exported paths from `glex_core`.
       - In `python` crate, ensure `use` paths point to `crate::...` for its own modules and `glex_core::...` for shared logic.
     - Remove any now-unused modules or duplicate definitions that may have been created during the move.
   - **Changes**:
     - Systematic search-and-replace with careful verification, especially in:
       - Former `src/lib.rs` (now `python/src/lib.rs`).
       - `src/xgboost/mod.rs` and related files.
       - Any tests or benchmarks referring to original crate structure (including `benchmarks/rust_benchmarks`).
   - **Dependencies**:
     - Steps 1–3 completed.
   - **Testing**:
     - `cargo build --workspace`.
     - Run Rust benches if applicable (`cargo bench` in `benchmarks/rust_benchmarks`).
   - **Commit**:
     - Suggested message: `refactor: update imports for workspace crates`
   - **Progress**: NOT BEGUN!

### 7) Preserve and validate functionality via tests and CI alignment — Verification
   - **AI Actions**:
     - Run test suites:
       - `cargo test --workspace`.
       - `maturin develop` (with virtualenv activated) and `pytest` for Python tests.
     - Fix any regressions that show up, focusing on:
       - Behavior of `FastPDPy.from_xgboost`, `FastPDPy.pd_function`, `FastPDPy.predict`.
       - `extract_trees_from_xgboost` correctness, especially base_score extraction.
     - Adjust CI scripts or local helper scripts if they rely on old crate paths, ensuring behavior is preserved.
   - **Changes**:
     - Minor adjustments to error types or function signatures only where required by the structural change, not by behavior.
   - **Dependencies**:
     - All structural steps (1–6) completed.
   - **Testing**:
     - As above, plus any benchmarks used to track performance regressions.
   - **Commit**:
     - Suggested message: `test: ensure workspace passes Rust and Python test suites`
   - **Progress**: NOT BEGUN!

## AI Agent Quality Gates

**Smart Validation Strategy:**
- [ ] **Functionality**: All existing features behave identically, as confirmed by Rust and Python tests.
- [ ] **Performance**: No measurable regressions in PD benchmarks (e.g., `benchmarks/rust_benchmarks` and `pytest-benchmark` tests).
- [ ] **Safety**: No panics in normal usage; error handling preserved or improved; consistent `PyErr` messages.
- [ ] **Integration**: 
  - `cargo build --workspace` and `cargo test --workspace` run cleanly.
  - `maturin develop` builds `python` crate and installs `glex_rust` module correctly.
- [ ] **Observability**: Existing logging or diagnostics remain intact; no regression in error messages.

## Strategic Considerations

**Performance & Risk Assessment:**
- [ ] **Critical Paths**:
  - `FastPD` computation and caching in `fastpd` module.
  - XGBoost tree parsing and JSON handling (`xgboost::parser`, `json_schema`, `types`, `tree_model`).
  - Bridge functions between Python XGBoost models and Rust structures (`python` crate).
- [ ] **Failure Modes**:
  - Misconfigured module names causing `ImportError` in Python.
  - Incorrect dependency split leading to missing symbols at link time.
  - Changes to type visibility in `core` causing compile-time errors.
- [ ] **Breaking Changes**:
  - Avoid renaming public Python functions or classes.
  - Ensure that `FastPD` alias and behavior are preserved.
- [ ] **Memory Management**:
  - Maintain zero-copy patterns and minimal allocations where possible in `core`.
  - Ensure that `numpy` array conversions and lifetime handling in `python` crate follow PyO3 patterns for safety.

## AI Agent Execution

**Tool Usage & Validation:**
- [ ] Use `run_terminal_cmd` for `cargo build --workspace`, `cargo test --workspace`, and `maturin develop` + `pytest`.
- [ ] Apply `read_lints` on updated files to catch lints early.
- [ ] Leverage `codebase_search`/`read_file` for validating that all moved modules satisfy dependencies.
- [ ] Maintain a clear rollback path: the original monolithic crate is preserved in git history; structural changes are grouped into atomic commits (workspace introduction, core extraction, binding refactor, packaging updates).
