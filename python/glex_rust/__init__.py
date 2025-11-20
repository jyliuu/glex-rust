"""
Python package wrapper for the Rust extension.

When built with maturin using `python-source = "python"` and a Rust library
named `glex_rust`, the compiled extension is installed as the submodule
`glex_rust.glex_rust`. This file re-exports the Rust functions at the
package level so callers can simply `import glex_rust` and use them.
"""

from .glex_rust import FastPDPy, extract_trees_from_xgboost
from .visualization import plot_1d_components, plot_2d_interactions

# Alias FastPDPy as FastPD for cleaner API
FastPD = FastPDPy

# Add visualization methods to FastPD class
FastPD.plot_1d_components = plot_1d_components
FastPD.plot_2d_interactions = plot_2d_interactions

__all__ = [
    "extract_trees_from_xgboost",
    "FastPD",
    "FastPDPy",
]
