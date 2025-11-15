"""
Python package wrapper for the Rust extension.

When built with maturin using `python-source = "python"` and a Rust library
named `glex_rust`, the compiled extension is installed as the submodule
`glex_rust.glex_rust`. This file re-exports the Rust functions at the
package level so callers can simply `import glex_rust` and use them.
"""

from .glex_rust import get_projection_beta, extract_trees_from_xgboost

__all__ = ["get_projection_beta", "extract_trees_from_xgboost"]
