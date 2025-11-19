"""Type stubs for glex_rust Rust extension module."""

from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

class XGBoostTreeModel:
    """XGBoost tree model extracted from an XGBoost ensemble."""

    def __repr__(self) -> str:
        """Returns a string representation of the tree model."""
        ...

    def format_tree(self) -> str:
        """Returns a formatted string showing the tree structure."""
        ...

    def num_nodes(self) -> int:
        """Returns the number of nodes in the tree."""
        ...

    def root(self) -> int:
        """Returns the root node index."""
        ...

class FastPDPy:
    """Python-facing wrapper for FastPD.

    This class provides efficient computation of partial dependence functions
    for tree-based models.
    """

    @classmethod
    def from_xgboost(
        cls,
        model: Any,
        background_samples: NDArray[np.float64],
        n_threads: int = 1,
    ) -> "FastPDPy":
        """Create a FastPD instance from an XGBoost model.

        Args:
            model: XGBoost model (Booster or XGBModel)
            background_samples: Background samples for PD estimation
                shape: (n_background, n_features)
            n_threads: Number of threads to use for parallelization (default: 1)

        Returns:
            FastPDPy instance
        """
        ...

    def pd_function(
        self,
        evaluation_points: NDArray[np.float64],
        feature_subset: List[int],
    ) -> NDArray[np.float32]:
        """Compute PD function v_S(x_S) for a single feature subset.

        Args:
            evaluation_points: Points at which to evaluate PD
                shape: (n_evaluation_points, n_features)
            feature_subset: Indices of features in subset S

        Returns:
            PD values at each evaluation point
                shape: (n_evaluation_points,)
        """
        ...

    def predict(
        self,
        evaluation_points: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Predicts the output for given input points.

        Args:
            evaluation_points: Points at which to predict
                shape: (n_evaluation_points, n_features)

        Returns:
            Predictions at each evaluation point
                shape: (n_evaluation_points,)
        """
        ...

    def num_trees(self) -> int:
        """Returns the number of trees in the ensemble."""
        ...

    def n_background(self) -> int:
        """Returns the number of background samples."""
        ...

    def n_features(self) -> int:
        """Returns the number of features."""
        ...

    def pd_functions_up_to_order(
        self,
        evaluation_points: NDArray[np.float64],
        max_order: int,
    ) -> Tuple[NDArray[np.float32], List[List[int]]]:
        """Compute plain partial dependence surfaces v_S(x_S) for all subsets S.

        Args:
            evaluation_points: Points at which to evaluate PD
                shape: (n_evaluation_points, n_features)
            max_order: Maximum interaction order (e.g., 1 for main effects, 2 for pairwise, etc.)

        Returns:
            Tuple of (pd_values, subsets) where:
            - pd_values: 2D numpy array of shape (n_eval, n_subsets) with one column per subset S
            - subsets: List of lists, where each inner list contains the feature indices for a subset
        """
        ...

    def functional_decomp_up_to_order(
        self,
        evaluation_points: NDArray[np.float64],
        max_order: int,
    ) -> Tuple[NDArray[np.float32], List[List[int]]]:
        """Compute functional decomposition components f_S(x_S) for all subsets S.

        Args:
            evaluation_points: Points at which to evaluate
                shape: (n_evaluation_points, n_features)
            max_order: Maximum interaction order

        Returns:
            Tuple of (comp_values, subsets) where:
            - comp_values: 2D numpy array of shape (n_eval, n_subsets) with one column per component f_S
            - subsets: List of lists, where each inner list contains the feature indices for a subset
        """
        ...

def extract_trees_from_xgboost(model: Any) -> List[XGBoostTreeModel]:
    """Extract trees from an XGBoost model.

    Args:
        model: XGBoost model (Booster or XGBModel)

    Returns:
        List of XGBoostTreeModel instances, one per tree in the ensemble
    """
    ...
