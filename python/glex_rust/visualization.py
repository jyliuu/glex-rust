"""
Visualization functions for functional decomposition components.

These functions are added as methods to the FastPD class.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX-style math rendering in matplotlib
plt.rcParams["mathtext.default"] = "regular"

# Default number of evaluation points for default plotting
DEFAULT_N_POINTS = 1000  # For 1D components


def _resolve_feature(fastpd, feature: Union[str, int]) -> int:
    """Resolve feature name or index to feature index.

    Args:
        fastpd: FastPD instance
        feature: Feature name (str) or index (int)

    Returns:
        Feature index (int)

    Raises:
        ValueError: If feature name not found
        IndexError: If feature index out of range
    """
    if isinstance(feature, int):
        n_features = fastpd.n_features()
        if not 0 <= feature < n_features:
            raise IndexError(f"Feature index {feature} out of range [0, {n_features})")
        return feature
    elif isinstance(feature, str):
        feature_names = fastpd.feature_names()
        if feature_names is None:
            raise ValueError(f"Feature names not available, cannot resolve '{feature}'")
        try:
            return feature_names.index(feature)
        except ValueError:
            raise ValueError(f"Feature '{feature}' not found in feature names")
    else:
        raise TypeError(f"Feature must be str or int, got {type(feature)}")


def _generate_default_evaluation_points_1d(fastpd) -> np.ndarray:
    """Generate default evaluation points for all 1D components.

    For each feature, generates DEFAULT_N_POINTS evenly spaced points
    along that feature's range, with other features set to their mean.

    Args:
        fastpd: FastPD instance

    Returns:
        Evaluation points array of shape (DEFAULT_N_POINTS * n_features, n_features)
        as float64, contiguous array
    """
    background_samples = np.asarray(fastpd.get_background_samples(), dtype=np.float64)
    n_features = fastpd.n_features()
    feature_means = np.mean(background_samples, axis=0)

    eval_points_list = []
    for feature_idx in range(n_features):
        feature_values = background_samples[:, feature_idx]
        feature_points = np.linspace(
            feature_values.min(), feature_values.max(), DEFAULT_N_POINTS
        )
        eval_matrix = np.tile(feature_means, (DEFAULT_N_POINTS, 1))
        eval_matrix[:, feature_idx] = feature_points
        eval_points_list.append(eval_matrix)

    return np.ascontiguousarray(np.vstack(eval_points_list), dtype=np.float64)


def _generate_default_evaluation_points_2d(fastpd) -> np.ndarray:
    """Generate default evaluation points for all 2D interactions.

    Uses the actual background samples (all rows) as evaluation points.
    For each pair, we'll extract the two feature columns when plotting.

    Args:
        fastpd: FastPD instance

    Returns:
        Evaluation points array of shape (n_background, n_features)
        as float64, contiguous array (just the background samples)
    """
    return np.ascontiguousarray(
        np.asarray(fastpd.get_background_samples(), dtype=np.float64)
    )


def _extract_component_by_subset(
    comp_values: np.ndarray,
    subsets: List[List[int]],
    subset: List[int],
) -> np.ndarray:
    """Extract component values for a specific subset.

    Args:
        comp_values: Component values array of shape (n_eval, n_subsets)
        subsets: List of feature subsets, parallel to comp_values columns
        subset: Feature subset to find (e.g., [0, 1] for interaction)

    Returns:
        Component values array of shape (n_eval,)

    Raises:
        ValueError: If subset not found
    """
    subset_sorted = sorted(subset)
    for idx, cached_subset in enumerate(subsets):
        if sorted(cached_subset) == subset_sorted:
            return comp_values[:, idx]
    raise ValueError(f"Subset {subset} not found in cached components")


def _flatten_axes(axes) -> List:
    """Flatten axes from plt.subplots to a list."""
    if not hasattr(axes, "__len__"):
        return [axes]
    return axes.flatten().tolist() if hasattr(axes, "flatten") else list(axes)


def _create_subplots(n_plots: int, n_cols: int, figsize: Tuple[int, int]):
    """Create subplot grid and return figure and flattened axes list.

    For single plots, creates a single subplot that fills the entire figure.
    """
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, [ax]
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    return fig, _flatten_axes(axes)


def _ensure_components_cached(fastpd, max_order: int) -> None:
    """Ensure components are cached up to the specified order.

    If not cached or cached order insufficient, computes and caches components.

    Args:
        fastpd: FastPD instance
        max_order: Maximum order needed (1 for 1D, 2 for 2D, etc.)
    """
    cached = fastpd.get_cached_components()
    if cached is not None and cached[3] >= max_order:
        return  # Already cached with sufficient order

    eval_points = (
        _generate_default_evaluation_points_1d(fastpd)
        if max_order == 1
        else _generate_default_evaluation_points_2d(fastpd)
        if max_order == 2
        else None
    )
    if eval_points is None:
        raise ValueError(f"max_order {max_order} not yet supported (only 1 and 2)")

    comp_values, subsets = fastpd.functional_decomp_up_to_order(
        evaluation_points=eval_points, max_order=max_order
    )

    fastpd.set_cached_components(
        comp_values=np.ascontiguousarray(comp_values, dtype=np.float32),
        subsets=subsets,
        eval_points=np.ascontiguousarray(eval_points, dtype=np.float32),
        max_order=max_order,
    )


def plot_1d_components(
    fastpd,
    features: Optional[Union[str, int, List[str], List[int]]] = None,
    evaluation_points: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5),
    n_cols: int = 3,
    **kwargs,
):
    """Plot 1D functional decomposition components.

    Args:
        fastpd: FastPD instance
        features: Features to plot. None for all, single str/int, or list of str/int
        evaluation_points: Optional custom evaluation points (bypasses cache)
        figsize: Figure size tuple
        n_cols: Number of columns in subplot grid
        **kwargs: Additional arguments passed to plot()

    Returns:
        matplotlib.figure.Figure
    """
    # Resolve features to indices
    if features is None:
        feature_indices = list(range(fastpd.n_features()))
    elif isinstance(features, (str, int)):
        feature_indices = [_resolve_feature(fastpd, features)]
    else:
        feature_indices = [_resolve_feature(fastpd, f) for f in features]

    # Get components and evaluation points
    if evaluation_points is not None:
        eval_points = np.ascontiguousarray(evaluation_points, dtype=np.float64)
        comp_values, subsets = fastpd.functional_decomp_up_to_order(
            evaluation_points=eval_points, max_order=1
        )
    else:
        _ensure_components_cached(fastpd, max_order=1)
        cached = fastpd.get_cached_components()
        if cached is None:
            raise RuntimeError("Failed to cache components")
        comp_values, subsets, eval_points, _ = cached

    # Filter to 1D components and extract requested features
    plot_data = []
    for feature_idx in feature_indices:
        try:
            comp_vals = _extract_component_by_subset(
                comp_values, subsets, [feature_idx]
            )
            if evaluation_points is None:
                start_idx = feature_idx * DEFAULT_N_POINTS
                feature_eval_points = eval_points[
                    start_idx : start_idx + DEFAULT_N_POINTS, feature_idx
                ]
            else:
                feature_eval_points = eval_points[:, feature_idx]
            plot_data.append(
                (comp_vals, feature_eval_points, fastpd.get_feature_name(feature_idx))
            )
        except ValueError:
            continue

    if not plot_data:
        raise ValueError("No components found for requested features")

    # Create subplot grid
    fig, axes = _create_subplots(len(plot_data), n_cols, figsize)

    # Plot each component
    for idx, (comp_vals, eval_pts, feat_name) in enumerate(plot_data):
        ax = axes[idx]
        sort_idx = np.argsort(eval_pts)
        ax.plot(eval_pts[sort_idx], comp_vals[sort_idx], **kwargs)
        ax.set_xlabel(feat_name)
        ax.set_title(f"$f_{{{feat_name}}}$")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    # Hide unused subplots
    for idx in range(len(plot_data), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_2d_interactions(
    fastpd,
    features: Optional[
        Union[
            Tuple[str, str],
            Tuple[int, int],
            List[Tuple[str, str]],
            List[Tuple[int, int]],
        ]
    ] = None,
    evaluation_points: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5),
    n_cols: int = 2,
    cmap: str = "RdBu_r",
    alpha: float = 0.9,
    **kwargs,
):
    """Plot 2D functional decomposition interaction components.

    Args:
        fastpd: FastPD instance
        features: Interaction pairs to plot. None for all, single tuple, or list of tuples
        evaluation_points: Optional custom evaluation points (bypasses cache)
        figsize: Figure size tuple
        n_cols: Number of columns in subplot grid
        cmap: Colormap for scatter plot
        alpha: Transparency for scatter points
        **kwargs: Additional arguments passed to scatter()

    Returns:
        matplotlib.figure.Figure
    """
    # Resolve feature pairs to indices
    if features is None:
        n_features = fastpd.n_features()
        feature_pairs = [
            (i, j) for i in range(n_features) for j in range(i + 1, n_features)
        ]
    elif isinstance(features, tuple) and len(features) == 2:
        feature_pairs = [
            (
                _resolve_feature(fastpd, features[0]),
                _resolve_feature(fastpd, features[1]),
            )
        ]
    else:
        feature_pairs = [
            (_resolve_feature(fastpd, f[0]), _resolve_feature(fastpd, f[1]))
            for f in features
        ]

    # Get components and evaluation points
    if evaluation_points is not None:
        eval_points = np.ascontiguousarray(evaluation_points, dtype=np.float64)
        comp_values, subsets = fastpd.functional_decomp_up_to_order(
            evaluation_points=eval_points, max_order=2
        )
        eval_points_source = eval_points
    else:
        _ensure_components_cached(fastpd, max_order=2)
        cached = fastpd.get_cached_components()
        if cached is None:
            raise RuntimeError("Failed to cache components")
        comp_values, subsets, _, _ = cached
        eval_points_source = np.asarray(
            fastpd.get_background_samples(), dtype=np.float64
        )

    # Filter to 2D components and extract requested interactions
    plot_data = []
    for i, j in feature_pairs:
        try:
            comp_vals = _extract_component_by_subset(comp_values, subsets, [i, j])
            pair_eval_points = eval_points_source[:, [i, j]]
            plot_data.append(
                (
                    comp_vals,
                    pair_eval_points,
                    (fastpd.get_feature_name(i), fastpd.get_feature_name(j)),
                )
            )
        except ValueError:
            continue

    if not plot_data:
        raise ValueError("No interaction components found for requested feature pairs")

    # Create subplot grid
    n_plots = len(plot_data)
    fig, axes = _create_subplots(n_plots, n_cols, figsize)

    # Plot each interaction
    for idx, (comp_vals, eval_pts, (feat_name1, feat_name2)) in enumerate(plot_data):
        ax = axes[idx]

        # Calculate extended range for colorbar (5% padding on each side)
        v_min, v_max = comp_vals.min(), comp_vals.max()
        v_range = v_max - v_min
        padding = v_range * 0.05 if v_range > 0 else 0.01
        v_min_extended = v_min - padding
        v_max_extended = v_max + padding

        scatter = ax.scatter(
            eval_pts[:, 0],
            eval_pts[:, 1],
            c=comp_vals,
            cmap=cmap,
            alpha=alpha,
            vmin=v_min_extended,
            vmax=v_max_extended,
            edgecolors="none",  # Remove edge colors to prevent darkening from overlap
            **kwargs,
        )
        ax.set_xlabel(feat_name1)
        ax.set_ylabel(feat_name2)
        ax.set_title(f"$f_{{{feat_name1},{feat_name2}}}$")

        # Add colorbar with clear ticks (using actual data range for ticks)
        cbar = plt.colorbar(scatter, ax=ax)
        ticks = np.linspace(v_min, v_max, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick:.3f}" for tick in ticks])

    # Hide unused subplots (only needed when n_plots > 1)
    if n_plots > 1:
        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

    plt.tight_layout()
    return fig
