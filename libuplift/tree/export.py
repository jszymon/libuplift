"""Tree export functionality for uplift trees.

This module provides functions to export uplift tree structures as text,
similar to scikit-learn's export_text function.
"""


def export_text(tree, feature_names=None, decimals=2, spacing=3):
    """Build a text report showing the rules of an uplift tree.
    
    Parameters
    ----------
    tree : UpliftTreeBase instance
        The uplift tree estimator to be exported.
    
    feature_names : array-like of shape (n_features,), default=None
        Names of each of the features.
        If None, generic names will be used ("feature_0", "feature_1", ...).
    
    decimals : int, default=2
        Number of decimal digits to display.
    
    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.
    
    Returns
    -------
    report : str
        Text summary of all the rules in the uplift tree.
    """
    if tree.tree_ is None:
        return "Tree not fitted yet."
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree.n_features_in_)]
    
    # Build the tree text representation
    lines = []
    _build_tree_text(tree.tree_, feature_names, lines, "", decimals, spacing)
    
    return "\n".join(lines)


def _build_tree_text(node, feature_names, lines, prefix, decimals, spacing):
    """Recursively build text representation of a tree node."""
    if node.is_leaf():
        # Leaf node
        uplift_str = ", ".join([f"{val:.{decimals}f}" for val in node.uplift])
        lines.append(f"{prefix}|--- leaf: uplift = [{uplift_str}]")
        lines.append(f"{prefix}     samples = {node.n_samples}")
        lines.append(f"{prefix}     samples_per_treatment = {node.n_samples_per_treatment}")
    else:
        # Internal node
        feature_name = feature_names[node.feature_idx]
        threshold_str = f"{node.threshold:.{decimals}f}"
        
        # Left child (samples <= threshold)
        left_prefix = prefix + "|" + (" " * spacing)
        lines.append(f"{prefix}|--- {feature_name} <= {threshold_str}")
        _build_tree_text(node.left, feature_names, lines, left_prefix, decimals, spacing)
        
        # Right child (samples > threshold)
        right_prefix = prefix + (" " * (spacing+1))
        lines.append(f"{prefix}|--- {feature_name} >  {threshold_str}")
        _build_tree_text(node.right, feature_names, lines, right_prefix, decimals, spacing)
