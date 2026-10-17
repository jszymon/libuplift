"""Tree export functionality for uplift trees.

This module provides functions to export uplift tree structures as
text or plot them using matplotlib, similar to scikit-learn's
export_text and plot_tree functions.
"""

import numbers

import numpy as np


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
        Number of decimal digits to display.  Split gains are shown
        with this many significant digits.
    
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
        gain_str = f"   (gain = {node.gain:.{decimals}g})" if node.gain is not None else ""

        # Left child (samples <= threshold)
        left_prefix = prefix + "|" + (" " * spacing)
        lines.append(f"{prefix}|--- {feature_name} <= {threshold_str}{gain_str}")
        _build_tree_text(node.left, feature_names, lines, left_prefix, decimals, spacing)
        
        # Right child (samples > threshold)
        right_prefix = prefix + (" " * (spacing+1))
        lines.append(f"{prefix}|--- {feature_name} >  {threshold_str}")
        _build_tree_text(node.right, feature_names, lines, right_prefix, decimals, spacing)


def plot_tree(tree, *, max_depth=None, feature_names=None, label="all",
              filled=False, cmap="RdBu", node_ids=False, proportion=False,
              rounded=False, precision=3, ax=None, fontsize=None):
    """Plot an uplift tree using matplotlib.

    The style follows :func:`sklearn.tree.plot_tree`.  Each internal
    node shows its split rule and gain, every node shows the number
    of samples (overall and per treatment) and the uplift estimate.
    The left child of a node contains samples satisfying the split
    rule.

    Parameters
    ----------
    tree : UpliftTreeBase instance
        The fitted uplift tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation.  If None, the tree
        is fully drawn.  Subtrees below this depth are shown as
        ``(...)``.

    feature_names : array-like of shape (n_features,), default=None
        Names of each of the features.  If None, the names seen
        during fit are used if available, otherwise generic names
        ("feature_0", "feature_1", ...).

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels such as ``samples =``.
        'all' shows them at every node, 'root' only at the root node
        and 'none' at no node.

    filled : bool, default=False
        When True, paint nodes according to their uplift.  The uplift
        of the first treatment for the last class (the positive class
        for binary targets) is used.  Colors are scaled symmetrically
        around zero.

    cmap : str or matplotlib Colormap, default='RdBu'
        Diverging colormap used when ``filled=True``.  The lower end
        corresponds to negative and the upper end to positive uplift.

    node_ids : bool, default=False
        When True, show the ID number on each node.

    proportion : bool, default=False
        When True, show sample counts as proportions of the total
        number of samples.

    rounded : bool, default=False
        When True, draw node boxes with rounded corners.

    precision : int, default=3
        Number of digits of precision for floating point values.

    ax : matplotlib Axes, default=None
        Axes to plot to.  If None, use the current axes.

    fontsize : int, default=None
        Size of text font.  If None, determined automatically to fit
        the figure.

    Returns
    -------
    annotations : list of matplotlib Annotation
        List containing the artists for the annotation boxes making
        up the tree.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Colormap

    if getattr(tree, "tree_", None) is None:
        raise ValueError("The uplift tree is not fitted yet.")
    if label not in ("all", "root", "none"):
        raise ValueError(f"label must be 'all', 'root' or 'none', got {label!r}")
    if max_depth is not None and (not isinstance(max_depth, numbers.Integral)
                                  or max_depth < 0):
        raise ValueError(f"max_depth must be None or a non-negative integer,"
                         f" got {max_depth!r}")
    if feature_names is None:
        feature_names = getattr(tree, "feature_names_in_", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree.n_features_in_)]
    if ax is None:
        ax = plt.gca()
    _disconnect_rescale(ax)
    ax.clear()
    ax.set_axis_off()
    if not isinstance(cmap, Colormap):
        cmap = colormaps[cmap]

    root = tree.tree_
    total_samples = root.n_samples

    # Compute layout: leaves (and truncated subtrees) are placed at
    # successive horizontal positions, internal nodes are centered
    # above their children.  Each entry is
    # (node, node_id, depth, x, parent_index); node is None for
    # truncated subtrees.
    layout = []
    n_leaves = [0]
    next_id = [0]
    def _layout(node, depth, parent):
        node_id = next_id[0]
        next_id[0] += 1
        idx = len(layout)
        layout.append(None)
        if node is None or node.is_leaf():
            x = n_leaves[0]
            n_leaves[0] += 1
        elif max_depth is not None and depth >= max_depth:
            _layout(None, depth + 1, idx)
            x = layout[idx + 1][3]
            _skip_ids(node.left)
            _skip_ids(node.right)
        else:
            left_x = _layout(node.left, depth + 1, idx)
            right_x = _layout(node.right, depth + 1, idx)
            x = (left_x + right_x) / 2
        layout[idx] = (node, node_id, depth, x, parent)
        return x
    def _skip_ids(node):
        # keep node ids consistent with the full tree
        next_id[0] += 1
        if not node.is_leaf():
            _skip_ids(node.left)
            _skip_ids(node.right)
    _layout(root, 0, None)
    tree_depth = max(entry[2] for entry in layout)

    def _fill_value(node):
        u = np.asarray(node.uplift, dtype=float)
        return u.ravel()[-1] if u.ndim == 1 else u[0, -1]
    if filled:
        fill_vals = [_fill_value(n) for n, *_ in layout if n is not None]
        vmax = np.nanmax(np.abs(fill_vals)) if np.any(np.isfinite(fill_vals)) else 0
    def _color(node):
        if not filled or node is None:
            return "white"
        v = _fill_value(node)
        if not np.isfinite(v):
            return "white"
        v = v / vmax if vmax > 0 else 0.0
        # stay away from colormap extremes to keep text readable
        return cmap(0.5 + 0.35 * v)

    def _fmt(v):
        v = float(v)
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.{precision}g}"
    def _fmt_array(a):
        a = np.asarray(a)
        return "[" + ", ".join(f"{v:.{precision}f}" for v in a) + "]"

    def _node_text(node, node_id):
        if node is None:
            return "(...)"
        show_labels = label == "all" or (label == "root" and node is root)
        lines = []
        if node_ids:
            lines.append(("node #" if show_labels else "#") + str(node_id))
        if not node.is_leaf():
            lines.append(f"{feature_names[node.feature_idx]} <= "
                         f"{node.threshold:.{precision}g}")
            if node.gain is not None:
                lines.append(("gain = " if show_labels else "")
                             + f"{node.gain:.{precision}g}")
        if proportion:
            n = f"{100 * node.n_samples / total_samples:.{precision}g}%"
            n_trt = _fmt_array(np.asarray(node.n_samples_per_treatment)
                               / node.n_samples)
        else:
            n = _fmt(node.n_samples)
            n_trt = "[" + ", ".join(_fmt(v) for v in
                                    node.n_samples_per_treatment) + "]"
        lines.append(("samples = " if show_labels else "") + n)
        lines.append(("samples per trt = " if show_labels else "") + n_trt)
        uplift = np.asarray(node.uplift)
        if uplift.ndim == 1:
            uplift_str = _fmt_array(uplift)
        else:
            uplift_str = "[" + ", ".join(_fmt_array(u) for u in uplift) + "]"
        lines.append(("uplift = " if show_labels else "") + uplift_str)
        return "\n".join(lines)

    # node centers are on an integer grid: leaves at successive x
    # positions, depth growing downwards.  Data coordinates are used so
    # that zooming and panning move the nodes.
    def _pos(entry):
        _, _, depth, x, _ = entry
        return x, -depth

    # feature names may contain '$', do not interpret them as mathtext
    kwargs = {"ha": "center", "va": "center", "annotation_clip": False,
              "parse_math": False}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    boxstyle = "round" if rounded else "square"
    arrow_args = {"arrowstyle": "<-, head_length=0.8, head_width=0.4"}

    annotations = []
    for entry in layout:
        node, node_id, depth, _, parent = entry
        bbox = {"boxstyle": boxstyle, "fc": _color(node), "ec": "black"}
        xy = _pos(entry)
        text = _node_text(node, node_id)
        # parent boxes cover the arrows pointing to their children
        zorder = 100 - 10 * depth
        if parent is None:
            ann = ax.annotate(text, xy, bbox=bbox, zorder=zorder, **kwargs)
        else:
            ann = ax.annotate(text, _pos(layout[parent]), xytext=xy,
                              bbox=bbox, arrowprops=arrow_args,
                              zorder=zorder, **kwargs)
        # do not draw parts of the tree zoomed out of the axes
        ann.set_clip_on(True)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_clip_box(ax.bbox)
        annotations.append(ann)

    # True/False labels on the edges from the root, as in scikit-learn.
    # They are placed above the middle of each edge, on its outer side,
    # so that they do not overlap the arrows.
    edge_labels = []
    root_children = [e for e in layout if e[4] == 0]
    if len(root_children) == 2:
        rx, ry = _pos(layout[0])
        for child, txt, ha, dx in zip(root_children, ("True", "False"),
                                      ("right", "left"), (-3, 3)):
            cx, cy = _pos(child)
            edge_labels.append(ax.annotate(
                txt, ((rx + cx) / 2, (ry + cy) / 2), xytext=(dx, 3),
                textcoords="offset points", ha=ha, va="bottom",
                fontsize=fontsize, annotation_clip=False, clip_on=True))

    # Choose font scale and axes limits.  Boxes are measured at the
    # initial font size; with font scaled by k, horizontal spacing
    # between leaves is at least 1.1 box widths and vertical spacing
    # between levels at least 2 box heights (room for arrows).  The
    # limits leave just enough space for the outermost boxes.
    renderer = ax.figure.canvas.get_renderer()
    for ann in annotations:
        ann.update_bbox_position_size(renderer)
    extents = [ann.get_bbox_patch().get_window_extent()
               for ann in annotations]
    pad = 2  # pixels
    box_w = max(e.width for e in extents) + pad
    box_h = max(e.height for e in extents) + pad
    ax_w, ax_h = ax.bbox.width, ax.bbox.height
    n_x, n_y = n_leaves[0] - 1, tree_depth
    if fontsize is None:
        k = min(1.0, ax_w / (box_w * (1 + 1.1 * n_x)),
                ax_h / (box_h * (1 + 2 * n_y)))
        fontsize = annotations[0].get_fontsize() * k
        for artist in annotations + edge_labels:
            artist.set_fontsize(fontsize)
    else:
        k = 1.0
    def _margin(n, ax_size, box_size):
        """Margin in data units around n+1 grid positions."""
        if n == 0:
            return 0.5
        units = (ax_size - box_size) / n  # pixels per data unit
        if units <= 0:  # boxes do not fit, overlap them
            units = box_size
        return box_size / 2 / units
    mx = _margin(n_x, ax_w, k * box_w)
    my = _margin(n_y, ax_h, k * box_h)
    ax.set_xlim(-mx, n_x + mx)
    ax.set_ylim(-n_y - my, my)

    # rescale fonts when zooming or resizing so that the tree keeps
    # its proportions
    def _inches_per_unit():
        dpi = ax.figure.dpi
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return (ax.bbox.width / dpi / abs(x1 - x0),
                ax.bbox.height / dpi / abs(y1 - y0))
    base_scale = _inches_per_unit()
    def _rescale(_=None):
        sx, sy = _inches_per_unit()
        factor = min(sx / base_scale[0], sy / base_scale[1])
        for artist in annotations + edge_labels:
            artist.set_fontsize(fontsize * factor)
    _connect_rescale(ax, _rescale)

    return annotations


def _disconnect_rescale(ax):
    """Remove font rescaling callbacks left by an earlier plot_tree
    call on the same axes."""
    cids = getattr(ax, "_plot_tree_cids", None)
    if cids is not None:
        ax_cids, fig_cid = cids
        for cid in ax_cids:
            ax.callbacks.disconnect(cid)
        ax.figure.canvas.mpl_disconnect(fig_cid)
        ax._plot_tree_cids = None


def _connect_rescale(ax, callback):
    """Connect plot_tree font rescaling callback to axes."""
    ax._plot_tree_cids = (
        [ax.callbacks.connect("xlim_changed", callback),
         ax.callbacks.connect("ylim_changed", callback)],
        ax.figure.canvas.mpl_connect("resize_event", callback))
