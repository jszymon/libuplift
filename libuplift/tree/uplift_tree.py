"""Uplift Tree implementation with recursive partitioning.
"""

import numbers

import numpy as np
#from scipy.stats import entropy # too slow!
from scipy.special import rel_entr

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree._tree import Tree

from ..base import UpliftClassifierMixin, UpliftRegressorMixin
from ..utils.validation import check_trt, check_consistent_length

class UpliftTreeNode:
    """Node in the uplift tree."""

    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None,
                 trt_y_ct=None, gain=None):
        self.feature_idx = feature_idx  # feature index for splitting
        self.threshold = threshold      # threshold for splitting
        self.left = left                # left child (samples <= threshold)
        self.right = right              # right child (samples > threshold)
        self.trt_y_ct = trt_y_ct        # treatment x y contingency table
        self.gain = gain                # split gain for the node
        # computed attributes
        # array of sample counts per treatment group
        self.n_samples_per_treatment = self.trt_y_ct.sum(axis=1)
        # number of samples in this node
        self.n_samples = self.n_samples_per_treatment.sum()
        # response probabilities per treatment
        with np.errstate(divide='ignore', invalid="ignore"):
            self.resp_P = self.trt_y_ct / self.n_samples_per_treatment.reshape(-1,1)
        # uplift value if leaf node
        ctrl_P = self.resp_P[0]
        if self.trt_y_ct.shape[0] == 2: # single treatment
            self.uplift = self.resp_P[1] - ctrl_P
        else:
            self.uplift = self.resp_P[1:] - ctrl_P.reshape(1,-1)

    def is_leaf(self):
        """Check if this node is a leaf."""
        return self.left is None and self.right is None


class UpliftTreeBase(BaseEstimator):
    """Base class for proper uplift trees with recursive partitioning."""

    def __init__(self, splitting_criterion="E", max_depth=None,
                 min_samples_split=100, min_samples_leaf=100,
                 min_weight_fraction_leaf=None, max_features=None, random_state=None,
                 max_bins=None):
        super().__init__()
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        # if not None, features are binned into at most max_bins
        # quantile bins and split points are searched using
        # histograms (faster, exact for features with at most
        # max_bins distinct values)
        self.max_bins = max_bins

    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        """Fit the uplift tree."""
        # These will be set during fitting
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.tree_ = None

        # Store feature names before X is converted to an array
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()

        # Validate and prepare data
        X, y = check_X_y(X, y)
        trt, n_trt = check_trt(trt, n_trt)
        if sample_weight is None:
            check_consistent_length(X, y, trt)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(X, y, trt, sample_weight)
        self._set_fit_params(y, trt, n_trt)

        self._check_splitting_criterion(self.splitting_criterion)

        # Store feature information
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # compute the number of selected features for random forest
        if self.max_features is None:
            self.max_features_ = None
        elif self.max_features == 'sqrt':
            self.max_features_ = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            self.max_features_ = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, numbers.Integral):
            self.max_features_ = max(1, min(self.max_features, n_features))
        elif isinstance(self.max_features, numbers.Real):
            # fraction of features
            self.max_features_ = max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features!r}")
        if self.max_bins is not None and \
           (not isinstance(self.max_bins, numbers.Integral) or self.max_bins < 2):
            raise ValueError(f"max_bins must be None or an integer >= 2,"
                             f" got {self.max_bins!r}")
        self._rng = check_random_state(self.random_state)

        # combined treatment x class code of each sample, classes
        # encoded as 0..n_classes_-1
        y_enc = np.searchsorted(self.classes_, y)
        trt_y_code = trt * self.n_classes_ + y_enc

        # Build the tree using recursive partitioning
        if self.max_bins is None:
            # sort each feature once; children inherit sorted orders by
            # stable partitioning, so no sorting is needed below the root
            # smallest integer type holding all sample indices
            order = np.argsort(X, axis=0, kind="stable").T.astype(
                np.min_scalar_type(X.shape[0] - 1))
            go_left = np.empty(X.shape[0], dtype=bool)
            self.tree_ = self._build_tree(X, trt_y_code, sample_weight, order, go_left, depth=0)
        else:
            X_binned = self._bin_features(X)
            self.tree_ = self._build_tree_hist(X, X_binned, trt_y_code, sample_weight,
                                               np.arange(X.shape[0]), depth=0)
        del self._rng

        return self

    def _build_tree(self, X, trt_y_code, sample_weight, order, go_left, depth, trt_y_ct=None):
        """Recursively build the uplift tree.

        order is a (n_features, n_node_samples) array of sample indices
        of the node sorted by each feature.  go_left is a work buffer
        shared by all nodes.
        """
        # numpy converts index arrays to intp when indexing, do it once
        idx = order[0].astype(np.intp)

        # compute global statistics
        if trt_y_ct is None:
            trt_y_ct = self._calculate_trt_y_ct(trt_y_code[idx],
                sample_weight[idx] if sample_weight is not None else None)

        # Check stopping conditions
        if self._stop_splitting(trt_y_ct, depth):
            return UpliftTreeNode(trt_y_ct=trt_y_ct)

        # Find best split
        best_split, best_uplift_gain, best_left_ct, best_right_ct =\
            self._find_best_split(X, trt_y_code, sample_weight, order)

        if best_split is None:
            # No good split found, create leaf node
            return UpliftTreeNode(trt_y_ct=trt_y_ct)

        feature_idx, threshold = best_split

        # Split the sorted orders of all features, keeping them sorted
        go_left[idx] = X[idx, feature_idx] <= threshold
        left_mask = go_left[order]
        n_left = np.count_nonzero(left_mask[0])
        left_order = order[left_mask].reshape(order.shape[0], n_left)
        right_order = order[~left_mask].reshape(order.shape[0], -1)
        # free before recursing, otherwise every ancestor keeps them alive
        del left_mask, idx

        # Create left and right children
        left_child = self._build_tree(X, trt_y_code, sample_weight, left_order, go_left,
                                      depth + 1, trt_y_ct=best_left_ct)
        right_child = self._build_tree(X, trt_y_code, sample_weight, right_order, go_left,
                                       depth + 1, trt_y_ct=best_right_ct)

        # Create internal node
        utn = UpliftTreeNode(feature_idx=feature_idx, threshold=threshold,
                             left=left_child, right=right_child,
                             trt_y_ct=trt_y_ct, gain=float(best_uplift_gain))
        return utn

    def _stop_splitting(self, trt_y_ct, depth):
        """Check stopping conditions for a node."""
        return (self.max_depth is not None and depth >= self.max_depth) or \
               (trt_y_ct.sum(axis=1).min() < self.min_samples_split)

    def _bin_features(self, X):
        """Replace each feature by its bin number, at most max_bins
        bins.  Features with at most max_bins distinct values get one
        bin per value, others are binned using quantiles."""
        n_samples = X.shape[0]
        # smallest integer type holding all bin numbers (uint8 for
        # max_bins <= 256)
        X_binned = np.empty(X.shape, dtype=np.min_scalar_type(self.max_bins - 1))
        # quantile positions among order statistics, interpolated
        # linearly as in np.quantile
        q = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        pos = q * (n_samples - 1)
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, n_samples - 1)
        t = pos - lo
        for feature_idx in range(X.shape[1]):
            u, inv, counts = np.unique(X[:, feature_idx], return_inverse=True,
                                       return_counts=True)
            if len(u) <= self.max_bins:
                X_binned[:, feature_idx] = inv
                continue
            # order statistics from the sorted unique values, the k-th
            # one is the first value whose cumulative count exceeds k
            cum_counts = np.cumsum(counts)
            a = u[np.searchsorted(cum_counts, lo, side="right")]
            b = u[np.searchsorted(cum_counts, hi, side="right")]
            edges = np.unique(np.where(t < 0.5, a + (b - a) * t, b - (b - a) * (1 - t)))
            # bin b contains edges[b-1] < x <= edges[b]; bin each
            # unique value, then map to samples
            X_binned[:, feature_idx] = np.searchsorted(edges, u, side="left")[inv]
        return X_binned

    def _build_tree_hist(self, X, X_binned, trt_y_code, sample_weight, idx, depth, trt_y_ct=None):
        """Recursively build the uplift tree using binned features.

        idx contains indices of samples in the node."""
        # compute global statistics
        if trt_y_ct is None:
            trt_y_ct = self._calculate_trt_y_ct(trt_y_code[idx],
                sample_weight[idx] if sample_weight is not None else None)

        # Check stopping conditions
        if self._stop_splitting(trt_y_ct, depth):
            return UpliftTreeNode(trt_y_ct=trt_y_ct)

        # Find best split
        best_split, best_uplift_gain, best_left_ct, best_right_ct =\
            self._find_best_split_hist(X_binned, trt_y_code, sample_weight, idx)

        if best_split is None:
            # No good split found, create leaf node
            return UpliftTreeNode(trt_y_ct=trt_y_ct)

        feature_idx, best_bin = best_split

        # Split the data, threshold halfway between the largest value
        # going left and the smallest value going right
        left_mask = X_binned[idx, feature_idx] <= best_bin
        x = X[idx, feature_idx]
        threshold = (x[left_mask].max() + x[~left_mask].min()) / 2

        # Create left and right children
        left_child = self._build_tree_hist(X, X_binned, trt_y_code, sample_weight, idx[left_mask],
                                           depth + 1, trt_y_ct=best_left_ct)
        right_child = self._build_tree_hist(X, X_binned, trt_y_code, sample_weight, idx[~left_mask],
                                            depth + 1, trt_y_ct=best_right_ct)

        # Create internal node
        utn = UpliftTreeNode(feature_idx=feature_idx, threshold=threshold,
                             left=left_child, right=right_child,
                             trt_y_ct=trt_y_ct, gain=float(best_uplift_gain))
        return utn

    def _find_best_split_hist(self, X_binned, trt_y_code, sample_weight, idx):
        """Find the best split using per feature histograms.  Returns
        ((feature_idx, bin), gain, left_ct, right_ct), the left child
        gets samples in bins <= bin."""
        n_bins = self.max_bins
        n_trt_y = (self.n_trt_ + 1) * self.n_classes_

        # Determine which features to consider
        feature_indices = np.asarray(self._select_feature_subset(X_binned.shape[1]))
        n_feat = len(feature_indices)

        # treatment x class histograms of all features with a single
        # bincount, cell index is (feature, bin, treatment x class),
        # computed in place to avoid (n, n_feat) int64 temporaries
        cell = X_binned[np.ix_(idx, feature_indices)].astype(np.intp)
        cell += np.arange(n_feat) * n_bins
        cell *= n_trt_y
        cell += trt_y_code[idx, None]
        weights = np.repeat(sample_weight[idx], n_feat) if sample_weight is not None else None
        hist = np.bincount(cell.ravel(), weights=weights, minlength=n_feat * n_bins * n_trt_y)
        if sample_weight is None:
            hist = hist.astype(np.int64)

        # cumulative counts over bins, (feature, bin, treatment, class)
        trt_y_cum = hist.reshape(n_feat, n_bins, self.n_trt_ + 1, self.n_classes_).cumsum(axis=1)
        trt_y_tot = trt_y_cum[:, -1]
        n_cum_left = trt_y_cum.sum(axis=3)
        n_cum_right = trt_y_tot.sum(axis=2)[:, None, :] - n_cum_left

        # candidate splits with enough samples of each treatment in
        # both leaves
        cand_mask = (n_cum_left.min(axis=2) >= self.min_samples_leaf) & \
                    (n_cum_right.min(axis=2) >= self.min_samples_leaf)
        cand_f, cand_b = np.nonzero(cand_mask)
        if cand_f.size == 0:
            return None, -np.inf, None, None

        left_ct = trt_y_cum[cand_f, cand_b].transpose(1, 2, 0)
        total_ct = trt_y_tot[cand_f].transpose(1, 2, 0)
        gain = self._compute_gain(left_ct, total_ct,
                                  n_cum_left[cand_f, cand_b].T, n_cum_right[cand_f, cand_b].T)
        gain[np.isnan(gain)] = -np.inf

        best_j = np.argmax(gain)
        if gain[best_j] == -np.inf:
            return None, -np.inf, None, None
        best_split = (feature_indices[cand_f[best_j]], cand_b[best_j])
        best_left_ct = left_ct[:, :, best_j].copy()
        best_right_ct = total_ct[:, :, best_j] - best_left_ct
        return best_split, gain[best_j], best_left_ct, best_right_ct

    def _select_feature_subset(self, n_features):
        """Select a subset of features for random forest."""
        if self.max_features_ is not None and self.max_features_ < n_features:
            # Randomly select features
            feature_indices = self._rng.choice(n_features, self.max_features_, replace=False)
        else:
            feature_indices = range(n_features)
        return feature_indices

    def _find_best_split(self, X, trt_y_code, sample_weight, order):
        """Find the best split that maximizes uplift."""
        best_gain = -np.inf
        best_split = None
        best_left_ct = None
        best_right_ct = None
        n_trt_y = (self.n_trt_ + 1) * self.n_classes_
        n_node = order.shape[1]
        cum_dtype = np.int64 if sample_weight is None else np.float64

        # Determine which features to consider
        feature_indices = self._select_feature_subset(X.shape[1])

        # Try each feature
        for feature_idx in feature_indices:
            sort_idx = order[feature_idx].astype(np.intp)
            x_sorted = X[:,feature_idx][sort_idx]

            # cumulative treatment x class counts, layout (n_trt+1,
            # n_classes, n_node) so that reductions over the small
            # treatment/class axes are cheap
            trt_y_cum = np.zeros((n_trt_y, n_node), dtype=cum_dtype)
            if sample_weight is not None:
                trt_y_cum[trt_y_code[sort_idx], np.arange(n_node)] = sample_weight[sort_idx]
            else:
                trt_y_cum[trt_y_code[sort_idx], np.arange(n_node)] = 1
            np.cumsum(trt_y_cum, axis=1, out=trt_y_cum)
            trt_y_cum = trt_y_cum.reshape(self.n_trt_ + 1, self.n_classes_, n_node)
            n_cum_left = trt_y_cum.sum(axis=1)
            n_cum_right = n_cum_left[:, -1:] - n_cum_left

            # candidate thresholds: only where the feature changes and
            # there are enough samples of each treatment in both leaves
            cand_mask = np.empty(n_node, dtype=bool)
            cand_mask[:-1] = x_sorted[1:] != x_sorted[:-1]
            cand_mask[-1] = False
            cand_mask &= n_cum_left.min(axis=0) >= self.min_samples_leaf
            cand_mask &= n_cum_right.min(axis=0) >= self.min_samples_leaf
            cand = np.flatnonzero(cand_mask)

            # if no splits can be valid skip feature
            if cand.size == 0:
                continue

            # with many candidates (e.g. continuous features) use a
            # contiguous range of positions (views, no copying) and
            # mask non-candidates afterwards, otherwise gather the
            # candidate positions
            dense = cand.size > n_node // 4
            sel = slice(cand[0], cand[-1] + 1) if dense else cand
            left_ct = trt_y_cum[:, :, sel]
            gain = self._compute_gain(left_ct, trt_y_cum[:, :, -1:],
                                      n_cum_left[:, sel], n_cum_right[:, sel])
            if dense:
                gain[~cand_mask[sel]] = -np.inf
            gain[np.isnan(gain)] = -np.inf

            best_j = np.argmax(gain)
            gain_i = gain[best_j]

            if gain_i > best_gain:
                best_i = cand[0] + best_j if dense else cand[best_j]
                best_gain = gain_i
                best_split = (feature_idx, (x_sorted[best_i] + x_sorted[best_i+1]) / 2)
                # copy, so that tree nodes do not keep the large
                # cumulative count arrays alive
                best_left_ct = trt_y_cum[:, :, best_i].copy()
                best_right_ct = trt_y_cum[:, :, -1] - best_left_ct

        return best_split, best_gain, best_left_ct, best_right_ct

    def _compute_gain(self, left_ct, total_ct, n_left, n_right):
        """Compute split gains.

        left_ct are (n_trt+1, n_classes, n_splits) counts in the left
        child for each candidate split, total_ct are node counts
        broadcastable to left_ct, n_left, n_right are (n_trt+1,
        n_splits) per treatment totals of the left/right child.
        """
        if self.n_trt_ == 1 and self.n_classes_ == 2 and \
           self.splitting_criterion in ("E", "DeltaDeltaP"):
            return self._compute_gain_binary(left_ct, total_ct, n_left, n_right)
        right_ct = total_ct - left_ct
        # Outcome probabilities used by all measures
        with np.errstate(divide='ignore', invalid="ignore"):
            P_left = left_ct / n_left[:, None, :]
            P_right = right_ct / n_right[:, None, :]
            if self.splitting_criterion == "DeltaDeltaP":
                gain = np.abs((P_left[1:] - P_left[:1])
                              - (P_right[1:] - P_right[:1])).max(axis=(0,1))
            elif self.splitting_criterion in ["E", "KL", "Chi2"]:
                if self.splitting_criterion == "E":
                    D_left  = np.square(P_left[1:] - P_left[:1]).sum(axis=(0,1))
                    D_right = np.square(P_right[1:] - P_right[:1]).sum(axis=(0,1))
                if self.splitting_criterion == "Chi2":
                    D_left  = np.where(P_left[:1] != 0,
                                       np.square(P_left[1:] - P_left[:1]) / P_left[:1],
                                       np.inf).sum(axis=(0,1))
                    D_right = np.where(P_right[:1] != 0,
                                       np.square(P_right[1:] - P_right[:1]) / P_right[:1],
                                       np.inf).sum(axis=(0,1))
                if self.splitting_criterion == "KL":
                    D_left  = rel_entr(P_left[1:], P_left[:1]).sum(axis=(0,1))
                    D_right = rel_entr(P_right[1:], P_right[:1]).sum(axis=(0,1))
                n_left_tot = n_left.sum(axis=0)
                P_avg_left = n_left_tot / (n_left_tot + n_right.sum(axis=0))
                P_avg_right = 1 - P_avg_left
                gain = P_avg_left * D_left + P_avg_right * D_right
            else:
                raise NotImplementedError()
        return gain

    def _compute_gain_binary(self, left_ct, total_ct, n_left, n_right):
        """_compute_gain for a single treatment and binary outcome.

        Uses closed forms based on the response rates (class 1
        probabilities) only, since P(y=0) = 1 - P(y=1).  Only for E
        and DeltaDeltaP: for KL and Chi2 using 1 - P(y=1) instead of
        P(y=0) changes results where the gain is infinite (class
        probabilities 0 or 1)."""
        left_1 = left_ct[:, 1]
        right_1 = total_ct[:, 1] - left_1
        with np.errstate(divide='ignore', invalid="ignore"):
            # response rates, (treatment, n_splits)
            p_left = left_1 / n_left
            p_right = right_1 / n_right
            if self.splitting_criterion == "DeltaDeltaP":
                return np.abs((p_left[1] - p_left[0]) - (p_right[1] - p_right[0]))
            if self.splitting_criterion == "E":
                # both classes contribute the same squared difference
                D_left = 2 * np.square(p_left[1] - p_left[0])
                D_right = 2 * np.square(p_right[1] - p_right[0])
            else:
                raise NotImplementedError()
            n_left_tot = n_left.sum(axis=0)
            P_avg_left = n_left_tot / (n_left_tot + n_right.sum(axis=0))
            return P_avg_left * D_left + (1 - P_avg_left) * D_right

    def _calculate_trt_y_ct(self, trt_y_code, sample_weight=None):
        """Calculate treatment x outcome contingency table."""
        trt_y_ct = np.bincount(trt_y_code, weights=sample_weight,
                               minlength=(self.n_trt_ + 1) * self.n_classes_)
        return trt_y_ct.reshape(self.n_trt_ + 1, self.n_classes_)

    def _predict_uplift(self, X):
        """Predict uplift for each sample by passing sample index sets
        down the tree."""
        predictions = np.empty((X.shape[0],) + np.shape(self.tree_.uplift))
        stack = [(self.tree_, np.arange(X.shape[0]))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf():
                predictions[idx] = node.uplift
                continue
            # Traverse to child nodes
            left_mask = X[idx, node.feature_idx] <= node.threshold
            stack.append((node.left, idx[left_mask]))
            stack.append((node.right, idx[~left_mask]))
        return predictions

class UpliftTreeClassifier(UpliftClassifierMixin, UpliftTreeBase):
    """Uplift Tree Classifier with proper recursive partitioning."""

    def predict(self, X):
        """Predict uplift for each sample."""
        # TODO!!!  needs tags in base uplift
        #check_is_fitted(self, ['tree_'])
        X = check_array(X)
        return self._predict_uplift(X)

    def _check_splitting_criterion(self, criterion):
        if criterion not in ["DeltaDeltaP", "E", "KL", "Chi2"]:
            raise ValueError(f"UpliftTreeClassifier: wrong spliting criterion"
                             f" {criterion}. Available criteria: DeltaDeltaP,"
                             f" E, KL, Chi2")

# class UpliftTreeRegressor(UpliftTreeBase, UpliftRegressorMixin):
#     """Uplift Tree Regressor with proper recursive partitioning."""
#
#     def predict(self, X):
#         """Predict uplift for each sample."""
#         # TODO!!!  needs tags in base uplift
#         #check_is_fitted(self, ['tree_'])
#         X = check_array(X)
#         return self._predict_uplift(X)
