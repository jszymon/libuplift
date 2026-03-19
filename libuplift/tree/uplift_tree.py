"""Uplift Tree implementation with recursive partitioning.
"""

import numpy as np
#from scipy.stats import entropy # too slow!
from scipy.special import rel_entr

from sklearn.base import BaseEstimator
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
                 min_weight_fraction_leaf=None, max_features=None, random_state=None):
        super().__init__()
        self.splitting_criterion = splitting_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        """Fit the uplift tree."""
        # These will be set during fitting
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.tree_ = None

        # Validate and prepare data
        X, y = check_X_y(X, y, accept_sparse="csr")
        trt, n_trt = check_trt(trt, n_trt)
        if sample_weight is None:
            check_consistent_length(X, y, trt)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(X, y, trt, sample_weight)
        self._set_fit_params(y, trt, n_trt)

        self._check_splitting_criterion(self.splitting_criterion)

        # Store feature information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()

        # compute the number of selected features for random forest
        if self.max_features is not None:
            if isinstance(self.max_features, (int, float)):
                self.max_features_ = min(int(self.max_features), n_features)
            elif self.max_features == 'sqrt':
                self.max_features_ = max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                self.max_features_ = max(1, int(np.log2(n_features)))
            else:
                self.max_features_ = n_features
        else:
            self.max_features_ = None

        # Build the tree using recursive partitioning
        self.tree_ = self._build_tree(X, y, trt, sample_weight, depth=0)
        
        return self

    def _build_tree(self, X, y, trt, sample_weight, depth, trt_y_ct=None):
        """Recursively build the uplift tree."""
        n_samples = X.shape[0]

        # compute global statistics
        if trt_y_ct is None:
            trt_y_ct = self._calculate_trt_y_ct(y, trt, sample_weight)
        n_samples_per_treatment = trt_y_ct.sum(axis=1)

        # Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (min(n_samples_per_treatment) < self.min_samples_split):
            return UpliftTreeNode(trt_y_ct=trt_y_ct)
        
        # Find best split
        best_split, best_uplift_gain, best_left_ct, best_right_ct =\
            self._find_best_split(X, y, trt, sample_weight)

        if best_split is None:
            # No good split found, create leaf node
            return UpliftTreeNode(trt_y_ct=trt_y_ct)
        
        feature_idx, threshold = best_split
        
        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Create left and right children
        left_child = self._build_tree(X[left_mask], y[left_mask], trt[left_mask], 
                                     sample_weight[left_mask] if sample_weight is not None else None,
                                     depth + 1, trt_y_ct=best_left_ct)
        right_child = self._build_tree(X[right_mask], y[right_mask], trt[right_mask],
                                      sample_weight[right_mask] if sample_weight is not None else None,
                                      depth + 1, trt_y_ct=best_right_ct)
        
        # Create internal node
        utn = UpliftTreeNode(feature_idx=feature_idx, threshold=threshold,
                             left=left_child, right=right_child,
                             trt_y_ct=trt_y_ct)
        return utn

    def _select_feature_subset(self, n_features):
        """Select a subset of features for random forest."""
        if self.max_features_ is not None:
            # Randomly select features
            if self.random_state is not None:
                np.random.seed(self.random_state + hash(tuple(X.shape)) % 1000)
            feature_indices = np.random.choice(n_features, self.max_features__, replace=False)
        else:
            feature_indices = range(n_features)
        return feature_indices

    def _find_best_split(self, X, y, trt, sample_weight):
        """Find the best split that maximizes uplift."""
        best_gain = -np.inf
        best_split = None
        best_left_ct = None
        best_right_ct = None
        
        # Determine which features to consider
        feature_indices = self._select_feature_subset(X.shape[1])
        
        # Try each feature
        for feature_idx in feature_indices:
            x_values = X[:, feature_idx]
            sort_idx = np.argsort(x_values)
            x_sorted = x_values[sort_idx]
            y_sorted = y[sort_idx]
            trt_sorted = trt[sort_idx]
            # values masked from thresholding
            # only consider thresholds where feature changes
            x_diff_mask = np.concatenate([x_sorted[1:] == x_sorted[:-1], [True]])

            # compute cumulative sums
            trt_ohe = np.eye(self.n_trt_ + 1, dtype=np.int8)[trt_sorted]
            if sample_weight is not None:
                sample_weight_sorted = sample_weight[sort_idx]
                y_ohe = np.eye(self.n_classes_)[y_sorted]
                y_ohe *= sample_weight_sorted.reshape(-1,1)
            else:
                y_ohe = np.eye(self.n_classes_, dtype=np.int8)[y_sorted]
            trt_y = np.einsum("ij,ik->ijk", trt_ohe, y_ohe)
            trt_y_cum = trt_y.cumsum(axis=0)
            n_cum_per_trt = trt_y_cum.sum(axis=2)

            # Remove cases where there is not enough samples in leaf
            n_cum_min_trt = np.minimum(n_cum_per_trt.min(axis=1), (n_cum_per_trt[-1,:] - n_cum_per_trt).min(axis=1))
            x_diff_mask[n_cum_min_trt < self.min_samples_leaf] = True

            # if no splits can be valid skip feature
            if (~x_diff_mask).all():
                continue
          
            gain = self._compute_gain(trt_y_cum, n_cum_per_trt)
            gain[x_diff_mask] = -np.inf

            best_i = np.nanargmax(gain)
            gain_i = gain[best_i]
            threshold = (x_sorted[best_i] + x_sorted[best_i+1]) / 2
            left_ct = trt_y_cum[best_i]
            right_ct = trt_y_cum[-1] - left_ct
            
            if gain_i > best_gain:
                best_gain = gain_i
                best_split = (feature_idx, threshold)
                best_left_ct = left_ct
                best_right_ct = right_ct
                
        return best_split, best_gain, best_left_ct, best_right_ct

    def _compute_gain(self, trt_y_cum, n_cum_per_trt):
        # Cumulative outcome  probabilities used by all measures
        with np.errstate(divide='ignore', invalid="ignore"):
            P_cum_left = trt_y_cum / n_cum_per_trt.reshape(-1, self.n_trt_+1 ,1)
            P_cum_right = (trt_y_cum[-1] - trt_y_cum) / (n_cum_per_trt[-1] - n_cum_per_trt).reshape(-1, self.n_trt_+1, 1)
            #P_cum_right_new = (trt_y_cum[-1] - trt_y_cum) / (trt_y_cum[-1] - trt_y_cum).sum(axis=2).reshape(-1, self.n_trt_+1, 1)
        if self.splitting_criterion == "DeltaDeltaP":
            if self.n_trt_ == 1 and self.n_classes_ == 1:
                gain = np.abs((P_cum_left[:,1,1] - P_cum_left[:,0,1])
                              - (P_cum_right[:,1,1] - P_cum_right[:,0,1]))
            else:
                gain = np.max(np.abs((P_cum_left[:,1:,:] - P_cum_left[:,:1,:])
                                     - (P_cum_right[:,1:,:] - P_cum_right[:,:1,:])), axis=(1,2))
        elif self.splitting_criterion in ["E", "KL", "Chi2"]:
            if self.splitting_criterion == "E":
                D_left  = np.square(P_cum_left[:,1:,:] - P_cum_left[:,:1,:]).sum(axis=(1,2))
                D_right = np.square(P_cum_right[:,1:,:] - P_cum_right[:,:1,:]).sum(axis=(1,2))
            if self.splitting_criterion == "Chi2":
                mask_left = (P_cum_left[:,:1,:] != 0)
                D_left  = np.square(P_cum_left[:,1:,:] - P_cum_left[:,:1,:])
                D_left[mask_left] /= P_cum_left[:,:1,:][mask_left]
                D_left[~mask_left] = np.inf
                D_left = D_left.sum(axis=(1,2))
                mask_right = (P_cum_right[:,:1,:] != 0)
                D_right = np.square(P_cum_right[:,1:,:] - P_cum_right[:,:1,:])
                D_right[mask_right] /= P_cum_right[:,:1,:][mask_right]
                D_right[~mask_right] = np.inf
                D_right = D_right.sum(axis=(1,2))
            if self.splitting_criterion == "KL":
                D_left  = rel_entr(P_cum_left[:,1:], P_cum_left[:,:1]).sum(axis=(1,2))
                D_right = rel_entr(P_cum_right[:,1:], P_cum_right[:,:1]).sum(axis=(1,2))
            n_cum = n_cum_per_trt.sum(axis=1)
            P_avg_left_cum = n_cum / n_cum[-1]
            P_avg_right_cum = 1 - P_avg_left_cum
            gain = P_avg_left_cum * D_left + P_avg_right_cum * D_right
        else:
            raise NotImplementedError()
        return gain

    def _calculate_trt_y_ct(self, y, trt, sample_weight=None):
        """Calculate treatment x outcome contingency table."""
        trt_y_ct = []
        for treatment_val in range(0, self.n_trt_ + 1):
            trt_mask = (trt == treatment_val)
            if sample_weight is not None:
                y_cnt = np.bincount(y[trt_mask], weights=sample_weight[trt_mask])
            else:
                y_cnt = np.bincount(y[trt_mask])
            trt_y_ct.append(y_cnt)
        return np.array(trt_y_ct)

    def _predict_sample(self, x, node):
        """Predict uplift for a single sample by traversing the tree."""
        if node.is_leaf():
            return node.uplift
        
        # Traverse to child node
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
class UpliftTreeClassifier(UpliftClassifierMixin, UpliftTreeBase):
    """Uplift Tree Classifier with proper recursive partitioning."""
    
    def predict(self, X):
        """Predict uplift for each sample."""
        # TODO!!!  needs tags in base uplift
        #check_is_fitted(self, ['tree_'])
        X = check_array(X)
        
        predictions = []
        for i in range(X.shape[0]):
            pred = self._predict_sample(X[i], self.tree_)
            predictions.append(pred)
        
        return np.array(predictions)

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
#         
#         predictions = []
#         for i in range(X.shape[0]):
#             pred = self._predict_sample(X[i], self.tree_)
#             predictions.append(pred)
#         
#         return np.array(predictions)
