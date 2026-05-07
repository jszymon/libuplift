"""R grf package causal forest wrapper.

This module provides Python wrappers for the causal_forest function from the R grf package.
Requires rpy2 to be installed.
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.validation import check_consistent_length

from ..utils import check_trt
from ..base import UpliftRegressorMixin, UpliftClassifierMixin


class CausalForestUpliftBase(BaseEstimator):
    """Wrapper for grf::causal_forest.
    
    This estimator trains a causal forest to estimate conditional average 
    treatment effects (CATE). It wraps the R grf package's causal_forest 
    function using rpy2.
    
    Parameters
    ----------
    num_trees : int, default=2000
        Number of trees grown in the forest.
    n_estimators : int, default=None
        A synonym for num_trees for scikit learn compatibility
    sample_fraction : float, default=0.5
        Fraction of the data used to build each tree.
    mtry : int or None, default=None
        Number of variables tried for each split. If None, uses 
        min(ceiling(sqrt(ncol(X)) + 20), ncol(X)).
    min_node_size : int, default=5
        Minimum number of observations in each leaf.
    honesty : bool, default=True
        Whether to use honest (sample-splitting-based) confidence intervals.
    honesty_fraction : float, default=0.5
        Fraction of samples used for honesty.
    honesty_prune_leaves : bool, default=True
        Whether to prune leaves for honesty.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    imbalance_penalty : float, default=0
        Penalty for imbalanced splits.
    stabilize_splits : bool, default=True
        Whether to stabilize splits.
    num_threads : int or None, default=None
        Number of threads for training. If None, uses all available.
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    grf_forest_ : rpy2 object
        The trained causal forest from grf.
    n_features_in_ : int
        Number of features.
    feature_names_in_ : list
        Feature names.
    
    Notes
    -----
    Requires the R package 'grf' to be installed and rpy2 to be installed in Python.
    
    Examples
    --------
    >>> from libuplift.tree import CausalForestRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> trt = np.random.randint(0, 2, 100)
    >>> model = CausalForestRegressor(num_trees=100)
    >>> model.fit(X, y, trt)
    >>> predictions = model.predict(X)
    """
    
    def __init__(self, num_trees=2000, n_estimators=None, sample_fraction=0.5,
                 mtry=None, min_node_size=5, honesty=True, honesty_fraction=0.5,
                 honesty_prune_leaves=True, alpha=0.05, imbalance_penalty=0,
                 stabilize_splits=True, num_threads=None, seed=None):
        self.num_trees = num_trees
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.num_threads = num_threads
        self.seed = seed
    
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        """Fit the causal forest.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        trt : array-like of shape (n_samples,)
            Treatment assignment. Must be binary (0/1) or numeric.
        n_trt : int or None, default=None
            Number of treatment groups. Inferred from trt if None.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        trt, n_trt = check_trt(trt, n_trt)
        
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_1d=True)
            check_consistent_length(X, y, trt, sample_weight)
        else:
            check_consistent_length(X, y, trt)

        # Store feature info
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = None

        # activate R interface
        self._import_rpy2()
        
        # Prepare parameters for grf
        # Convert numpy arrays to R vectors
        X_r = self.ro_.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = self.ro_.r.matrix(y, nrow=y.shape[0], ncol=1)
        W_r = self.ro_.r.matrix(trt, nrow=trt.shape[0], ncol=1)
        
        # Build parameter dict, removing None values
        params = {
            'num.trees': self.num_trees,
            'sample.fraction': self.sample_fraction,
            'min.node.size': self.min_node_size,
            'honesty': self.honesty,
            'honesty.fraction': self.honesty_fraction,
            'honesty.prune.leaves': self.honesty_prune_leaves,
            'alpha': self.alpha,
            'imbalance.penalty': self.imbalance_penalty,
            'stabilize.splits': self.stabilize_splits,
        }
        if self.n_estimators is not None:
            params['num.trees'] = self.n_estimators
        
        if self.mtry is not None:
            params['mtry'] = self.mtry
        if self.num_threads is not None:
            params['num.threads'] = self.num_threads
        if self.seed is not None:
            params['seed'] = self.seed
        if sample_weight is not None:
            params['sample.weights'] = sample_weight
        
        # Train the forest
        self.grf_forest_ = self.grf_.causal_forest(X_r, y_r, W_r, **params)
        
        # Set treatment info
        self._set_fit_params(y, trt, n_trt)
        
        return self
    
    def predict(self, X):
        """Predict conditional average treatment effects (CATE) for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Estimated CATE for each sample.
        """
        
        check_array(X)
        
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but CausalForestRegressor "
                f"was fitted with {self.n_features_in_} features."
            )
        
        # activate R interface if not already active
        self._import_rpy2()

        X_r = self.ro_.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        
        # Predict
        predictions_r = self.grf_.predict_causal_forest(self.grf_forest_, X_r)
        predictions = np.array(predictions_r)
        
        # The predictions from grf are a matrix with column 'predictions'
        # Extract the predictions column
        if predictions.ndim == 2:
            predictions = predictions[0]
        
        return predictions

    def _import_rpy2(self):
        """Import rpy2 if needed and set apropriate attributes for
        reuse in different methods .

        """
        if hasattr(self, "ro_") and hasattr(self, "grf_"):
            return
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
        except ImportError as e:
            raise ImportError(
                f"rpy2 is required for {self.__class__.__name__}. "
                "Install it with: pip install rpy2"
            ) from e
        
        # Set up rpy2 conversion
        numpy2ri.activate()
        
        try:
            grf = importr('grf')
        except Exception as e:
            raise ImportError(
                "The R package 'grf' is required but not installed. "
                "Install it in R with: install.packages('grf')"
            ) from e

        # set as attributes for reuse in different methods
        self.ro_ = ro
        self.grf_ = grf
        
class CausalForestUpliftRegressor(UpliftRegressorMixin, CausalForestUpliftBase):
    pass

class CausalForestUpliftClassifier(UpliftClassifierMixin, CausalForestUpliftBase):
    """Wrapper for grf::causal_forest for binary classification
    outcomes.
    
    This estimator trains a causal regression forest and interprets
    predictions as difference in probabilities.  Prections outside
    [-1,1] are clipped.  Only two class problems are allowed.

    All parameters are the same."""
    
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        """Fit the causal forest.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (should be 0/1 for binary classification).
        trt : array-like of shape (n_samples,)
            Treatment assignment. Must be binary (0/1) or numeric.
        n_trt : int or None, default=None
            Number of treatment groups. Inferred from trt if None.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.
        
        Returns
        -------
        self : object
            Returns self.
        """
        super().fit(X, y, trt, n_trt=None, sample_weight=None)
        if self.n_classes_ > 2:
            raise ValueError("CausalForestUpliftClassifier: only two class "
                             "problems are supported")
        return self
    
    def predict(self, X):
        predictions = super().predict(X)
        predictions = np.clip(predictions, -1, 1)
        predictions = np.column_stack([-predictions, predictions])
        return predictions
