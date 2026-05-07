"""R grf package causal forest wrapper.

This module provides Python wrappers for the causal_forest function from the R grf package.
Requires rpy2 to be installed.
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

from ..base import UpliftRegressorMixin


class CausalForestRegressor(UpliftRegressorMixin, BaseEstimator):
    """Wrapper for grf::causal_forest for continuous outcomes.
    
    This estimator trains a causal forest to estimate conditional average 
    treatment effects (CATE). It wraps the R grf package's causal_forest 
    function using rpy2.
    
    Parameters
    ----------
    num_trees : int, default=2000
        Number of trees grown in the forest.
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
    
    def __init__(self, num_trees=2000, sample_fraction=0.5, mtry=None,
                 min_node_size=5, honesty=True, honesty_fraction=0.5,
                 honesty_prune_leaves=True, alpha=0.05, imbalance_penalty=0,
                 stabilize_splits=True, num_threads=None, seed=None):
        self.num_trees = num_trees
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
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
        except ImportError as e:
            raise ImportError(
                "rpy2 is required for CausalForestRegressor. "
                "Install it with: pip install rpy2"
            ) from e
        
        try:
            grf = importr('grf')
        except Exception as e:
            raise ImportError(
                "The R package 'grf' is required but not installed. "
                "Install it in R with: install.packages('grf')"
            ) from e
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        trt = check_array(trt, ensure_1d=True)
        
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_1d=True)
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = None
        
        # Set up rpy2 conversion
        numpy2ri.activate()
        
        # Prepare parameters for grf
        # Convert numpy arrays to R vectors
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.r.matrix(y, nrow=y.shape[0], ncol=1)
        W_r = ro.r.matrix(trt, nrow=trt.shape[0], ncol=1)
        
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
        
        if self.mtry is not None:
            params['mtry'] = self.mtry
        if self.num_threads is not None:
            params['num.threads'] = self.num_threads
        if self.seed is not None:
            params['seed'] = self.seed
        if sample_weight is not None:
            params['sample.weights'] = sample_weight
        
        # Train the forest
        self.grf_forest_ = grf.causal_forest(X_r, Y_r, W_r, **params)
        
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
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
        except ImportError as e:
            raise ImportError(
                "rpy2 is required for CausalForestRegressor. "
                "Install it with: pip install rpy2"
            ) from e
        
        check_array(X)
        
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but CausalForestRegressor "
                f"was fitted with {self.n_features_in_} features."
            )
        
        numpy2ri.activate()
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        
        # Predict
        predictions_r = self.grf_forest_.predict(X_r)
        predictions = np.array(predictions_r)
        
        # The predictions from grf are a matrix with column 'predictions'
        # Extract the predictions column
        if predictions.ndim == 2:
            predictions = predictions[:, 0]
        
        return predictions


class CausalForestClassifier(UpliftRegressorMixin, BaseEstimator):
    """Wrapper for grf::causal_forest for binary classification outcomes.
    
    This estimator trains a causal forest to estimate conditional average 
    treatment effects (CATE) for binary outcomes. It wraps the R grf 
    package's causal_forest function using rpy2.
    
    Note: grf treats the outcome as numeric, so for binary classification,
    the outcome should be encoded as 0/1.
    
    Parameters
    ----------
    num_trees : int, default=2000
        Number of trees grown in the forest.
    sample_fraction : float, default=0.5
        Fraction of the data used to build each tree.
    mtry : int or None, default=None
        Number of variables tried for each split.
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
        Number of threads for training.
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
    
    See Also
    --------
    CausalForestRegressor : For continuous outcomes.
    
    Notes
    -----
    Requires the R package 'grf' to be installed and rpy2 to be installed in Python.
    
    Examples
    --------
    >>> from libuplift.tree import CausalForestClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> trt = np.random.randint(0, 2, 100)
    >>> model = CausalForestClassifier(num_trees=100)
    >>> model.fit(X, y, trt)
    >>> predictions = model.predict(X)
    """
    
    def __init__(self, num_trees=2000, sample_fraction=0.5, mtry=None,
                 min_node_size=5, honesty=True, honesty_fraction=0.5,
                 honesty_prune_leaves=True, alpha=0.05, imbalance_penalty=0,
                 stabilize_splits=True, num_threads=None, seed=None):
        self.num_trees = num_trees
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
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
        except ImportError as e:
            raise ImportError(
                "rpy2 is required for CausalForestClassifier. "
                "Install it with: pip install rpy2"
            ) from e
        
        try:
            grf = importr('grf')
        except Exception as e:
            raise ImportError(
                "The R package 'grf' is required but not installed. "
                "Install it in R with: install.packages('grf')"
            ) from e
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        trt = check_array(trt, ensure_1d=True)
        
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_1d=True)
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = None
        
        # Set up rpy2 conversion
        numpy2ri.activate()
        
        # Prepare parameters for grf
        # Convert numpy arrays to R vectors/matrices
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.r.matrix(y, nrow=y.shape[0], ncol=1)
        W_r = ro.r.matrix(trt, nrow=trt.shape[0], ncol=1)
        
        # Build parameter dict
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
        
        if self.mtry is not None:
            params['mtry'] = self.mtry
        if self.num_threads is not None:
            params['num.threads'] = self.num_threads
        if self.seed is not None:
            params['seed'] = self.seed
        if sample_weight is not None:
            params['sample.weights'] = sample_weight
        
        # Train the forest
        self.grf_forest_ = grf.causal_forest(X_r, Y_r, W_r, **params)
        
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
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
        except ImportError as e:
            raise ImportError(
                "rpy2 is required for CausalForestClassifier. "
                "Install it with: pip install rpy2"
            ) from e
        
        check_array(X)
        
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but CausalForestClassifier "
                f"was fitted with {self.n_features_in_} features."
            )
        
        numpy2ri.activate()
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        
        # Predict
        predictions_r = self.grf_forest_.predict(X_r)
        predictions = np.array(predictions_r)
        
        # The predictions from grf are a matrix with column 'predictions'
        if predictions.ndim == 2:
            predictions = predictions[:, 0]
        
        return predictions
