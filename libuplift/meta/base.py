"""Base classe for uplift meta models."""

import numpy as np

from sklearn.base import clone
from sklearn.utils import check_X_y, check_consistent_length
from sklearn.utils.metaestimators import _BaseComposition

from ..utils import check_trt

class UpliftMetaModelBase(_BaseComposition):
    """Base class for uplift meta estimators.

    Checks input consistency, builds classifiers on subsets of data.


    Derived classess need to overwride the _get_model_names_list and
    _iter_training_subsets methods.  The predict method needs to be
    implemented as well.

    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    def _get_model_names_list(self, X=None, y=None, trt=None):
        """Return a list of names of constituent
        classification/regression models.

        This method should be overridden such that the number of
        models can be determined by _check_base_estimator.  If model
        name starts with '_' None will be put in model list instead of
        a real model and _ removed from name (useful to keep the list
        of given size even if some models are not used).
        """
        raise NotImplementedError()
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        """Return training sets for all models in the meta model.

        Each iteration returns a triple of predictor matrix, target
        vector, and sample weights (possibly None).  While iterating
        i-th subset it may be assumed that models on previous subsets
        have been fitted.

        """
        raise NotImplementedError()
    def _check_base_estimator(self, model_names):
        if hasattr(self.base_estimator, "fit"):
            estimator_list = []
            for m_name in model_names:
                if m_name.startswith("_"):
                    estimator_list.append((m_name[1:], None))
                else:
                    estimator_list.append((m_name, clone(self.base_estimator)))
        else:
            # full model list is provided, check length and names
            estimator_list = self.base_estimator
            new_estimator_list = []
            for m_name, (user_m_name, model) in zip(model_names, estimator_list):
                if m_name != user_m_name:
                    raise RuntimeError(f"Expected model name {m_name}, got {user_m_name}")
                if m_name.startswith("_") and model is not None:
                    raise RuntimeError(f"Model name {m_name} starts with '_' but the model is not None")
                if not m_name.startswith("_") and model is None:
                    raise RuntimeError(f"Model name {m_name} does not start with '_' but the model is None")
                new_estimator_list.append((m_name, clone(model)))
            estimator_list = new_estimator_list
        return estimator_list
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        trt, n_trt = check_trt(trt, n_trt)
        check_consistent_length(X, y, trt)
        self._set_fit_params(y, trt, n_trt)

        self.models_ = self._check_base_estimator(self._get_model_names_list(X, y, trt))
        self.n_models_ = len(self.models_)
        self.n_ = np.zeros(self.n_models_, dtype=int)
        for i, (X_i, y_i, w_i) in enumerate(self._iter_training_subsets(X, y, trt, n_trt, sample_weight)):
            m_name, m_i = self.models_[i]
            if m_i is not None:
                if w_i is None:
                    self.n_[i] = X_i.shape[0]
                    m_i.fit(X_i, y_i)
                else:
                    self.n_[i] = w_i.sum()
                    m_i.fit(X_i, y_i, sample_weight=w_i)
        return self
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        if hasattr(self.base_estimator, "fit"):
            return super().get_params(deep=deep)
        return self._get_params('base_estimator', deep=deep)
    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        if hasattr(self.base_estimator, "fit"):
            return super().set_params(**kwargs)
        else:
            self._set_params('base_estimator', **kwargs)
        return self
