"""Inverse propensity score-reweighted uplift model."""

from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.metaestimators import available_if

class WrappedUpliftEstimator(_BaseComposition):
    """Wrap upift estimator to modify its functionality.

    If wrapped_predict is set it specifies the method to be called by
    wrapped model's predict.  This way standard scikit functions can
    access uplift specific predictions such as predict_action.
    """
    _uplift_model = True
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    @property
    def _estimator_type(self):
        return self.base_estimator._estimator_type
    def fit(self, X, y, trt, n_trt, **kwargs):
        return self.base_estimator.fit(X, y, trt, n_trt, **kwargs)
    def score(self, X, y, trt, n_trt, **kwargs):
        return self.base_estimator.score(X, y, trt, n_trt, *args, **kwargs)
    @available_if(_estimator_has_predict())
    def predict(self, X):
        return self.base_estimator.predict(X)
    @available_if(_estimator_has("predict_action"))
    def predict_action(self, X):
        return self.base_estimator.predict_action(X)
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        return self.base_estimator.predict_log_proba(X)
    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        return self.base_estimator.decision_function(X)
    @available_if(_estimator_has("transform"))
    def transform(self, X):
        return self.base_estimator.transform(X)
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        return self.base_estimator.inverse_transform(Xt)
    # generic parameters access
    def get_params(self, deep=True):
        if hasattr(self.base_estimator, "fit"):
            return super().get_params(deep=deep)
        return self._get_params('base_estimator', deep=deep)
    def set_params(self, **kwargs):
        if hasattr(self.base_estimator, "fit"):
            return super().set_params(**kwargs)
        else:
            self._set_params('base_estimator', **kwargs)
        return self
    def __getattr__(self, arg):
        return getattr(self.base_estimator, arg)


class IPTW_UpliftEstimator(WrappedUpliftEstimator):
    pass
