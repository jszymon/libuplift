"""Use a classifier as a regressor.

By default return predicted probabilities as numeric predictions.

"""


from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.base import clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
)

class ClassifierAsRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """Wraps a classifier such that it behaves like a regressor.

    The predict method returns by default predicted probability for
    class specified by ``pos_label`` (default 1).  The method used for
    prediction can be changed by passing the response_method
    argument.

    If ``response_method returns a vector`` (e.g. ``decision_function``)
    ``pos_label`` will be ignored.

    Parameters
    ----------

    estimator : a scikit-klearn classifier
        Classifier to wrap in a regessor interface.

    response_method : string, default='predict_proba'
        Classifier's method to use for making predictions.

    pos_label : integer, default=1
        Label whose probability should be returned by regressor's
        predict method.

    """
    def __init__(self, estimator, response_method='predict_proba',
                 pos_label=1):
        super().__init__()
        self.estimator = estimator
        self.response_method = response_method
        self.pos_label = pos_label
    def fit(self, *args, **kwargs):
        self.fitted_estimator_ = clone(self.estimator).fit(*args, **kwargs)
        return self
    def predict(self, *args, **kwargs):
        resp_method = getattr(self.fitted_estimator_, self.response_method)
        preds = resp_method(*args, **kwargs)
        pred_ndim = len(preds.shape)
        if pred_ndim > 2:
            raise RuntimeError("ClassifierAsRegressor: response method"
                               " must return a vector or a matrix.")
        elif pred_ndim == 2:
            preds = preds[:,self.pos_label]
        return preds
