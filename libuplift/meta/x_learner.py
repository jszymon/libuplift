"""The X-learner model from Künzel et al. 'Metalearners for estimating
heterogeneous treatment effects using machine learning'.

Currently available only for regression.  Can be applied to
classification problems when treating class variable as numeric.

"""

import numpy as np

from sklearn.linear_model import LinearRegression

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin

class XLearnerUpliftRegressor(UpliftRegressorMixin, UpliftMetaModelBase):
    def __init__(self, base_estimator = LinearRegression()):
        super().__init__(base_estimator=base_estimator)
    def _get_model_names_list(self, X=None, y=None, trt=None):
        if self.n_trt_ > 1:
            raise ValueError("XLearner is only supported for single treatments")
        m_names = ["model_c", "model_t", "model_c_hat", "model_t_hat"]
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        for i in range(self.n_trt_ + 1):
            mask = (trt==i)
            if sample_weight is None:
                wi = None
            else:
                wi = sample_weight[mask]
            Xi = X[mask]
            yi = y[mask]
            yield Xi, yi, wi
        c_mask = (trt==0)
        t_mask = (trt==1)
        # c_hat model
        if sample_weight is None:
            wi = None
        else:
            wi = sample_weight[c_mask]
        y_c_pred = self.models_[1][1].predict(X[c_mask])
        yield X[c_mask], y_c_pred - y[c_mask], wi
        # t_hat model
        if sample_weight is None:
            wi = None
        else:
            wi = sample_weight[t_mask]
        y_t_pred = self.models_[0][1].predict(X[t_mask])
        yield X[t_mask], y[t_mask] - y_t_pred, wi
    def predict(self, X):
        n = self.n_[0] + self.n_[1]
        pc = self.n_[0] / n
        pt = self.n_[1] / n
        pred_c_hat = self.models_[2][1].predict(X)
        pred_t_hat = self.models_[3][1].predict(X)
        y = pc * pred_c_hat + pt * pred_t_hat
        return y
