"""Uplift models based on target transform."""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..base import UpliftClassifierMixin


class _TargetTransformUpliftModelBase(UpliftMetaModelBase):
    def _get_model_names_list(self, X=None, y=None, trt=None):
        m_names = []
        for i in range(self.n_trt_):
            name = "model_ct"
            if self.n_trt_ > 1:
                name += str(i)
            m_names.append(name)
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        c_mask = (trt==0)
        for i in range(self.n_models_):
            t_mask = (trt==(i+1))
            mask = t_mask|c_mask
            if sample_weight is None:
                w_i = None
            else:
                w_i = sample_weight[mask]
            X_i = X[mask]
            y_i = y[mask]
            trt_i = trt[mask]
            X_i, y_i, w_i = self._transform(X_i, y_i, trt_i, n_trt, w_i, y)
            yield X_i, y_i, w_i

    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        raise NotImplementedError()

class TargetTransformUpliftRegressor(UpliftRegressorMixin,
                                     _TargetTransformUpliftModelBase):
    def __init__(self, base_estimator=LinearRegression()):
        super().__init__(base_estimator=base_estimator)
    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        n = trt.shape[0]
        mask_c = (trt==0)
        mask_t = ~mask_c
        nt = mask_t.sum()
        nc = mask_c.sum()
        n = nt + nc
        y = np.asarray(y, float) # allow classification problems
        if np.may_share_memory(y, full_y):
            y = y.copy()
        y[mask_c] *= (-n/nc)
        y[mask_t] *= (n/nt)
        return X, y, sample_weight
    def predict(self, X):
        preds = [m_i.predict(X) for _, m_i in self.models_]
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.column_stack(preds)
        return y


class TargetTransformUpliftClassifier(UpliftClassifierMixin,
                                      _TargetTransformUpliftModelBase):
    def __init__(self, base_estimator=LogisticRegression(),
                 balance_treatments=True):
        """Target transform based uplift classifier.

        If balance_treatments is True sample weights are
        introduced/modified to ensure weighted probability of both
        treatments is 1/2.  This is necessary for correct uplift
        estimation.  For multiple treatments this is ensured in all
        submodels.

        """
        super().__init__(base_estimator=base_estimator)
        self.balance_treatments = balance_treatments
    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        if self.balance_treatments:
            if sample_weight is None:
                n = X.shape[0]
                nt = trt.sum()
            else:
                n = sample_weight.sum()
                nt = sample_weight[trt==1].sum()
            nc = n-nt
            pt = nt/n
            pc = nc/n
            k = pc/pt
            if sample_weight is None:
                sample_weight = np.ones(n)
            sample_weight[trt==1] *= k
        if np.may_share_memory(y, full_y):
            y = y.copy()
        y[trt == 0] = 1-y[trt == 0]
        return X, y, sample_weight
    def predict(self, X):
        preds = [2 * m_i.predict_proba(X) - 1 for _, m_i in self.models_]
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.dstack(preds)
        return y
