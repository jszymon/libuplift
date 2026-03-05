"""The S-learner meta model.

Simply add the treatment variable to a classifier/regressor.
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ..utils import safe_hstack

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..base import UpliftClassifierMixin



class _SLearnerBase(UpliftMetaModelBase):
    def __init__(self, base_estimator, treatment_encoding="one_hot"):
        super().__init__(base_estimator=base_estimator)
        self.treatment_encoding = treatment_encoding
    def _get_model_names_list(self, X=None, y=None, trt=None):
        m_names = ["model"]
        return m_names
    def _encode_treatment(self, trt):
        if self.n_trt_ == 1 or self.treatment_encoding == "int":
            trt = trt.reshape(-1, 1).astype(float)
        elif self.treatment_encoding == "one_hot":
            trt = np.eye(self.n_trt_+1)[:,1:]
        else:
            raise ValueError(f"Unsupported threatment encoding"
                             "({self.treatment_encoding}) for SLearner")
        return trt
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        trt = self._encode_treatment(trt)
        X = safe_hstack([X, trt])
        yield X, y, sample_weight
    def _predict_diffs(self, X, prediction_method):
        n = X.shape[0]
        p = X.shape[1]
        trt_0 = self._encode_treatment(np.zeros(n, dtype=int))
        X_aug = safe_hstack([X, trt_0])
        y_0 = getattr(self.models_[0][1], prediction_method)(X_aug)
        pred_diffs = []
        for i in range(self.n_trt_):
            trt_i = self._encode_treatment(np.full(n, i+1, dtype=int))
            X_aug[:,p:] = trt_i
            y_i = getattr(self.models_[0][1], prediction_method)(X_aug)
            pred_diffs.append(y_i - y_0)
        return pred_diffs

class SLearnerUpliftRegressor(UpliftRegressorMixin, _SLearnerBase):
    def __init__(self, base_estimator=LinearRegression(),
                 treatment_encoding="one_hot"):
        """The S-learner meta regressor.

        treatment can be encoded either as integer or using one-hot
        encoding (with control indicator skipped).

        """
        super().__init__(base_estimator=base_estimator,
                         treatment_encoding=treatment_encoding)
    def predict(self, X):
        pred_diffs = self._predict_diffs(X, "predict")
        if self.n_trt_ == 1:
            y = pred_diffs[0]
        else:
            y = np.column_stack(pred_diffs)
        return y

class SLearnerUpliftClassifier(UpliftClassifierMixin, _SLearnerBase):
    def __init__(self, base_estimator=LogisticRegression(),
                 treatment_encoding="one_hot"):
        """The S-learner meta regressor.

        treatment can be encoded either as integer or using one-hot
        encoding (with control indicator skipped).

        """
        super().__init__(base_estimator=base_estimator,
                         treatment_encoding=treatment_encoding)
    def predict(self, X):
        pred_diffs = self._predict_diffs(X, "predict_proba")
        if self.n_trt_ == 1:
            y = pred_diffs[0]
        else:
            y = np.dstack(pred_diffs)
        return y
