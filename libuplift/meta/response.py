"""'Fake' uplift models based on response classifiers."""

import numpy as np

from sklearn.linear_model import LogisticRegression

from .multi_model import _MultimodelUpliftClassifierBase

class _ResponseModelBase(_MultimodelUpliftClassifierBase):
    def __init__(self, base_estimator, reverse, ignore_control):
        super().__init__(base_estimator=base_estimator,
                         ignore_control=ignore_control)
        self.reverse = reverse
    def predict(self, X):
        y = super().predict(X)
        if self.reverse:
            y = 1-y
        return y

class TreatmentUpliftClassifier(_ResponseModelBase):
    """Predict uplift based on treatment classifiers.

    Ignore control."""
    def __init__(self, base_estimator=LogisticRegression(), reverse=False):
        super().__init__(base_estimator, reverse=reverse,
                         ignore_control=True)
class ResponseUpliftClassifier(TreatmentUpliftClassifier):
    """Predict uplift using a classifier built on full data.

    Ignore causal nature of the data."""
    def __init__(self, base_estimator=LogisticRegression(), reverse=False):
        super().__init__(base_estimator, reverse=reverse)
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        n = X.shape[0]
        trt = np.ones(n, dtype=np.int32)
        n_trt = 1
        super().fit(X, y, trt, n_trt, sample_weight=sample_weight)
class ControlUpliftClassifier(TreatmentUpliftClassifier):
    """Predict uplift based on a control classifier.

    Ignore treatment data.  If reverse is True lower classification
    scores are assumed to correspond to higher uplift.

    """
    def __init__(self, base_estimator=LogisticRegression(), reverse=True):
        super().__init__(base_estimator=base_estimator, reverse=reverse)
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        mask = (trt == 0)
        n = mask.sum()
        trt = np.ones(n, dtype=np.int32)
        n_trt = 1
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
        super().fit(X[mask], y[mask], trt, n_trt, sample_weight=sample_weight)

