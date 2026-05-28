"""R-learner."""

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_predict

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..model_selection import uplift_check_cv

class RLearnerUpliftRegressor(UpliftRegressorMixin, UpliftMetaModelBase):
    def __init__(self, base_estimator=LinearRegression(),
                 mean_estimator=None, cv=None):
        self.mean_estimator = mean_estimator
        self.cv = cv
        super().__init__(base_estimator)

    def _get_model_names_list(self, X=None, y=None, trt=None, y_stratify=None):
        return ["model_tau"]
    def _make_cv(self, y, trt, y_stratify):
        if y_stratify is None:
            cv_classifier=False
        else:
            cv_classifier=True
        cv_classifier = False # TODO no way to pass y_stratify to cv
                              # inside cross_val_predict
        self.cv_, y_stratify = uplift_check_cv(self.cv, y_stratify,
                                               trt, self.n_trt_,
                                               classifier=cv_classifier)
        self.y_stratify_ = y_stratify # TODO: more elegant way to pass to _iter_training_subsets?

    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight,
                               y_stratify=None):
        if n_trt > 1:
            raise ValueError("RLearner is only supported for a single treatment.")
        self._make_cv(y, trt, self.y_stratify_)
        if self.mean_estimator is not None:
            mean_estimator = self.mean_estimator
        else:
            mean_estimator = self.base_estimator
        # TODO: cross_val_predict does not support sample_weight
        # TODO: no way to pass stratified Y
        y_mean = cross_val_predict(mean_estimator, X, y, cv=self.cv_)
        y = np.asarray(y, float) # allow classification problems
        y = y - y_mean
        # TODO: refactor with other methods
        mask_c = (trt==0)
        mask_t = ~mask_c
        if sample_weight is None:
            nt = mask_t.sum()
            nc = mask_c.sum()
        else:
            nt = sample_weight[mask_t].sum()
            nc = sample_weight[mask_c].sum()
        n = nt + nc
        w = np.where(mask_t, nt/n, -nc/n)
        y /= w
        if sample_weight is None:
            sample_weight = w*w
        else:
            sample_weight = sample_weight * (w*w)
        yield X, y, sample_weight

    def fit(self, X, y, trt, n_trt=None, sample_weight=None, *, y_stratify=None):
        self.y_stratify_ = y_stratify # TODO: more elegant way to pass this
        super().fit(X, y, trt, n_trt, sample_weight=sample_weight,
                    y_stratify=y_stratify)
        del self.y_stratify_
    def predict(self, X):
        return self.models_[0][1].predict(X)
