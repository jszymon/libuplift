"""The DR-learner model from Kennedy 2004."""

import numpy as np

from sklearn.linear_model import LinearRegression

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin

from ..model_selection import uplift_check_cv



class DRLearnerUpliftRegressor(UpliftRegressorMixin, UpliftMetaModelBase):
    def __init__(self, base_estimator=LinearRegression(),
                 mean_estimator=None, cv=2):
        super().__init__(base_estimator)
        self.mean_estimator = mean_estimator
        self.cv = cv
    def _get_model_names_list(self, X, y, trt):
        if self.n_trt_ > 1:
            raise ValueError("DRLearner is only supported for single treatment.")
        if self.mean_estimator is None:
            self.mean_estimator_ = self.base_estimator
        else:
            self.mean_estimator_ = self.mean_estimator
        self.cv_, y_stratify = uplift_check_cv(self.cv, y, trt, self.n_trt_,
                                               classifier=False)
        self.y_stratify_ = y_stratify # more elegant way to pass to _iter_training_subsets?
        self.n_splits_ = self.cv_.get_n_splits(X, self.y_stratify_)
        if self.n_splits_ > 1:
            m_names = []
            for i in range(self.n_splits_):
                m_names += [f"mean_model_c_fold_{i}",
                           f"mean_model_t_fold_{i}",
                           f"tau_model_fold_{i}"]
        else:
            m_names = ["mean_model_c", "mean_model_t", "tau_model"]
        if self.mean_estimator is not None:
            m_names_w_models = []
            for i, m_name in enumerate(m_names):
                if i % 3 == 0 or i % 3 == 1:
                    m_names_w_models.append((m_name, self.mean_estimator))
                else:
                    m_names_w_models.append((m_name, self.base_estimator))
            m_names = m_names_w_models
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        for i, (mean_idx, tau_idx) in enumerate(self.cv_.split(X, self.y_stratify_)):
            X_mean = X[mean_idx]
            y_mean = y[mean_idx]
            mean_trt = trt[mean_idx]
            mask_c = (mean_trt==0)
            mask_t = ~mask_c
            if sample_weight is not None:
                w_mean = sample_weight[mean_idx]
                yield X_mean[mask_c], y_mean[mask_c], w_mean[mask_c]
                yield X_mean[mask_t], y_mean[mask_t], w_mean[mask_t]
            else:
                yield X_mean[mask_c], y_mean[mask_c], None
                yield X_mean[mask_t], y_mean[mask_t], None
            X_tau = X[tau_idx]
            y_tau = y[tau_idx]
            # allow use for classification problems
            y_tau = np.array(y_tau, dtype=float)
            if np.may_share_memory(y_tau, y):
                y_tau = y_tau.copy()
            tau_trt = trt[tau_idx]
            if sample_weight is not None:
                w_tau = sample_weight[tau_idx]
            else:
                w_tau = None
            mask_c = (tau_trt==0)
            mask_t = ~mask_c
            y0_hat = self.models_[i*3][1].predict(X_tau)
            y1_hat = self.models_[i*3+1][1].predict(X_tau)
            # create pseudooutcomes
            # TODO: share code with target transform
            nt = mask_t.sum()
            nc = mask_c.sum()
            n = nt + nc
            y_tau[mask_c] -= y0_hat[mask_c]
            y_tau[mask_t] -= y1_hat[mask_t]
            y_tau[mask_c] *= (-n/nc)
            y_tau[mask_t] *= (n/nt)
            tau_hat = y1_hat - y0_hat
            y_tau += tau_hat
            yield X_tau, y_tau, w_tau
    def predict(self, X):
        for i in range(self.n_splits_):
            if i == 0:
                pred = self.models_[3*i+2][1].predict(X)
            else:
                pred += self.models_[3*i+2][1].predict(X)
        return pred / self.n_splits_
