"""Uplift models based on multiple classification/regression models."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import LinearModel

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..base import UpliftClassifierMixin



class _MultimodelUpliftModel(UpliftMetaModelBase):
    """Base class fot multimodels (T-learners).

    if ignore_control is set to True, response model on treatment is
    built.

    """
    def __init__(self, base_estimator, prediction_method,
                 ignore_control=False):
        super().__init__(base_estimator)
        self.prediction_method = prediction_method
        self.ignore_control = ignore_control
    def _get_model_names_list(self, X=None, y=None, trt=None):
        control_m_name = "_model_c" if self.ignore_control else "model_c"
        m_names = []
        for i in range(self.n_trt_ + 1):
            if i == 0:
                name = control_m_name
            else:
                name = "model_t"
                if self.n_trt_ > 1:
                    name += str(i-1)
            m_names.append(name)
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        """Return training sets for all models in the meta model.

        Each iteration returns a triple of predictor matrix, target
        vector, and sample weights (possibly None).

        """
        for i in range(self.n_models_):
            if self.ignore_control and i == 0:
                yield None, None, None
            else:
                mask = (trt==i)
                if sample_weight is None:
                    wi = None
                else:
                    wi = sample_weight[mask]
                Xi = X[mask]
                yi = y[mask]
                yield Xi, yi, wi
    def _predict_diffs(self, X):
        """Predict differences between model predictions for each
        treatment."""
        if self.ignore_control:
            y_control = 0
        else:
            y_control = getattr(self.models_[0][1], self.prediction_method)(X)
        pred_diffs = [getattr(self.models_[i+1][1], self.prediction_method)(X)
                      - y_control for i in range(self.n_trt_)]
        return pred_diffs


class MultimodelUpliftRegressor(UpliftRegressorMixin, _MultimodelUpliftModel):
    """Multimodel uplift regressor.

    Build separate models for control and all treatments, subtract
    control predictions from treatment predictions.

    Parameters
    ----------

    base_estimator : a sklearn regressor or list of (name, regressor)
        tuples.  If a list is provided the first model is used for
        control, successive one for treatments.  If different parameters are
        to be used per each model, the list version must be given.  If a
        single estimator is given it will be cloned for every treatment.
    """
    def __init__(self, base_estimator=LinearRegression(), ignore_control=False):
        super().__init__(base_estimator, "predict",
                         ignore_control=ignore_control)
    def predict(self, X):
        pred_diffs = self._predict_diffs(X)
        if self.n_trt_ == 1:
            y = pred_diffs[0]
        else:
            y = np.column_stack(pred_diffs)
        return y
class TLearnerUpliftRegressor(MultimodelUpliftRegressor):
    pass

class _MultimodelUpliftClassifierBase(UpliftClassifierMixin, _MultimodelUpliftModel):
    def __init__(self, base_estimator, ignore_control=False):
        super().__init__(base_estimator, prediction_method="predict_proba",
                         ignore_control=ignore_control)
    def predict(self, X):
        pred_diffs = self._predict_diffs(X)
        if self.n_trt_ == 1:
            y = pred_diffs[0]
        else:
            y = np.dstack(pred_diffs)
        return y
    """Abstract base for several multimodel and response uplift
    classifiers.

    Implements the predict method."""

class MultimodelUpliftClassifier(_MultimodelUpliftClassifierBase):
    """Multimodel uplift classifier.

    Build separate models for control and all treatments, subtract
    control predicted probs from treatment predicted probs.

    Parameters
    ----------

    base_estimator : a sklearn classifier supporting predict_proba or
        list of (name, regressor) tuples.  If a list is provided the
        first model is used for control, successive one for
        treatments.  If different parameters are to be used per each
        model, the list version must be given.  If a single estimator
        is given it will be cloned for every treatment.

    """
    def __init__(self, base_estimator=LogisticRegression()):
        super().__init__(base_estimator)
class TLearnerUpliftClassifier(MultimodelUpliftClassifier):
    pass

class MultimodelUpliftLinearRegressor(MultimodelUpliftRegressor, LinearModel):
    """Uplift regressor with ``coef_`` and ``intercept_`` fields."""
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                               " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ui = self.models_[i+1][1].coef_ - c0
            ii = self.models_[i+1][1].intercept_ - i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)

class MultimodelUpliftLinearRegressorJamesSeparate(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        coef_all0 = np.concatenate((np.array([i0]),c0))
        alpha0 = (1-(self.p-3)*self.sigma[0]**2/((coef_all0-np.mean(coef_all0))@self.X_2[0]@(coef_all0-np.mean(coef_all0))))
        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            coef_alli = np.concatenate((np.array([ii2]),ci))
            alphai = (1-(self.p-3)*self.sigma[i+1]**2/((coef_alli-np.mean(coef_alli))@self.X_2[i+1]@(coef_alli-np.mean(coef_alli))))
            ui = alphai*ci - alpha0*c0
            ii = alphai*ii2 - alpha0*i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)

class MultimodelUpliftLinearRegressorJamesU(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_

        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            ui = ci - c0
            ii = ii2 - i0
            coef_all = np.concatenate((np.array([ii]),ui))
            alpha = (1-(self.p-3)/((coef_all -np.mean(coef_all))@np.linalg.pinv(self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])+self.sigma[0]**2*np.linalg.pinv(self.X_2[0]))@(coef_all-np.mean(coef_all))))        
            coef[i,:] = alpha*(ui-np.mean(ui))+np.mean(ui)
            intercept[i] = alpha*(ii-np.mean(ii))+np.mean(ii)
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)


class MultimodelUpliftLinearRegressorMSESeparate(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        var_beta0 = self.sigma[0]**2*np.linalg.pinv(self.X_2[0])
        coef_all0 = np.concatenate((np.array([i0]),c0))
        W0 = self.X_2[0]/self.X_2[0].shape[0]
        alpha0 = coef_all0.T@W0@coef_all0/(coef_all0.T@W0@coef_all0+np.trace(W0@var_beta0))
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            var_betai = self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])
            coef_alli = np.concatenate((np.array([ii2]),ci))
            Wi = self.X_2[i+1]/self.X_2[i+1].shape[0]
            alphai = coef_alli.T@Wi@coef_alli/(coef_alli.T@Wi@coef_alli+np.trace(Wi@var_betai))
            ui = alphai*ci - alpha0*c0
            #print(c0)
            ii = alphai*ii2 - alpha0*i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)
    
    
class MultimodelUpliftLinearRegressorMSEU(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        var_beta0 = self.sigma[0]**2*np.linalg.pinv(self.X_2[0])
        coef_all0 = np.concatenate((np.array([i0]),c0))
        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            var_betai = self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])
            coef_alli = np.concatenate((np.array([ii2]),ci))
            ui = ci - c0
            #print(c0)
            ii = ii2 - i0
            coef_all = np.concatenate((np.array([ii]),ui))
            W = (self.X_2[0]+self.X_2[i+1])/(self.n_[0]+self.n_[i+1])
            A = np.column_stack((np.vstack((coef_alli.T@W@coef_alli+np.trace(W@var_betai),-coef_all0.T@W@coef_alli)),
                         np.vstack((-coef_alli.T@W@coef_all0,coef_all0.T@W@coef_all0+np.trace(W@var_beta0)))))
    
            b = np.array([coef_alli.T@W@coef_all,-coef_all0.T@W@coef_all]) 
            alphas = np.linalg.pinv(A)@b
    
            alpha_t = alphas[0]
            alpha_c = alphas[1]
            
            coef[i,:] = alpha_t*ci - alpha_c*c0
            intercept[i] = alpha_t*ii2 - alpha_c*i0
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)    
    
