# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:12:29 2021

@author: Krzysztof Rudaś
"""

"""Uplift models based on multiple classification/regression models."""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
from scipy.linalg.misc import LinAlgError, LinAlgWarning
from scipy import linalg
from scipy.linalg import lstsq
import scipy.sparse as sp
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y, check_consistent_length
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from sklearn.utils.metaestimators import _BaseComposition
from ..base import UpliftRegressorMixin
from ..utils import check_trt

def Gaussian(x,Sigma):
    return -x@np.linalg.inv(Sigma)@x.T

def GaussianMatrix(X,y,Sigma):
    row,col=X.shape
    G_x=np.zeros(shape=(row,col))
    G_y=np.zeros(shape=row)
    g = np.zeros(shape=row)
    X=np.asarray(X)
    GG = Gaussian(X,Sigma)
    j=0
    for v_j in X:
        gg = np.exp(np.diag(GG)+GG[j,j]-2*GG[:,j])
        g = g+gg
        G_x=G_x+gg[None].T@v_j[None]
        G_y=G_y+gg*y[j]
        j+=1
    G_x=G_x/g[:,None]
    G_y=G_y/g
    return (G_x,G_y)


def GaussianMatrix2(X,y,Sigma,beta):
    row,col=X.shape
    G_g=np.zeros(shape=row)
    g = np.zeros(shape=row)
    X=np.asarray(X)
    GG = Gaussian(X,Sigma)
    j=0
    for v_j in X:
        gg = np.exp(np.diag(GG)+GG[j,j]-2*GG[:,j])
        g = g+gg
        G_g=G_g+gg*(y[j]-v_j.T@beta)
        j+=1
    G_g=G_g/g
    return (G_g)


class MultimodelUpliftRegressorRobinson(_BaseComposition, UpliftRegressorMixin):
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
    def __init__(self, base_estimator=LinearRegression()):
        self.base_estimator = base_estimator
    def _check_base_estimator(self, n_models):
        if hasattr(self.base_estimator, "fit"):
            estimator_list = []
            for i in range(n_models):
                name = "model_c" if i == 0 else "model_t" + str(i-1)
                estimator_list.append((name, clone(self.base_estimator)))
        else:
            estimator_list = self.base_estimator
        return estimator_list
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        self.n_models_ = self.n_trt_ + 1
        self.models_ = self._check_base_estimator(self.n_models_)
        self.n_ = np.empty(self.n_models_, dtype=int)
        self.p = X.shape[1]
        self.sigma = np.empty(self.n_models_)
        #import pdb; pdb.set_trace()
        for i in range(self.n_models_):
            mi = self.models_[i][1]
            ind = (trt==i)
            self.n_[i] = ind.sum()
            Xi = X[ind]
            yi = y[ind]
            Sigma = np.eye(self.p)
            M= GaussianMatrix(Xi,yi,3*Sigma)
            #mi.fit(Xi-M[0], yi-M[1])
            mi.fit(Xi, yi)
            mi.coef_=np.linalg.inv((Xi-M[0]).T@(Xi-M[0]))@(Xi-M[0]).T@(yi-M[1])
            #mi.intercept_=0
            #g = GaussianMatrix2(Xi,yi,3*Sigma,mi.coef_)
            #mi.fit(Xi, yi-g)
            #mi.coef_
        return self
    def predict(self, X):
        y_control = self.models_[0][1].predict(X)
        cols = [self.models_[i+1][1].predict(X) - y_control
                    for i in range(self.n_trt_)]
        if self.n_trt_ == 1:
            y = cols[0]
        else:
            y = np.column_stack(cols)
        return y
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


class MultimodelUpliftLinearRegressorRobinson(MultimodelUpliftRegressorRobinson, LinearModel):
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
            #print(c0)
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
