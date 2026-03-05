"""Base classes for all estimators and trasformers."""

import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels

from .metrics import e_sate

class _BaseUpliftMixin:
    def predict_action(self, X):
        """Predict most beneficial action."""
        y = self.predict(X)
        if self.n_trt_ == 1:
            a = (y > 0)*1
        else:
            a = np.argmax(y, axis=1) + 1
            best_y = np.max(y, axis=1)
            a[best_y <= 0] == 0
        return a
    def _set_fit_params(self, y, trt, n_trt):
        """Set attributes related to fitted data common to all models
        of a given type."""
        self.trt_, self.n_trt_ = trt, n_trt

class UpliftRegressorMixin(_BaseUpliftMixin, RegressorMixin):
    """Mixin class for all uplift regression estimators in
    libuplift."""

    _estimator_type = "regressor"
    _uplift_model = True

    def score(self, X, y, trt, n_trt=None, sample_weight=None):
        """Score test data.

        By default difference between model predicted and sample ATE
        is returned (e_sate).

        """
        return -e_sate(y, self.predict(X), trt, n_trt=self.n_trt_)

class UpliftClassifierMixin(_BaseUpliftMixin, ClassifierMixin):
    """Mixin class for all uplift classification estimators in
    libuplift."""

    _estimator_type = "classifier"
    _uplift_model = True

    def predict_action(self, X, pos_label=None):
        """Predict most beneficial action.

        Only supported for binary classification or when pos_label is
        set.  pos_label must be an intereger between 0 and
        ``self.n_classes_-1``.

        """
        if pos_label is None and self.n_classes_ > 2:
            raise RuntimeError("predict_action only available for"\
                               "binary classifiers or when positive label is explicitly"\
                               "specified.")
        if pos_label is None:
            pos_label = 1
        try:
            range(self.n_classes_)[pos_label]
            assert pos_label >= 0
        except (IndexError, AssertionError):
            raise ValueError("pos_label must be an intereger between 0 and"\
                             "self.n_classes_-1.")
        y = self.predict(X)
        if self.n_trt_ == 1:
            a = (y[:,pos_label] > 0)*1
        else:
            a = np.argmax(y, axis=1) + 1
            best_y = np.max(y, axis=1)
            a[best_y <= 0] == 0
        return a
    def _set_fit_params(self, y, trt, n_trt):
        """Also set class list and number of classes."""
        super()._set_fit_params(y, trt, n_trt)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
    def score(self, X, y, trt, n_trt=None, sample_weight=None):
        """Score test data.

        By default difference between model predicted and sample ATE
        is returned (e_sate).

        """
        return -e_sate(y, self.predict(X), trt, n_trt=self.n_trt_)


class UpliftTransformerMixin(object):
    def fit_transform(self, X, y=None, trt=None, n_trt=None, **fit_params):
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            fitted = self.fit(X, trt, n_trt, **fit_params)
            return fitted.transform(X, trt, n_trt)
        else:
            # fit method of arity 2 (supervised transformation)
            fitted = self.fit(X, y, trt, n_trt, **fit_params)
            return fitted.transform(X, trt, n_trt)

def is_uplift(estimator):
    """Returns True if the given estimator is an uplift model.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an uplift model and False otherwise.
    """
    return getattr(estimator, "_uplift_model", None) is True
