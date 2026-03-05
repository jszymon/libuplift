"""Nested models where control outcome predictions are used in an
uplift model."""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ..utils import safe_hstack

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..base import UpliftClassifierMixin

class NestedMeanUpliftRegressor(UpliftRegressorMixin, UpliftMetaModelBase):
    """Nested regression model.

    First builds a model on controls, then subtracts its training
    predictions from target.  An uplift model is then build on the new
    target.

    Only available for regression models.

    """
    def __init__(self, base_estimator = LinearRegression()):
        super().__init__(base_estimator=base_estimator)
    def _get_model_names_list(self, X=None, y=None, trt=None):
        m_names = ["model_c"]
        for i in range(self.n_trt_):
            name = "model_u"
            if self.n_trt_ > 1:
                name += str(i-1)
            m_names.append(name)
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        c_mask = (trt==0)
        y_c = y[c_mask]
        if sample_weight is not None:
            yield X[c_mask], y_c, sample_weight[c_mask]
        else:
            yield X[c_mask], y_c, None
        # assume the control model is already fitted
        m_c = self.models_[0][1]
        for i in range(self.n_trt_):
            t_mask = (trt==(i+1))
            X_i = X[t_mask]
            y_i_pred = m_c.predict(X_i)
            y_i = y[t_mask] - y_i_pred
            if sample_weight is not None:
                w_i = sample_weight[t_mask]
            else:
                w_i = None
            yield X_i, y_i, w_i
    def predict(self, X):
        preds = [m_i.predict(X) for _, m_i in self.models_[1:]]
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.column_stack(preds)
        return y


class DDRUpliftClassifier(UpliftClassifierMixin, UpliftMetaModelBase):
    """Dependent Data Representation metamodel.  It is a double model
    where control predictions are added as a variable in the treatment
    model.

    The model was proposed in A. Betlei, E. Diemert, and M.-R. Amini
    Uplift Prediction with Dependent Feature Representation in
    Imbalanced Treatment and Control Conditions, ICONIP, 2018.

    direction : string, default="C->T" "C->T" means control
        predictions are used as an additional predictor for the
        treatment model, "T->C" means the reverse: predictions of all
        treatment models are used (jointly) as predictors for the
        control model.

    """
    def __init__(self, base_estimator=LogisticRegression(),
                 feature_prediction_method="predict_proba",
                 direction="C->T"):
        """A DDR uplift """
        super().__init__(base_estimator=base_estimator)
        self.feature_prediction_method = feature_prediction_method
        self.direction = direction
    def _get_model_names_list(self, X=None, y=None, trt=None):
        c_names = ["model_c"]
        t_names = []
        for i in range(self.n_trt_):
            name = "model_t"
            if self.n_trt_ > 1:
                name += str(i-1)
            t_names.append(name)
        if self.direction == "C->T":
            m_names = c_names + t_names
        elif self.direction == "T->C":
            m_names = t_names + c_names
        else:
            raise ValueError(f"The direction parameter for the DDR model must be 'C->T' or 'T->C', got {self.direction}")
        return m_names
    def _prediction_feature(self, m, X):
        if self.feature_prediction_method == "predict_proba":
            y = m.predict_proba(X)[:,1:]
        elif self.feature_prediction_method == "predict":
            y = m.predict(X)
        elif self.feature_prediction_method == "decision_function":
            y = m.decision_function(X)
        return y
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        c_mask = (trt==0)
        if self.direction == "C->T":
            if sample_weight is not None:
                yield X[c_mask], y[c_mask], sample_weight[c_mask]
            else:
                yield X[c_mask], y[c_mask], None
            m_c = self.models_[0][1]
            for i in range(self.n_trt_):
                t_mask = (trt==(i+1))
                X_i = X[t_mask]
                y_i = y[t_mask]
                # assume the control model is already fitted
                y_i_pred = self._prediction_feature(m_c, X_i)
                X_i = safe_hstack([X_i, y_i_pred])
                if sample_weight is not None:
                    w_i = sample_weight[t_mask]
                else:
                    w_i = None
                yield X_i, y_i, w_i
        else:
            X_c = X[c_mask]
            # train treatment models first
            t_preds = []
            for i in range(self.n_trt_):
                t_mask = (trt==(i+1))
                X_i = X[t_mask]
                y_i = y[t_mask]
                # assume the control model is already fitted
                if sample_weight is not None:
                    w_i = sample_weight[t_mask]
                else:
                    w_i = None
                yield X_i, y_i, w_i
                y_i_pred = self._prediction_feature(self.models_[i][1], X_c)
                t_preds.append(y_i_pred)
            # train the control model
            X_c = safe_hstack([X_c] + t_preds)
            y_c = y[c_mask]
            if sample_weight is not None:
                w_c = sample_weight[c_mask]
            else:
                w_c = None
            yield X_c, y_c, w_c
            
    def predict(self, X):
        if self.direction == "C->T":
            pred_feature = self._prediction_feature(self.models_[0][1], X)
            pred_c = self.models_[0][1].predict_proba(X)
            preds = []
            for i in range(self.n_trt_):
                X_i = safe_hstack([X, pred_feature])
                pred_i = self.models_[i+1][1].predict_proba(X_i) - pred_c
                preds.append(pred_i)
        else:
            pred_features = [self._prediction_feature(m[1], X) for m in self.models_[:-1]]
            preds = [m[1].predict_proba(X) for m in self.models_[:-1]]
            X_c = safe_hstack([X] + pred_features)
            pred_c = self.models_[-1][1].predict_proba(X_c)
            for i in range(self.n_trt_):
                preds[i] -= pred_c
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.dstack(preds)
        return y
