from pytest import approx

import numpy as np

from sklearn.linear_model import LogisticRegression

from libuplift.classifiers import ClassifierAsRegressor

def _make_data():
    X = [[1.0], [1], [1], [0], [0], [0]]
    y = [1,1,0, 0,0,1]
    return X, y
    
def test_classifier_as_regressor():
    lr = LogisticRegression(penalty=None)
    lr_reg = ClassifierAsRegressor(lr)
    X, y = _make_data()
    lr_reg.fit(X, y)
    
    pred_y = lr_reg.predict(X)
    assert pred_y[:3] == approx(2/3, abs=1e-3)
    assert pred_y[3:] == approx(1/3, abs=1e-3)

def test_classifier_as_regressor_pos_label():
    lr = LogisticRegression(penalty=None)
    lr_reg = ClassifierAsRegressor(lr, pos_label=0)
    X, y = _make_data()
    lr_reg.fit(X, y)
    
    pred_y = lr_reg.predict(X)
    assert pred_y[:3] == approx(1/3, abs=1e-3)
    assert pred_y[3:] == approx(2/3, abs=1e-3)

def test_classifier_as_regressor_log_proba():
    lr = LogisticRegression(penalty=None)
    lr_reg = ClassifierAsRegressor(lr, response_method="predict_log_proba")
    X, y = _make_data()
    lr_reg.fit(X, y)
    
    pred_y = lr_reg.predict(X)
    assert pred_y[:3] == approx(np.log(2/3), abs=1e-3)
    assert pred_y[3:] == approx(np.log(1/3), abs=1e-3)

def test_classifier_as_regressor_decision_function():
    lr = LogisticRegression(penalty=None)
    lr_reg = ClassifierAsRegressor(lr, response_method="decision_function")
    X, y = _make_data()
    lr_reg.fit(X, y)
    
    pred_y = lr_reg.predict(X)
    assert pred_y[:3] == approx(np.log(2), abs=1e-3)
    assert pred_y[3:] == approx(np.log(1/2), abs=1e-3)
