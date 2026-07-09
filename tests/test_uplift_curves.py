import pytest

import numpy as np

from libuplift.metrics import uplift_curve, Qini_curve
from libuplift.utils import area_under_curve
from libuplift.metrics import optimal_uplift_curve, optimal_Qini_curve
from libuplift.metrics import Qini_coefficient
from libuplift.metrics import area_under_uplift_curve, area_under_Qini_curve


@pytest.fixture
def small_uplift_data():
    """Sample data for testing uplift and Qini curves."""
    y_true = [0, 1, 0, 0, 0, 1, 1]
    score = [1, 2, 3, 1, 2, 2, 3]
    trt = [0, 0, 0, 1, 1, 1, 1]
    return y_true, score, trt


def test_small_uplift_curve(small_uplift_data):
    y_true, score, trt = small_uplift_data
    x, u = uplift_curve(y_true, score, trt)
    assert x[0] == 0
    assert u[0] == 0
    assert x[-1] == 1
    assert np.allclose(u[-1], 1 / 2 - 1 / 3)


def test_small_Qini_curve(small_uplift_data):
    y_true, score, trt = small_uplift_data
    x, u = Qini_curve(y_true, score, trt)
    assert x[0] == 0
    assert u[0] == 0
    assert x[-1] == 1
    assert np.allclose(u[-1], (1 / 2 - 1 / 3) * 4)

def test_optimal_uplift_curve(small_uplift_data):
    y_true, _, trt = small_uplift_data
    x, u = optimal_uplift_curve(y_true, trt)
    assert np.allclose(x, [0, 0.5, 2/3, 1])
    assert np.allclose(u, [0, 0.5, 0.5, 1/6])
    auuc = area_under_curve(x, u, subtract_diag=False)
    assert np.allclose(auuc, 23/72)


def test_optimal_Qini_curve(small_uplift_data):
    y_true, _, trt = small_uplift_data
    x, u = optimal_Qini_curve(y_true, trt)
    assert np.allclose(x, [0, 0.5, 2/3, 1])
    assert np.allclose(u, [0, 2, 2, 2/3])
    auqc = area_under_curve(x, u, subtract_diag=False)
    assert np.allclose(auqc, 23/18)


def test_Qini_coefficient(small_uplift_data):
    y_true, score, trt = small_uplift_data
    qc = Qini_coefficient(y_true, score, trt)
    assert 0 <= qc <= 1
    assert np.allclose(qc, 27/68)


def test_area_under_uplift_curve(small_uplift_data):
    y_true, score, trt = small_uplift_data
    assert np.allclose(area_under_uplift_curve(y_true, score, trt, subtract_diag=False), 17/96)
    assert np.allclose(area_under_uplift_curve(y_true, score, trt, subtract_diag=True), 3/32)


def test_area_under_Qini_curve(small_uplift_data):
    y_true, score, trt = small_uplift_data
    assert np.allclose(area_under_Qini_curve(y_true, score, trt, subtract_diag=False), 17/24)
    assert np.allclose(area_under_Qini_curve(y_true, score, trt, subtract_diag=True), 3/8)
