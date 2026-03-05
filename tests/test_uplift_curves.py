from libuplift.metrics import uplift_curve

import numpy as np

def test_small_uplift_curve():
    y_true = [0,1,0,0,0,1,1]
    score  = [1,2,3,1,2,2,3]
    trt    = [0,0,0,1,1,1,1]
    x, u = uplift_curve(y_true, score, trt)
    assert x[0] == 0
    assert u[0] == 0
    assert x[-1] == 1
    assert np.allclose(u[-1], 1/2 - 1/3)
