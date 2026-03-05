import numpy as np

from libuplift.metrics.bins import iter_quantiles

def test_iter_quantiles():
    trt =    np.array([0,0,0,0,0, 1,1,1,1,1])
    scores = np.array([1,2,3,4,5, 5,4,3,2,1], dtype=float)

    for i, q in enumerate(iter_quantiles(scores, trt, 1, n=5)):
        assert len(q) == 2
        assert q[0].shape == (1,)
        assert q[1].shape == (1,)
        assert q[0].item() == (4-i)
        assert q[1].item() == 9-(4-i)

def test_iter_quantiles_weighted():
    trt =     np.array([0,0,0,0,0, 1,1,1,1,1])
    scores =  np.array([1,2,3,4,5, 5,4,3,2,1], dtype=float)
    weights = np.array([1,1,1,1,1, 1,1,1,1,1], dtype=float)

    for i, q in enumerate(iter_quantiles(scores, trt, 1, n=5,
                                         sample_weight=weights)):
        assert len(q) == 2
        assert q[0].shape == (1,)
        assert q[1].shape == (1,)
        assert q[0].item() == (4-i)
        assert q[1].item() == 9-(4-i)

def test_iter_quantiles_weighted2():
    trt =     np.array([0,0,0,0,0, 1,1,1,1,1])
    scores =  np.array([1,2,3,4,5, 5,4,3,2,1], dtype=float)
    weights = np.array([1,1,0,0,0, 0,0,0,1,1], dtype=float)

    for i, q in enumerate(iter_quantiles(scores, trt, 1, n=2,
                                         sample_weight=weights)):
        assert len(q) == 2
        if i == 0:
            assert q[0].shape == (4,)
            assert q[1].shape == (4,)
            assert set(q[0]) == set([1,2,3,4])
            assert set(q[1]) == set([5,6,7,8])
        elif i == 1:
            assert q[0].shape == (1,)
            assert q[1].shape == (1,)
            assert set(q[0]) == set([0])
            assert set(q[1]) == set([9])
        else:
            assert False
