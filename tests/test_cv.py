import numpy as np

from sklearn.model_selection import train_test_split

from libuplift.utils.multi_array import MultiArray

def _prepare_data():
    n = 100
    p = 10
    X = np.repeat(np.arange(n, dtype=float).reshape(-1,1), p, axis=1)
    y = np.arange(n)
    w = np.arange(n, dtype=float)
    trt = np.arange(n) % 3
    n_trt = 3
    m = MultiArray(X, {"w":w, "trt":trt}, {"n_trt":n_trt})
    return m    
def test_multiarray1():
    """Test simple indexing."""
    m = _prepare_data()
    # test ranges and boolean indices
    for s in [slice(1,2), slice(50,100), (np.arange(m.shape[0]) % 2 == 1)]:
        ms = m[s]
        assert ms.scalar_dict == m.scalar_dict
        if isinstance(s, slice):
            r = np.arange(s.start, s.stop, s.step, dtype=float)
        else:
            r = np.arange(m.shape[0], dtype=float)[s]
        assert np.all(ms.main_array == r.reshape(-1,1))
        assert np.all(ms.array_dict["trt"] == r.astype(int) % 3)
        assert np.all(ms.array_dict["w"] == r)
        assert ms.shape[0] == r.shape[0]
        assert ms.shape[1] == m.shape[1]
def test_multiarray2():
    """Test crossvalidation."""
    m = _prepare_data()
    # test crossvalidation
    m1, m2 = train_test_split(m)
    w = set(m1.array_dict["w"]) | set(m2.array_dict["w"])
    w = list(w)
    w.sort()
    assert np.all(np.array(w) == np.arange(m.shape[0]))
