"""Utilities for input validation."""

import numpy as np

from sklearn.utils import column_or_1d, check_consistent_length

def check_trt(trt, n_trt=None, y=None):
    trt = column_or_1d(trt)
    if not np.issubdtype(trt.dtype, np.integer):
        raise ValueError("Treatment values must be integers")
    if (trt < 0).any():
        raise ValueError("Treatment values must be >= 0")
    if n_trt is not None:
        if np.max(trt) > n_trt:
            raise ValueError("Treatment values must be <= n_trt")
    else:
        n_trt = np.max(trt)
    return trt, n_trt

