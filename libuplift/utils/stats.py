"""Utility functions for statistical computations."""

import numpy as np

def quantile(a, q, axis=None, out=None, overwrite_input=False,
             method='linear', keepdims=False, *, weights=None,
             interpolation=None):
    """Weighted quantiles functions.

    When sample_weight is None np.quantile is used.  Also used for
    numpy >= 2.0 since inverse_cdf strategy gives strange results for
    small samples.

    Warning: results for sample_weight=None and
    sample_weight=[1,...,1] might not be identical since a different
    strategy.

    """
    if weights is None:
        return np.quantile(a, q, axis=axis, out=out,
                           overwrite_input=overwrite_input,
                           method=method, keepdims=keepdims,
                           interpolation=interpolation)

    def weighted_quantile(data, quantiles, weights):
        """
        from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
        """
        ix = np.argsort(data)
        data = data[ix] # sort data
        weights = weights[ix] # sort weights
        cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
        q = np.interp(quantiles, cdf, data)
        return q
    arr = np.asanyarray(a)
    return weighted_quantile(arr, q, weights)


