"""Utility functions used for metric computation."""

import numpy as np

def area_under_curve(xs, ys, subtract_diag=True):
    """Compute area under a curve given by xs and ys.

    If subtract_diag is True area under the diagonal is subtracted.

    """
    auc = np.trapz(ys, xs)
    if subtract_diag:
        a = xs[-1] - xs[0]
        h = ys[-1]
        auc -= a*h/2
    return auc
