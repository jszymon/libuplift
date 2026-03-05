import numpy as np
import scipy.sparse as sp

def safe_hstack(Xs):
    """hstack which works for dense and sparse arrays."""
    if any(sp.issparse(f) for f in Xs):
        Xs = sp.hstack(Xs, format="csr")
    else:
        Xs = np.hstack(Xs)
    return Xs
