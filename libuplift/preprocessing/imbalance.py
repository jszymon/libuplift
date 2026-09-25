"""Methods and wrappers for correcting class imbalance in uplift
modeling."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from ..base import UpliftClassifierMixin
from ..utils import check_trt


def _class_flip_sample(y, trt, *, K=None, sample_weight=None,
                      random_state=None):
    n_trt, n_classes = K.shape
    rng = check_random_state(random_state)
    for c in range(n_classes):
        y_mask = (y==c)
        for t in range(n_trt):
            k = K[t, c]
            if k == 1:
                continue
            if trt is not None:
                mask = ((trt==t) & (y_mask))
            else:
                mask = y_mask # for classification
            n = mask.sum()
            n_flip = int(n * (1-k))
            mask_idx = np.flatnonzero(mask)
            if sample_weight is not None:
                weights = sample_weight[mask_idx]
                weights = weights / weights.sum()
                mask_flip = rng.choice(mask_idx, n_flip, p=weights)
            else:
                mask_flip = rng.choice(mask_idx, n_flip)
            y[mask_flip] = 1-y[mask_flip]
    return y

def _class_flip_weighted(X, y, trt, K=None, sample_weight=None,
                        interleave=True):
    """y_unflipped contains original y's multiplied.  Useful e.g. for stratification."""
    n_trt, n_classes = K.shape
    if sample_weight is None:
        w = np.ones(X.shape[0])
    else:
        w = sample_weight.copy()
        assert(sample_weight.shape[0] == X.shape[0])
    new_records = []
    new_ys = []
    new_ws = []
    new_trts = []
    new_unflipped_ys = []
    if interleave:
        insert_idxs = []
    for c in range(n_classes):
        y_mask = (y==c)
        for t in range(n_trt):
            k = K[t, c]
            if k == 1:
                continue
            if trt is not None:
                mask = ((trt==t) & (y_mask))
            else:
                mask = y_mask # for classification
            mask_idx = np.flatnonzero(mask)
            if interleave:
                insert_idxs.append(mask_idx)
            X_new = X[mask_idx]
            y_unflipped_new = y[mask_idx]
            y_new = 1-y_unflipped_new
            w_new = w[mask_idx]*(1-k)
            trt_new = trt[mask_idx]
            w[mask_idx] *= k
            new_records.append(X_new)
            new_ys.append(y_new)
            new_ws.append(w_new)
            new_trts.append(trt_new)
            new_unflipped_ys.append(y_unflipped_new)
    if len(new_records) == 0:
        return X, y, w, trt, y.copy()
    if interleave:
        insert_idxs = np.concatenate(insert_idxs)
        X = np.insert(X, insert_idxs, np.concatenate(new_records), axis=0)
        y_unflipped = np.insert(y, insert_idxs, np.concatenate(new_unflipped_ys), axis=0)
        y = np.insert(y, insert_idxs, np.concatenate(new_ys), axis=0)
        w = np.insert(w, insert_idxs, np.concatenate(new_ws), axis=0)
        trt = np.insert(trt, insert_idxs, np.concatenate(new_trts), axis=0)
    else:
        X = np.concatenate([X] + new_records)
        y_unflipped = np.concatenate([y] + new_unflipped_ys)
        y = np.concatenate([y] + new_ys)
        w = np.concatenate([w] + new_ws)
        trt = np.concatenate([trt] + new_trts)
    return X, y, w, trt, y_unflipped

def class_flip(X, y, trt, n_trt=None, *, K=None, sample_weight=None,
               interleave=True, random_state=None, method="weights"):
    """Low level class flipping function.

    Parameters
    ----------
    K : matrix
        Gives flipping proportion (1-k) for every treatment x class
        pair.
    interlreave: boolean, default=True
        If True flipped records are added right after original
        records, not at the end.  Only applicable if method="weights"
    method: string
        If "sample" class records are actually flipped.  If "weights"
        a weighting and record copying scheme is used instead.

    """
    if K is None:
        raise RuntimeError("The flipping proportion matrix not provided.")
    K = np.asarray(K)
    if len(K.shape) != 2:
        raise RuntimeError("The flipping proportions must be a two diemnsional array")
    if np.any(K < 0) or np.any(K > 1):
        raise RuntimeError("Class flipping factors must be in [0,1]")
    if method == "weights":
        Xf, yf, wf, trtf, y_unflipped = _class_flip_weighted(X, y, trt, K,
                                                             sample_weight=sample_weight,
                                                             interleave=interleave)
    elif method == "sample":
        yf = _class_flip_sample(y.copy(), trt, K=K,
                                sample_weight=sample_weight,
                                random_state=random_state)
        Xf = X
        trtf = trt
        wf = sample_weight
        y_unflipped = y
    else:
        raise RuntimeError("Wrong class flipping method")
    return Xf, yf, wf, trtf, y_unflipped


class UpliftBalancerBase(UpliftClassifierMixin, BaseEstimator):
    """Base class for wrapping uplift models with imbalance
    correction."""
    def __init__(self, base_estimator, method="weights", k="balance",
                 handle_balancing_error="warn"):
        super().__init__()
        self.base_estimator = base_estimator
        self.method = method
        self.k = k
        self.handle_balancing_error = handle_balancing_error
    def _compute_probs(self, y, trt, sample_weight=None):
        """Compute class probability distributions within each
        treatment."""
        shape = (self.n_trt_ + 1, self.n_classes_)
        linear_indices = np.ravel_multi_index((trt, y), shape)
        n_counts = shape[0]*shape[1]
        if sample_weight is not None:
            counts = np.bincount(linear_indices, minlength=n_counts,
                                 weights=sample_weight)
        else:
            counts = np.bincount(linear_indices, minlength=n_counts)
        counts_2d = counts.reshape(shape)
        counts_trt = counts_2d.sum(axis=1)
        n = counts_trt.sum()
        P_trt = counts_trt / n
        P_y_cond_trt = counts_2d / counts_trt.reshape(-1, 1)
        return P_y_cond_trt, P_trt
    def _print_balancing_error(self, err_msg):
        if self.handle_balancing_error == "error":
            raise RuntimeError(err_msg)
        if self.handle_balancing_error == "warn":
            print("Warning: " + err_msg)
    def _majority_class(self, P_y_cond_trt):
        """Find majority class.  Ensure it is the same across all
        treatments."""
        maj_class_per_t = P_y_cond_trt.argmax(axis=1)
        if np.any(maj_class_per_t != maj_class_per_t[0]):
            err_msg = "Different majority classes for different treatments"
            self._print_balancing_error(err_msg)            
            maj_class = None
        else:
            maj_class = maj_class_per_t[0]
        return maj_class
    def _check_binary(self):
        """Ensure binary treatment and class."""
        if self.n_trt_ > 1:
            raise RuntimeError(f"{self.__class__.__name__} only"
                               f" supports binary treatments.")
        if self.n_classes_ != 2:
            raise RuntimeError(f"{self.__class__.__name__} only"
                               f" supports binary classification.")
    def fit(self, X, y, trt, n_trt):
        if self.handle_balancing_error not in ["error", "warn", "ignore"]:
            raise RuntimeError("Invalid parameter value."
                               " handle_balancing_error must"
                               " be: error, ignore or warn.")

class StratifiedUndersampledUpliftClassifier(UpliftBalancerBase):
    """Wraps an uplift model with stratified undersampling imbalance
    correction from [1]_.

    Undersamples the majority to inflate the minority class k times.
    Both treatments are undersampled at the same rate.  When
    k='balance' tries to balance classes as much as possible described
    in [2]_.

    References
    ----------

    .. [1] O Nyberg, A Klami, "Exploring uplift modeling with high
       class imbalance", Data Mining and Knowledge Discovery 37(2),
       736-766, 2023

    .. [2] K. Rudaś, S. Jaroszewicz, "Class flipping for uplift
       modeling and Heterogeneous Treatment Effect estimation on
       imbalanced RCT data", 2025

    """
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        trt, n_trt = check_trt(trt, n_trt)
        self._set_fit_params(y, trt, n_trt)
        super().fit(X, y, trt, n_trt)
        self._check_binary()
        P_y_cond_trt, _P_trt = self._compute_probs(y, trt, sample_weight)
        maj_class = self._majority_class(P_y_cond_trt)
        if maj_class is None:
            # fit the model without balancing
            self.k_ = 1
        else:
            if self.k == "balance":
                # 1/(pt0 + pc0) or 1/(pt1 + pc1)
                self.k_ = 1.0 / P_y_cond_trt[:,1-maj_class].sum() 
            else:
                self.k_ = self.k
        S = (1.0 / self.k_ - P_y_cond_trt[:,1-maj_class]) / P_y_cond_trt[:,maj_class]
        if np.any(S < 0):
            self._print_balancing_error("negative undersampling rate.  Using 0 instead.")
        if sample_weight is not None:
            w = sample_weight.copy()
        else:
            w = np.ones_like(y, dtype=float)
        maj_mask = (y == maj_class)
        w[(maj_mask * (1-trt)) == 1] *= S[0]
        w[(maj_mask * trt) == 1] *= S[1]

        self.base_estimator.fit(X, y, trt, n_trt=n_trt, sample_weight=w)
    def predict(self, X):
        base_pred = self.base_estimator.predict(X)
        return base_pred / self.k_

class FlippedUpliftClassifier(UpliftBalancerBase):
    """Wraps an uplift model with class flipping imbalance
    correction [1]_.

    Flips the majority class labels in both the treatment and the
    control group so as to inflate the minority class k times.  When
    k='balance' tries to balance classes.

    .. [1] K. Rudaś, S. Jaroszewicz, "Class flipping for uplift
       modeling and Heterogeneous Treatment Effect estimation on
       imbalanced RCT data", 2025

    """
    def __init__(self, base_estimator, method="weights", k="balance",
                 stratify_orig_y=False):
        super().__init__(base_estimator, method=method, k=k)
        self.stratify_orig_y = stratify_orig_y
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        trt, n_trt = check_trt(trt, n_trt)
        self._set_fit_params(y, trt, n_trt)
        super().fit(X, y, trt, n_trt)
        self._check_binary()
        
        P_y_cond_trt, _P_trt = self._compute_probs(y, trt, sample_weight)
        maj_class = self._majority_class(P_y_cond_trt)
        K = np.ones((2, 2))
        if maj_class is None:
            # fit the model without balancing
            self.k_ = 1
        else:
            if self.k == "balance":
                # 1/(pt0 + pc0) or 1/(pt1 + pc1)
                self.k_ = 1.0 / P_y_cond_trt[:,maj_class].sum()
            else:
                self.k_ = self.k
            K[:, maj_class] = self.k_
            if np.any((K[:, maj_class] < 0) | (K[:, maj_class] > 1)):
                self._print_balancing_error("flipping rate out of range.  "
                                            "Using 0 or 1 instead.")
                K[:, maj_class] = np.clip(K[:, maj_class], 0, 1)
        Xf, yf, wf, trtf, y_unflipped = class_flip(
            X, y, trt, n_trt, K=K, sample_weight=sample_weight,
            method=self.method)
        if self.stratify_orig_y:
            self.base_estimator.fit(Xf, yf, trtf, n_trt=n_trt,
                                    sample_weight=wf, y_stratify=y_unflipped)
        else:
            self.base_estimator.fit(Xf, yf, trtf, n_trt=n_trt,
                                    sample_weight=wf)
    def predict(self, X):
        base_pred = self.base_estimator.predict(X)
        return base_pred / self.k_
