"""Measures based on comparing treatment and control statistics within
bins, such as quantiles."""

import numpy as np

from sklearn.utils.validation import check_array, check_consistent_length

from ..utils.validation import check_trt
from ..utils.stats import quantile

def iter_quantiles(scores, trt, n_trt, n=10, joint=False,
                   sample_weight=None):
    """Iterate simultaneously over quantiles of score vectors for all
    treatments.

    Returns a generator which, for each quantile, returns a list of
    index arrays for scores within each treatment.

    If joint is True, quantiles are computed jointly for all
    treatments.

    If sample_weight is not None, weighted quantiles are used.

    """
    # sort by treatment
    t_idx = np.argsort(trt)
    t_counts = np.r_[[0], np.cumsum(np.bincount(trt, minlength=n_trt+1))]
    counts = []
    idxs = []
    if joint:
        q = quantile(scores, np.linspace(0,1,n, endpoint=False),
                     weights=sample_weight)
    for ti in range(n_trt+1):
        s = scores[t_idx[t_counts[ti]:t_counts[ti+1]]]
        if not joint:
            if sample_weight is not None:
                w = sample_weight[t_idx[t_counts[ti]:t_counts[ti+1]]]
            else:
                w = None
            q = quantile(s, np.linspace(0,1,n, endpoint=False),
                         weights=w)
        b = (np.digitize(s, q)-1).astype(np.min_scalar_type(n+1))
        c = np.r_[[0], np.cumsum(np.bincount(b, minlength=n))]
        idx = np.argsort(b)
        counts.append(c)
        idxs.append(t_idx[t_counts[ti]:t_counts[ti+1]][idx])
    for i in range(n-1, -1, -1):
        yield [idx[c[i]:c[i+1]] for idx, c in zip(idxs, counts)]

def _binned_measure(per_q_func, aggreg_func, name,
                    y_true, y_pred, trt, n_trt=None, sample_weight=None,
                    n_bins=10, allow_nans=False, joint_quantiles=False):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    trt, n_trt = check_trt(trt, n_trt)
    if sample_weight is None:
        check_consistent_length(y_true, y_pred, trt)
    else:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y_true, y_pred, trt, sample_weight)
    if n_trt > 1:
        raise ValueError(name + " only supported for a single treatment.")

    stat_q_s = []
    for q_idxs in iter_quantiles(y_pred, trt, n_trt, n_bins, joint=joint_quantiles):
        idx_c, idx_t = q_idxs
        stat_q = per_q_func(y_true, y_pred, idx_t, idx_c)
        stat_q_s.append(stat_q)
    stat = aggreg_func(stat_q_s, allow_nans)
    if np.isnan(stat):
        raise RuntimeError("NaN's present in " + name + " computation")
    return stat

def _q_aggreg_mean(stats_q_s, allow_nans):
    if allow_nans:
        stat = np.nanmean(stats_q_s)
    else:
        stat = np.mean(stats_q_s)
    return stat
def _q_aggreg_max(stats_q_s, allow_nans):
    if allow_nans:
        stat = np.nanmax(stats_q_s)
    else:
        stat = np.max(stats_q_s)
    return stat

def QMSE(y_true, y_pred, trt, n_trt=None, sample_weight=None,
         n_bins=10, allow_nans=False, joint_quantiles=False):
    """The per-quantile MSE measure by Rudaś, Jaroszewicz."""
    def _per_q_qmse(y_true, y_pred, idx_t, idx_c):
        if len(idx_c) > 0:
            mi_c = y_true[idx_c].mean()
        else:
            return np.nan
        if len(idx_t) > 0:
            mi_t = y_true[idx_t].mean()
        else:
            return np.nan
        if len(idx_t) > 0:
            mse_j = np.mean((y_pred[idx_t] - (mi_t-mi_c))**2)
        else:
            mse_j = np.nan
        return mse_j
    return _binned_measure(_per_q_qmse, _q_aggreg_mean, "QMSE",
                           y_true, y_pred, trt, n_trt=n_trt, sample_weight=sample_weight,
                           n_bins=10, allow_nans=allow_nans, joint_quantiles=joint_quantiles)
def QMSE_j(y_true, y_pred, trt, n_trt=None, sample_weight=None,
         n_bins=10, allow_nans=False):
    return QMSE(y_true, y_pred, trt, n_trt=None, sample_weight=None,
                n_bins=10, allow_nans=False, joint_quantiles=True)

def _per_q_euce(y_true, y_pred, idx_t, idx_c):
    if len(idx_c) > 0:
        mi_c = y_true[idx_c].mean()
    else:
        return np.nan
    if len(idx_t) > 0:
        mi_t = y_true[idx_t].mean()
    else:
        return np.nan
    idx = np.concatenate([idx_c, idx_t])
    mi_p = np.mean(y_pred[idx])
    d_j = np.abs(mi_p - (mi_t-mi_c))
    return d_j

def EUCE(y_true, y_pred, trt, n_trt=None, sample_weight=None,
         n_bins=100, allow_nans=False, joint_quantiles=True):
    """The EUCE measure by Nyberg and Klami."""
    return _binned_measure(_per_q_euce, _q_aggreg_mean, "EUCE",
                           y_true, y_pred, trt, n_trt=n_trt, sample_weight=sample_weight,
                           n_bins=10, allow_nans=allow_nans, joint_quantiles=joint_quantiles)
def MUCE(y_true, y_pred, trt, n_trt=None, sample_weight=None,
         n_bins=100, allow_nans=False, joint_quantiles=True):
    """The MUCE measure by Nyberg and Klami."""
    return _binned_measure(_per_q_euce, _q_aggreg_max, "MUCE",
                           y_true, y_pred, trt, n_trt=n_trt, sample_weight=sample_weight,
                           n_bins=10, allow_nans=allow_nans, joint_quantiles=joint_quantiles)
