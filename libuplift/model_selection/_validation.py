"""Model validation functions for uplift models."""


import numpy as np

from sklearn.base import is_classifier, BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils import indexable
from sklearn.utils.metaestimators import available_if
from sklearn.model_selection import check_cv
from sklearn.model_selection import cross_validate as _sklearn_cross_validate
from sklearn.model_selection import cross_val_predict as _sklearn_cross_val_predict
from sklearn.model_selection import permutation_test_score as _sklearn_permutation_test_score
from sklearn.model_selection import learning_curve as _sklearn_learning_curve
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.metaestimators import _BaseComposition

from ..utils import check_trt
from ..utils import MultiArray

from ..metrics import check_uplift_scoring


__all__ = ['cross_validate', 'cross_val_score', 'cross_val_predict',
           'permutation_test_score', 'learning_curve', 'validation_curve']

# Adapted from sklearn:
def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    """

    def check(self):
        # raise an AttributeError if `attr` does not exist
        getattr(self.base_estimator, attr)
        return True

    return check
def _estimator_has_predict():
    """Check the predict method of wrapped uplift estimator.
    """

    def check(self):
        # raise an AttributeError if `attr` does not exist
        if self.wrapped_predict is None:
            getattr(self.base_estimator, "predict")
        else:
            getattr(self.base_estimator, self.wrapped_predict)
        return True

    return check

def _extract_uplift_arrays(X):
        """Extract data for uplift training from MultiArray."""
        real_X = X.main_array
        y = X.array_dict["y"]
        trt = X.array_dict["trt"]
        n_trt = X.scalar_dict["n_trt"]
        return real_X, y, trt, n_trt
def _extract_main_array(X):
        """Extract main array if input is a MultiArray, otherwise, just return X.

        Useful in X is used for prediction on a fitted model."""
        if isinstance(X, MultiArray):
            X = X.main_array
        return X

class _WrappedUpliftEstimator(_BaseComposition, MetaEstimatorMixin, BaseEstimator):
    """Wrap upift estimator inside a sklearn estimator interface.

    If wrapped_predict is set it specifies the method to be called by
    wrapped model's predict.  This way standard scikit functions can
    access uplift specific predictions such as predict_action.
    """
    _uplift_model = True
    def __init__(self, base_estimator, wrapped_predict=None):
        self.base_estimator = base_estimator
        self.wrapped_predict = wrapped_predict
        if (self.wrapped_predict is not None and not
            hasattr(self.base_estimator, wrapped_predict)):
            raise RuntimeError(f"Can't wrap uplift model: {self.wrapped_predict} method not available.")
    @property
    def _estimator_type(self):
        return self.base_estimator._estimator_type
    def fit(self, X, y, **kwargs):
        real_X, y, trt, n_trt = _extract_uplift_arrays(X)
        return self.base_estimator.fit(real_X, y, trt, n_trt, **kwargs)
    def score(self, X, y, *args, **kwargs):
        real_X, y, trt, n_trt = _extract_uplift_arrays(X)
        return self.base_estimator.score(real_X, y, trt, n_trt, *args, **kwargs)
    @available_if(_estimator_has_predict())
    def predict(self, X):
        X = _extract_main_array(X)
        if self.wrapped_predict is not None:
            return getattr(self.base_estimator, self.wrapped_predict)(X)
        return self.base_estimator.predict(X)
    @available_if(_estimator_has("predict_action"))
    def predict_action(self, X):
        X = _extract_main_array(X)
        return self.base_estimator.predict_action(X)
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        X = _extract_main_array(X)
        return self.base_estimator.predict_proba(X)
    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        X = _extract_main_array(X)
        return self.base_estimator.predict_log_proba(X)
    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        X = _extract_main_array(X)
        return self.base_estimator.decision_function(X)
    @available_if(_estimator_has("transform"))
    def transform(self, X):
        X = _extract_main_array(X)
        return self.base_estimator.transform(X)
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        X = _extract_main_array(X)
        return self.base_estimator.inverse_transform(Xt)
    def get_params(self, deep=True):
        if hasattr(self.base_estimator, "fit"):
            return super().get_params(deep=deep)
        return self._get_params('base_estimator', deep=deep)
    def set_params(self, **kwargs):
        if hasattr(self.base_estimator, "fit"):
            return super().set_params(**kwargs)
        else:
            self._set_params('base_estimator', **kwargs)
        return self
    def get_metadata_routing(self):
        return self.base_estimator.get_metadata_routing()
    def __getattr__(self, arg):
        """Delegate to real grid search."""
        return getattr(self.base_estimator, arg)

class _WrappedScoring:
    """Wrap uplift scoring into a sklearn scoring interface."""
    def __init__(self, uplift_scoring):
        self.uplift_scoring = uplift_scoring
    def __call__(self, estimator, X, y, *args, **kwargs):
        real_X, y, trt, n_trt = _extract_uplift_arrays(X)
        return self.uplift_scoring(estimator.base_estimator, real_X, y, trt, n_trt, *args, **kwargs)

# copied from sklearn to avoid use of private sklearn functions
def _check_multimetric_scoring(estimator, scoring):
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, (list, tuple, set)):
        err_msg = (
            "The list/tuple elements must be unique strings of predefined scorers. "
        )
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e

        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            scorers = {
                scorer: check_uplift_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {
            key: check_uplift_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        raise ValueError(err_msg_generic)
    return scorers


def uplift_check_cv(cv, y, trt, n_trt, *, classifier=False):
    """Return a correct cv and y_stratify to pass to

    _sklearn_cross_validate.

    y_stratify is used only for stratification since y is contained in a
    MultiArray."""
    
    # always stratify on treatment and, if available, also on class
    if classifier:
        le = LabelEncoder()
        y_stratify = le.fit_transform(y)
        y_stratify = y_stratify * (n_trt+1) + trt
    else:
        y_stratify = trt
    # classifier=True ensures stratification
    cv = check_cv(cv, y_stratify, classifier=True)
    return cv, y_stratify

        
def cross_validate(estimator, X, y, trt, n_trt=None, groups=None, scoring=None, cv=None,
                       *args, **kwargs):
    X, y, trt, groups = indexable(X, y, trt, groups)
    trt, n_trt = check_trt(trt, n_trt)
    # y_stratify is used only for stratification
    cv, y_stratify = uplift_check_cv(cv, y, trt, n_trt, classifier=is_classifier(estimator))
    # multiarray to pass additional data
    Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
    # wrapped estimator
    wrapped_est = _WrappedUpliftEstimator(estimator)
    # wrapped scoring
    # check_cv explicitly to stratify on treatment even for regression
    scorers = _check_multimetric_scoring(estimator, scoring=scoring)
    wrapped_scorers = {name:_WrappedScoring(scoring) for name, scoring in scorers.items()}
    return _sklearn_cross_validate(wrapped_est, Xm, y_stratify, groups=groups, scoring=wrapped_scorers,
                                   cv=cv, *args, **kwargs)




def cross_val_score(estimator, X, y, trt, n_trt=None, groups=None, scoring=None, cv=None,
                    *args, **kwargs):
    # To ensure multimetric format is not supported
    scorer = check_uplift_scoring(estimator, scoring=scoring)
    cv_results = cross_validate(estimator=estimator, X=X, y=y, trt=trt,
                                n_trt=n_trt, groups=groups,
                                scoring={'score': scorer}, cv=cv,
                                *args, **kwargs)
    return cv_results['test_score']


def cross_val_predict(estimator, X, y, trt, n_trt=None, groups=None,
                      cv=None, method='predict', *args, **kwargs):
    X, y, trt, groups = indexable(X, y, trt, groups)
    trt, n_trt = check_trt(trt, n_trt)
    # y_stratify is used only for stratification
    cv, y_stratify = uplift_check_cv(cv, y, trt, n_trt, classifier=is_classifier(estimator))
    # multiarray to pass additional data
    Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
    # wrapped estimator
    if method not in ["predict", "predict_proba", "predict_log_proba",
                      "decision_function", "transform", "inverse_transform"]:
        wrapped_predict = method
        method = "predict"
    else:
        wrapped_predict = None
    wrapped_est = _WrappedUpliftEstimator(estimator, wrapped_predict=wrapped_predict)
    return _sklearn_cross_val_predict(wrapped_est, Xm, y_stratify, groups=groups, cv=cv,
                                      method=method, *args, **kwargs)


# permutation_test_score needs a modified approach to wrapping after
# wrapping y contains only stratification information which is
# permuted by this function

class _FixedStrataCV:
    """Wrap a crossvalidator such that statification variable is the same
    for each iteration.
    """
    def __init__(self, cv, strata, ignore_groups=False):
        self.cv = cv
        self.strata = strata
        self.ignore_groups = ignore_groups
    def get_n_splits(self, X, y=None, groups=None):
        if self.ignore_groups:
            return self.cv.get_n_splits(X=X, y=y)
        return self.cv.get_n_splits(X=X, y=y, groups=groups)
    def split(self, X, y=None, groups=None):
        if self.ignore_groups:
            return self.cv.split(X=X, y=self.strata)
        return self.cv.split(X=X, y=self.strata, groups=groups)

class _WrappedUpliftEstimatorOrigY(_WrappedUpliftEstimator):
    """Modification of wrapped uplift estimator which uses original y for
    training.
    """
    def fit(self, X, y, **kwargs):
        real_X, y_wrapped, trt, n_trt = _extract_uplift_arrays(X)
        return self.base_estimator.fit(real_X, y, trt, n_trt, **kwargs)
    def score(self, X, y, *args, **kwargs):
        real_X, y_wrapped, trt, n_trt = _extract_uplift_arrays(X)
        return self.base_estimator.score(real_X, y, trt, n_trt, *args, **kwargs)

    
def permutation_test_score(estimator, X, y, trt, n_trt=None, stratify_on_trt=True,
                           groups=None, cv=None,
                           scoring=None, *args, **kwargs):
    """Permutation score test for uplift models.

    If stratify_on_trt is true, permutations are performed separately
    for each treatment.  This way constant treatment effect will not
    affect the results.
    """
    X, y, trt, groups = indexable(X, y, trt, groups)
    trt, n_trt = check_trt(trt, n_trt)
    # y_stratify is used only for stratification
    cv, y_stratify = uplift_check_cv(cv, y, trt, n_trt, classifier=is_classifier(estimator))
    # don't permute strata
    # don't do group CV if stratify_on_trt is True
    wrapped_cv = _FixedStrataCV(cv, y_stratify, ignore_groups=stratify_on_trt)
    if stratify_on_trt:
        assert groups is None
        groups = trt
    Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
    wrapped_est = _WrappedUpliftEstimatorOrigY(estimator)
    scorer = check_uplift_scoring(estimator, scoring=scoring)
    wrapped_scorer = _WrappedScoring(scorer)
    return _sklearn_permutation_test_score(wrapped_est, Xm, y, groups=groups,
                                           cv=wrapped_cv, scoring=wrapped_scorer,
                                           *args, **kwargs)



def learning_curve(estimator, X, y, trt, n_trt=None, groups=None,
                   cv=None, scoring=None,
                   *args, **kwargs):
    X, y, trt, groups = indexable(X, y, trt, groups)
    trt, n_trt = check_trt(trt, n_trt)
    # y_stratify is used only for stratification
    cv, y_stratify = uplift_check_cv(cv, y, trt, n_trt, classifier=is_classifier(estimator))
    # multiarray to pass additional data
    Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
    wrapped_est = _WrappedUpliftEstimator(estimator)
    scorer = check_uplift_scoring(estimator, scoring=scoring)
    wrapped_scorer = _WrappedScoring(scorer)
    return _sklearn_learning_curve(wrapped_est, Xm, y_stratify, groups=groups, cv=cv,
                                   scoring = wrapped_scorer, *args, **kwargs)




def validation_curve(estimator, X, y, param_name, param_range, groups=None,
                     cv='warn', scoring=None, n_jobs=None, pre_dispatch="all",
                     verbose=0, error_score='raise-deprecating'):
    """Validation curve.

    Determine training and test scores for varying parameter values.

    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.

    Read more in the :ref:`User Guide <learning_curve>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" `cv` instance
        (e.g., `GroupKFold`).

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' | 'raise-deprecating' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If set to 'raise-deprecating', a FutureWarning is printed before the
        error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        Default is 'raise-deprecating' but from version 0.22 it will change
        to np.nan.

    Returns
    -------
    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_uplift_scoring(estimator, scoring=scoring)

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)
    out = parallel(delayed(_fit_and_score)(
        clone(estimator), X, y, scorer, train, test, verbose,
        parameters={param_name: v}, fit_params=None, return_train_score=True,
        error_score=error_score)
        # NOTE do not change order of iteration to allow one time cv splitters
        for train, test in cv.split(X, y, groups) for v in param_range)
    out = np.asarray(out)
    n_params = len(param_range)
    n_cv_folds = out.shape[0] // n_params
    out = out.reshape(n_cv_folds, n_params, 2).transpose((2, 1, 0))

    return out[0], out[1]


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Examples
    --------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}
