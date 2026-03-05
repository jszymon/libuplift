"""Implement scorers for uplift modeling."""

import copy

from sklearn.base import is_classifier

from .regression import e_sate, e_satt
from .curves import area_under_uplift_curve, area_under_uplift_curve_j
from .bins import QMSE, QMSE_j, EUCE, MUCE

class _BaseUpliftScorer:
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

class _UpliftPredictScorer(_BaseUpliftScorer):
    def __call__(self, estimator, X, y_true, trt, n_trt=None, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = estimator.predict(X)
        if is_classifier(estimator):
            if "pos_label" in self._kwargs:
                pos_label = self._kwargs["pos_label"]
            else:
                pos_label = 1
            y_pred = y_pred[:,pos_label]
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, trt, n_trt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, trt, n_trt,
                                                 **self._kwargs)
class _UpliftDecisionScorer(_BaseUpliftScorer):
    def _score(self, estimator, X, y_true, trt, n_trt=None, sample_weight=None):
        """Evaluate predicted treatment decisions for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        a = estimator.predict_action(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, a, trt, n_trt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, a, trt, n_trt,
                                                 **self._kwargs)
class _PassthroughUpliftScorer:
    def __init__(self, estimator):
        self._estimator = estimator

    def __call__(self, estimator, *args, **kwargs):
        """Method that wraps estimator.score"""
        return estimator.score(*args, **kwargs)

    #def get_metadata_routing(self):
    #    """Get requested data properties.
    #    .. versionadded:: 1.2
    #    Returns
    #    -------
    #    routing : MetadataRouter
    #        A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
    #        routing information.
    #    """
    #    # This scorer doesn't do any validation or routing, it only exposes the
    #    # score requests to the parent object. This object behaves as a
    #    # consumer rather than a router.
    #    res = MetadataRequest(owner=self._estimator.__class__.__name__)
    #    res.score = get_routing_for_object(self._estimator).score
    #    return res

def get_uplift_scorer(scoring):
    """Get an uplift scorer from string.
    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.
    Returns
    -------
    scorer : callable
        The scorer.
    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_uplift_scorer` twice for the same scorer results in two
    separate scorer objects.
    """
    if isinstance(scoring, str):
        try:
            scorer = copy.deepcopy(_UPLIFT_SCORERS[scoring])
        except KeyError:
            raise ValueError(
                "%r is not a valid uplift scoring value. "
                "Use usklearn.metrics.get_uplift_scorer_names() "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer

def check_uplift_scoring(estimator, scoring=None, *, allow_none=False):
    """Determine uplift scorer from user options.
    A TypeError will be thrown if the estimator cannot be scored.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y, trt, n_trt)``.
        If None, the provided estimator object's `score` method is used.
    allow_none : bool, default=False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.
    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "estimator should be an estimator implementing 'fit' method, %r was passed"
            % estimator
        )
    if isinstance(scoring, str):
        return get_uplift_scorer(scoring)
    elif callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("usklearn.metrics.")
            and not module.startswith("usklearn.metrics._scorer")
            and not module.startswith("usklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_uplift_scorer(scoring)
    elif scoring is None:
        if hasattr(estimator, "score"):
            return _PassthroughUpliftScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )
    elif isinstance(scoring, Iterable):
        raise ValueError(
            "For evaluating multiple scores, use "
            "usklearn.model_selection.cross_validate instead. "
            "{0} was passed.".format(scoring)
        )
    else:
        raise ValueError(
            "uplift scoring value should either be a callable, string or None. %r was passed"
            % scoring
        )



def make_uplift_scorer(score_func, greater_is_better=True, needs_decision=False,
                       needs_proba=False, needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.
    Read more in the :ref:`User Guide <scoring>`.
    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.
        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    if needs_proba or needs_threshold:
        raise NotImplementedError()
    sign = 1 if greater_is_better else -1
    if needs_proba + needs_decision + needs_threshold > 1:
        raise ValueError("Set only one of needs_proba, needs_threshold or"
                         " needs_decision to True.")
    if needs_proba:
        cls = _UpliftProbaScorer
    elif needs_threshold:
        cls = _UpliftThresholdScorer
    elif needs_decision:
        cls = _UpliftDecisionScorer
    else:
        cls = _UpliftPredictScorer
    return cls(score_func, sign, kwargs)


_UPLIFT_SCORERS = dict(
    e_sate=make_uplift_scorer(e_sate, greater_is_better=False),
    e_satt=make_uplift_scorer(e_satt, greater_is_better=False),
    auuc=make_uplift_scorer(area_under_uplift_curve, greater_is_better=True),
    auuc_j=make_uplift_scorer(area_under_uplift_curve_j, greater_is_better=True),
    QMSE=make_uplift_scorer(QMSE, greater_is_better=False),
    QMSE_j=make_uplift_scorer(QMSE_j, greater_is_better=False),
    EUCE=make_uplift_scorer(EUCE, greater_is_better=False),
    MUCE=make_uplift_scorer(MUCE, greater_is_better=False),
)

def get_uplift_scorer_names():
    """Get the names of all available scorers.
    These names can be passed to :func:`~usklearn.metrics.get_uplift_scorer` to
    retrieve the scorer object.
    Returns
    -------
    list of str
        Names of all available scorers.
    """
    return sorted(_UPLIFT_SCORERS.keys())
