"""A memoized classifier class.

Used to avoid recomputing the same classifier twice e.g. when both
T-learner and Response models are computed.

"""

import os
from tempfile import gettempdir

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.validation import check_memory

def _do_fit(estimator, model_params, *args, **kwargs):
    return clone(estimator).fit(*args, **kwargs)

class MemoizedClassifier(BaseEstimator):
    """Creates a memoized version of an estimator.

    Subsequent calls to fit with the same arguments will reuse a
    prefitted model.

    memory is either a path or a joblib.Memory object.  If None a
    default path is used: "usklearn_cache" in systems default
    temporary directory.

    Parameters
    ----------

    estimator : a scikit-klearn 
        EstimatClassifier to wrap in a regessor interface.

    memory : a joblib.Memory object, default=None
    
    """

    def __init__(self, estimator, memory=None):
        self.estimator = estimator
        self.memory = memory
    def fit(self, *args, **kwargs):
        memory = self.memory
        if memory is None:
            memory = os.path.join(gettempdir(), "usklearn_cache")
        self.memory_ = check_memory(memory)
        if not hasattr(self, "do_fit_cached_"):
            self.do_fit_cached_ = self.memory_.cache(_do_fit, ignore=["estimator"])
        self.fitted_estimator_ = self.do_fit_cached_(self.estimator,
                                                     self.estimator.get_params(),
                                                     *args, **kwargs)
        return self
    
    def __getattr__(self, name):
        if name in ["fitted_estimator_", "do_fit_cached_"]:
            try:
                return self.__dict__[name]
            except:
                raise AttributeError(f"MemoizedClassifier has no attribute {name}")
        if "fitted_estimator_" not in self.__dict__:
            return getattr(self.estimator, name)
        return getattr(self.fitted_estimator_, name)

