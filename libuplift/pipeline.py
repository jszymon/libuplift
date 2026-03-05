import sklearn.pipeline
from sklearn.utils.metaestimators import if_delegate_has_method

class Pipeline(sklearn.pipeline.Pipeline):
    """Uplift version of sklearn's Pipeline.

    allows trt, n_trt args in fit method and passes them to the final
    estimator. (it makes no sense to pass them to transforms since trt
    is not available for future data).
    """
    
    def fit(self, X, y=None, trt=None, n_trt=None, **fit_params):
        n, _ = self.steps[-1]
        fit_params[n + "__trt"] = trt
        fit_params[n + "__n_trt"] = n_trt
        super().fit(X, y, **fit_params)
    # a copy of score method was necessary since it does not accept
    # score_params as arguments
    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, trt=None, n_trt=None, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        trt : vector of treatments
        n_trt : number of different treatments
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, trt, n_trt, **score_params)
    @if_delegate_has_method(delegate='_final_estimator')
    def predict_action(self, X, y=None, **predict_action_params):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        predict_action_params : additional parameters to pass to
            predict_action method of final estimator

        Returns
        -------
        score : float
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_action(Xt,
                                        **predict_action_params)
