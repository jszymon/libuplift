import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """Two-part hurdle regression metamodel for targets with a point
    mass at zero.

    Unlike other hurdle models negative target values are allowed,
    useful for uplift models based on target variable transformation.

    Stage 1 (the hurdle): a classifier models P(y != 0 | X), where a value
    counts as "zero" if ``|y| <= threshold``.
    Stage 2 (the non-zero part): a regressor is fit only on rows with
    ``|y| > threshold`` and models ``E[y | y != 0, X]``. Its output can be negative.

    The prediction is the unconditional mean (near-zero values count as 0):

        E[y | X] = P(y != 0 | X) * E[y | y != 0, X]

    Parameters
    ----------
    classifier : estimator with fit/predict_proba, default=None
        Model for the zero / non-zero decision. Defaults to LogisticRegression.
    regressor : estimator with fit/predict, default=None
        Model for the non-zero part. Defaults to LinearRegression. Any
        regressor that handles signed targets works. If your non-zero values
        are all one sign, you can use e.g. PoissonRegressor / GammaRegressor
        (for positive y) instead.
    threshold : float, default=0.0
        Rows with ``|y| <= threshold`` are treated as zeros (hurdle not crossed).
        Must be >= 0. With 0.0, only exact zeros are zeros.

    Attributes
    ----------
    ``classifier_`` : estimator or None
        Fitted hurdle classifier (None if only one class was seen).
    ``regressor_`` : estimator or None
        Fitted non-zero regressor (None if no non-zero rows).
    ``p_nonzero_const_`` : float
        Share of non-zero rows (used if ``classifier_`` is None).
    ``n_features_in_`` : int
        Number of features seen during fit.

    """

    def __init__(self, classifier=None, regressor=None, threshold=0.0):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0.")
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        nonzero = np.abs(y) > self.threshold
        fit_kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}

        # Stage 1: hurdle classifier
        self.p_nonzero_const_ = float(nonzero.mean())
        if nonzero.all() or not nonzero.any():
            self.classifier_ = None  # degenerate: constant probability
        else:
            clf = clone(self.classifier) if self.classifier is not None \
                else LogisticRegression(max_iter=1000)
            self.classifier_ = clf.fit(X, nonzero.astype(int), **fit_kw)

        # Stage 2: regressor on the non-zero rows
        if nonzero.any():
            reg = clone(self.regressor) if self.regressor is not None \
                else LinearRegression()
            kw = {} if sample_weight is None else {"sample_weight": fit_kw["sample_weight"][nonzero]}
            self.regressor_ = reg.fit(X[nonzero], y[nonzero], **kw)
        else:
            self.regressor_ = None
        return self

    def predict_proba_nonzero(self, X):
        """``P(|y| > threshold | X)``."""
        check_is_fitted(self, "p_nonzero_const_")
        X = check_array(X)
        if self.classifier_ is None:
            return np.full(X.shape[0], self.p_nonzero_const_)
        return self.classifier_.predict_proba(X)[:, 1]

    def predict_nonzero(self, X):
        """``E[y | |y| > threshold, X]``."""
        check_is_fitted(self, "p_nonzero_const_")
        X = check_array(X)
        if self.regressor_ is None:
            return np.zeros(X.shape[0])
        return self.regressor_.predict(X)

    def predict(self, X):
        """Predict mean E[y | X]."""
        return self.predict_proba_nonzero(X) * self.predict_nonzero(X)


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(0.5 + 1.2 * X[:, 0])))
    nonzero = rng.random(n) < p
    signed = 1.0 + 2.0 * X[:, 1] - 1.5 * X[:, 2] + rng.normal(scale=0.5, size=n)
    signed[np.abs(signed) < 0.1] = 0.5          # keep the non-zero part away from 0
    y = np.where(nonzero, signed, 0.0)
    y[rng.random(n) < 0.05] = rng.normal(scale=0.01)   # some near-zero noise

    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    model = make_pipeline(StandardScaler(), HurdleRegressor(threshold=0.05))
    model.fit(Xtr, ytr)
    print("R^2 test:", round(model.score(Xte, yte), 3))
    print("share negative targets:", round((y < 0).mean(), 2))
    print("CV R^2:", cross_val_score(HurdleRegressor(threshold=0.05), X, y, cv=3).round(3))
