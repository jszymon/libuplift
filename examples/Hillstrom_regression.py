"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from sklearn.linear_model import Ridge

from libuplift.datasets import fetch_Hillstrom
from libuplift.meta import MultimodelUpliftRegressor
from libuplift.meta import MultimodelUpliftLinearRegressor
from libuplift.metrics import e_sate, e_satt
from libuplift.model_selection import cross_validate, cross_val_score
from libuplift.model_selection import GridSearchCV

def encode_features(D):
    """Convert features to float matrix.

    Use K-1 encoding for categorical variables."""
    X = D.data
    cols = []
    for c in D.feature_names:
        if c not in D.categ_values:
            cols.append(np.asarray(X[c], float))
        else:
            n_categs = len(D.categ_values[c])
            x = np.eye(n_categs)[X[c]]
            cols.append(x[:,:-1]) # skip last category
    return np.column_stack(cols)

D = fetch_Hillstrom(as_frame=True)
X = encode_features(D)
y = np.log1p(D.target_spend)
trt = D.treatment

r = MultimodelUpliftRegressor()
r.fit(X, y, trt)
print(r)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, r.predict(X), trt, n_trt=2))
print("crossval SATE:", cross_val_score(r, X, y, trt, n_trt=2, cv=10))
print("crossval SATE:", cross_validate(r, X, y, trt,
                                       n_trt=2, cv=10,
                                       scoring=["e_sate"])["test_e_sate"])

# linear regression uplift model with coef_ and intercept_
rlin = MultimodelUpliftLinearRegressor()
rlin.fit(X, y, trt)
print("\n\n")
print(rlin, rlin.coef_.shape, rlin.intercept_.shape)
print("training SATE:", rlin.score(X, y, trt))
print("training SATT:", e_satt(y, rlin.predict(X), trt, n_trt=2))

# merge treatments
trt[trt==2] = 1
r.fit(X, y, trt)
print("\n\n")
print(r)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, r.predict(X), trt))
print("crossval SATE:", cross_val_score(r, X, y, trt, n_trt=1, cv=10))
print("crossval SATE:", cross_validate(r, X, y, trt,
                                       n_trt=1, cv=10,
                                       scoring=["e_sate"])["test_e_sate"])


rlin.fit(X, y, trt)
print("\n\n")
print(rlin, rlin.coef_.shape, rlin.intercept_.shape)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, rlin.predict(X), trt, n_trt=1))
print("crossval SATE:", cross_val_score(rlin, X, y, trt, n_trt=1, cv=10))

# tuned ridge regression
rridge = MultimodelUpliftRegressor(base_estimator=Ridge())
rr = GridSearchCV(rridge,
                  {"base_estimator__alpha":[0,1e-3,1e-2,1e-1,1,1e1,1e2,1e3]},
                  cv=3, n_jobs=-1)
rr.fit(X, y, trt)
print("\n\n")
print(rr)
print("best alpha:", rr.best_params_)
print("training SATE:", rr.score(X, y, trt))
print("training SATT:", e_satt(y, rr.predict(X), trt))
print("crossval SATE:", cross_val_score(rr, X, y, trt, n_trt=1, cv=10))

#tune T/C ridge separately
rridge2 = MultimodelUpliftRegressor(base_estimator = [("model_c", Ridge()),
                                                ("model_t", Ridge())])
rr2 = GridSearchCV(rridge2,
                    {"model_c__alpha":[0,1e-3,1e-2,1e-1,1,1e1,1e2,1e3],
                    "model_t__alpha":[0,1e-3,1e-2,1e-1,1,1e1,1e2,1e3]},
                    cv=3)
rr2.fit(X, y, trt)
print("\n\n")
print(rr)
print("best alpha:", rr2.best_params_)
print("training SATE:", rr2.score(X, y, trt))
print("training SATT:", e_satt(y, rr2.predict(X), trt))
print("crossval SATE:", cross_val_score(rr2, X, y, trt, n_trt=1, cv=10, n_jobs=-1))

# permutation test
from libuplift.model_selection import permutation_test_score
print(permutation_test_score(r, X, y, trt, n_trt=1, cv=3, n_permutations=100, n_jobs=-1))
