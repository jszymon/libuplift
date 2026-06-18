"""
=======
Lalonde
=======

Build uplift regression models on Lalonde data.
"""

import numpy as np

from sklearn.linear_model import Ridge

from libuplift.datasets import fetch_Lalonde
from libuplift.meta import MultimodelUpliftRegressor
from libuplift.meta import MultimodelUpliftLinearRegressor
from libuplift.metrics import e_sate, e_satt
from libuplift.model_selection import cross_validate, cross_val_score
from libuplift.model_selection import GridSearchCV

DA = fetch_Lalonde("A")
print(DA.data.shape)
DB = fetch_Lalonde("B")
print(DB.data.shape)
D = DB
X = D.data
y = np.log1p(D.target_RE78)
trt = D.treatment

r = MultimodelUpliftRegressor()
r.fit(X, y, trt)
print(r)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, r.predict(X), trt, n_trt=1))
print("crossval SATE:", cross_val_score(r, X, y, trt, n_trt=1, cv=10))
print("crossval SATE:", cross_validate(r, X, y, trt,
                                       n_trt=1, cv=10,
                                       scoring=["e_sate"])["test_e_sate"])

# linear regression uplift model with coef_ and intercept_
rlin = MultimodelUpliftLinearRegressor()
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
                  cv=3)
rr.fit(X, y, trt)
print("\n\n")
print(rr)
print("best alpha:", rr.best_params_)
print("training SATE:", rr.score(X, y, trt))
print("training SATT:", e_satt(y, rr.predict(X), trt))
print("crossval SATE:", cross_val_score(rr, X, y, trt, n_trt=1, cv=10))
#print("crossval SATE:", cross_validate(rr, X, y, trt,
#                                       n_trt=1, cv=10,
#                                       scoring="e_sate")["test_score"])

#: tune T/C ridge separately
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
print("crossval SATE:", cross_val_score(rr2, X, y, trt, n_trt=1, cv=10))

