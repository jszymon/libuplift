"""Demo of some model selection and evaluation functions."""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

from libuplift.datasets import fetch_Lalonde, fetch_Hillstrom
from libuplift.meta import MultimodelUpliftRegressor
from libuplift.model_selection import (cross_val_score,
                                      cross_val_predict,
                                      permutation_test_score,
                                      learning_curve)

if False:
    D = fetch_Lalonde("B", as_frame=False)
    X = D.data
    y = np.log1p(D.target_RE78)
    trt = D.treatment

if True:
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
    #y = D.target_visit
    trt = D.treatment
    trt[trt==2] = 1


r = MultimodelUpliftRegressor()
print("Crossvalidated predictions:")
print(cross_val_predict(r, X, y, trt, n_trt=1, cv=10))

print("\n\nPermutation based model score")
score, permutation_scores, pv = permutation_test_score(r, X, y, trt, n_trt=1, cv=10,
                                                       n_permutations=100, n_jobs=-1,
                                                       verbose=10,
                                                       #scoring="auuc"
                                                       )
r.fit(X, y, trt, n_trt=1)
print("score:", r.score(X, y, trt, n_trt=1))
plt.hist(permutation_scores, density=True, label=f"p-value={pv}")
plt.axvline(score, color="r")
plt.show()

print("\n\nLearning curve")
train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(r, X, y, trt, n_trt=1, return_times=True)
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.fill_between(train_sizes,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.1, color='r'
                 )
plt.plot(train_sizes, train_scores_mean, 'ro-',
                 label="Train score")
plt.fill_between(train_sizes,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.1, color='g'
                 )
plt.plot(train_sizes, test_scores_mean, 'go-',
                 label="Test score")
plt.legend()
plt.show()

#auucs = cross_val_score(r, X, y, trt, n_trt=1, scoring="auuc")
#auucs_j = cross_val_score(r, X, y, trt, n_trt=1, scoring="auuc_j")
#print("crossval AUUC:", auucs)
