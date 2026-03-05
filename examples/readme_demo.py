"""Demo from uplift-sklearn README.md"""


### necessary imports

import numpy as np
np.random.seed(123)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from libuplift.meta import TLearnerUpliftClassifier


### fetch and prepare data

from libuplift.datasets import fetch_Hillstrom
D = fetch_Hillstrom(as_frame=True)
trt = D.treatment
# encode categorical features, standardize numerical features
ct = ColumnTransformer([("ohe", OneHotEncoder(), list(D.categ_values.keys()))],
                       remainder=StandardScaler())
X = ct.fit_transform(D.data)
# keep only women's campaign
mask = ~(trt == 1)
X = X[mask]
y = D.target_visit[mask]
trt = (trt[mask] == 2)*1


### fit model and draw uplift curve


X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(X, y, trt, train_size=0.7)
m = TLearnerUpliftClassifier(base_estimator=LogisticRegression())
m.fit(X_train, y_train, trt_train, n_trt=1)

import matplotlib.pyplot as plt
from libuplift.metrics import uplift_curve, area_under_uplift_curve

score = m.predict(X_test)[:,1]
print("AUUC=", area_under_uplift_curve(y_test, score, trt_test, n_trt=1))
cx, cy = uplift_curve(y_test, score, trt_test, n_trt=1)
plt.plot(cx, cy)
plt.plot([0,1], [0,cy[-1]], "k-")
plt.show()


### tune model parameters using crossvalidation

# import those from libuplift instead of sklearn
from libuplift.model_selection import cross_val_score
from libuplift.model_selection import GridSearchCV

m1 = TLearnerUpliftClassifier(base_estimator=LogisticRegression())
m_cv1 = GridSearchCV(m1,
                     {"base_estimator__C":[1e-1,1,1e1,1e2,1e3]},
                     cv=3, n_jobs=-1)
# tune regularization of treatment/control models separately
m2 = TLearnerUpliftClassifier(base_estimator=[("model_c", LogisticRegression()),
                                              ("model_t", LogisticRegression())])
m_cv2 = GridSearchCV(m2,
                    {"model_c__C":[1e-1,1,1e1,1e2,1e3],
                    "model_t__C":[1e-1,1,1e1,1e2,1e3]},
                    cv=3, n_jobs=-1)

auuc_m1 = np.mean(cross_val_score(m_cv1, X, y, trt, n_trt=1, cv=5, scoring="auuc"))
auuc_m2 = np.mean(cross_val_score(m_cv2, X, y, trt, n_trt=1, cv=5, scoring="auuc"))
print("crossval AUUC m1:", auuc_m1)
print("crossval AUUC m2:", auuc_m2)

# refit and find best regularization params
m_cv1.fit(X, y, trt, n_trt=1)
print("best params: ", m_cv1.best_params_)



### verify model significance using permutation test, draw learning curve

# those functions are thin wrappers around original sklearn functions,
# so they accept the same set of parameters
from libuplift.model_selection import permutation_test_score, learning_curve

score, permutation_scores, pv =\
    permutation_test_score(m, X, y, trt, n_trt=1, cv=3,
                           n_permutations=100, scoring="auuc",
                           verbose=10, n_jobs=-1)

fix, (ax0, ax1) = plt.subplots(ncols=2)
ax0.hist(permutation_scores, density=True, label=f"p-value={pv}")
ax0.axvline(score, color="r")
ax0.set_title("Permutation test")

train_sizes, train_scores, test_scores = learning_curve(m, X, y, trt, n_trt=1, scoring="auuc")

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
ax1.fill_between(train_sizes,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.1, color='r')
ax1.plot(train_sizes, train_scores_mean, 'ro-', label="Train score")
ax1.fill_between(train_sizes,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.1, color='g')
ax1.plot(train_sizes, test_scores_mean, 'go-', label="Test score")
ax1.legend()
ax1.yaxis.tick_right()
ax1.set_title("Learning curve")
plt.show()
