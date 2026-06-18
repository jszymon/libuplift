"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libuplift.datasets import fetch_Hillstrom
from libuplift.meta import MultimodelUpliftClassifier
from libuplift.tree import UpliftTreeClassifier, export_text

from libuplift.metrics import uplift_curve, uplift_curve_j
from libuplift.model_selection import cross_validate, cross_val_score, uplift_check_cv

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except:
    def tqdm(x, total=np.inf):
        return x


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
y = D.target_visit
trt = D.treatment
# keep women's campaign
mask = ~(trt == 1)
X = X[mask]
y = y[mask]
trt = (trt[mask] == 2)*1
n_trt = 1

n_iter = 10

base_classifier = Pipeline([("scaler", StandardScaler()),
                            ("logistic", LogisticRegression(max_iter=1000))])
# base_classifier = HistGradientBoostingClassifier()
# # Memoize gradient boosting to avoid recomputing same models
# base_classifier = MemoizedClassifier(base_classifier)

tree = UpliftTreeClassifier(max_depth=2, min_samples_leaf=100)
tree.fit(X, y, trt, n_trt)
print(export_text(tree))


models = [MultimodelUpliftClassifier(base_estimator=base_classifier),
          UpliftTreeClassifier(max_depth=3, min_samples_leaf=100)
          ]
cv, y_stratify = uplift_check_cv(StratifiedShuffleSplit(test_size=0.3,
                                                        n_splits=n_iter,
                                                        random_state=123),
                                 y, trt, n_trt, classifier=True)

colors = list("rgbkcym") + ["orange", "lime", "grey", "brown", "pink", "gold", "purple"]
avg_x = np.linspace(0,1,1000)
avg_u = np.zeros((len(models), len(avg_x)))

for train_index, test_index in tqdm(cv.split(X, y_stratify), total=n_iter):
    for mi, m in enumerate(models):
        m.fit(X[train_index], y[train_index], trt[train_index], n_trt)
        score = m.predict(X[test_index])
        if is_classifier(m) or True:
            score = score[:,1]
        x, u = uplift_curve_j(y[test_index], score, trt[test_index], n_trt)
        plt.plot(x, u, color=colors[mi], alpha=0.5/n_iter)

        avg_u[mi] += np.interp(avg_x, x, u)

for mi, m in enumerate(models):
    plt.plot(avg_x, avg_u[mi]/n_iter, color=colors[mi], lw=3)
plt.plot([0,1], [0,avg_u[0,-1]/n_iter], "k-")
plt.show()
