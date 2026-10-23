"""
=========
Hillstrom
=========

Build uplift classification and regression models on Telecom churn data.
"""

import numpy as np

from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from libuplift.datasets import fetch_TelecomChurn
from libuplift.meta import MultimodelUpliftRegressor
from libuplift.meta import MultimodelUpliftClassifier
from libuplift.meta import TreatmentUpliftClassifier
from libuplift.meta import ResponseUpliftClassifier
from libuplift.meta import ControlUpliftClassifier
from libuplift.meta import TargetTransformUpliftRegressor
from libuplift.meta import TargetTransformUpliftClassifier

from libuplift.metrics import uplift_curve
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
        if not c.startswith("FACTOR"):
            cols.append(np.asarray(X[c]))
        else:
            x = OneHotEncoder(sparse_output=False, drop='first').fit_transform(X[[c]])
            cols.append(x)
    return np.column_stack(cols)

D = fetch_TelecomChurn(as_frame=True)
X = encode_features(D)
y = D.target
trt = D.treatment
n_trt = 1

n_iter = 100

base_classifier = Pipeline([("scaler", StandardScaler()),
                            ("logistic", LogisticRegression(max_iter=1000, C=0.1, random_state=234))])
#base_classifier = RandomForestClassifier(n_estimators=10)
models = [MultimodelUpliftRegressor(),
          MultimodelUpliftClassifier(base_estimator=base_classifier),
          TreatmentUpliftClassifier(base_estimator=base_classifier, reverse=False),
          TreatmentUpliftClassifier(base_estimator=base_classifier, reverse=True),
          ResponseUpliftClassifier(base_estimator=base_classifier, reverse=True),
          #ControlUpliftClassifier(base_estimator=base_classifier, reverse=True),
          #ControlUpliftClassifier(base_estimator=base_classifier, reverse=False),
          TargetTransformUpliftRegressor(),
          TargetTransformUpliftClassifier(),
          ]
cv, y_stratify = uplift_check_cv(StratifiedShuffleSplit(test_size=0.1,
                                                        n_splits=n_iter,
                                                        random_state=123),
                                 y, trt, n_trt, classifier=True)

colors = list("rgbkcym") + ["orange"]
avg_x = np.linspace(0,1,1000)
avg_u = np.zeros((len(models), len(avg_x)))

for train_index, test_index in tqdm(cv.split(X, y_stratify), total=n_iter):
    for mi, m in enumerate(models):
        m.fit(X[train_index], y[train_index], trt[train_index], n_trt)
        #test_index = train_index
        score = m.predict(X[test_index])
        if is_classifier(m):
            score = score[:,0] # pos_label is 0
        else:
            score = -score # for regressor: pos_label is 0
        x, u = uplift_curve(y[test_index], score, trt[test_index], n_trt, pos_label=0)
        plt.plot(x, u, color=colors[mi], alpha=0.05)

        avg_u[mi] += np.interp(avg_x, x, u)

for mi, m in enumerate(models):
    plt.plot(avg_x, avg_u[mi]/n_iter, color=colors[mi], lw=3)
plt.plot([0,1], [0,avg_u[0,-1]/n_iter], "k-")
plt.show()
