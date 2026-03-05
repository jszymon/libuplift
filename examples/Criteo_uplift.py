"""
Build uplift classification and regression models on Criteo data.
"""

import numpy as np

from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from libuplift.datasets import fetch_Criteo
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

D = fetch_Criteo()
X = D.data#.astype(np.float32)
y = D.target_visit
trt = D.treatment
n_trt = 1

del D # free some memory

n_iter = 10

base_classifier = Pipeline([("scaler", StandardScaler()),
                            ("logistic", LogisticRegression(max_iter=100, C=1, random_state=234,
                                                            solver="newton-cholesky"))])
#base_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
models = [#MultimodelUpliftRegressor(base_estimator=LinearRegression(copy_X=False)),
          #MultimodelUpliftClassifier(base_estimator=base_classifier),
          #TreatmentUpliftClassifier(base_estimator=base_classifier, reverse=False),
          #TreatmentUpliftClassifier(base_estimator=base_classifier, reverse=True),
          #ResponseUpliftClassifier(base_estimator=base_classifier),
          #ControlUpliftClassifier(base_estimator=base_classifier, reverse=True),
          #ControlUpliftClassifier(base_estimator=base_classifier, reverse=False),
          TargetTransformUpliftRegressor(),
          TargetTransformUpliftClassifier(base_estimator=base_classifier),
          ]
cv, y_stratify = uplift_check_cv(StratifiedShuffleSplit(test_size=0.3,
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
            score = score[:,1]
        x, u = uplift_curve(y[test_index], score, trt[test_index], n_trt)
        u = np.interp(avg_x, x, u)
        plt.plot(avg_x, u, color=colors[mi], alpha=0.05)
        avg_u[mi] += u

for mi, m in enumerate(models):
    plt.plot(avg_x, avg_u[mi]/n_iter, color=colors[mi], lw=3)
plt.plot([0,1], [0,avg_u[0,-1]/n_iter], "k-")
plt.show()
