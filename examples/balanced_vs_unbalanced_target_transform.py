"""Target transform uplift classifier with and without weighting to
make treatment distributions equal."""

import numpy as np
np.random.seed(123)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from libuplift.datasets import fetch_Hillstrom
from libuplift.meta import TargetTransformUpliftClassifier
from libuplift.model_selection import cross_val_score

D = fetch_Hillstrom(as_frame=True)

ohe = OneHotEncoder()
ct = ColumnTransformer([("one_hot", ohe, list(D.categ_values.keys()))],
                       remainder=StandardScaler())
X = ct.fit_transform(D.data)
y = D.target_visit
trt = D.treatment
# merge both campaigns for unequal treatment probability
trt[trt==2] = 1
n_trt = 1

m1 = TargetTransformUpliftClassifier(balance_treatments=False)
m2 = TargetTransformUpliftClassifier()

auucs1 = cross_val_score(m1, X, y, trt, n_trt, scoring="auuc", cv=10)
auucs2 = cross_val_score(m2, X, y, trt, n_trt, scoring="auuc", cv=10)

print("Unbalanced: ", np.mean(auucs1))
print("Balanced:   ", np.mean(auucs2))
