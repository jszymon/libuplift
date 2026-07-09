"""==================================
Qini and uplift curves
==================================

libuplift follows the convention that Qini curves are drawn with the
number of net successes on the y-axis (scaled to treatment group size)
and uplift curves with net gain in success probabilites.  This
conventions follow original publications where the curves were
introduced.

"""

####################################
# The necessary imports
####################################

import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from libuplift.meta import TLearnerUpliftClassifier
from libuplift.metrics import uplift_curve, Qini_curve
from libuplift.metrics import optimal_uplift_curve, optimal_Qini_curve
from libuplift.metrics import Qini_coefficient



####################################
# Fetch and prepare data
####################################

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


#############
# Fit a model
#############

X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(X, y, trt, train_size=0.7)
m = TLearnerUpliftClassifier(base_estimator=LogisticRegression())
m.fit(X_train, y_train, trt_train, n_trt=1)

#############################
# Draw uplift and Qini curves
#############################

score = m.predict(X_test)[:,1]

plt.figure(figsize=(10,5))

cx, cy = Qini_curve(y_test, score, trt_test, n_trt=1)
cx_opt, cy_opt = optimal_Qini_curve(y_test, trt_test, n_trt=1)
plt.subplot(1,2,1)
plt.plot(cx, cy)
plt.plot(cx_opt, cy_opt, "r-")
plt.plot([0,1], [0,cy[-1]], "k-")
plt.title("Qini curve (success count)")

cx, cy = uplift_curve(y_test, score, trt_test, n_trt=1)
cx_opt, cy_opt = optimal_uplift_curve(y_test, trt_test, n_trt=1)
plt.subplot(1,2,2)
plt.plot(cx, cy)
plt.plot(cx_opt, cy_opt, "r-")
plt.plot([0,1], [0,cy[-1]], "k-")
plt.title("Uplift curve (success prob.)")
plt.show()

# Now print the Qini coefficient as defined by Radcliffe and Surry
print("Qini coefficient ", Qini_coefficient(y_test, score, trt_test, n_trt=1))

# The coefficient is the same for Qini and uplift curves

# Dependence of Qini curves on sample size 
