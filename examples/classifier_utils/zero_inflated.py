"""=============================================
Uplift regression with zero inflated targets
=============================================

Many uplift regression problems involve targets which are zero for
most cases, e.g. spend on a marketing campaign where most customers
do not buy anything.  Ordinary least squares regression ignores this
structure.  This example compares several models better suited to
zero inflated targets on the Hillstrom data with ``spend`` as the
target:

* T-learners with Tweedie regression (``power=1.5``) which allows for
  a point mass at zero and a continuous positive part,
* T-learners with a hurdle model: logistic regression predicts whether
  spend is nonzero and Gamma regression predicts the amount spent by
  buying customers,
* target transformation models with standard linear regression,
* target transformation models with a hurdle model based on linear
  regression.  The transformed target is also zero inflated, but its
  nonzero values may be negative, so Gamma regression cannot be used
  for the nonzero part.

Note that Gamma regression cannot be used directly as the base model
of the T-learner since the Gamma distribution does not allow zero
values.
"""

# %%
# The necessary imports
#######################

import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import GammaRegressor, TweedieRegressor

from libuplift.classifiers import HurdleRegressor
from libuplift.meta import TLearnerUpliftRegressor
from libuplift.meta import TargetTransformUpliftRegressor
from libuplift.metrics import uplift_curve
from libuplift.metrics import area_under_uplift_curve


# %%
# Fetch and prepare data
########################
#
# We use the women's e-mail campaign and the control group.  Only about
# 1% of customers spent any money.

from libuplift.datasets import fetch_Hillstrom
D = fetch_Hillstrom(as_frame=True)
trt = D.treatment

# encode categorical features, standardize numerical features
ct = ColumnTransformer([("ohe", OneHotEncoder(), list(D.categ_values.keys()))],
                       remainder=StandardScaler())
X = ct.fit_transform(D.data)

# keep only women's campaign
mask = (trt != 1)
X = X[mask]
y = np.asarray(D.target_spend[mask], float)
trt = (trt[mask] == 2) * 1

print(f"fraction of nonzero spend: {(y > 0).mean():.4f}")

X_train, X_test, y_train, y_test, trt_train, trt_test =\
    train_test_split(X, y, trt, train_size=0.5, stratify=trt)

# %%
# Define the models
###################
#
# Small regularization is used for the generalized linear models.
# The hurdle model's classifier predicts P(spend > 0), its regressor
# is fit only on records with nonzero spend.

models = {
    "T-learner, Tweedie (p=1.5)":
        TLearnerUpliftRegressor(base_estimator=TweedieRegressor(
            power=1.5, link="log", alpha=1e-4, max_iter=1000)),
    "T-learner, hurdle Gamma":
        TLearnerUpliftRegressor(base_estimator=HurdleRegressor(
            classifier=LogisticRegression(max_iter=1000),
            regressor=GammaRegressor(alpha=1e-4, max_iter=1000))),
    "Target transform, linear":
        TargetTransformUpliftRegressor(base_estimator=LinearRegression()),
    "Target transform, hurdle linear":
        TargetTransformUpliftRegressor(base_estimator=HurdleRegressor(
            classifier=LogisticRegression(max_iter=1000),
            regressor=LinearRegression())),
}

# %%
# Fit the models and compute uplift curves on the test set
##########################################################

curves = {}
auucs = {}
for name, m in models.items():
    m.fit(X_train, y_train, trt_train, n_trt=1)
    score = m.predict(X_test)
    curves[name] = uplift_curve(y_test, score, trt_test, n_trt=1)
    auucs[name] = area_under_uplift_curve(y_test, score, trt_test, n_trt=1)
    print(f"{name:35s} AUUC = {auucs[name]:.4f}")

# %%
# Plot uplift curves
####################
#
# The y-axis is the average net gain in spend per customer when the
# given fraction of customers with highest predicted uplift is
# targeted.

fig, ax = plt.subplots(figsize=(8, 6))
for name, (cx, cy) in curves.items():
    ax.plot(cx, cy, label=f"{name} (AUUC={auucs[name]:.4f})")
ax.plot([0, 1], [0, cy[-1]], "k--", label="random")
ax.set_xlabel("fraction of customers targeted")
ax.set_ylabel("net gain in spend per customer")
ax.set_title("Uplift curves, Hillstrom women's campaign (spend)")
ax.legend()
plt.show()

# %%
# All models perform better than random targeting.  Models which
# account for the large number of zero targets worked better than
# plain linear regression on the transformed target, which has the
# lowest AUUC and does not beat random selection for the first 25% of
# customers.  The hurdle models performed best: the target
# transformation model with a hurdle linear regressor achieved the
# highest AUUC, followed by the T-learner with a hurdle Gamma model,
# which gives the best results for small fractions of targeted
# customers.  The Tweedie T-learner falls in between.
#
# Note, however, that only about 1% of customers made a purchase, so
# the curves are noisy and the differences between models are small.
# The ranking may change for a different train/test split.
