"""======================================================
Accelerating multi-model training with MemoizedClassifier
======================================================

In uplift modeling, multiple meta-models often fit identical base estimators on
the same data subsets. For example, :class:`~libuplift.meta.TLearnerUpliftClassifier`
fits separate models on the control group and treatment groups, while
:class:`~libuplift.meta.TreatmentUpliftClassifier` fits a model on the treatment group only.

When evaluating both models together, the treatment model would normally be trained
twice from scratch. :class:`~libuplift.classifiers.MemoizedClassifier` wraps any
scikit-learn estimator to cache fitted models using ``joblib.Memory``. Subsequent
calls to ``fit`` with identical training data and parameters reuse the prefitted
model from cache, avoiding redundant computations and significantly speeding up
the workflow.

This example builds a :class:`~libuplift.meta.TLearnerUpliftClassifier` and a
:class:`~libuplift.meta.TreatmentUpliftClassifier` without and with memoization
on the Hillstrom dataset to demonstrate the speedup.

"""

# %%
# The necessary imports
#######################

import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from libuplift.classifiers import MemoizedClassifier
from libuplift.datasets import fetch_Hillstrom
from libuplift.meta import TLearnerUpliftClassifier, TreatmentUpliftClassifier

# %%
# Fetch and prepare Hillstrom data
##################################
#
# We load the Hillstrom dataset, encode categorical features, standardize
# numerical features, and select the control and women's visit campaign.

D = fetch_Hillstrom(as_frame=True)
trt = D.treatment

ct = ColumnTransformer(
    [("ohe", OneHotEncoder(), list(D.categ_values.keys()))],
    remainder=StandardScaler(),
)
X = ct.fit_transform(D.data)

# Keep only control (0) and women's campaign (2)
mask = ~(trt == 1)
X = X[mask]
y = D.target_visit[mask]
trt = (trt[mask] == 2) * 1
n_trt = 1

print(f"Number of samples: {X.shape[0]} (Control: {(trt == 0).sum()}, Treatment: {(trt == 1).sum()})")
print(f"Number of features: {X.shape[1]}")

# %%
# Training without memoization
##############################
#
# We train both models using an unmemoized :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
# Notice that ``TLearnerUpliftClassifier`` trains a model for control and a model
# for treatment. Then, ``TreatmentUpliftClassifier`` trains a treatment model
# on the exact same treatment subset again.

base_classifier = HistGradientBoostingClassifier(random_state=42)

# Fit TLearner without memoization
t0 = time.perf_counter()
tlearner_unmemo = TLearnerUpliftClassifier(base_estimator=base_classifier)
tlearner_unmemo.fit(X, y, trt, n_trt=n_trt)
t_tlearner_unmemo = time.perf_counter() - t0

# Fit TreatmentUpliftClassifier without memoization
t0 = time.perf_counter()
treatment_unmemo = TreatmentUpliftClassifier(base_estimator=base_classifier)
treatment_unmemo.fit(X, y, trt, n_trt=n_trt)
t_treatment_unmemo = time.perf_counter() - t0

t_total_unmemo = t_tlearner_unmemo + t_treatment_unmemo

print("Without memoization:")
print(f"  TLearner fit time:            {t_tlearner_unmemo:.3f} s")
print(f"  TreatmentUplift fit time:     {t_treatment_unmemo:.3f} s")
print(f"  Total fit time:               {t_total_unmemo:.3f} s")

# %%
# Training with memoization
###########################
#
# Now we wrap the base estimator in :class:`~libuplift.classifiers.MemoizedClassifier`.
# We specify a temporary directory as cache storage.
#
# When ``TLearnerUpliftClassifier`` fits, it caches both control and treatment models.
# When ``TreatmentUpliftClassifier`` is fitted next, it requires the treatment model on
# the same subset; ``MemoizedClassifier`` retrieves it directly from disk cache,
# completing almost instantaneously!

cache_dir = tempfile.mkdtemp()

try:
    memo_classifier = MemoizedClassifier(base_classifier, memory=cache_dir)

    # Fit TLearner with memoization
    t0 = time.perf_counter()
    tlearner_memo = TLearnerUpliftClassifier(base_estimator=memo_classifier)
    tlearner_memo.fit(X, y, trt, n_trt=n_trt)
    t_tlearner_memo = time.perf_counter() - t0

    # Fit TreatmentUpliftClassifier with memoization
    t0 = time.perf_counter()
    treatment_memo = TreatmentUpliftClassifier(base_estimator=memo_classifier)
    treatment_memo.fit(X, y, trt, n_trt=n_trt)
    t_treatment_memo = time.perf_counter() - t0

    t_total_memo = t_tlearner_memo + t_treatment_memo

    print("With memoization:")
    print(f"  TLearner fit time:            {t_tlearner_memo:.3f} s")
    print(f"  TreatmentUplift fit time:     {t_treatment_memo:.3f} s (cache hit!)")
    print(f"  Total fit time:               {t_total_memo:.3f} s")
    print()
    print(f"Speedup for TreatmentUplift:    {t_treatment_unmemo / t_treatment_memo:.1f}x faster")
    print(f"Overall training speedup:       {t_total_unmemo / t_total_memo:.2f}x faster")

    # %%
    # Verify identical predictions
    ##############################
    #
    # We confirm that memoization does not alter predictions. Both models produce
    # bit-for-bit identical outputs to their unmemoized counterparts.

    pred_tlearner_unmemo = tlearner_unmemo.predict(X)
    pred_tlearner_memo = tlearner_memo.predict(X)
    np.testing.assert_allclose(pred_tlearner_unmemo, pred_tlearner_memo)

    pred_treatment_unmemo = treatment_unmemo.predict(X)
    pred_treatment_memo = treatment_memo.predict(X)
    np.testing.assert_allclose(pred_treatment_unmemo, pred_treatment_memo)

    print("Predictions verified: memoized models produce identical results.")

    # %%
    # Subsequent fits reuse all models
    ##################################
    #
    # In scenarios like grid search or repeated evaluations where the same model
    # is fit on the same data again, all fits result in cache hits:

    t0 = time.perf_counter()
    tlearner_memo.fit(X, y, trt, n_trt=n_trt)
    t_tlearner_refit = time.perf_counter() - t0

    t0 = time.perf_counter()
    treatment_memo.fit(X, y, trt, n_trt=n_trt)
    t_treatment_refit = time.perf_counter() - t0

    print(f"Subsequent fit times (all cached):")
    print(f"  TLearner refit time:          {t_tlearner_refit:.4f} s")
    print(f"  TreatmentUplift refit time:   {t_treatment_refit:.4f} s")

    # %%
    # Visualizing training times
    ############################
    #
    # Finally, we compare the training times visually.

    labels = ["TLearner", "TreatmentUplift", "Total"]
    unmemo_times = [t_tlearner_unmemo, t_treatment_unmemo, t_total_unmemo]
    memo_times = [t_tlearner_memo, t_treatment_memo, t_total_memo]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, unmemo_times, width, label="Without Memoization", color="#d9534f")
    rects2 = ax.bar(x + width / 2, memo_times, width, label="With Memoization", color="#5cb85c")

    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Model Training Time With and Without Memoization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}s",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

finally:
    shutil.rmtree(cache_dir, ignore_errors=True)
