# ci: run
"""Experiments of Chapter 5: our metrics vs scikit-learn, imbalance, hold-out vs k-fold, data leakage."""
import numpy as np
from sklearn import metrics as skm
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as sk_cv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import metrics as M

rng = np.random.default_rng(0)

# 1. Our functions agree with scikit-learn ---------------------------------------------------
y = rng.integers(0, 2, 500)
s = np.clip(0.35 * y + rng.normal(0.4, 0.25, 500), 0, 1)           # a noisy score
p = (s >= 0.5).astype(int)
checks = {
    "accuracy": (M.accuracy(y, p), skm.accuracy_score(y, p)),
    "precision": (M.precision(y, p), skm.precision_score(y, p)),
    "recall": (M.recall(y, p), skm.recall_score(y, p)),
    "F1": (M.fbeta(y, p), skm.f1_score(y, p)),
    "F2": (M.fbeta(y, p, 2), skm.fbeta_score(y, p, beta=2)),
    "ROC AUC": (M.auc(*M.roc_curve(y, s)), skm.roc_auc_score(y, s)),
    "average precision": (M.average_precision(y, s), skm.average_precision_score(y, s)),
}
for name, (a, b) in checks.items():
    assert abs(a - b) < 1e-9, name
    print(f"{name:18s} ours {a:.4f}  sklearn {b:.4f}")
yr = rng.normal(size=100) * 5 + 50
hr = yr + rng.normal(size=100) * 3
assert abs(M.r2(yr, hr) - skm.r2_score(yr, hr)) < 1e-12 and abs(M.rmse(yr, hr) - np.sqrt(skm.mean_squared_error(yr, hr))) < 1e-12

# 2. The accuracy paradox: 1 % positives -----------------------------------------------------
X, y = make_classification(n_samples=20000, n_features=10, n_informative=4, weights=[0.99, 0.01],
                           class_sep=1.0, flip_y=0.0, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
pd = dummy.predict(Xte)
print("\ntest set: %d examples, %d positives" % (len(yte), yte.sum()))
print("always-negative : accuracy %.4f precision %.2f recall %.2f" % (M.accuracy(yte, pd), M.precision(yte, pd), M.recall(yte, pd)))
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
sc = lr.predict_proba(Xte)[:, 1]
for thr in (0.5, 0.2, 0.1, 0.05):
    pr = (sc >= thr).astype(int)
    print("logreg, thr %.2f : TP FP FN TN = %s  acc %.4f prec %.3f rec %.3f F1 %.3f"
          % (thr, M.confusion(yte, pr), M.accuracy(yte, pr), M.precision(yte, pr), M.recall(yte, pr), M.fbeta(yte, pr)))
print("ROC AUC %.3f   average precision %.3f   (positive rate %.4f)" %
      (M.auc(*M.roc_curve(yte, sc)), M.average_precision(yte, sc), yte.mean()))

# 3. One hold-out split against 10-fold cross-validation ---------------------------------------
Xb, yb = make_classification(n_samples=300, n_features=10, n_informative=4, class_sep=0.8, random_state=3)
single = []
for seed in range(200):
    a, b, c, d = train_test_split(Xb, yb, test_size=0.2, random_state=seed)
    single.append(LogisticRegression(max_iter=1000).fit(a, c).score(b, d))
single = np.array(single)
cv = []
for seed in range(200):
    cv.append(M.cross_val_score(lambda A, c, B: LogisticRegression(max_iter=1000).fit(A, c).predict(B),
                                Xb, yb, k=10, seed=seed).mean())
cv = np.array(cv)
print("\n200 repetitions, 300 examples:")
print("hold-out 80/20 : mean %.3f  sd %.3f  range %.2f-%.2f" % (single.mean(), single.std(), single.min(), single.max()))
print("10-fold CV     : mean %.3f  sd %.3f  range %.2f-%.2f" % (cv.mean(), cv.std(), cv.min(), cv.max()))

# 4. Data leakage: feature selection on pure noise ----------------------------------------------
rng = np.random.default_rng(5)
Xn = rng.normal(size=(100, 5000)); yn = rng.integers(0, 2, 100)          # labels independent of features
wrong = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)               # selection sees all labels
acc_wrong = sk_cv(LogisticRegression(max_iter=1000), wrong, yn, cv=5).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
acc_right = sk_cv(pipe, Xn, yn, cv=5).mean()
print("\nnoise data, true accuracy 0.50: selection before CV %.3f, selection inside CV %.3f" % (acc_wrong, acc_right))
