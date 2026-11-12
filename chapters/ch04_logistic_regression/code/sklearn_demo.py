# ci: run
"""The lecture notebook's experiments with our own code and with scikit-learn."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from logreg import add_bias, fit, fit_one_vs_rest, fit_softmax, predict, predict_one_vs_rest, softmax

# --- Binary problem of the notebook: 500 examples, 20 features, random_state=42 -------------------
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=4)
A_tr, A_te = add_bias(Xtr), add_bias(Xte)

theta, hist = fit(A_tr, ytr.reshape(-1, 1), lr=0.1, iters=5000)
pred = predict(A_te, theta).ravel()
print("scratch   : J_train = %.4f  test acc = %.3f  macro-F1 = %.3f  precision = %.3f  recall = %.3f"
      % (hist[-1], accuracy_score(yte, pred), f1_score(yte, pred, average="macro"),
         precision_score(yte, pred, average="macro"), recall_score(yte, pred, average="macro")))
print("confusion matrix (rows = true, columns = predicted):\n", confusion_matrix(yte, pred))

# scikit-learn without regularisation gives (almost) the same model; with the default C = 1 it differs.
for name, clf in [("sklearn, no penalty", LogisticRegression(C=np.inf, max_iter=5000)),
                  ("sklearn, C = 1 (default)", LogisticRegression(max_iter=5000))]:
    clf.fit(Xtr, ytr)
    print(f"{name:26s}: test acc = {clf.score(Xte, yte):.3f}")

# --- Multi-class problem of the notebook: 3 classes, 2 informative features ------------------------
X3, y3 = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                             n_classes=3, n_clusters_per_class=1, random_state=42)
X3tr, X3te, y3tr, y3te = train_test_split(X3, y3, test_size=0.2, random_state=42)
B_tr, B_te = add_bias(X3tr), add_bias(X3te)

thetas = fit_one_vs_rest(B_tr, y3tr, 3, lr=0.1, iters=1000)
print("one-vs-rest (ours)  : test acc = %.3f" % accuracy_score(y3te, predict_one_vs_rest(B_te, thetas)))
Theta = fit_softmax(B_tr, y3tr, 3, lr=0.1, iters=1000)
print("softmax (ours)      : test acc = %.3f" % accuracy_score(y3te, np.argmax(B_te @ Theta, axis=1)))
sk = make_pipeline(StandardScaler(), LogisticRegression()).fit(X3tr, y3tr)
print("sklearn pipeline    : test acc = %.3f" % sk.score(X3te, y3te))
lr = LogisticRegression().fit(B_tr, y3tr)          # the notebook's model: default C = 1, column of ones included
pl = lr.predict(B_te)
print("notebook LogisticRegression: macro-F1 = %.3f  accuracy = %.3f"
      % (f1_score(y3te, pl, average="macro"), accuracy_score(y3te, pl)))
print(confusion_matrix(y3te, pl))
P = softmax(B_te @ Theta)
print("softmax rows sum to one:", np.allclose(P.sum(axis=1), 1.0))
