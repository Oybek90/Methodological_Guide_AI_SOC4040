# ci: skip
"""Generate the figures of Chapter 5 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import metrics as M  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. One large error: MAE, RMSE and R^2 of the notebook data, before and after -------------------
h = np.array([11, 19, 28, 35, 48, 60, 66, 78, 91, 99], float)
y = np.array([10, 20, 30, 40, 40, 60, 70, 80, 90, 100], float)
h2 = h.copy(); h2[-1] = 140.0                         # the last prediction is now 40 too large
vals = {"MAE": (M.mae(y, h), M.mae(y, h2)), "RMSE": (M.rmse(y, h), M.rmse(y, h2))}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
x = np.arange(2); w = 0.35
ax[0].bar(x - w / 2, [v[0] for v in vals.values()], w, color="0.75", edgecolor="k", label="original")
ax[0].bar(x + w / 2, [v[1] for v in vals.values()], w, color="0.25", edgecolor="k", label="one error of 40")
ax[0].set_xticks(x); ax[0].set_xticklabels(list(vals)); ax[0].set_ylabel("error (target units)"); ax[0].legend(frameon=False, fontsize=8)
for xi, (a, b) in zip(x, vals.values()):
    ax[0].text(xi - w / 2, a + 0.2, f"{a:.2f}", ha="center", fontsize=7); ax[0].text(xi + w / 2, b + 0.2, f"{b:.2f}", ha="center", fontsize=7)
ax[1].plot([0, 140], [0, 140], color="0.6", lw=0.8)
ax[1].scatter(y, h, c="white", edgecolors="k", s=22, zorder=3)
ax[1].scatter(y[-1:], h2[-1:], c="k", s=26, marker="^", zorder=3)
ax[1].set_xlabel("actual $y$"); ax[1].set_ylabel(r"predicted $h(x)$")
save(fig, "outlier_metrics.pdf")
print("MAE/RMSE original", [round(v[0], 3) for v in vals.values()], "with outlier", [round(v[1], 3) for v in vals.values()],
      "R2", round(M.r2(y, h), 4), round(M.r2(y, h2), 4))

# 2. Precision, recall and F1 against the threshold (balanced binary data of Chapter 4) ---------
X, yy = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.2, random_state=4)
sc = LogisticRegression(C=np.inf, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
ths = np.linspace(0.05, 0.95, 91)
P = [M.precision(yte, (sc >= t).astype(int)) for t in ths]
R = [M.recall(yte, (sc >= t).astype(int)) for t in ths]
F = [M.fbeta(yte, (sc >= t).astype(int)) for t in ths]
fig, ax = plt.subplots(figsize=(4.8, 2.9))
ax.plot(ths, P, "k--", lw=1.1, label="precision"); ax.plot(ths, R, "k:", lw=1.3, label="recall"); ax.plot(ths, F, "k-", lw=1.1, label="$F_1$")
ax.axvline(0.5, color="0.7", lw=0.6); ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.legend(frameon=False, fontsize=8, loc="lower left")
save(fig, "threshold_tradeoff.pdf")
print("best F1 at threshold", ths[int(np.argmax(F))], "F1", round(max(F), 3))

# 3. ROC and precision-recall curves on a 1 % problem ------------------------------------------
X, y1 = make_classification(n_samples=20000, n_features=10, n_informative=4, weights=[0.99, 0.01],
                            class_sep=1.0, flip_y=0.0, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y1, test_size=0.25, stratify=y1, random_state=1)
s1 = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
fpr, tpr = M.roc_curve(yte, s1); prec, rec = M.precision_recall_curve(yte, s1)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax[0].plot(fpr, tpr, "k-", lw=1.2); ax[0].plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.8)
ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate (recall)")
ax[0].set_title(f"ROC, AUC = {M.auc(fpr, tpr):.3f}", fontsize=9)
ax[1].plot(rec, prec, "k-", lw=1.2); ax[1].axhline(yte.mean(), color="0.6", ls="--", lw=0.8)
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_ylim(0, 1)
ax[1].set_title(f"precision-recall, AP = {M.average_precision(yte, s1):.3f}", fontsize=9)
save(fig, "roc_pr.pdf")

# 4. One hold-out split against 10-fold cross-validation ----------------------------------------
Xb, yb = make_classification(n_samples=300, n_features=10, n_informative=4, class_sep=0.8, random_state=3)
single, cvm = [], []
for seed in range(200):
    a, b, c, d = train_test_split(Xb, yb, test_size=0.2, random_state=seed)
    single.append(LogisticRegression(max_iter=1000).fit(a, c).score(b, d))
    cvm.append(M.cross_val_score(lambda A, c_, B: LogisticRegression(max_iter=1000).fit(A, c_).predict(B), Xb, yb, k=10, seed=seed).mean())
fig, ax = plt.subplots(figsize=(4.8, 2.8))
bins = np.linspace(0.6, 0.9, 31)
ax.hist(single, bins=bins, color="0.75", edgecolor="k", lw=0.4, label="one 80/20 split")
ax.hist(cvm, bins=bins, color="0.2", edgecolor="k", lw=0.4, label="10-fold CV mean")
ax.set_xlabel("estimated accuracy"); ax.set_ylabel("count (200 runs)"); ax.legend(frameon=False, fontsize=8)
save(fig, "holdout_vs_cv.pdf")
