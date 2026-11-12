# ci: skip
"""Figures for Chapter 10 (grayscale-safe vector PDFs)."""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from prep_utils import Robust, Standard, MinMax, iqr_bounds, smote

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
T = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic.csv"))


def save(name):
    plt.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close()


# 1. pipeline: what is learned from the training part only
def box(ax, x, y, w, h, text, fc="w", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
box(ax, 0.1, 1.6, 1.4, 1.0, "raw\ndata", "0.92")
box(ax, 2.2, 2.9, 1.5, 0.9, "train\n(fit here)", "0.80")
box(ax, 2.2, 0.4, 1.5, 0.9, "test\n(locked)", "1.0")
ax.annotate("", (2.15, 3.3), (1.55, 2.4), arrowprops=dict(arrowstyle="->"))
ax.annotate("", (2.15, 0.85), (1.55, 1.8), arrowprops=dict(arrowstyle="->"))
box(ax, 4.5, 2.9, 3.2, 0.9, "impute, encode, scale, select\nstatistics learned: fit()", "0.80")
box(ax, 4.5, 0.4, 3.2, 0.9, "same operations, same numbers\ntransform() only", "1.0")
ax.annotate("", (4.45, 3.35), (3.8, 3.35), arrowprops=dict(arrowstyle="->"))
ax.annotate("", (4.45, 0.85), (3.8, 0.85), arrowprops=dict(arrowstyle="->"))
ax.annotate("", (4.45, 0.95), (6.1, 2.85), arrowprops=dict(arrowstyle="->", ls="--"))
ax.text(6.3, 1.95, "stored means, medians,\nfences, categories", fontsize=7.5, ha="left", va="center")
box(ax, 8.3, 2.9, 1.5, 0.9, "model\nfit", "0.80")
box(ax, 8.3, 0.4, 1.5, 0.9, "score\n(once)", "1.0")
ax.annotate("", (8.25, 3.35), (7.8, 3.35), arrowprops=dict(arrowstyle="->"))
ax.annotate("", (8.25, 0.85), (7.8, 0.85), arrowprops=dict(arrowstyle="->"))
ax.annotate("", (9.05, 1.35), (9.05, 2.85), arrowprops=dict(arrowstyle="->"))
save("pipeline_fit_transform")

# 2. missing values: three imputers on Titanic age
age = T["age"]; obs = age.dropna()
Xk = KNNImputer(n_neighbors=5).fit_transform(T[["age", "pclass", "sibsp", "parch", "fare"]])[:, 0]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for ax, (v, t) in zip(axs, [(age.fillna(age.mean()), "mean"), (age.fillna(age.median()), "median"), (pd.Series(Xk), "KNN (k=5)")]):
    ax.hist(obs, bins=np.arange(0, 85, 5), color="0.85", ec="0.5", label="observed")
    ax.hist(v[age.isna()], bins=np.arange(0, 85, 5), color="k", alpha=0.75, label="imputed")
    ax.set_title(f"{t}: sd {v.std():.1f}"); ax.set_xlabel("age")
axs[0].set_ylabel("passengers"); axs[0].legend(frameon=False, fontsize=7.5)
plt.tight_layout(); save("imputation_age")

# 3. outliers: IQR fences and z-score on a heavy-tailed variable
fare = T["fare"]; lo, hi = iqr_bounds(fare); z = (fare - fare.mean()) / fare.std(ddof=0)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.hist(fare, bins=np.arange(0, 520, 10), color="0.8", ec="0.5")
ax.axvline(hi, color="k", ls="--"); ax.axvline(fare.mean() + 3 * fare.std(ddof=0), color="k", ls=":")
ax.text(hi + 4, 150, f"IQR fence {hi:.0f}\n{(fare > hi).sum()} points above", fontsize=8)
ax.text(fare.mean() + 3 * fare.std(ddof=0) + 4, 60, f"mean + 3 sd = {fare.mean() + 3 * fare.std(ddof=0):.0f}\n{(z > 3).sum()} points above", fontsize=8)
ax.set_xlabel("fare"); ax.set_ylabel("passengers"); ax.set_yscale("symlog", linthresh=10)
save("outlier_rules")

# 4. scalers on a feature with one outlier
r = np.random.default_rng(0); x = np.r_[r.normal(50, 10, 99), 400.0]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
for ax, (v, t) in zip(axs, [(x, "raw"), (MinMax().fit(x[:, None]).transform(x[:, None]).ravel(), "Min-Max"),
                            (Standard().fit(x[:, None]).transform(x[:, None]).ravel(), "Z-score"), (Robust().fit(x[:, None]).transform(x[:, None]).ravel(), "Robust")]):
    ax.hist(v[:-1], bins=15, color="0.6", ec="k", lw=0.4); ax.axvline(v[-1], color="k", lw=1.5)
    ax.set_title(f"{t}\nbulk width {np.ptp(v[:-1]):.2f}", fontsize=8.5); ax.set_yticks([])
plt.tight_layout(); save("scalers_outlier")

# 5. SMOTE
X, y = make_classification(300, 2, n_redundant=0, n_informative=2, n_clusters_per_class=1, weights=[0.9], class_sep=1.2, random_state=3)
Xa, ya = smote(X, y, seed=0); new = Xa[len(X):]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True, sharey=True)
for ax, t in zip(axs, ["original (10 % minority)", "after SMOTE (train only)"]):
    ax.scatter(*X[y == 0].T, facecolors="none", edgecolors="0.5", s=10, lw=0.6); ax.scatter(*X[y == 1].T, c="k", s=14); ax.set_title(t)
axs[1].scatter(*new.T, marker="x", c="k", s=14, lw=0.8)
plt.tight_layout(); save("smote_demo")

# 6. leakage: selection outside vs inside cross-validation, as the number of candidate features grows
rng = np.random.default_rng(0); cv = StratifiedKFold(5, shuffle=True, random_state=0); ps = [20, 100, 500, 2000]
o, i = [], []
for p in ps:
    Xn = rng.normal(size=(100, p)); yn = rng.integers(0, 2, 100)
    o.append(cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(f_classif, k=min(20, p)).fit_transform(Xn, yn), yn, cv=cv).mean())
    i.append(cross_val_score(make_pipeline(SelectKBest(f_classif, k=min(20, p)), LogisticRegression(max_iter=1000)), Xn, yn, cv=cv).mean())
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(ps, o, "k-o", label="selection before CV (leak)"); ax.plot(ps, i, "k--s", mfc="w", label="selection inside pipeline")
ax.axhline(0.5, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("number of noise features p (n = 100)"); ax.set_ylabel("CV accuracy")
ax.legend(frameon=False, fontsize=8); save("leakage_selection")

# 7. Titanic stage-by-stage results (values printed by titanic_pipeline.py)
st = ["0 raw", "1 KNN imp.", "2 one-hot", "3 scaling", "4 IQR clip", "5 features", "6 select 8", "7 SMOTE"]
rf = [0.717, 0.721, 0.829, 0.829, 0.824, 0.824, 0.832, 0.825]; rfs = [0.036, 0.039, 0.026, 0.025, 0.026, 0.026, 0.030, 0.030]
lr = [0.700, 0.698, 0.803, 0.804, 0.797, 0.802, 0.795, 0.780]; lrs = [0.025, 0.023, 0.028, 0.028, 0.030, 0.028, 0.027, 0.031]
fig, ax = plt.subplots(figsize=(6.4, 2.8)); xx = np.arange(8)
ax.errorbar(xx - 0.05, rf, rfs, fmt="k-o", capsize=2, label="random forest"); ax.errorbar(xx + 0.05, lr, lrs, fmt="k--s", mfc="w", capsize=2, label="logistic regression")
ax.set_xticks(xx); ax.set_xticklabels(st, rotation=25, ha="right"); ax.set_ylabel("accuracy (5x4 CV, mean +- sd)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
save("titanic_stages")
