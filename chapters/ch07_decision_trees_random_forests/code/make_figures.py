# ci: skip
"""Figures for Chapter 7 (grayscale-safe vector PDFs)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tree import DecisionTree, entropy
from forest import RandomForest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# 1. impurity curves
p = np.linspace(0.0005, 0.9995, 400)
H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(p, H, "k-", label="entropy $E$")
ax.plot(p, 2 * p * (1 - p), "k--", label="Gini $G$ (scaled by 2)")
ax.plot(p, 1 - np.maximum(p, 1 - p), "k:", label="misclassification (scaled by 2)".replace("scaled by 2", "x2")) if False else None
ax.plot(p, 2 * (1 - np.maximum(p, 1 - p)), color="0.55", ls="-.", label="error rate (scaled by 2)")
ax.set_xlabel("proportion of positives $p$"); ax.set_ylabel("impurity"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/impurity_curves.pdf"); plt.close(fig)

# 2. decision boundaries: depth-1, depth-4, full tree, forest
Xtr, ytr = make_moons(300, noise=0.3, random_state=3)
xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 300), np.linspace(-1.3, 1.8, 300))
G = np.c_[xx.ravel(), yy.ravel()]
models = [("depth 1 (stump)", DecisionTree("gini", max_depth=1)), ("depth 4", DecisionTree("gini", max_depth=4)),
          ("fully grown tree", DecisionTree("gini")), ("forest, 100 trees", RandomForest(100, max_features=1, seed=0))]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.1), sharey=True)
for ax, (name, m) in zip(axs, models):
    m.fit(Xtr, ytr)
    ax.contourf(xx, yy, m.predict(G).reshape(xx.shape), levels=[-0.5, 0.5, 1.5], colors=["0.85", "0.6"])
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=np.where(ytr == 1, "k", "w"), edgecolors="k", s=7, linewidths=0.5)
    ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/tree_boundaries.pdf"); plt.close(fig)

# 3. complexity vs error, and forest size vs error
Xa, ya = make_moons(400, noise=0.3, random_state=5)
Xb, yb = make_moons(4000, noise=0.3, random_state=6)
depths = range(1, 16)
tr = [1 - DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xa, ya).score(Xa, ya) for d in depths]
te = [1 - DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xa, ya).score(Xb, yb) for d in depths]
sizes = [1, 2, 5, 10, 25, 50, 100, 200]
oob = [1 - RandomForest(m, seed=0).fit(Xa, ya).oob_score() for m in sizes]
fte = [1 - np.mean(RandomForest(m, seed=0).fit(Xa, ya).predict(Xb) == yb) for m in sizes]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
axs[0].plot(depths, tr, "k--o", ms=3, label="training"); axs[0].plot(depths, te, "k-s", ms=3, label="test")
axs[0].set_xlabel("max depth"); axs[0].set_ylabel("error rate"); axs[0].legend(frameon=False); axs[0].set_title("single tree", fontsize=9)
axs[1].plot(sizes, oob, "k--o", ms=3, label="OOB"); axs[1].plot(sizes, fte, "k-s", ms=3, label="test")
axs[1].set_xscale("log"); axs[1].set_xlabel("number of trees"); axs[1].legend(frameon=False); axs[1].set_title("random forest", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/depth_forest_size.pdf"); plt.close(fig)

# 4. regression tree
rng = np.random.default_rng(0)
xs = np.sort(rng.uniform(0, 6, 80))[:, None]; ys = np.sin(xs[:, 0]) + rng.normal(0, 0.2, 80)
xg = np.linspace(0, 6, 400)[:, None]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.1), sharey=True)
for ax, d in zip(axs, (2, 4, None)):
    t = DecisionTree("mse", max_depth=d).fit(xs, ys)
    ax.scatter(xs, ys, s=7, c="0.5"); ax.plot(xg, np.sin(xg), "k:", lw=1); ax.plot(xg, t.predict(xg), "k-", lw=1.2)
    ax.set_title("depth " + (str(d) if d else "unlimited") + f", {t.n_leaves()} leaves", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/regression_tree.pdf"); plt.close(fig)
print("ok")
