# ci: run
"""Experiments for Chapter 7: from-scratch trees/forest vs scikit-learn, overfitting, bagging, OOB."""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from forest import RandomForest
from tree import DecisionTree

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("== 1. from scratch vs scikit-learn (breast cancer, stratified 80/20) ==")
for crit in ("gini", "entropy"):
    mine = DecisionTree(crit).fit(Xtr, ytr)
    ref = DecisionTreeClassifier(criterion=crit, random_state=0).fit(Xtr, ytr)
    print(f"{crit:8s} mine: depth {mine.depth():2d} leaves {mine.n_leaves():2d} acc {np.mean(mine.predict(Xte) == yte):.4f} | "
          f"sklearn: depth {ref.get_depth():2d} leaves {ref.get_n_leaves():2d} acc {ref.score(Xte, yte):.4f} | "
          f"predictions agree {np.mean(mine.predict(Xte) == ref.predict(Xte)):.3f}")

print("== 2. complexity vs error (single tree, gini) ==")
for d in (1, 2, 3, 4, 6, 8, None):
    t = DecisionTree("gini", max_depth=d).fit(Xtr, ytr)
    print(f"max_depth {str(d):4s} leaves {t.n_leaves():3d} train {np.mean(t.predict(Xtr) == ytr):.3f} test {np.mean(t.predict(Xte) == yte):.3f}")

print("== 3. one tree vs forest, same criterion (gini), 5 random splits ==")
rows = []
for s in range(5):
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
    tr = DecisionTreeClassifier(random_state=s).fit(a, c).score(b, d)
    rf = RandomForestClassifier(100, random_state=s).fit(a, c).score(b, d)
    mine = RandomForest(25, seed=s).fit(a, c)
    rows.append((tr, rf, np.mean(mine.predict(b) == d), mine.oob_score()))
r = np.array(rows)
print("tree %.3f+-%.3f | sklearn forest(100) %.3f+-%.3f | my forest(25) %.3f+-%.3f | my OOB estimate %.3f" %
      (r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std(), r[:, 2].mean(), r[:, 2].std(), r[:, 3].mean()))

print("== 4. forest size vs OOB and test accuracy (my forest) ==")
for m in (1, 5, 25, 100):
    f = RandomForest(m, seed=1).fit(Xtr, ytr)
    print(f"trees {m:3d} OOB {f.oob_score():.3f} test {np.mean(f.predict(Xte) == yte):.3f}")

print("== 5. variance of trees vs forest on moons (noise 0.3), 20 training sets ==")
Xt, yt = make_moons(2000, noise=0.3, random_state=0)
a, b = [], []
for s in range(20):
    Xs, ys = make_moons(200, noise=0.3, random_state=100 + s)
    a.append(DecisionTreeClassifier(random_state=s).fit(Xs, ys).score(Xt, yt))
    b.append(RandomForestClassifier(100, random_state=s).fit(Xs, ys).score(Xt, yt))
print("tree  acc %.3f +- %.3f\nforest acc %.3f +- %.3f" % (np.mean(a), np.std(a), np.mean(b), np.std(b)))

print("== 6. top impurity-based importances (my forest) ==")
f = RandomForest(50, seed=1).fit(Xtr, ytr)
names = load_breast_cancer().feature_names
for i in np.argsort(-f.feature_importances())[:5]:
    print(f"{names[i]:25s} {f.feature_importances()[i]:.3f}")

print("== 7. regression tree on y = sin(x) + noise ==")
rng = np.random.default_rng(0)
xs = np.sort(rng.uniform(0, 6, 80))[:, None]
ys = np.sin(xs[:, 0]) + rng.normal(0, 0.2, 80)
xg = np.linspace(0, 6, 300)[:, None]
for d in (2, 4, None):
    t = DecisionTree("mse", max_depth=d).fit(xs, ys)
    print(f"depth {str(d):4s} leaves {t.n_leaves():3d} MSE vs true sin: {np.mean((t.predict(xg) - np.sin(xg[:, 0])) ** 2):.3f}")
