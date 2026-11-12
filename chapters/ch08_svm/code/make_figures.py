# ci: skip
"""Figures for Chapter 8 (grayscale-safe vector PDFs)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def scatter(ax, X, y):
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=14, marker="o")
    ax.scatter(X[y != 1, 0], X[y != 1, 1], facecolors="w", edgecolors="k", s=14, marker="o", linewidths=0.8)


def lines(ax, m, X, lo=None, hi=None):
    lo = lo or X[:, 0].min() - 0.5
    hi = hi or X[:, 0].max() + 0.5
    x = np.array([lo, hi]); w, b = m.coef_[0], m.intercept_[0]
    for off, ls, lw in ((0, "-", 1.4), (1, "--", 0.9), (-1, "--", 0.9)):
        ax.plot(x, (off - b - w[0] * x) / w[1], "k", ls=ls, lw=lw)
    ax.scatter(*m.support_vectors_.T, s=60, facecolors="none", edgecolors="0.35", linewidths=1.0)


# 1. margin anatomy (separable data)
X, y = make_classification(60, 2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=2.5, random_state=7)
m = SVC(kernel="linear", C=1e3).fit(X, y)
fig, ax = plt.subplots(figsize=(4.4, 3.4))
scatter(ax, X, y); lines(ax, m, X)
ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5); ax.set_ylim(X[:, 1].min() - 0.8, X[:, 1].max() + 0.8)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("solid: $\\mathbf{w}^\\top\\mathbf{x}+b=0$; dashed: $\\pm1$; circled: support vectors", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/margin_anatomy.pdf"); plt.close(fig)

# 2. effect of C
Xb, yb = make_classification(200, 2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=1.0, random_state=3)
Xb = StandardScaler().fit_transform(Xb)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), sharex=True, sharey=True)
for ax, C in zip(axs, (0.01, 1, 1000)):
    mm = SVC(kernel="linear", C=C).fit(Xb, yb)
    scatter(ax, Xb, yb); lines(ax, mm, Xb)
    ax.set_title(f"$C={C:g}$: width {2 / np.linalg.norm(mm.coef_[0]):.2f}, {mm.n_support_.sum()} SVs", fontsize=8)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/c_effect.pdf"); plt.close(fig)

# 3. lifting: 1-D data that no threshold separates become separable after x -> (x, x^2)
rng = np.random.default_rng(0)
x1 = np.r_[rng.uniform(-3, -1.6, 12), rng.uniform(1.6, 3, 12)]; x0 = rng.uniform(-1, 1, 20)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
axs[0].scatter(x1, np.zeros_like(x1), c="k", s=16); axs[0].scatter(x0, np.zeros_like(x0), facecolors="w", edgecolors="k", s=16)
axs[0].set_yticks([]); axs[0].set_xlabel("$x$"); axs[0].set_title("input space: no single threshold works", fontsize=8)
axs[1].scatter(x1, x1 ** 2, c="k", s=16); axs[1].scatter(x0, x0 ** 2, facecolors="w", edgecolors="k", s=16)
axs[1].axhline(2.0, color="k", lw=1.2); axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$x^2$")
axs[1].set_title("feature space $(x,x^2)$: the line $x^2=2$ separates", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/lifting.pdf"); plt.close(fig)

# 4. kernels and gamma on moons
Xm, ym = make_moons(300, noise=0.25, random_state=1)
xx, yy = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.5, 2, 250)); G = np.c_[xx.ravel(), yy.ravel()]


def panel(ax, model, title):
    model.fit(Xm, ym)
    ax.contourf(xx, yy, model.predict(G).reshape(xx.shape), levels=[-0.5, 0.5, 1.5], colors=["0.88", "0.62"])
    scatter(ax, Xm, ym); ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
panel(axs[0], SVC(kernel="linear", C=1), "linear")
panel(axs[1], SVC(kernel="poly", degree=3, coef0=1, C=1), "polynomial, degree 3")
panel(axs[2], SVC(kernel="rbf", gamma=1, C=1), "RBF, $\\gamma=1$")
fig.tight_layout(); fig.savefig(f"{OUT}/kernel_boundaries.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
for ax, (g, C) in zip(axs, ((0.01, 1), (1, 1), (100, 1000))):
    panel(ax, SVC(kernel="rbf", gamma=g, C=C), f"RBF, $\\gamma={g:g}$, $C={C:g}$")
fig.tight_layout(); fig.savefig(f"{OUT}/gamma_effect.pdf"); plt.close(fig)

# 5. losses
z = np.linspace(-2, 3, 400)
fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(z, (z < 0).astype(float), color="0.55", ls="-.", label="0--1 loss")
ax.plot(z, np.maximum(0, 1 - z), "k-", label="hinge $\\max(0,1-z)$")
ax.plot(z, np.log2(1 + np.exp(-z)), "k--", label="logistic $\\log_2(1+e^{-z})$")
ax.set_ylim(0, 3); ax.set_xlabel("$z=y\\,f(\\mathbf{x})$"); ax.set_ylabel("loss"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/losses.pdf"); plt.close(fig)

# 6. SVR tube
rng = np.random.default_rng(0)
xs = np.sort(rng.uniform(0, 6, 80))[:, None]; ys = np.sin(xs[:, 0]) + rng.normal(0, 0.2, 80)
xg = np.linspace(0, 6, 300)[:, None]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
for ax, eps in zip(axs, (0.01, 0.3, 0.6)):
    r = SVR(C=10, gamma=1.0, epsilon=eps).fit(xs, ys); p = r.predict(xg)
    ax.fill_between(xg[:, 0], p - eps, p + eps, color="0.85")
    ax.scatter(xs, ys, s=8, c="0.5"); ax.scatter(xs[r.support_], ys[r.support_], s=26, facecolors="none", edgecolors="k", linewidths=0.7)
    ax.plot(xg, p, "k-", lw=1.2)
    ax.set_title(f"$\\varepsilon={eps}$: {len(r.support_)} support vectors", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/svr_tube.pdf"); plt.close(fig)
print("ok")
