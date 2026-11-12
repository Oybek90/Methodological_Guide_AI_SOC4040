# ci: skip
"""Figures for Chapter 9 (grayscale-safe vector PDFs)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from kernels import rbf_kernel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def xor(n=300, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(-1, 1, (n, 2)); return X, (X[:, 0] * X[:, 1] > 0).astype(int)


DATA = {"XOR": xor(), "Circles": make_circles(300, noise=0.1, factor=0.4, random_state=0), "Moons": make_moons(300, noise=0.2, random_state=0)}


def scatter(ax, X, y):
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=11)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], facecolors="w", edgecolors="k", s=11, linewidths=0.7)


def region(ax, model, X, y, title):
    p = 0.3
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - p, X[:, 0].max() + p, 250), np.linspace(X[:, 1].min() - p, X[:, 1].max() + p, 250))
    Z = model.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-1e9, 0, 1e9], colors=["0.88", "1.0"])
    ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1.3)
    scatter(ax, X, y); ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])


def fitted(X, y, **kw):
    a, b, c, e = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    m = make_pipeline(StandardScaler(), SVC(**kw)).fit(a, c)
    return m, m.score(b, e)


def dec(m):
    class W:
        def decision_function(self, Z): return m.decision_function(Z)
    return W()


# 1. linear SVM on the three datasets
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, (name, (X, y)) in zip(axes, DATA.items()):
    m, acc = fitted(X, y, kernel="linear")
    region(ax, dec(m), X, y, f"{name}: linear, test {acc:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/linear_fails.pdf"); plt.close(fig)

# 2. polynomial degree on circles
X, y = DATA["Circles"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, d in zip(axes, (1, 2, 5)):
    m, acc = fitted(X, y, kernel="poly", degree=d, coef0=1, gamma=1, C=1)
    region(ax, dec(m), X, y, f"degree {d}: test {acc:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/poly_degree.pdf"); plt.close(fig)

# 3. RBF similarity profile and a decision function made of bumps
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
d = np.linspace(0, 3, 200)
for g, ls in zip((0.1, 0.5, 2, 10), ("-", "--", "-.", ":")):
    a1.plot(d, np.exp(-g * d ** 2), "k", ls=ls, label=fr"$\gamma={g}$")
a1.set_xlabel(r"$\|\mathbf{x}-\mathbf{z}\|$"); a1.set_ylabel(r"$k(\mathbf{x},\mathbf{z})$"); a1.legend(frameon=False, fontsize=7.5)
xs = np.linspace(-4, 5, 400); tot = np.zeros_like(xs)
for c, s in zip((-2.0, -0.5, 1.0, 2.5), (1, -1, 1, -1)):
    b = s * np.exp(-(xs - c) ** 2); tot += b; a2.plot(xs, b, color="0.6", lw=0.8, ls="--")
a2.plot(xs, tot, "k", lw=1.8); a2.axhline(0, color="0.5", lw=0.6)
a2.set_xlabel(r"$x$"); a2.set_ylabel(r"$f(x)=\sum_i \alpha_i y_i k(x_i,x)$")
fig.tight_layout(); fig.savefig(f"{OUT}/rbf_bumps.pdf"); plt.close(fig)

# 4. Gram matrix of moons, sorted by class
X, y = make_moons(60, noise=0.1, random_state=0)
o = np.argsort(y); Xs = StandardScaler().fit_transform(X)[o]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
for ax, g in zip(axes, (0.2, 2.0)):
    im = ax.imshow(rbf_kernel(Xs, Xs, g), cmap="Greys", vmin=0, vmax=1)
    ax.set_title(fr"RBF Gram matrix, $\gamma={g}$", fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(29.5, color="k", lw=0.5); ax.axvline(29.5, color="k", lw=0.5)
fig.colorbar(im, ax=axes, shrink=0.8, label=r"$k(\mathbf{x}_i,\mathbf{x}_j)$")
fig.savefig(f"{OUT}/gram_matrix.pdf", bbox_inches="tight"); plt.close(fig)

# 5. joint (C, gamma) heat map of 5-fold CV accuracy on moons (n=200, noise=0.12)
X, y = make_moons(200, noise=0.12, random_state=2)
Cs, Gs = 10.0 ** np.arange(-2, 4), 10.0 ** np.arange(-3, 3)
S = np.array([[cross_val_score(make_pipeline(StandardScaler(), SVC(C=C, gamma=g)), X, y, cv=5).mean() for g in Gs] for C in Cs])
fig, ax = plt.subplots(figsize=(4.6, 3.2))
im = ax.imshow(S, cmap="Greys", origin="lower", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(Gs))); ax.set_xticklabels([f"$10^{{{int(np.log10(g))}}}$" for g in Gs])
ax.set_yticks(range(len(Cs))); ax.set_yticklabels([f"$10^{{{int(np.log10(c))}}}$" for c in Cs])
for i in range(len(Cs)):
    for j in range(len(Gs)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if S[i, j] > 0.8 else "k")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("$C$"); fig.colorbar(im, label="CV accuracy")
fig.tight_layout(); fig.savefig(f"{OUT}/grid_search.pdf"); plt.close(fig)

# 6. kernel x dataset grid
ks = [("linear", dict(kernel="linear")), ("poly d=2", dict(kernel="poly", degree=2, coef0=1, gamma=1)), ("RBF", dict(kernel="rbf")), ("sigmoid", dict(kernel="sigmoid"))]
fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.3))
for r, (name, (X, y)) in enumerate(DATA.items()):
    for c, (kn, kw) in enumerate(ks):
        m, acc = fitted(X, y, **kw)
        region(axes[r, c], dec(m), X, y, f"{name} | {kn}\ntest {acc:.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/kernel_grid.pdf"); plt.close(fig)

# 7. kernel approximation: accuracy against the number of random features
Xl, yl = make_moons(6000, noise=0.25, random_state=0)
a, b, c, e = train_test_split(Xl, yl, test_size=0.3, stratify=yl, random_state=0)
exact = SVC(gamma=2.0).fit(a, c).score(b, e); lin = LinearSVC(random_state=0).fit(a, c).score(b, e)
Ds = [2, 5, 10, 20, 50, 100, 200]
ny = [make_pipeline(Nystroem(gamma=2.0, n_components=D, random_state=0), LinearSVC(random_state=0)).fit(a, c).score(b, e) for D in Ds]
rf = [make_pipeline(RBFSampler(gamma=2.0, n_components=D, random_state=0), LinearSVC(random_state=0)).fit(a, c).score(b, e) for D in Ds]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(Ds, ny, "k-o", ms=3, label="Nystroem"); ax.plot(Ds, rf, "k--s", ms=3, label="random Fourier")
ax.axhline(exact, color="0.4", ls=":", label=f"exact RBF SVC ({exact:.3f})"); ax.axhline(lin, color="0.7", ls="-", label=f"LinearSVC ({lin:.3f})")
ax.set_xscale("log"); ax.set_xlabel("number of features $D$"); ax.set_ylabel("test accuracy"); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/approx_features.pdf"); plt.close(fig)
print("done", {"exact": exact, "lin": lin, "ny": ny, "rf": rf}, "\nbest grid", S.max(), np.unravel_index(S.argmax(), S.shape))
