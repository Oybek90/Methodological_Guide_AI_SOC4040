# ci: skip
"""Generate the figures of Chapter 6 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification, make_moons  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from knn import IRIS12, IRIS12_LABELS, QUERY, minkowski  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
MARK = {"Setosa": ("o", "white"), "Versicolor": ("s", "0.6"), "Virginica": ("^", "black")}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. Unit balls of the Minkowski distances -------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 3.4))
t = np.linspace(0, 2 * np.pi, 400)
for p, ls, lab in ((1, "-", r"$p=1$ (Manhattan)"), (2, "--", r"$p=2$ (Euclidean)"), (np.inf, ":", r"$p=\infty$ (Chebyshev)")):
    pts = np.c_[np.cos(t), np.sin(t)]
    r = np.array([1.0 / minkowski(pt, [0, 0], p) for pt in pts])
    ax.plot(r * pts[:, 0], r * pts[:, 1], ls, color="k", lw=1.2, label=lab)
ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "distance_balls.pdf")

# 2. The twelve flowers and the query -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 3.4))
for name, (mk, fc) in MARK.items():
    m = IRIS12_LABELS == name
    ax.scatter(IRIS12[m, 0], IRIS12[m, 1], marker=mk, c=fc, edgecolors="k", s=36, label=name, zorder=3)
ax.scatter(*QUERY, marker="*", s=150, c="k", zorder=4, label="query")
d3 = sorted(minkowski(QUERY, x) for x in IRIS12)[2]
ax.add_patch(plt.Circle(QUERY, d3, fill=False, ls="--", lw=0.9, color="k"))
ax.text(QUERY[0] + 0.03, QUERY[1] - d3 - 0.13, "$k=3$: radius 0.608", fontsize=7)
ax.set_aspect("equal"); ax.set_xlim(4.5, 7.8); ax.set_ylim(2.0, 4.2)
ax.set_xlabel("sepal length"); ax.set_ylabel("sepal width"); ax.legend(frameon=False, fontsize=7, loc="upper right")
save(fig, "knn_example.pdf")

# 3. Decision boundaries for three values of k ----------------------------------------------------------
Xm, ym = make_moons(1000, noise=0.35, random_state=1)
Xa, Xb, ya, yb = train_test_split(Xm, ym, test_size=0.3, random_state=1, stratify=ym)
gx, gy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2.1, 300))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
for ax, k in zip(axs, (1, 15, 300)):
    m = KNeighborsClassifier(k).fit(Xa, ya)
    z = m.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=["0.92", "0.7"])
    ax.contour(gx, gy, z, levels=[0.5], colors="k", linewidths=0.8)
    s = slice(0, 150)
    ax.scatter(Xa[s][ya[s] == 0, 0], Xa[s][ya[s] == 0, 1], marker="o", c="white", edgecolors="k", s=9, lw=0.5)
    ax.scatter(Xa[s][ya[s] == 1, 0], Xa[s][ya[s] == 1, 1], marker="^", c="k", s=9, lw=0.5)
    ax.set_title(f"$k={k}$: train err {1 - m.score(Xa, ya):.2f}, test err {1 - m.score(Xb, yb):.2f}", fontsize=8)
    ax.set_xlabel("$x_1$")
axs[0].set_ylabel("$x_2$")
save(fig, "knn_boundaries.pdf")

# 4. Error against k ---------------------------------------------------------------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
ks = list(range(1, 62, 2))
tr, cvs, te = [], [], []
for k in ks:
    m = KNeighborsClassifier(k).fit(Xa, ya)
    tr.append(1 - m.score(Xa, ya)); te.append(1 - m.score(Xb, yb))
    cvs.append(1 - cross_val_score(KNeighborsClassifier(k), Xa, ya, cv=cv).mean())
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(ks, tr, "-", color="0.55", marker="o", ms=3, label="training error")
ax.plot(ks, cvs, "-", color="k", marker="s", ms=3, label="5-fold CV error")
ax.plot(ks, te, "--", color="k", marker="^", ms=3, label="test error")
kb = ks[int(np.argmin(cvs))]
ax.axvline(kb, color="0.5", lw=0.7, ls=":"); ax.text(kb + 1, 0.02, f"CV minimum $k={kb}$", fontsize=7)
ax.set_xlabel("number of neighbours $k$ (small $k$: flexible, large $k$: smooth)"); ax.set_ylabel("error rate")
ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3)
ax.set_ylim(0, 0.35)
save(fig, "k_selection.pdf")

# 5. Curse of dimensionality ---------------------------------------------------------------------------------
rng = np.random.default_rng(0)
dims = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
rc = []
for d in dims:
    P = rng.random((1000, d)); q = rng.random(d)
    dist = np.sqrt(((P - q) ** 2).sum(1)); rc.append((dist.max() - dist.min()) / dist.min())
extras = [0, 5, 10, 20, 50, 100, 200]
accK, accN = [], []
for e in extras:
    Xn = np.hstack([Xm, rng.standard_normal((1000, e))]) if e else Xm
    accK.append(cross_val_score(KNeighborsClassifier(15), Xn, ym, cv=cv).mean())
    accN.append(cross_val_score(GaussianNB(), Xn, ym, cv=cv).mean())
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
ax[0].loglog(dims, rc, "-o", color="k", ms=3)
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel(r"$(D_{\max}-D_{\min})/D_{\min}$"); ax[0].set_title("distances become alike", fontsize=8)
ax[1].plot(extras, accK, "-s", color="k", ms=3, label="15-NN")
ax[1].plot(extras, accN, "--^", color="0.4", ms=3, label="Gaussian NB")
ax[1].set_xlabel("number of added noise features"); ax[1].set_ylabel("5-fold CV accuracy"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("two moons plus noise", fontsize=8)
save(fig, "curse.pdf")

# 6. Gaussian Naive Bayes in one dimension -------------------------------------------------------------------
x = np.linspace(-4, 8, 400)
pdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
pri0, pri1 = 0.6, 0.4
j0, j1 = pri0 * pdf(x, 1.0, 1.2), pri1 * pdf(x, 4.0, 1.8)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax[0].plot(x, j0, "-", color="k", label=r"$P(y{=}0)\,p(x\mid y{=}0)$")
ax[0].plot(x, j1, "--", color="0.4", label=r"$P(y{=}1)\,p(x\mid y{=}1)$")
ax[0].set_xlabel("$x$"); ax[0].legend(frameon=False, fontsize=7)
post = j1 / (j0 + j1)
ax[1].plot(x, post, "-", color="k"); ax[1].axhline(0.5, color="0.6", lw=0.6, ls=":")
xb = x[np.argmin(np.abs(post - 0.5)[100:300]) + 100]
ax[1].axvline(xb, color="0.6", lw=0.6, ls=":"); ax[1].text(xb + 0.15, 0.1, f"$x\\approx{xb:.2f}$", fontsize=7)
ax[1].set_xlabel("$x$"); ax[1].set_ylabel(r"$P(y{=}1\mid x)$")
save(fig, "nb_gaussian.pdf")
print("1D boundary at x =", round(float(xb), 3))

# 7. Repeated feature: confidence against accuracy ---------------------------------------------------------
Xc, yc = make_classification(3000, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8,
                             random_state=3, n_clusters_per_class=1, flip_y=0.05)
Xc1, Xc2, yc1, yc2 = train_test_split(Xc, yc, test_size=0.5, random_state=0)
cop = [1, 2, 3, 5, 10]
acc, conf, ll = [], [], []
for c in cop:
    f = lambda A: np.hstack([A[:, [0]]] * c + [A[:, [1]]])
    m = GaussianNB().fit(f(Xc1), yc1); p = m.predict_proba(f(Xc2))
    acc.append(m.score(f(Xc2), yc2)); conf.append(p.max(1).mean()); ll.append(log_loss(yc2, np.clip(p, 1e-12, 1)))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
ax[0].plot(cop, acc, "-o", color="k", ms=3, label="accuracy")
ax[0].plot(cop, conf, "--s", color="0.4", ms=3, label="mean confidence")
ax[0].set_xlabel("copies of feature $x_1$"); ax[0].legend(frameon=False, fontsize=7)
ax[1].plot(cop, ll, "-^", color="k", ms=3); ax[1].set_xlabel("copies of feature $x_1$"); ax[1].set_ylabel("log-loss")
save(fig, "nb_dependence.pdf")
print("copies", cop, "acc", np.round(acc, 3), "conf", np.round(conf, 3), "ll", np.round(ll, 3))
