# ci: skip
"""Generate the figures of Chapter 4 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification, make_moons  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

from logreg import (X_LEC, Y_LEC, add_bias, cost, fit, fit_one_vs_rest, fit_softmax,  # noqa: E402
                    predict_one_vs_rest, sigmoid)

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. The sigmoid and the two branches of the cross-entropy loss -------------------------------
z = np.linspace(-8, 8, 400)
h = np.linspace(0.001, 0.999, 400)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax[0].plot(z, sigmoid(z), "k-", lw=1.2)
ax[0].axhline(0.5, color="0.6", lw=0.6, ls=":"); ax[0].axvline(0, color="0.6", lw=0.6, ls=":")
ax[0].set_xlabel(r"$z=\boldsymbol{\theta}^{\top}\mathbf{x}$"); ax[0].set_ylabel(r"$g(z)$")
ax[1].plot(h, -np.log(h), "k-", lw=1.2, label=r"$y=1$:  $-\log h$")
ax[1].plot(h, -np.log(1 - h), "k--", lw=1.2, label=r"$y=0$:  $-\log(1-h)$")
ax[1].set_ylim(0, 5); ax[1].set_xlabel(r"prediction $h_\theta(\mathbf{x})$"); ax[1].set_ylabel("loss")
ax[1].legend(frameon=False, fontsize=8, loc="upper center")
save(fig, "sigmoid_loss.pdf")

# 2. Why not a line? Linear regression with a threshold vs logistic regression ----------------
rng = np.random.default_rng(3)
xs = np.concatenate([rng.uniform(0.5, 4.5, 14), rng.uniform(5.5, 9.5, 14)])
ys = np.concatenate([np.zeros(14), np.ones(14)])
x_out, y_out = 40.0, 1.0                       # one very large tumour, clearly malignant
grid = np.linspace(0, 42, 400)


def line_fit(x, y):
    return np.polyfit(x, y, 1)


def cross(coef):                               # x where the fitted line crosses 0.5
    return (0.5 - coef[1]) / coef[0]


c1 = line_fit(xs, ys)
c2 = line_fit(np.append(xs, x_out), np.append(ys, y_out))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for a, ttl in zip(ax, ["linear regression + threshold", "logistic regression"]):
    a.scatter(xs, ys, c="white", edgecolors="k", s=18, zorder=3)
    a.scatter([x_out], [y_out], c="k", s=22, zorder=3)
    a.axhline(0.5, color="0.7", lw=0.6, ls=":")
    a.set_xlabel("tumour size"); a.set_title(ttl, fontsize=9)
ax[0].set_ylabel("malignant (1) / benign (0)")
ax[0].plot(grid, np.polyval(c1, grid), color="0.6", lw=1.2, label="without the large tumour")
ax[0].plot(grid, np.polyval(c2, grid), "k-", lw=1.2, label="with the large tumour")
ax[0].axvline(cross(c1), color="0.6", lw=0.8, ls="--"); ax[0].axvline(cross(c2), color="k", lw=0.8, ls="--")
ax[0].set_ylim(-0.3, 1.5); ax[0].legend(frameon=False, fontsize=7, loc="upper left")
for X_, Y_, col in [(xs, ys, "0.6"), (np.append(xs, x_out), np.append(ys, y_out), "k")]:
    t, _ = fit(add_bias(((X_ - 5) / 3).reshape(-1, 1)), Y_.reshape(-1, 1), lr=0.5, iters=3000)
    ax[1].plot(grid, sigmoid(t[0, 0] + t[1, 0] * (grid - 5) / 3), color=col, lw=1.2)
    print("logistic boundary at size", round(5 - 3 * t[0, 0] / t[1, 0], 2))
save(fig, "linear_vs_logistic.pdf")
print("line threshold without / with outlier:", round(cross(c1), 2), round(cross(c2), 2))

# 3. Decision boundary: the slide example theta = (-3, 1, 1) and a fitted model -----------------
rng = np.random.default_rng(1)
n = 40
P0 = rng.normal([1.2, 1.2], 0.6, (n, 2)); P1 = rng.normal([3.2, 3.0], 0.6, (n, 2))
Xb = np.vstack([P0, P1]); yb = np.r_[np.zeros(n), np.ones(n)].reshape(-1, 1)
t, _ = fit(add_bias(Xb), yb, lr=0.3, iters=20000)
gx = np.linspace(-0.5, 5, 200); gy = np.linspace(-0.5, 5, 200)
GX, GY = np.meshgrid(gx, gy)
Pg = sigmoid(t[0, 0] + t[1, 0] * GX + t[2, 0] * GY)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
ax[0].plot([-0.5, 3.5], [3.5, -0.5], "k-", lw=1.2); ax[0].fill_between(np.linspace(-0.5, 5, 50), 3 - np.linspace(-0.5, 5, 50), 5, color="0.9")
ax[0].scatter([2, 1], [2.3, 0.5], c="k", s=0)  # keep limits
ax[0].text(2.4, 3.6, r"$y=1$: $-3+x_1+x_2\geq0$", fontsize=8)
ax[0].text(0.15, 0.25, r"$y=0$", fontsize=8)
ax[0].set_title(r"slide example $\boldsymbol{\theta}=(-3,1,1)$", fontsize=9)
ax[1].contourf(GX, GY, Pg, levels=[0, 0.1, 0.3, 0.7, 0.9, 1], cmap="Greys", alpha=0.55)
ax[1].contour(GX, GY, Pg, levels=[0.5], colors="k", linewidths=1.4)
ax[1].set_title("fitted model: lines of constant probability", fontsize=9)
for a in ax:
    a.set_xlim(-0.5, 5); a.set_ylim(-0.5, 5); a.set_xlabel(r"$x_1$")
ax[1].scatter(*P0.T, c="white", edgecolors="k", s=16, zorder=3)
ax[1].scatter(*P1.T, c="k", marker="^", s=16, zorder=3)
ax[0].set_ylabel(r"$x_2$")
save(fig, "boundary.pdf")

# 4. Cost vs iteration for several learning rates; cross-entropy vs squared error ------------
Xn, yn = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
from sklearn.model_selection import train_test_split  # noqa: E402

Xtr, _, ytr, _ = train_test_split(Xn, yn, test_size=0.2, random_state=4)
A = add_bias(Xtr); yv = ytr.reshape(-1, 1)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lr, ls in [(0.01, ":"), (0.1, "-"), (1.0, "--")]:
    _, hh = fit(A, yv, lr=lr, iters=1000)
    ax[0].plot(hh, "k", ls=ls, lw=1.1, label=rf"$\alpha={lr}$")
    print("lr", lr, "J after 1000:", round(hh[-1], 4))
ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"cost $J$"); ax[0].legend(frameon=False, fontsize=8)
# 1-D example, one parameter: cross-entropy is convex, squared error of the sigmoid is not.
x1 = np.array([-2, -1, 1, 2, 3.0]); y1 = np.array([1, 0, 1, 0, 0.0])   # a hard, partly mislabelled set
th = np.linspace(-6, 6, 400)
Z = th[:, None] * x1[None, :]
H = sigmoid(Z)
ce = np.mean(np.logaddexp(0, Z) - y1 * Z, axis=1)
se = 0.5 * np.mean((H - y1) ** 2, axis=1)
ax[1].plot(th, ce, "k-", lw=1.2, label="cross-entropy")
ax[1].plot(th, se * (ce.max() / se.max()) * 0.5, "k--", lw=1.2, label="squared error (rescaled)")
ax[1].set_xlabel(r"$\theta_1$"); ax[1].set_ylabel("cost"); ax[1].legend(frameon=False, fontsize=8)
d2 = np.diff(se, 2)
print("squared error convex? min 2nd difference", d2.min(), " cross-entropy:", np.diff(ce, 2).min())
save(fig, "cost_curves.pdf")

# 5. Separable data: unregularised weights grow, L2 keeps them finite --------------------------
iters = 20000
_, h0 = fit(X_LEC, Y_LEC, lr=0.1, iters=iters, lam=0.0)
norms = {}
for lam in [0.0, 0.1, 1.0]:
    th_, _ = fit(X_LEC, Y_LEC, lr=0.1, iters=iters, lam=lam)
    norms[lam] = th_.ravel()
    print("lambda", lam, "theta =", np.round(th_.ravel(), 3), " J =", round(cost(X_LEC, Y_LEC, th_, lam), 4))
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for lam, ls in [(0.0, "-"), (0.1, "--"), (1.0, ":")]:
    path = []
    t_ = np.zeros((3, 1))
    from logreg import gradient  # noqa: E402
    for k in range(iters):
        t_ = t_ - 0.1 * gradient(X_LEC, Y_LEC, t_, lam)
        if k % 50 == 0:
            path.append(np.linalg.norm(t_[1:]))
    ax.plot(np.arange(len(path)) * 50, path, "k", ls=ls, lw=1.2, label=rf"$\lambda={lam:g}$")
ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|(\theta_1,\theta_2)\|$"); ax.legend(frameon=False, fontsize=8)
save(fig, "separable_weights.pdf")

# 6. Under- and overfitting of the decision boundary, and the effect of regularisation -------
Xm, ym = make_moons(n_samples=60, noise=0.35, random_state=2)
Xt_, yt_ = make_moons(n_samples=2000, noise=0.35, random_state=7)   # large test set
gx = np.linspace(-2, 3, 250); gy = np.linspace(-1.6, 2.2, 250)
GX, GY = np.meshgrid(gx, gy)
grid2 = np.c_[GX.ravel(), GY.ravel()]
configs = [(1, 1e6, "degree 1 (underfit)"), (9, 1e6, r"degree 9, $\lambda\approx0$ (overfit)"),
           (9, 1.0 / 0.1, r"degree 9, $\lambda=0.1$")]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, (deg, C, ttl) in zip(ax, configs):
    pipe = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(),
                         LogisticRegression(C=C, max_iter=20000)).fit(Xm, ym)
    a.contourf(GX, GY, pipe.predict(grid2).reshape(GX.shape), levels=[-0.5, 0.5, 1.5], colors=["0.92", "0.7"])
    a.scatter(*Xm[ym == 0].T, c="white", edgecolors="k", s=14, zorder=3)
    a.scatter(*Xm[ym == 1].T, c="k", marker="^", s=14, zorder=3)
    a.set_title(ttl, fontsize=8); a.set_xlabel(r"$x_1$")
    print(ttl, "train acc", round(pipe.score(Xm, ym), 3), "test acc", round(pipe.score(Xt_, yt_), 3),
          "max|w|", round(np.abs(pipe[-1].coef_).max(), 1))
ax[0].set_ylabel(r"$x_2$")
save(fig, "overfit_boundary.pdf")

# 7. Multi-class: the three-class data set, one-vs-rest and softmax regions --------------------
X3, y3 = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                             n_classes=3, n_clusters_per_class=1, random_state=42)
B = add_bias(X3)
thetas = fit_one_vs_rest(B, y3, 3, lr=0.1, iters=1000)
Theta = fit_softmax(B, y3, 3, lr=0.1, iters=1000)
gx = np.linspace(X3[:, 0].min() - 0.5, X3[:, 0].max() + 0.5, 250)
gy = np.linspace(X3[:, 1].min() - 0.5, X3[:, 1].max() + 0.5, 250)
GX, GY = np.meshgrid(gx, gy)
G = add_bias(np.c_[GX.ravel(), GY.ravel()])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
shades = ["0.95", "0.8", "0.62"]; marks = ["o", "^", "s"]
regions = [None, predict_one_vs_rest(G, thetas), np.argmax(G @ Theta, axis=1)]
for a, reg, ttl in zip(ax, regions, ["training data", "one-vs-rest", "softmax"]):
    if reg is not None:
        a.contourf(GX, GY, reg.reshape(GX.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=shades)
    for c in range(3):
        a.scatter(*X3[y3 == c].T, c="white" if c == 0 else "k", edgecolors="k", marker=marks[c], s=9, lw=0.5, zorder=3)
    a.set_title(ttl, fontsize=9); a.set_xlabel(r"$x_1$")
ax[0].set_ylabel(r"$x_2$")
save(fig, "multiclass.pdf")
print("OvR train acc", round(np.mean(predict_one_vs_rest(B, thetas) == y3), 3),
      "softmax train acc", round(np.mean(np.argmax(B @ Theta, axis=1) == y3), 3))
