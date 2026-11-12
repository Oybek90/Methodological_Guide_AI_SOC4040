# ci: skip
"""Generate the figures of Chapter 2 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

from linreg import cost, gradient_descent, gradient_descent_1d  # noqa: E402
from multivariate import A, gd, hist_s, price  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. Three problem types on synthetic data --------------------------------------------------
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
Xc, yc = make_blobs(n_samples=60, centers=[(-1.5, -1), (1.5, 1)], cluster_std=0.9, random_state=3)
for k, mk in zip((0, 1), ("o", "s")):
    axs[0].scatter(*Xc[yc == k].T, marker=mk, s=16, facecolor="none" if k == 0 else "0.3", edgecolor="k", lw=0.7)
axs[0].plot([-2.0, 2.0], [3.0, -3.0], "k--", lw=0.9); axs[0].set_ylim(-3.5, 3.5)
axs[0].set_title("Classification\n(labels: two classes)")
xr = rng.uniform(0, 10, 40)
yr = 2 + 0.8 * xr + rng.normal(0, 1.0, 40)
axs[1].scatter(xr, yr, s=14, color="0.3")
axs[1].plot([0, 10], [2, 10], "k-", lw=1.0)
axs[1].set_title("Regression\n(label: a number)")
Xu, _ = make_blobs(n_samples=90, centers=[(-2, 0), (2, 2), (2, -2)], cluster_std=0.7, random_state=5)
axs[2].scatter(*Xu.T, s=14, color="0.3")
for c, lab in (((-2, 0), "?"), ((2, 2), "?"), ((2, -2), "?")):
    axs[2].annotate("", xy=c, xytext=c)
axs[2].set_title("Clustering\n(no labels)")
for a in axs:
    a.set_xticks([]); a.set_yticks([])
save(fig, "problem_types.pdf")

# 2. Temperature data + three hypotheses -----------------------------------------------------
t = np.array([10.0, 15, 20, 25, 30]); f = 32 + 1.8 * t
fig, ax = plt.subplots(figsize=(4.4, 2.9))
ax.scatter(t, f, color="k", zorder=3, label="training examples")
xs = np.linspace(5, 40, 50)
for (a, b), ls in (((32, 1.8), "-"), ((20, 1.8), "--"), ((50, 1.0), ":")):
    ax.plot(xs, a + b * xs, "k" + ls, lw=1.0 if ls == "-" else 0.9, label=rf"$\theta_0={a},\ \theta_1={b}$")
ax.set_xlabel(r"temperature $x$ ($^\circ$C)"); ax.set_ylabel(r"$y$ ($^\circ$F)")
ax.legend(frameon=False, fontsize=7.5)
save(fig, "hypotheses.pdf")

# 3. Toy J(theta1) and gradient-descent steps ---------------------------------------------------
x3 = np.array([1.0, 2, 3]); y3 = x3.copy()
th = np.linspace(-0.2, 2.2, 200)
J = np.array([cost(0.0, v, x3, y3) for v in th])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
axs[0].plot(th, J, "k-", lw=1.0)
for v in (0, 0.5, 1, 1.5, 2):
    axs[0].plot(v, cost(0, v, x3, y3), "ko", ms=4)
    axs[0].annotate(f"{cost(0, v, x3, y3):.2f}", (v, cost(0, v, x3, y3)), textcoords="offset points", xytext=(4, 5), fontsize=7.5)
axs[0].set_xlabel(r"$\theta_1$"); axs[0].set_ylabel(r"$J(\theta_1)$")
path = gradient_descent_1d(x3, y3, alpha=0.1, steps=12, theta1=2.5)
p1 = [p[0] for p in path]
axs[1].plot(np.linspace(0, 2.6, 200), [cost(0, v, x3, y3) for v in np.linspace(0, 2.6, 200)], "0.6", lw=0.8)
axs[1].plot(p1, [cost(0, v, x3, y3) for v in p1], "ko-", ms=3, lw=0.7)
axs[1].set_xlabel(r"$\theta_1$"); axs[1].set_ylabel(r"$J(\theta_1)$")
axs[1].set_title(r"$\alpha=0.1$: steps shrink near the minimum", fontsize=8.5)
save(fig, "cost_toy.pdf")

# 4. Contours of J(theta0, theta1) for the temperature data with GD paths -------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
g0 = np.linspace(0, 60, 150); g1 = np.linspace(-0.5, 4.0, 150)
G0, G1 = np.meshgrid(g0, g1)
JJ = np.array([[cost(a, b, t, f) for a in g0] for b in g1])
raw = gradient_descent(t, f, alpha=0.001, steps=3000)
axs[0].contour(G0, G1, JJ, levels=np.geomspace(1, 2500, 12), colors="0.55", linewidths=0.6)
axs[0].plot([p[0] for p in raw], [p[1] for p in raw], "k-", lw=0.9)
axs[0].plot(*raw[0][:2], "ko", ms=4); axs[0].plot(32, 1.8, "k*", ms=9)
axs[0].set_xlabel(r"$\theta_0$"); axs[0].set_ylabel(r"$\theta_1$")
axs[0].set_title(r"raw inputs, $\alpha=0.001$, 3000 steps", fontsize=8.5)
mu, sd = t.mean(), t.std(); z = (t - mu) / sd
sc = gradient_descent(z, f, alpha=0.3, steps=30, theta0=0.0, theta1=0.0)
# draw in the (b0, b1) space of the scaled problem
b0 = np.linspace(0, 80, 150); b1 = np.linspace(-5, 30, 150)
B0, B1 = np.meshgrid(b0, b1)
JS = np.array([[cost(a, b, z, f) for a in b0] for b in b1])
axs[1].contour(B0, B1, JS, levels=np.geomspace(1, 2500, 12), colors="0.55", linewidths=0.6)
axs[1].plot([p[0] for p in sc], [p[1] for p in sc], "ko-", ms=2.5, lw=0.8)
axs[1].plot(68, 1.8 * sd, "k*", ms=9)
axs[1].set_xlabel(r"$\theta_0$ (scaled input)"); axs[1].set_ylabel(r"$\theta_1$ (scaled input)")
axs[1].set_title(r"standardised inputs, $\alpha=0.3$, 30 steps", fontsize=8.5)
save(fig, "contours_gd.pdf")
print("raw GD final:", [round(v, 3) for v in raw[-1]], " scaled GD final:", [round(v, 3) for v in sc[-1]])

# 5. Learning rate on the toy problem ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for a, ls in ((0.01, ":"), (0.1, "-"), (0.4, "--"), (0.5, "-.")):
    h = [p[1] for p in gradient_descent_1d(x3, y3, alpha=a, steps=30, theta1=2.5)]
    ax.plot(h, "k" + ls, lw=1.0, label=rf"$\alpha={a}$")
ax.set_yscale("log"); ax.set_ylim(1e-8, 1e6)
ax.set_xlabel("iteration"); ax.set_ylabel(r"cost $J$ (log scale)"); ax.legend(frameon=False, fontsize=8)
save(fig, "learning_rates.pdf")
for a in (0.01, 0.1, 0.4, 0.5):
    print("alpha", a, "J after 30:", f"{gradient_descent_1d(x3, y3, alpha=a, steps=30, theta1=2.5)[-1][1]:.3e}")

# 6. Multivariate: scaled vs raw features ---------------------------------------------------------
_, hraw = gd(A, price, alpha=1e-5, steps=200)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(hist_s, "k-", lw=1.0, label=r"standardised features, $\alpha=0.1$")
ax.plot(hraw, "k--", lw=1.0, label=r"raw features, $\alpha=10^{-5}$ (near the stability limit)")
ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel(r"cost $J$ (log scale)")
ax.legend(frameon=False, fontsize=8)
save(fig, "feature_scaling.pdf")
