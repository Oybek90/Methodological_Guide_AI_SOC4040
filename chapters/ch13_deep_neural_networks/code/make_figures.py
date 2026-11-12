# ci: skip
"""Figures for Chapter 13 (grayscale-safe vector PDFs)."""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from nn_numpy import MLP, ACT, sigmoid

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# 1. activation functions and derivatives -----------------------------------------------------
z = np.linspace(-5, 5, 400)
names = ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "selu", "swish"]
styles = [("0.0", "-"), ("0.0", "--"), ("0.35", "-"), ("0.35", "--"), ("0.6", "-"), ("0.6", "--"), ("0.0", ":")]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
for n, (c, ls) in zip(names, styles):
    axs[0].plot(z, ACT[n][0](z), c=c, ls=ls, lw=1.4, label=n.replace("_", " "))
    axs[1].plot(z, ACT[n][1](z), c=c, ls=ls, lw=1.4)
axs[0].set_ylim(-2, 4); axs[0].set_title("$g(z)$", fontsize=9)
axs[1].set_ylim(-0.2, 1.8); axs[1].set_title("$g'(z)$", fontsize=9)
for a in axs:
    a.axhline(0, c="0.8", lw=0.6); a.axvline(0, c="0.8", lw=0.6); a.set_xlabel("$z$")
axs[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
save(fig, "fig_activations.pdf")

# 2. the 2-2-1 network of the slides with the numbers of the worked example ----------------------
fig, ax = plt.subplots(figsize=(7.0, 3.3))
pos = {"x1": (0, 1.2), "x2": (0, -1.2), "a1": (3.4, 1.2), "a2": (3.4, -1.2), "y": (6.8, 0)}
edges = [("x1", "a1", "$w_{11}=0.2$", 0.5, 0.18), ("x1", "a2", "$w_{12}=-0.5$", 0.27, -0.0), ("x2", "a1", "$w_{13}=0.4$", 0.27, 0.0),
         ("x2", "a2", "$w_{14}=0.3$", 0.5, -0.18), ("a1", "y", "$w_{21}=0.6$", 0.45, 0.25), ("a2", "y", "$w_{22}=-0.1$", 0.45, -0.25)]
for u, v, lab, t, dy in edges:
    (x0, y0), (x1_, y1) = pos[u], pos[v]
    ax.plot([x0, x1_], [y0, y1], c="0.5", lw=1, zorder=1)
    ax.text(x0 + t * (x1_ - x0), y0 + t * (y1 - y0) + dy, lab, fontsize=7.5, ha="center", va="center", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.8))
txt = {"x1": "$x_1$\n$1$", "x2": "$x_2$\n$2$", "a1": "$a_1=g(z_1)$\n$z_1=1.10$\n$a_1=1.10$", "a2": "$a_2=g(z_2)$\n$z_2=0.30$\n$a_2=0.30$",
       "y": "$\\hat y=0.68$\n$L=0.1024$"}
for k, (x, y) in pos.items():
    r = 0.55 if k.startswith("x") else 0.85
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec="k", zorder=3))
    ax.text(x, y, txt[k], ha="center", va="center", fontsize=7.5, zorder=4)
ax.text(3.4, 2.6, "biases $b_{11}=0.1,\\ b_{12}=0.2$", ha="center", fontsize=7.5)
ax.text(6.8, 1.3, "bias $b_2=0.05$", ha="center", fontsize=7.5)
ax.text(0, -2.4, "input layer", ha="center"); ax.text(3.4, -2.4, "hidden layer (ReLU)", ha="center"); ax.text(6.8, -2.4, "output (linear)", ha="center")
ax.set_xlim(-1, 7.9); ax.set_ylim(-2.7, 3.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "fig_network221.pdf")

# 3. XOR: input space and hidden space --------------------------------------------------------------
P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
lab = np.array([0, 1, 1, 0])
H = np.maximum(0, P @ np.array([[1, 1], [1, 1]]) + np.array([0, -1]))
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for a, pts, ttl, xl, yl in ((axs[0], P, "input space: no line separates", "$x_1$", "$x_2$"), (axs[1], H, "hidden space $(a_1,a_2)$: separable", "$a_1$", "$a_2$")):
    for c, mk, fc in ((0, "o", "white"), (1, "s", "0.2")):
        m = lab == c
        a.scatter(pts[m, 0], pts[m, 1], s=90, marker=mk, c=fc, edgecolor="k", zorder=3, label=f"XOR = {c}")
    a.set_title(ttl, fontsize=8.5); a.set_xlabel(xl); a.set_ylabel(yl)
axs[1].plot([-0.2, 2.2], [0.4 + 0.0, 0.4 + 0.0 + 0], alpha=0)           # keep limits
axs[1].plot([0.0, 2.2], [0.0 + 0.0, 0.0], alpha=0)
xx = np.linspace(-0.2, 2.2, 10)
axs[1].plot(xx, 0.5 * xx - 0.25 + 0.0, c="k", ls="--", lw=1)          # a2 = (a1 - 0.5)/2 separates (0,0),(2,1) from (1,0)
axs[1].set_xlim(-0.3, 2.3); axs[1].set_ylim(-0.3, 1.3)
axs[0].legend(fontsize=7, frameon=False, loc="center")
fig.tight_layout()
save(fig, "fig_xor.pdf")

# 4. decision boundaries on two moons ------------------------------------------------------------------
Xm, ym = make_moons(1000, noise=0.2, random_state=0)
Ym = ym.reshape(-1, 1)
gx, gy = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.5, 2, 250))
grid = np.c_[gx.ravel(), gy.ravel()]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5))
for a, (act, ttl) in zip(axs, (("identity", "no activation"), ("sigmoid", "sigmoid"), ("relu", "ReLU"))):
    net = MLP([2, 16, 16, 1], act=act, out="sigmoid", init="he" if act == "relu" else "xavier", seed=0)
    net.fit(Xm, Ym, epochs=150, lr=0.05, batch=32)
    acc = np.mean((net.predict(Xm) >= 0.5) == Ym)
    Z = net.predict(grid).reshape(gx.shape)
    a.contourf(gx, gy, Z, levels=[0, 0.5, 1], colors=["0.92", "0.7"])
    a.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1)
    for c, mk, fc in ((0, "o", "white"), (1, "s", "0.2")):
        a.scatter(Xm[ym == c, 0], Xm[ym == c, 1], s=6, marker=mk, c=fc, edgecolor="k", linewidth=0.3)
    a.set_title(f"{ttl}: train acc. {acc:.3f}", fontsize=8.5); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
save(fig, "fig_moons.pdf")

# 5. vanishing gradients ---------------------------------------------------------------------------------
rng = np.random.default_rng(0)
Xd = rng.normal(size=(256, 20))
Yd = rng.integers(0, 2, (256, 1)).astype(float)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for (act, init), (c, ls, mk) in zip((("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "he")), (("0.0", "-", "o"), ("0.45", "--", "s"), ("0.0", ":", "^"))):
    n = MLP([20] + [50] * 10 + [1], act=act, out="sigmoid", init=init, seed=1)
    gW, _ = n.gradients(Xd, Yd)
    axs[0].semilogy(range(1, 12), [np.abs(w).mean() for w in gW], c=c, ls=ls, marker=mk, ms=4, label=f"{act} ({init})")
axs[0].set_xlabel("layer (1 = nearest the input)"); axs[0].set_ylabel("mean $|\\partial L/\\partial W|$"); axs[0].legend(fontsize=7, frameon=False)
axs[0].set_title("gradient size per layer at initialisation", fontsize=8.5)
Xi = rng.normal(size=(1000, 100))
for (name, init), (c, ls, mk) in zip((("N(0, 0.01$^2$)", 0.01), ("Xavier", "xavier"), ("He", "he")), (("0.6", "-", "o"), ("0.0", "--", "s"), ("0.0", "-", "^"))):
    n = MLP([100] * 11, act="relu", out="linear", init=init, seed=0)
    A, _ = n.forward(Xi)
    axs[1].semilogy(range(0, 10), [a.std() for a in A[:10]], c=c, ls=ls, marker=mk, ms=4, label=name)
axs[1].set_xlabel("layer"); axs[1].set_ylabel("std of activations"); axs[1].legend(fontsize=7, frameon=False)
axs[1].set_title("ReLU network, 100 units per layer", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_vanishing.pdf")

# 6. why cross-entropy: gradient with respect to the logit -----------------------------------------------
zz = np.linspace(-8, 8, 400)
p = sigmoid(zz)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
axs[0].plot(p, -np.log(np.clip(p, 1e-12, 1)), c="k", label="cross-entropy $-\\log\\hat y$")
axs[0].plot(p, (1 - p) ** 2, c="0.5", ls="--", label="squared error $(1-\\hat y)^2$")
axs[0].set_xlabel("predicted probability $\\hat y$ (target $y=1$)"); axs[0].set_ylabel("loss"); axs[0].set_ylim(0, 4); axs[0].legend(fontsize=7, frameon=False)
axs[1].plot(zz, p - 1, c="k", label="cross-entropy: $\\hat y-1$")
axs[1].plot(zz, 2 * (p - 1) * p * (1 - p), c="0.5", ls="--", label="squared error: $2(\\hat y-1)\\hat y(1-\\hat y)$")
axs[1].set_xlabel("logit $z$"); axs[1].set_ylabel("$\\partial L/\\partial z$"); axs[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
fig.tight_layout()
save(fig, "fig_loss_gradients.pdf")

# 7. approximation with one hidden layer --------------------------------------------------------------------
xs = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
ys = np.sin(xs)
fig, axs = plt.subplots(1, 4, figsize=(7.8, 2.0), sharey=True)
for a, h in zip(axs, (2, 4, 8, 32)):
    best = None
    for s in range(3):
        n = MLP([1, h, 1], act="relu", seed=s)
        L = n.fit(xs, ys, 3000, 0.02, opt="adam")[-1]
        if best is None or L < best[0]:
            best = (L, n)
    a.plot(xs, ys, c="0.6", lw=2); a.plot(xs, best[1].predict(xs), c="k", lw=1.2)
    a.set_title(f"$h={h}$, MSE {best[0]:.5f}", fontsize=8); a.set_xticks([])
fig.tight_layout()
save(fig, "fig_approximation.pdf")
print("figures written to", os.path.abspath(OUT))
