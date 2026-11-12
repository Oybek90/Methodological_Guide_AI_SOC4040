# ci: skip
"""Figures for Chapter 14 (grayscale-safe vector PDFs)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.datasets import load_digits
from cnn_numpy import conv2d, pool2d

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def grid(ax, M, fmt="%d", hl=None, shade=None, fs=8):
    h, w = M.shape
    ax.imshow(shade if shade is not None else np.zeros_like(M), cmap="Greys", vmin=0, vmax=3, extent=(0, w, h, 0))
    for i in range(h):
        for j in range(w):
            ax.text(j + .5, i + .5, fmt % M[i, j], ha="center", va="center", fontsize=fs)
    for k in range(h + 1):
        ax.plot([0, w], [k, k], color="0.5", lw=.5)
    for k in range(w + 1):
        ax.plot([k, k], [0, h], color="0.5", lw=.5)
    if hl:
        ax.add_patch(Rectangle(hl[:2], hl[2], hl[3], fill=False, lw=2, ec="k"))
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")


# 1. the 7x7 example of the lecture ----------------------------------------------------------
X = np.array([[2, 1, 1, 2, 4, 2, 1], [3, 2, 3, 2, 3, 1, 1], [1, 1, 2, 2, 2, 3, 3], [1, 2, 1, 3, 1, 2, 1],
              [2, 2, 2, 2, 2, 0, 2], [3, 2, 3, 1, 3, 2, 1], [2, 3, 2, 3, 1, 2, 1]], dtype=float)
K = np.array([[1, 2, 1], [0, 0, 0], [1, 2, 1]], dtype=float)
Y = conv2d(X[None, None], K[None, None], pad=1)[0, 0]
fig, ax = plt.subplots(1, 3, figsize=(8, 2.9), gridspec_kw={"width_ratios": [9, 3, 7]})
grid(ax[0], np.pad(X, 1), hl=(0, 0, 3, 3)); ax[0].set_title("input, zero padding $P=1$")
grid(ax[1], K, shade=np.ones((3, 3)) * 0.4); ax[1].set_title("filter $3\\times3$")
grid(ax[2], Y, hl=(0, 0, 1, 1)); ax[2].set_title("output $7\\times7$ (first cell $=8$)")
save(fig, "fig_conv_example.pdf")

# 2. filters on a digit --------------------------------------------------------------------------
img = load_digits().images[3] / 16.0
filt = {"original": None, "vertical edges": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "horizontal edges": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), "blur": np.ones((3, 3)) / 9}
fig, ax = plt.subplots(1, 4, figsize=(8, 2.3))
for a, (name, f) in zip(ax, filt.items()):
    out = img if f is None else conv2d(img[None, None], f[None, None].astype(float), pad=1)[0, 0]
    a.imshow(out, cmap="Greys"); a.set_title(name, fontsize=9); a.axis("off")
save(fig, "fig_filters.pdf")

# 3. parameter counts ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].bar(["fully connected", "convolution"], [31467520, 760], color=["0.55", "0.15"])
ax[0].set_yscale("log"); ax[0].set_ylabel("parameters")
for i, v in enumerate([31467520, 760]):
    ax[0].text(i, v * 1.4, f"{v:,}", ha="center", fontsize=8)
ax[0].set_ylim(1, 3e9); ax[0].set_title("$32\\times32\\times3\\to32\\times32\\times10$")
names = ["standard", "grouped\n(g=32)", "separable"]
vals = [18496, 640, 2432]
ax[1].bar(names, vals, color=["0.15", "0.55", "0.8"])
for i, v in enumerate(vals):
    ax[1].text(i, v + 300, f"{v:,}", ha="center", fontsize=8)
ax[1].set_ylabel("parameters"); ax[1].set_title("$32\\to64$ channels, $3\\times3$")
save(fig, "fig_params.pdf")

# 4. max / average pooling ------------------------------------------------------------------------
P = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]], dtype=float)
mx, av = pool2d(P[None, None], 2, 2, "max")[0, 0], pool2d(P[None, None], 2, 2, "avg")[0, 0]
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.4), gridspec_kw={"width_ratios": [4, 2, 2]})
grid(ax[0], P, hl=(0, 0, 2, 2)); ax[0].set_title("input $4\\times4$")
grid(ax[1], mx, shade=np.ones((2, 2)) * 0.4); ax[1].set_title("max pool $2\\times2$, $S=2$")
grid(ax[2], av, fmt="%.2f", shade=np.ones((2, 2)) * 0.4); ax[2].set_title("average pool")
save(fig, "fig_pooling.pdf")

# 5. receptive field of stacked 3x3 convolutions ----------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.3))
for a, r in zip(ax, (3, 5, 7)):
    a.add_patch(Rectangle((0, 0), 9, 9, fill=False, ec="0.6"))
    o = (9 - r) / 2
    a.add_patch(Rectangle((o, o), r, r, fc="0.8", ec="k", lw=1.5))
    a.add_patch(Rectangle((4, 4), 1, 1, fc="k"))
    a.set_xlim(-.5, 9.5); a.set_ylim(9.5, -.5); a.set_aspect("equal"); a.axis("off")
    a.set_title(f"{(r - 1) // 2} conv $3\\times3$: {r}$\\times${r}", fontsize=9)
save(fig, "fig_receptive.pdf")

# 6. LeNet-5 shape pipeline -------------------------------------------------------------------------
stages = [("input\n$1\\times28\\times28$", 28, 1), ("conv 5$\\times$5\n$6\\times24\\times24$", 24, 6),
          ("avg pool 2\n$6\\times12\\times12$", 12, 6), ("conv 5$\\times$5\n$16\\times8\\times8$", 8, 16),
          ("avg pool 2\n$16\\times4\\times4$", 4, 16)]
fig, ax = plt.subplots(figsize=(8.4, 2.6))
x0 = 0
for lab, s, c in stages:
    w, h = 0.04 * c + 0.3, s / 14
    for k in range(min(c, 4)):
        ax.add_patch(Rectangle((x0 + k * .08, -h / 2 + k * .08), .6, h, fc=str(0.9 - 0.12 * k), ec="k", lw=.8))
    ax.text(x0 + .3, -1.55, lab, ha="center", fontsize=7)
    x0 += 1.75
for lab, n in [("fc 256\n$\\to$120", 120), ("fc 120\n$\\to$84", 84), ("fc 84\n$\\to$10", 10)]:
    ax.add_patch(Rectangle((x0, -n / 150), .25, n / 75, fc="0.6", ec="k", lw=.8))
    ax.text(x0 + .12, -1.55, lab, ha="center", fontsize=7)
    x0 += 1.1
ax.set_xlim(-.3, x0); ax.set_ylim(-1.9, 1.2); ax.axis("off")
save(fig, "fig_lenet.pdf")
print("figures written to", os.path.normpath(OUT))
