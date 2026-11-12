# ci: skip
"""Figures for Chapter 15 (grayscale-safe vector PDFs)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from lightcnn import params_separable, params_grouped, macs_grouped, macs_separable, batchnorm_forward
import torch_demo as td

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# 1. which input channels feed which output channels (C = K = 8) -----------------------------------
def conn(g, C=8, K=8):
    M = np.zeros((K, C))
    for i in range(K):
        gi = i // (K // g)
        M[i, gi * (C // g):(gi + 1) * (C // g)] = 1
    return M


fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.6))
panels = [(conn(1), "standard\n$g=1$: 576 weights"), (conn(2), "grouped\n$g=2$: 288 weights"),
          (conn(8), "depthwise\n$g=8$: 72 weights"), (conn(1), "pointwise $1\\times1$\n64 weights")]
for a, (M, t) in zip(ax, panels):
    a.imshow(M, cmap="Greys", vmin=0, vmax=1.6)
    a.set_xticks(range(8)); a.set_yticks(range(8)); a.set_xticklabels(range(1, 9), fontsize=6)
    a.set_yticklabels(range(1, 9), fontsize=6); a.set_title(t, fontsize=8)
    a.set_xlabel("input channel"); a.tick_params(length=0)
    for k in range(9):
        a.axhline(k - .5, color="0.8", lw=.4); a.axvline(k - .5, color="0.8", lw=.4)
ax[0].set_ylabel("output channel")
save(fig, "fig_groups.pdf")

# 2. cost of separable vs standard -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
K = np.arange(8, 513)
for f, ls in ((3, "-"), (5, "--"), (7, ":")):
    ax[0].plot(K, 1 / (1 / K + 1 / f ** 2), "k", ls=ls, label=f"$F={f}$")
ax[0].set_xlabel("output channels $K$"); ax[0].set_ylabel("cost standard / separable")
ax[0].legend(frameon=False, loc="lower right"); ax[0].set_title("saving factor $1/(1/K+1/F^2)$")
labels = ["standard", "grouped\n($g=32$)", "separable"]
macs = [macs_grouped(32, 64, 3, 16, 16), macs_grouped(32, 64, 3, 16, 16, 32), macs_separable(32, 64, 3, 16, 16)]
ax[1].bar(labels, macs, color=["0.15", "0.55", "0.8"])
for i, v in enumerate(macs):
    ax[1].text(i, v * 1.02, f"{v / 1e6:.2f}M", ha="center", fontsize=8)
ax[1].set_ylabel("multiply-adds"); ax[1].set_title("$32\\to64$ channels, $3\\times3$, $16\\times16$ map")
save(fig, "fig_cost.pdf")

# 3. batch normalization: distributions and training curves -------------------------------------------
rng = np.random.default_rng(0)
xb = np.concatenate([rng.normal(5, 3, size=(16, 1, 6, 6)), rng.normal(-2, 0.5, size=(16, 1, 6, 6))], axis=1)
yb = batchnorm_forward(xb, np.ones(2), np.zeros(2))[0]
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7))
for c, ls in ((0, "-"), (1, "--")):
    ax[0].hist(xb[:, c].ravel(), bins=30, histtype="step", color="k", ls=ls, label=f"channel {c + 1}")
    ax[1].hist(yb[:, c].ravel(), bins=30, histtype="step", color="k", ls=ls, label=f"channel {c + 1}")
ax[0].set_title("before BN"); ax[1].set_title("after BN ($\\gamma=1,\\beta=0$)")
ax[0].legend(frameon=False, fontsize=7)
for lr, ls in ((0.01, "-"), (0.1, "--")):
    for bn, col in ((False, "0.6"), (True, "0.0")):
        hs = []
        for s in range(3):
            torch.manual_seed(s)
            hs.append(td.train(td.make_net(bn=bn), lr, 15, opt="sgd", seed=s)[1])
        ax[2].plot(np.mean(hs, 0), color=col, ls=ls, label=f"{'BN' if bn else 'no BN'}, lr {lr}")
ax[2].set_yscale("log"); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("training loss")
ax[2].legend(frameon=False, fontsize=6.5, loc="lower left"); ax[2].set_title("plain SGD (mean of 3 seeds)")
save(fig, "fig_bn.pdf")

# 4. ImageNet winners (data checked against Russakovsky et al. 2015 and the ILSVRC overviews) --------
yrs = ["2010\nNEC-UIUC", "2011\nXRCE", "2012\nAlexNet", "2013\nClarifai", "2014\nGoogLeNet", "2015\nResNet",
       "2016\nTrimps-\nSoushen", "2017\nSENet"]
err = [28.2, 25.8, 15.3, 11.7, 6.7, 3.57, 2.99, 2.25]
fig, ax = plt.subplots(figsize=(8, 2.9))
cols = ["0.65", "0.65"] + ["0.2"] * 6
ax.bar(range(8), err, color=cols)
for i, v in enumerate(err):
    ax.text(i, v + .6, f"{v:g}", ha="center", fontsize=8)
ax.axhline(5.1, color="k", ls="--", lw=1); ax.text(4.6, 7.2, "human 5.1", ha="right", fontsize=8)
ax.set_xticks(range(8)); ax.set_xticklabels(yrs, fontsize=7); ax.set_ylabel("top-5 test error (%)")
save(fig, "fig_imagenet.pdf")
print("figures written to", os.path.normpath(OUT))
