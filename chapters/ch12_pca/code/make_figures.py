# ci: skip
"""Generate all figures of Chapter 12 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from pca_classification import run  # noqa: E402
from pca_scratch import PCAScratch  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. Toy example: four points, PC1 axis, orthogonal projections ---------------
X = np.array([[4, 11], [8, 4], [13, 5], [7, 14]], dtype=float)
p = PCAScratch(1).fit(X)
R = p.inverse_transform(p.transform(X))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
t = np.linspace(-9, 9, 2)
line = p.mean_ + t[:, None] * p.components_[0]
ax.plot(line[:, 0], line[:, 1], "k-", lw=1, label="PC1 axis")
for a, b in zip(X, R):
    ax.plot([a[0], b[0]], [a[1], b[1]], "k:", lw=0.8)
ax.plot(*X.T, "ko", ms=6, label="data points")
ax.plot(*R.T, "ks", ms=5, mfc="white", label="projections")
ax.plot(*p.mean_, "k+", ms=10, mew=1.4, label="mean")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.set_xlim(0, 16); ax.set_ylim(0, 16); ax.legend(loc="lower right", fontsize=7.5, frameon=False)
save(fig, "toy_projection.pdf")

# 2. Scree + cumulative variance on digits --------------------------------------
Xd, yd = load_digits(return_X_y=True)
full = PCA(svd_solver="full").fit(Xd)
ev = full.explained_variance_ratio_
cum = np.cumsum(ev)
ks = {q: int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99)}
print("PC1, PC2 ratio:", ev[:2].round(4), "cum2:", cum[1].round(4), "ks:", ks)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
a1.plot(range(1, 65), ev, "ko-", ms=3, lw=0.8)
a1.set_xlabel("component index $i$"); a1.set_ylabel("explained variance ratio")
a2.plot(range(1, 65), cum, "k-", lw=1.2)
for q, ls in zip(ks, ("--", "-.", ":")):
    a2.axhline(q, color="gray", ls=ls, lw=0.8)
    a2.plot([ks[q]], [q], "ko", ms=4)
    a2.annotate(f"{int(q*100)}%: k={ks[q]}", (ks[q] + 1, q - 0.06), fontsize=7.5)
a2.set_xlabel("number of components $k$"); a2.set_ylabel("cumulative ratio")
a2.set_ylim(0, 1.02)
save(fig, "scree_cumulative.pdf")

# 3. Mean digit and first eight principal axes as images ------------------------
fig, axs = plt.subplots(1, 9, figsize=(7.2, 1.0))
axs[0].imshow(full.mean_.reshape(8, 8), cmap="gray_r"); axs[0].set_title("mean", fontsize=7)
for i in range(8):
    axs[i + 1].imshow(full.components_[i].reshape(8, 8), cmap="gray_r")
    axs[i + 1].set_title(f"PC{i+1}", fontsize=7)
for a in axs:
    a.set_xticks([]); a.set_yticks([])
save(fig, "digit_components.pdf")

# 4. Reconstruction quality ------------------------------------------------------
sel = [0, 1, 2, 3]
ks_r = [2, 5, 15, 30]
fig, axs = plt.subplots(len(sel), len(ks_r) + 1, figsize=(4.6, 3.6))
for r, idx in enumerate(sel):
    axs[r, 0].imshow(Xd[idx].reshape(8, 8), cmap="gray_r")
    for c, k in enumerate(ks_r):
        m = PCA(n_components=k, svd_solver="full").fit(Xd)
        rec = m.inverse_transform(m.transform(Xd[idx:idx + 1]))
        axs[r, c + 1].imshow(rec.reshape(8, 8), cmap="gray_r")
for c, title in enumerate(["original"] + [f"k={k}" for k in ks_r]):
    axs[0, c].set_title(title, fontsize=8)
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
save(fig, "reconstruction.pdf")
rmse = []
for k in (2, 5, 15, 30, 64):
    m = PCA(n_components=k, svd_solver="full").fit(Xd)
    rmse.append((k, float(np.sqrt(((Xd - m.inverse_transform(m.transform(Xd))) ** 2).mean()))))
print("RMSE per pixel:", [(k, round(v, 3)) for k, v in rmse])

# 5. 2-D embedding of four digit classes -----------------------------------------
Z = PCA(n_components=2, svd_solver="full").fit_transform(Xd)
fig, ax = plt.subplots(figsize=(4.4, 3.6))
for cls, mk, gray in ((0, "o", "0.0"), (1, "^", "0.45"), (4, "s", "0.7"), (7, "x", "0.0")):
    m = yd == cls
    ax.scatter(Z[m, 0], Z[m, 1], marker=mk, s=14, linewidths=0.8, c=gray if mk == "x" else "white",
               edgecolors=gray, label=f"digit {cls}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7.5, frameon=False)
save(fig, "digits_2d.pdf")

# 6. Multi-seed classification: with / without PCA, with / without scaling -----------
res = {s: [run(i, scale=s) for i in range(10)] for s in (False, True)}
labels, means, stds = [], [], []
for scale in (False, True):
    for m in ("SVM", "RF"):
        for tag in ("raw", "pca"):
            v = np.array([r[f"{m}-{tag}"] for r in res[scale]])
            labels.append(f"{m}\n{tag}" + ("\n+scale" if scale else ""))
            means.append(v.mean()); stds.append(v.std(ddof=1))
            print(scale, m, tag, round(v.mean(), 3), round(v.std(ddof=1), 3))
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = np.arange(len(labels))
hatch = ["", "///"] * 4
ax.bar(x, means, yerr=stds, color="white", edgecolor="black", hatch=hatch, capsize=2.5, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("test accuracy (10 seeds)"); ax.set_ylim(0.4, 1.0)
ax.axvline(3.5, color="gray", lw=0.6, ls=":")
save(fig, "seed_comparison.pdf")
print("done")
