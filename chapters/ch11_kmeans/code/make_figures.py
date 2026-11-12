# ci: skip
"""Figures for Chapter 11 (grayscale-safe vector PDFs)."""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, load_sample_image
from sklearn.metrics import silhouette_score
from kmeans import sq_dists, lloyd, random_init, kmeanspp_init

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
MK = ["o", "s", "^", "D", "v", "P"]
GR = ["0.15", "0.45", "0.7", "0.3", "0.85", "0.55"]


def scatter(ax, X, lab, C=None, s=12):
    for j in np.unique(lab):
        m = lab == j
        ax.scatter(X[m, 0], X[m, 1], s=s, marker=MK[j % 6], c=GR[j % 6], edgecolor="k", linewidth=0.3)
    if C is not None:
        ax.scatter(C[:, 0], C[:, 1], s=110, marker="X", c="white", edgecolor="k", linewidth=1.3, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


X5, y5 = make_blobs(500, centers=5, cluster_std=[1.0, 2.5, 0.5, 1.5, 0.8], random_state=42)

# 1. Lloyd iterations on 3 blobs ------------------------------------------------------------------
X3, _ = make_blobs(240, centers=3, cluster_std=1.0, random_state=7)
C = X3[[0, 1, 2]].copy()
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.0))
C0 = C.copy()
for t, ax in enumerate(axs):
    D = sq_dists(X3, C0); lab = D.argmin(1); J = D.min(1).sum()
    scatter(ax, X3, lab, C0)
    ax.set_title(("init" if t == 0 else f"after update {t}") + f"\n$J$={J:.0f}", fontsize=8)
    C0 = np.array([X3[lab == j].mean(0) if (lab == j).any() else C0[j] for j in range(3)])
    if t == 0:
        pass
save(fig, "fig_lloyd_steps.pdf")

# 2. local minimum vs good solution ---------------------------------------------------------------
res = []
for s in range(60):
    Cr, lab, h = lloyd(X5, random_init(X5, 5, np.random.default_rng(s)))
    res.append((h[-1], s, Cr, lab))
res.sort(key=lambda r: r[0])
good, bad = res[0], res[-1]
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
scatter(axs[0], X5, good[3], good[2]); axs[0].set_title(f"best of 60 runs: $J$={good[0]:.0f}", fontsize=8.5)
scatter(axs[1], X5, bad[3], bad[2]); axs[1].set_title(f"worst of 60 runs: $J$={bad[0]:.0f}", fontsize=8.5)
save(fig, "fig_local_minima.pdf")

# 3. k-means++ probabilities ----------------------------------------------------------------------------
P = np.array([[1, 2], [2, 1], [5, 4], [6, 5], [9, 8]], float)
names = list("ABCDE")
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.3))
ax = axs[0]
ax.scatter(P[:, 0], P[:, 1], c="0.5", edgecolor="k", s=40)
for n, (x, y) in zip(names, P):
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 4))
ax.scatter(*P[0], s=140, marker="X", c="white", edgecolor="k", linewidth=1.3)
ax.set_title("points; first centre $=A$", fontsize=8.5); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
for ax, d2, ttl in [(axs[1], sq_dists(P, P[:1])[:, 0], "2nd centre: $D^2$ to $A$"),
                    (axs[2], sq_dists(P, P[[0, 4]]).min(1), "3rd centre (after $A$,$E$)")]:
    p = d2 / d2.sum()
    ax.bar(names, p, color="0.55", edgecolor="k")
    for i, v in enumerate(p):
        ax.text(i, v + 0.015, f"{v:.3f}" if v else "0", ha="center", fontsize=7.5)
    ax.set_ylim(0, 0.75); ax.set_title(ttl, fontsize=8.5); ax.set_ylabel("selection probability")
fig.tight_layout()
save(fig, "fig_kmeanspp.pdf")

# 4. choosing K ---------------------------------------------------------------------------------------------
ks = np.arange(1, 10)
inert = [KMeans(k, n_init=10, random_state=0).fit(X5).inertia_ for k in ks]
sil = [silhouette_score(X5, KMeans(k, n_init=10, random_state=0).fit_predict(X5)) for k in ks[1:]]
rng = np.random.default_rng(0)
lo, hi = X5.min(0), X5.max(0)
gap, sk = [], []
for k in ks:
    w = np.log(KMeans(k, n_init=5, random_state=0).fit(X5).inertia_)
    ref = [np.log(KMeans(k, n_init=3, random_state=0).fit(rng.uniform(lo, hi, X5.shape)).inertia_) for _ in range(15)]
    gap.append(np.mean(ref) - w); sk.append(np.std(ref) * np.sqrt(1 + 1 / 15))
fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.3))
axs[0].plot(ks, inert, "ko-", ms=4); axs[0].set_xlabel("$K$"); axs[0].set_ylabel("inertia $J$"); axs[0].set_title("elbow", fontsize=8.5)
axs[1].plot(ks[1:], sil, "ks-", ms=4); axs[1].axvline(ks[1:][int(np.argmax(sil))], ls=":", c="0.4")
axs[1].set_xlabel("$K$"); axs[1].set_ylabel("mean silhouette"); axs[1].set_title("silhouette", fontsize=8.5)
axs[2].errorbar(ks, gap, yerr=sk, fmt="k^-", ms=4, capsize=2); axs[2].set_xlabel("$K$"); axs[2].set_ylabel("Gap($K$)"); axs[2].set_title("gap statistic", fontsize=8.5)
for a in axs:
    a.set_xticks(range(1, 10))
fig.tight_layout()
save(fig, "fig_choose_k.pdf")

# 5. failure gallery --------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
cases = []
Xu = np.vstack([rng.normal([0, 0], 1.0, (500, 2)), rng.normal([2.0, 0], 0.5, (30, 2))]); cases.append(("unequal sizes", Xu, 2))
Xd = np.vstack([rng.normal([0, 0], 0.3, (300, 2)), rng.normal([3, 0], 1.5, (300, 2))]); cases.append(("unequal densities", Xd, 2))
Xa, _ = make_blobs(600, centers=[[0, 0], [2.5, 1.2], [5, 2.4]], cluster_std=0.5, random_state=3)
cases.append(("anisotropic", Xa @ np.array([[0.6, -0.64], [-0.4, 0.85]]), 3))
cases.append(("non-convex (moons)", make_moons(400, noise=0.07, random_state=0)[0], 2))
Xs, _ = make_blobs(400, centers=[[0, 0], [0, 4]], cluster_std=1.0, random_state=1)
Xs2 = Xs.copy(); Xs2[:, 0] *= 1000; cases.append(("unscaled features", Xs2, 2))
Xo, _ = make_blobs(300, centers=[[0, 0], [6, 0]], cluster_std=0.8, random_state=5)
cases.append(("one outlier at (150,150)", np.vstack([Xo, [[150, 150]]]), 2))
fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.6))
for ax, (ttl, X, k) in zip(axs.ravel(), cases):
    m = KMeans(k, n_init=10, random_state=0).fit(X)
    scatter(ax, X, m.labels_, m.cluster_centers_, s=8)
    ax.set_title(ttl, fontsize=8.5)
fig.tight_layout()
save(fig, "fig_failures.pdf")

# 6. colour quantisation ----------------------------------------------------------------------------------
img = load_sample_image("china.jpg") / 255.0
px = img.reshape(-1, 3)
sub = px[np.random.default_rng(0).choice(len(px), 10000, replace=False)]
fig, axs = plt.subplots(1, 4, figsize=(7.8, 2.0))
axs[0].imshow(img); axs[0].set_title("original (24 bit)", fontsize=8.5)
for ax, k in zip(axs[1:], (4, 16, 64)):
    m = KMeans(k, n_init=1, random_state=0).fit(sub)
    ax.imshow(m.cluster_centers_[m.predict(px)].reshape(img.shape)); ax.set_title(f"$K$={k} ({int(np.log2(k))} bit)", fontsize=8.5)
for ax in axs:
    ax.axis("off")
save(fig, "fig_quantization.pdf")

# 7. Voronoi cells --------------------------------------------------------------------------------------------
m = KMeans(4, n_init=10, random_state=0).fit(X5[:, :2])
gx, gy = np.meshgrid(np.linspace(X5[:, 0].min() - 1, X5[:, 0].max() + 1, 300), np.linspace(X5[:, 1].min() - 1, X5[:, 1].max() + 1, 300))
Z = m.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.contourf(gx, gy, Z, levels=np.arange(-0.5, 4, 1), colors=["0.97", "0.88", "0.78", "0.68"])
ax.contour(gx, gy, Z, levels=np.arange(0.5, 4, 1), colors="k", linewidths=0.8)
scatter(ax, X5, m.labels_, m.cluster_centers_, s=7)
ax.set_title("$K=4$: Voronoi cells of the centroids", fontsize=8.5)
save(fig, "fig_voronoi.pdf")
print("figures written to", os.path.abspath(OUT))
