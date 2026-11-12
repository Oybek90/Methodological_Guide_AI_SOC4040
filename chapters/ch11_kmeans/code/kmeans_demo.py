# ci: run
"""Numbers quoted in Chapter 11: hand example, seeding, local minima, choosing K, failures."""

import time
import warnings
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from kmeans import sq_dists, lloyd, random_init, kmeanspp_init

warnings.filterwarnings("ignore")

# ---- 1. Hand example: A(1,2) B(2,1) C(5,4) D(6,5) E(9,8), K = 2 -----
P = np.array([[1, 2], [2, 1], [5, 4], [6, 5], [9, 8]], float)
print("== hand example, K=2, start at A and E")
C, lab, hist = lloyd(P, P[[0, 4]])
print("final centres:", C.tolist(), "labels:", lab.tolist())
print("J per step:", [round(float(h), 3) for h in hist])
C0 = P[[0, 4]].copy()
for t in range(1, 4):  # print each step explicitly
    lab_t = sq_dists(P, C0).argmin(1)
    C1 = np.array([P[lab_t == j].mean(0) for j in range(2)])
    print(f" step {t}: labels {lab_t.tolist()} -> centres {np.round(C1, 3).tolist()}")
    C0 = C1

# ---- 2. k-means++ probabilities -------------------------------------
print("== k-means++ D^2 probabilities (first centre A)")
d2 = sq_dists(P, P[:1])[:, 0]
print("D^2:", d2.tolist(), "sum", d2.sum(), "p:", np.round(d2 / d2.sum(), 3).tolist())
d2 = sq_dists(P, P[[0, 4]]).min(1)
print(
    "after E: min D^2:",
    d2.tolist(),
    "sum",
    d2.sum(),
    "p:",
    np.round(d2 / d2.sum(), 3).tolist(),
)
print("dist^2 to E:", sq_dists(P, P[4:5])[:, 0].tolist())

# ---- 3. local minima on blobs ---------------------------------------
print("== local minima: random vs k-means++, 200 single runs, 5 blobs")
X5, y5 = make_blobs(500, centers=5, cluster_std=[1.0, 2.5, 0.5, 1.5, 0.8], random_state=42)
best = KMeans(5, n_init=50, random_state=0).fit(X5).inertia_
for name, f in [("random", random_init), ("k-means++", kmeanspp_init)]:
    J = np.array([lloyd(X5, f(X5, 5, np.random.default_rng(s)))[2][-1] for s in range(200)])
    print(
        f" {name:10s} best J={best:.1f}  runs within 1% of best: {np.mean(J < 1.01 * best):.0%}  "
        f"worst/best = {J.max() / best:.2f}  median/best = {np.median(J) / best:.3f}"
    )

# ---- 4. choosing K --------------------------------------------------
print("== choosing K on the 5-blob data (n_init=10)")
print(" k   inertia   silhouette   CH      DB")
for k in range(2, 9):
    m = KMeans(k, n_init=10, random_state=0).fit(X5)
    print(
        f" {k}  {m.inertia_:9.1f}  {silhouette_score(X5, m.labels_):.3f}      "
        f"{calinski_harabasz_score(X5, m.labels_):7.1f}  {davies_bouldin_score(X5, m.labels_):.3f}"
    )


def gap(X, k, B=20, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    Wk = np.log(KMeans(k, n_init=5, random_state=0).fit(X).inertia_)
    ref = [
        np.log(
            KMeans(k, n_init=5, random_state=0).fit(rng.uniform(lo, hi, X.shape)).inertia_
        )
        for _ in range(B)
    ]
    return np.mean(ref) - Wk, np.std(ref) * np.sqrt(1 + 1 / B)


gs = [gap(X5, k) for k in range(1, 9)]
print(" gap:", [round(float(g), 3) for g, _ in gs])
print(
    " smallest k with Gap(k) >= Gap(k+1) - s(k+1):",
    next(k for k in range(1, 8) if gs[k - 1][0] >= gs[k][0] - gs[k][1]),
)

# ---- 5. failure modes -----------------------------------------------
print("== failure modes (ARI against the true grouping)")
rng = np.random.default_rng(0)
Xu = np.vstack([rng.normal([0, 0], 1.0, (500, 2)), rng.normal([2.0, 0], 0.5, (30, 2))])
yu = np.r_[np.zeros(500), np.ones(30)]
print(
    " unequal sizes:",
    round(adjusted_rand_score(yu, KMeans(2, n_init=10, random_state=0).fit_predict(Xu)), 3),
)
Xd = np.vstack([rng.normal([0, 0], 0.3, (300, 2)), rng.normal([3, 0], 1.5, (300, 2))])
yd = np.r_[np.zeros(300), np.ones(300)]
print(
    " unequal densities:",
    round(adjusted_rand_score(yd, KMeans(2, n_init=10, random_state=0).fit_predict(Xd)), 3),
)
Xa, ya = make_blobs(
    600, centers=[[0, 0], [2.5, 1.2], [5, 2.4]], cluster_std=0.5, random_state=3
)
Xa = Xa @ np.array([[0.6, -0.64], [-0.4, 0.85]])
print(
    " anisotropic:",
    round(adjusted_rand_score(ya, KMeans(3, n_init=10, random_state=0).fit_predict(Xa)), 3),
)
Xm, ym = make_moons(400, noise=0.07, random_state=0)
print(
    " moons:",
    round(adjusted_rand_score(ym, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)), 3),
)
Xs, ys = make_blobs(400, centers=[[0, 0], [0, 4]], cluster_std=1.0, random_state=1)
Xs_big = Xs.copy()
Xs_big[:, 0] *= 1000  # feature 0 (uninformative) in huge units
print(
    " scaling: raw ARI",
    round(
        adjusted_rand_score(ys, KMeans(2, n_init=10, random_state=0).fit_predict(Xs_big)), 3
    ),
    " standardised ARI",
    round(
        adjusted_rand_score(
            ys,
            KMeans(2, n_init=10, random_state=0).fit_predict(
                StandardScaler().fit_transform(Xs_big)
            ),
        ),
        3,
    ),
)
Xo, yo = make_blobs(300, centers=[[0, 0], [6, 0]], cluster_std=0.8, random_state=5)
J_clean = KMeans(2, n_init=10, random_state=0).fit(Xo)
Xo2 = np.vstack([Xo, [[150, 150]]])
mo = KMeans(2, n_init=10, random_state=0).fit(Xo2)
print(
    " outlier: ARI on the 300 regular points clean",
    round(adjusted_rand_score(yo, J_clean.labels_), 3),
    "with one outlier",
    round(adjusted_rand_score(yo, mo.labels_[:300]), 3),
)
print(
    " distance concentration (n=500 uniform points): (max-min)/min of distances to a query"
)
for d in (2, 10, 100, 1000):
    Z = np.random.default_rng(1).random((500, d))
    dist = np.sqrt(sq_dists(Z[1:], Z[:1])[:, 0])
    print(f"   d={d:5d}: {(dist.max() - dist.min()) / dist.min():.3f}")

# ---- 6. digits, mini-batch, GMM -------------------------------------
print("== digits (n=1797, d=64, K=10)")
Xd_, yd_ = load_digits(return_X_y=True)
for init in ("random", "k-means++"):
    m = KMeans(10, init=init, n_init=10, random_state=0).fit(Xd_)
    print(
        f" {init:10s} inertia {m.inertia_:.0f}  "
        f"ARI {adjusted_rand_score(yd_, m.labels_):.3f}  "
        f"NMI {normalized_mutual_info_score(yd_, m.labels_):.3f}"
    )
Xbig, _ = make_blobs(500000, centers=50, n_features=50, cluster_std=3, random_state=0)
t = time.perf_counter()
f = KMeans(50, n_init=1, random_state=0).fit(Xbig)
tf = time.perf_counter() - t
t = time.perf_counter()
mb = MiniBatchKMeans(50, n_init=1, batch_size=4096, random_state=0).fit(Xbig)
tm = time.perf_counter() - t
print(
    f" 500k x 50, K=50: full {tf:.2f}s J={f.inertia_:.4g}; "
    f"mini-batch {tm:.2f}s J={mb.inertia_:.4g}; ratio J {mb.inertia_ / f.inertia_:.3f}"
)
print("== k-means vs GMM on anisotropic data (ARI)")
print(
    " k-means",
    round(adjusted_rand_score(ya, KMeans(3, n_init=10, random_state=0).fit_predict(Xa)), 3),
    " GMM",
    round(
        adjusted_rand_score(
            ya,
            GaussianMixture(
                3, covariance_type="full", n_init=5, random_state=0
            ).fit_predict(Xa),
        ),
        3,
    ),
)

# ---- 7. colour quantisation -----------------------------------------
try:
    from sklearn.datasets import load_sample_image

    img = load_sample_image("china.jpg") / 255.0
    px = img.reshape(-1, 3)
    sub = px[np.random.default_rng(0).choice(len(px), 10000, replace=False)]
    for k in (4, 16, 64):
        m = KMeans(k, n_init=1, random_state=0).fit(sub)
        err = np.mean((px - m.cluster_centers_[m.predict(px)]) ** 2)
        print(
            f" quantisation k={k:3d}: MSE per channel {err:.5f}  "
            f"bits/pixel {np.log2(k):.0f} (vs 24)"
        )
except Exception as e:
    print(" load_sample_image unavailable:", type(e).__name__)
