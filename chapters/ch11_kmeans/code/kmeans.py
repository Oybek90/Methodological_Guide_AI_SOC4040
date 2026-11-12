# ci: run
"""K-means from scratch: Lloyd, k-means++, restarts, silhouette (NumPy only)."""

import numpy as np


def sq_dists(X, C):
    """Squared Euclidean distances, shape (n, k), without building an (n, k, d) array."""
    return (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]


def random_init(X, k, rng):
    """Forgy initialisation: k distinct data points chosen uniformly."""
    return X[rng.choice(len(X), k, replace=False)].copy()


def kmeanspp_init(X, k, rng):
    """k-means++ seeding: next centre drawn with probability proportional to D(x)^2."""
    n = len(X)
    C = [X[rng.integers(n)]]
    d2 = sq_dists(X, np.array(C))[:, 0].clip(0)
    for _ in range(1, k):
        C.append(X[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, sq_dists(X, np.array(C[-1:]))[:, 0].clip(0))
    return np.array(C)


def lloyd(X, C, max_iter=300, tol=1e-9):
    """Alternate assignment and update; returns centres, labels and the J history."""
    C = C.copy()
    hist = []
    for _ in range(max_iter):
        D = sq_dists(X, C)
        lab = D.argmin(1)
        hist.append(D[np.arange(len(X)), lab].sum())  # J before the update
        newC = C.copy()
        for j in range(len(C)):
            m = lab == j
            if m.any():
                newC[j] = X[m].mean(0)
            else:  # empty cluster: move it to the worst-fitted point
                newC[j] = X[D[np.arange(len(X)), lab].argmax()]
        if ((newC - C) ** 2).sum() <= tol:
            C = newC
            break
        C = newC
    D = sq_dists(X, C)
    lab = D.argmin(1)
    hist.append(D[np.arange(len(X)), lab].sum())
    return C, lab, hist


def kmeans(X, k, init="k-means++", n_init=10, seed=0, max_iter=300):
    """Best of n_init runs by inertia. Returns centres, labels, inertia."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C0 = kmeanspp_init(X, k, rng) if init == "k-means++" else random_init(X, k, rng)
        C, lab, hist = lloyd(X, C0, max_iter)
        if best is None or hist[-1] < best[2]:
            best = (C, lab, hist[-1])
    return best


def silhouette_samples(X, lab):
    """s(i) = (b - a) / max(a, b); a = mean distance within the own cluster, b = smallest
    mean distance to another cluster."""
    D = np.sqrt(sq_dists(X, X).clip(0))
    ks = np.unique(lab)
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        if own.sum() == 1:
            continue  # singleton: s = 0 by convention
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, lab == j].mean() for j in ks if j != lab[i])
        s[i] = (b - a) / max(a, b)
    return s


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_samples as sk_sil

    X, _ = make_blobs(300, centers=4, cluster_std=0.9, random_state=1)
    C, lab, J = kmeans(X, 4, n_init=10, seed=0)
    sk = KMeans(4, n_init=10, random_state=0).fit(X)
    print(f"ours J = {J:.4f}   sklearn inertia = {sk.inertia_:.4f}")
    assert abs(J - sk.inertia_) < 1e-6 * J
    assert np.allclose(silhouette_samples(X, lab), sk_sil(X, lab))
    _, _, h = lloyd(X, random_init(X, 4, np.random.default_rng(3)))
    assert all(a >= b - 1e-9 for a, b in zip(h, h[1:])), "J must never increase"
    print(
        "J history:",
        np.round(h, 2),
        "-> non-increasing; silhouettes agree with scikit-learn",
    )
